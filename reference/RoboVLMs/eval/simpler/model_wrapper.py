from typing import Optional, Sequence, Callable
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2 as cv
import json
from transforms3d.euler import euler2axangle
import sys
sys.path.append("/remote-home/jinminghao/structvla/reference/RoboVLMs")

from robovlms.train.base_trainer import BaseTrainer
from eval.calvin.model_wrapper import CustomModel
from queue import Queue
from PIL import Image


from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import sys
sys.path.append("/remote-home/jinminghao/structvla/reference/Emu3")
from emu3.mllm import Emu3Tokenizer, Emu3ForCausalLM, Emu3Processor
from emu3.mllm import Emu3MoE
from transformers import LogitsProcessor
from emu3.mllm.configuration_emu3 import Emu3Config, Emu3MoEConfig
class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        """
        :param allowed_token_ids: List of allowed token IDs
        """
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        # Build a mask: allowed token positions are True, others are False
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            mask[self.allowed_token_ids] = True
        else:
            mask[:, self.allowed_token_ids] = True
        
        # Set logits of disallowed tokens to negative infinity
        scores[~mask] = -float("inf")
        return scores

class EmuVLAInference(CustomModel):
    def __init__(
        self,
        emu_hub,
        vq_hub,
        vision_hub,
        device,
        policy_setup="widowx_bridge",
    ):

        self.emu_hub = emu_hub
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.device = device
        self.policy_setup = policy_setup

        if self.policy_setup == "google_robot":
            self.close_gripper_act = -1
            self.image_size = (160, 128)
        elif self.policy_setup == "widowx_bridge":
            self.close_gripper_act = 1
            self.image_size = (256,256)
        
        self.sticky_gripper_num_repeat = 2
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.close_gripper_num = 0
        self.late_close_gripper = 1 # 2

        ## hard code here
        #new! ========================================================================================================
        self.use_history = True
        self.use_action_history = True        
        self.window_size = 4
        self.predict_action_frames = 5
        self.context_frames = 1
        self.predict_frames = 1
        self.action_dim = 7
        self.use_gripper = False
        self.use_fast = True
        self.use_one_step = False
        self.eoa_token_id = 151845
        self.use_cot = False

        self.video_mode = True
    
        # load model and tokenizer
        self.init_config(device=device)
        self.image_processor.min_pixels = 80 * 80

        if self.use_cot:
            self.kwargs = dict(
                mode='VLA_COT',
                padding="longest",
            )
        else:
            self.kwargs = dict(
                mode='VLA',
                padding="longest",
            )
        if self.use_fast:
            self.GENERATION_CONFIG = GenerationConfig(
                    pad_token_id=self.model.config.pad_token_id,
                    bos_token_id=self.model.config.bos_token_id,
                    eos_token_id=self.eoa_token_id,
                    do_sample=False,
                )
        
        else:
            self.GENERATION_CONFIG = GenerationConfig(
                use_cache=True,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                max_new_tokens=800, # hard code here
                do_sample=True,
                top_k=2048,
                temperature=0.8,
            )


    def init_config(self, device):
        # 1) Config: raise the upper limit and enable RoPE extrapolation
        cfg = Emu3Config.from_pretrained(self.emu_hub)
        cfg.max_position_embeddings = 4400          # 5500 is also possible, but 3k~4k is a safer first step

        self.model = Emu3MoE.from_pretrained(
            self.emu_hub,
            config=cfg,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        #new    =======================================================================================================
        #self.model.config.max_position_embeddings = 5500

        self.model.to(device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.vq_hub,
            #model_max_length=self.model.config.max_position_embeddings,
            model_max_length=cfg.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True).to(device).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)

        # fast tokenization
        if self.policy_setup == "widowx_bridge":
            fast_path = "/remote-home/jinminghao/structvla/pretrain/fast_bridge_t5_s50"
        elif self.policy_setup == "google_robot":
            fast_path = "/remote-home/jinminghao/structvla/pretrain/fast_google_a5_s50"
        else:
            fast_path = "/remote-home/jinminghao/structvla/pretrain/fast"
        self.action_tokenizer = AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0

        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

    
    def add_image(self, image):
        if self.vision_queue.full():
            self.vision_queue.get()
        self.vision_queue.put(image)
    
    def get_history(self):
        return list(self.vision_queue.queue) 

    def add_action(self, action):
        if self.action_queue.full():
            self.action_queue.get()
        self.action_queue.put(action)
    
    def get_action_history(self):
        return list(self.action_queue.queue)


    def reset(self):

        self.rgb_list = []
        self.hand_rgb_list = []
        self.rollout_step_counter = 0
        self.action_hist_list = []

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.close_gripper_num = 0
        self.previous_gripper_action = None

        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()

    def preprocess(self, image):
        # preprocess image
        agent_view = image
        agent_view = Image.fromarray(agent_view)
        agent_view = agent_view.resize(self.image_size)
        image_x = self.image_processor(agent_view, return_tensors="pt")["pixel_values"].cuda()
        image_code = self.image_tokenizer.encode(image_x)

        gripper_code = None

        return (
            image_code,
            gripper_code,
        )

    def step(self, image, goal):
        input_dict = dict()
      
        image_code, gripper_code = self.preprocess(image)

        prompt,neg_prompt = goal,""

        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + prompt]
        text_tokens = self.processor.tokenizer(text_prompt)
        
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type='pt')

        if self.video_mode:
            kwargs = dict(
                    mode='VLA_Video',
                    padding="longest",
                )
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **kwargs)
        else:
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **self.kwargs)
        #new! ========================================================================================================
        if self.video_mode and self.use_history:
            self.add_image(pos_inputs)
            
            # Get history images and actions
            history = self.get_history()
            action_history = self.get_action_history()

            # Initialize input_ids, token_type_ids, and attention_mask
            all_input_ids = []
            all_token_type_ids = []
            all_attention_mask = []

            # Add text
            all_input_ids.append(text_tokens['input_ids'])
            all_token_type_ids.append(text_tokens['token_type_ids'])
            all_attention_mask.append(text_tokens['attention_mask'])

            # Iterate over history images
            for i in range(len(history)):
                img_input_ids = history[i]['input_ids']
                img_token_type_ids = history[i]['token_type_ids']
                img_attention_mask = history[i]['attention_mask']
                
                # Corresponding action
                if i < len(action_history):
                    act_input_ids = action_history[i]
                    
                    # Fill action token_type_ids with zeros and attention_mask with ones
                    act_token_type_ids = torch.zeros_like(act_input_ids)
                    act_attention_mask = torch.ones_like(act_input_ids)
                    
                    # Alternately append image and action data
                    all_input_ids.extend([img_input_ids, act_input_ids])
                    all_token_type_ids.extend([img_token_type_ids, act_token_type_ids])
                    all_attention_mask.extend([img_attention_mask, act_attention_mask])
                else:
                    # If there is no corresponding action, append image data only
                    all_input_ids.append(img_input_ids)
                    all_token_type_ids.append(img_token_type_ids)
                    all_attention_mask.append(img_attention_mask)
            # Concatenate all input_ids, token_type_ids, and attention_mask
            concatenated_input_ids = torch.cat(all_input_ids, dim=1)
            concatenated_token_type_ids = torch.cat(all_token_type_ids, dim=1)
            concatenated_attention_mask = torch.cat(all_attention_mask, dim=1)
            
            # Update pos_inputs
            final_inputs = pos_inputs.copy()
            final_inputs['input_ids'] = concatenated_input_ids
            final_inputs['token_type_ids'] = concatenated_token_type_ids
            final_inputs['attention_mask'] = concatenated_attention_mask
        else:
            final_inputs = pos_inputs

        if self.use_fast: 
            last_token_id = self.tokenizer.pad_token_id - 1
            allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)

            if self.use_cot:
                self.COT_CONFIG = GenerationConfig(
                    pad_token_id=self.model.config.pad_token_id,
                    bos_token_id=self.model.config.bos_token_id,
                    eos_token_id=151843,
                    do_sample=False,
                )
                cot_allowed_token_ids = list(range(0, self.tokenizer.pad_token_id)) + [151843]
                cot_id_processor = ActionIDConstraintLogitsProcessor(cot_allowed_token_ids)
                with torch.no_grad():
                    cot_outputs = self.model.generate(
                        final_inputs.input_ids.to(self.device),
                        self.COT_CONFIG,
                        max_new_tokens=500,
                        logits_processor=[cot_id_processor],
                        attention_mask=final_inputs.attention_mask.to(self.device),
                    )
                cot_text = cot_outputs[:, final_inputs.input_ids.shape[-1]:-1]
                reasoning = self.tokenizer.decode(cot_text[0], skip_special_tokens=True)

                boa = self.tokenizer.convert_tokens_to_ids(self.tokenizer.boa_token)
                final_inputs.input_ids = torch.cat([cot_outputs, torch.tensor([[boa]], device=cot_outputs.device)], dim=1)
                final_inputs.attention_mask = torch.ones_like(final_inputs.input_ids)
            with torch.no_grad():
                outputs = self.model.generate(
                    final_inputs.input_ids.to(self.device),
                    self.GENERATION_CONFIG,
                    max_new_tokens=100,
                    logits_processor=[action_id_processor],
                    attention_mask=final_inputs.attention_mask.to(self.device),
                )
            # omit the eoa token
            orig_outputs = outputs[:, final_inputs.input_ids.shape[-1]:]
            outputs = outputs[:, final_inputs.input_ids.shape[-1]:-1]
            last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
            processed_outputs = last_token_id_tensor - outputs
            action_outputs = self.action_tokenizer.decode(
                processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
            )
            action = action_outputs[0]
            #new!  ========================================================================================================
            if self.video_mode and self.use_history and self.use_action_history:
                self.add_action(orig_outputs.detach().cpu())

        else:
            pass

        # unnormalize action
        action = self.unormalize_action(action)

        if self.use_one_step:
            # only one step
            action_pred = action[0:3]
        else:
            # action chunk
            action_pred = action
        
        res = [self.transform_action(action[[i],:]) for i in range(action_pred.shape[0])]
        raw_actions = [_[0] for _ in res]
        env_actions = [_[1] for _ in res]

        return raw_actions, env_actions
    
    def unormalize_action(self, action):

        if self.policy_setup == "google_robot":
            action_high = np.array([
                0.17753939667156038,
                0.14669284061245857,
                0.2179850059450077,
                0.5882710857765758,
                0.35334834816471683,
                0.4470693223284772,
                0.99980000009996
            ])
            action_low = np.array([
                -0.22624227067544056,
                -0.15126218201085617,
                -0.23251856873127252,
                -0.3538952427136002,
                -0.4202906595250484,
                -0.43766197340888247,
                -1e-10
            ])
        else:
            if self.use_cot:
                action_high = np.array([
                    0.028418295088990186,
                    0.04023438128952678,
                    0.040332944217942646,
                    0.08114985513931172,
                    0.07819607377472315,
                    0.2025440130266567,
                    0.99980000009996
                ])
                action_low = np.array([
                    -0.029267571877272303,
                    -0.04094358115000042,
                    -0.026200699773410135,
                    -0.07969299698147969,
                    -0.0938352885296676,
                    -0.20914677715629448,
                    -1e-10
                ])
            else:
                # action_high = np.array([
                #     0.02819482065335177,
                #     0.04079562563528227,
                #     0.04015785568112329,
                #     0.08070396399877966,
                #     0.07745134926258301,
                #     0.2016542930635028,
                #     0.99980000009996
                # ])
                # action_low = np.array([
                #     -0.02887803485105428,
                #     -0.04178320091122349,
                #     -0.026113155505000457,
                #     -0.08117201867235568,
                #     -0.09309056401752747,
                #     -0.20778717060421048,
                #     -1e-10
                # ])
                #my
                action_low = np.array([
                    -0.02887803485105428,
                    -0.04178320091122349,
                    -0.026113155505000457,
                    -0.08117201867235568,
                    -0.09309056401752747,
                    -0.20914677715629448,
                    -1e-10
                ])
                action_high = np.array([
                    0.02819482065335177,
                    0.04079562563528227,
                    0.04015785568112329,
                    0.08070396399877966,
                    0.07745134926258301,
                    0.2025440130266567,
                    0.99980000009996
                ])
        action = 0.5 * (action + 1) * (action_high - action_low) + action_low
        return action
    
    def transform_action(self, raw_actions):
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(
                raw_actions[0, 6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] 
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle 

        if self.policy_setup == "google_robot":
            current_gripper_action = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                        self.previous_gripper_action - current_gripper_action
                )
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action
            # print(f'action gripper: {action["gripper"]}')

        elif self.policy_setup == "widowx_bridge":
            relative_gripper_action = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            if relative_gripper_action[0] > 0:
                self.close_gripper_num += 1
            else:
                self.close_gripper_num = 0

            if self.close_gripper_num >= self.late_close_gripper:
                relative_gripper_action[0] = 1
            else:
                relative_gripper_action[0] = -1

            action["gripper"] = relative_gripper_action

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

class BaseModelInference(CustomModel):
    def __init__(
        self,
        ckpt_path,
        configs,
        device,
        save_dir=None,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        exec_horizon=1,
    ):
        self.configs = configs
        self.dataset_stat = self.load_dataset_stat()
        self.model = BaseTrainer(configs=configs)
        self.policy = self.model

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.sticky_gripper_num_repeat = 2

        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        if self.policy_setup == "google_robot":
            self.close_gripper_act = -1
        elif self.policy_setup == "widowx_bridge":
            self.close_gripper_act = 1
        else:
            raise NotImplementedError

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        self.image_size = self.configs.get("image_size", 224)
        self.action_scale = self.configs.get("action_scale", 1.0)
        self.horizon = self.configs["window_size"]
        self.window_size = self.horizon
        self.pred_action_horizon = self.configs["fwd_pred_next_n"]
        self.exec_horizon = exec_horizon
        # repeat the closing gripper action for self.sticky_gripper_num_repeat times (google robot setting)
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.late_close_gripper = 2 
        self.close_gripper_num = 0

        self.task = None
        self.task_description = None
        self.num_image_history = 0

        self.init_config(ckpt_path, configs, device, save_dir)
        self.raw_calvin=True

    def reset(self):
        super().reset()

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.close_gripper_num = 0
        self.previous_gripper_action = None

    @staticmethod
    def load_dataset_stat():
        stat = {}

        with open(
            "configs/data/oxe_dataset_stats/dataset_statistics_google.json", "r"
        ) as f:
            google_info = json.load(f)
        stat["fractal20220817_data"] = google_info

        with open(
            "configs/data/oxe_dataset_stats/dataset_statistics_bridge.json", "r"
        ) as f:
            bridge_info = json.load(f)
        stat["bridge_orig"] = bridge_info

        return stat

    def transform_action(self, raw_actions):
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(
                raw_actions[0, 6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            # current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action
            # print(f'action gripper: {action["gripper"]}')

        elif self.policy_setup == "widowx_bridge":
            relative_gripper_action = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            if relative_gripper_action[0] > 0:
                self.close_gripper_num += 1
            else:
                self.close_gripper_num = 0

            if self.close_gripper_num >= self.late_close_gripper:
                relative_gripper_action[0] = 1
            else:
                relative_gripper_action[0] = -1

            action["gripper"] = relative_gripper_action

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(
            image,
            tuple((self.image_size, self.image_size)),
            interpolation=cv.INTER_AREA,
        )
        return image

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate(
                    [a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1
                )
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(
                pred_actions[:, action_dim], label="predicted action"
            )
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

    def step(self, image, goal):
        obs = {}
        obs['rgb_obs'] = {}
        obs["rgb_obs"]['rgb_static'] = image
        action = super().step(obs, goal)

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        if isinstance(action, torch.Tensor):
            action = action.squeeze()
            action = action.reshape(-1, action.shape[-1])
            action = action.numpy()

        action_norm_stats = self.dataset_stat[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )
        action = np.where(
            mask,
            0.5 * (action + 1) * (action_high - action_low) + action_low,
            action,
        )
        raw_action, env_action = self.transform_action(action)

        return raw_action, env_action
