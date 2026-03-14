import os
import json
import tempfile
import threading
import time
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

import torch
import sys


from typing import Optional, Sequence, Callable
import matplotlib.pyplot as plt

import cv2 as cv

from transforms3d.euler import euler2axangle

from queue import Queue

from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import sys
sys.path.append("/remote-home/jinminghao/structvla/reference/Emu3")
from emu3.mllm import Emu3Tokenizer, Emu3ForCausalLM, Emu3Processor
from emu3.mllm import Emu3MoE
from transformers import LogitsProcessor
from emu3.mllm.configuration_emu3 import Emu3Config,Emu3MoEConfig

class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            mask[self.allowed_token_ids] = True
        else:
            mask[:, self.allowed_token_ids] = True
        scores[~mask] = -float("inf")
        return scores


class EmuVLAInferenceFranka:
    """
    FRANKA real-robot inference:
    - input: dict(full_image, wrist_image) as numpy uint8 RGB
    - output: (T,7) with gripper in {0,1}, 1=open, 0=close
    """

    def __init__(
        self,
        emu_hub,
        vq_hub,
        vision_hub,
        device,
        fast_path="/remote-home/jinminghao/structvla/pretrain/fast",
        image_size=(256,144),
        image_size_wrist=(240, 135),  # (W,H) for resize
        crop_box_full=None,        # "l,t,r,b", (l,t,r,b), or None
        crop_box_wrist=None,       # "l,t,r,b", (l,t,r,b), or None
        window_size=1,
        predict_action_frames=5,
        context_frames=1,
        predict_frames=1,
        action_dim=7,
        use_action_history=False,  # Strongly recommended to keep False when window_size=1
    ):
        self.emu_hub = emu_hub
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.device = device

        # FRANKA input resolution
        self.image_size = tuple(image_size)
        self.image_size_wrist = tuple(image_size_wrist)  # (W,H) for resize
        # Rollout / chunk settings; these must match training and deployment
        self.window_size = int(window_size)
        self.predict_action_frames = int(predict_action_frames)
        self.context_frames = int(context_frames)
        self.predict_frames = int(predict_frames)
        self.action_dim = int(action_dim)

        # FRANKA uses a wrist camera
        self.use_gripper_cam = True

        # History settings; avoid action_history when window_size <= 1
        self.use_action_history = bool(use_action_history) and (self.window_size > 1)


        # Crop boxes can be passed as strings or tuples
        if isinstance(crop_box_full, str):
            crop_box_full = parse_crop_box(crop_box_full)
        if isinstance(crop_box_wrist, str):
            crop_box_wrist = parse_crop_box(crop_box_wrist)
        self.crop_box_full = crop_box_full
        self.crop_box_wrist = crop_box_wrist

        # generation
        self.use_fast = True
        self.eoa_token_id = 151845
        self.video_mode = True

        self._init_model(device=device, fast_path=fast_path)

        # queues
        self.vision_queue = Queue(maxsize=self.window_size)
        if self.window_size <= 1:
            self.action_queue = Queue(maxsize=1)  # Placeholder, not used
        else:
            self.action_queue = Queue(maxsize=self.window_size - 1)
        
        self.current_task_description = None
        self.current_exp_id = None
    def debug_token_and_vision(self):
        # --- action token id check ---
        pad_id = self.tokenizer.pad_token_id
        last_token_id = pad_id - 1
        vocab_size = self.action_tokenizer.vocab_size
        allowed_lo = last_token_id - vocab_size
        allowed_hi = last_token_id

        boa_id = self.tokenizer.encode(self.tokenizer.boa_token)[0]
        eoa_id = self.tokenizer.encode(self.tokenizer.eoa_token)[0]
        print("[TOK] pad:", pad_id, "last:", last_token_id)
        print("[TOK] fast vocab_size:", vocab_size, "allowed:", (allowed_lo, allowed_hi))
        print("[TOK] BOA:", boa_id, "in_allowed:", (allowed_lo <= boa_id <= allowed_hi))
        print("[TOK] EOA:", eoa_id, "in_allowed:", (allowed_lo <= eoa_id <= allowed_hi), "eoa_token_id(hard):", getattr(self, "eoa_token_id", None))

        # --- vision hub check ---
        print("[VIS] vision_hub:", getattr(self, "vision_hub", None))
        # You can also print the model class/config to confirm the same vision tokenizer is used
        print("[VIS] image_tokenizer class:", self.image_tokenizer.__class__)
    @torch.no_grad()
    def debug_encode_one(self, np_rgb):
        im = Image.fromarray(np_rgb).convert("RGB")
        x = self.image_processor(im, return_tensors="pt")["pixel_values"].to(self.device)
        code = self.image_tokenizer.encode(x)
        print("[CODE] shape:", tuple(code.shape), "dtype:", code.dtype)
        # Print numeric ranges; discrete codes are usually int/long or can be cast to int
        c = code.detach().cpu()
        print("[CODE] min/max:", int(c.min()), int(c.max()))
    def _init_model(self, device, fast_path):
        self.model = Emu3MoE.from_pretrained(
            self.emu_hub,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to(device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.vq_hub,
            model_max_length=self.model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )

        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
        self.image_processor.min_pixels = 80 * 80

        self.image_tokenizer = AutoModel.from_pretrained(
            self.vision_hub, trust_remote_code=True
        ).to(device).eval()

        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)

        # FAST tokenization (using the FRANKA tokenizer)
        self.action_tokenizer = AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)

        self.GENERATION_CONFIG = GenerationConfig(
            pad_token_id=self.model.config.pad_token_id,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.eoa_token_id,
            do_sample=False,
        )

    # ------------------- queues -------------------
    def reset(self):
        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()

    def add_image_tokens(self, tokens):
        if self.vision_queue.full():
            self.vision_queue.get()
        self.vision_queue.put(tokens)

    def add_action_tokens(self, act_tokens):
        if not self.use_action_history:
            return
        if self.action_queue.full():
            self.action_queue.get()
        self.action_queue.put(act_tokens)

    def get_history(self):
        return list(self.vision_queue.queue)

    def get_action_history(self):
        if not self.use_action_history:
            return []
        return list(self.action_queue.queue)

    # ------------------- preprocessing -------------------
    @torch.no_grad()
    def preprocess(self, obs: dict):
        """
        obs:
          - full_image: np.uint8 RGB
          - wrist_image: np.uint8 RGB
        return:
          - image_code: tensor
          - wrist_code: tensor
        """
        #full = Image.fromarray(obs["full_image"]).convert("RGB").resize(self.image_size)
        full = Image.fromarray(obs["full_image"]).convert("RGB")
        full = crop_then_resize(full, self.crop_box_full, self.image_size)
        full_x = self.image_processor(full, return_tensors="pt")["pixel_values"].to(self.device, non_blocking=True)
        full_code = self.image_tokenizer.encode(full_x)

        wrist_code = None
        if self.use_gripper_cam and ("wrist_image" in obs) and (obs["wrist_image"] is not None):
            #wrist = Image.fromarray(obs["wrist_image"]).convert("RGB").resize(self.image_size)
            wrist = Image.fromarray(obs["wrist_image"]).convert("RGB")
            wrist = crop_then_resize(wrist, self.crop_box_wrist, self.image_size_wrist)

            wrist_x = self.image_processor(wrist, return_tensors="pt")["pixel_values"].to(self.device, non_blocking=True)
            wrist_code = self.image_tokenizer.encode(wrist_x)
        print("full PIL size:", full.size)
        print("full pixel_values:", full_x.shape)
        if wrist_code is not None:
            print("wrist PIL size:", wrist.size)
            print("wrist pixel_values:", wrist_x.shape)
        return full_code, wrist_code

    # ------------------- core step -------------------
    @torch.no_grad()
    def step(self, obs: dict, task_description: str , exp_id: Optional[str]=None):
        if task_description != self.current_task_description or exp_id != self.current_exp_id:
            print(f"Task changed from '{self.current_task_description}' to '{task_description}'. Resetting history.")
            self.reset()
            self.current_task_description = task_description
            self.current_exp_id = exp_id
        full_code, wrist_code = self.preprocess(obs)

        # (B=1, T=1, ...)
        video_tokens = full_code.unsqueeze(1)
        gripper_tokens = wrist_code.unsqueeze(1) if (wrist_code is not None) else None

        # Text tokens are prepended separately
        text_prompt = [self.tokenizer.bos_token + task_description]
        text_tokens = self.processor.tokenizer(text_prompt)
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type="pt")

        # Build multimodal tokens for the current frame on the image side
        pos_inputs = self.processor.video_process(
            text=task_description,
            video_tokens=video_tokens,
            gripper_tokens=gripper_tokens,
            context_frames=self.context_frames,
            frames=self.predict_frames,
            return_tensors="pt",
            mode="VLA_Video",
            padding="longest",
        )

        # History concatenation: text + (img, act, img, act, ...)
        self.add_image_tokens(pos_inputs)
        history = self.get_history()
        action_history = self.get_action_history()

        all_input_ids = [text_tokens["input_ids"]]
        all_token_type_ids = [text_tokens["token_type_ids"]]
        all_attention_mask = [text_tokens["attention_mask"]]

        for i in range(len(history)):
            img_ids = history[i]["input_ids"]
            img_ttypes = history[i]["token_type_ids"]
            img_mask = history[i]["attention_mask"]

            all_input_ids.append(img_ids)
            all_token_type_ids.append(img_ttypes)
            all_attention_mask.append(img_mask)

            if i < len(action_history):
                act_ids = action_history[i]
                act_ttypes = torch.zeros_like(act_ids)
                act_mask = torch.ones_like(act_ids)
                all_input_ids.append(act_ids)
                all_token_type_ids.append(act_ttypes)
                all_attention_mask.append(act_mask)

        final_input_ids = torch.cat(all_input_ids, dim=1)
        final_attention_mask = torch.cat(all_attention_mask, dim=1)

        # fast action token constraint
        last_token_id = self.tokenizer.pad_token_id - 1
        allowed_token_ids = (
            list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1))
            + [self.eoa_token_id]
        )
        action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)

        outputs = self.model.generate(
            final_input_ids.to(self.device),
            self.GENERATION_CONFIG,
            max_new_tokens=100,
            logits_processor=[action_id_processor],
            attention_mask=final_attention_mask.to(self.device),
        )

        # decode to continuous actions in [-1,1]
        orig_outputs = outputs[:, final_input_ids.shape[-1]:]      # includes eoa
        action_tokens = outputs[:, final_input_ids.shape[-1]:-1]   # drop eoa

        last_token_id_tensor = torch.tensor(last_token_id, dtype=action_tokens.dtype, device=action_tokens.device)
        processed = last_token_id_tensor - action_tokens



        action_outputs = self.action_tokenizer.decode(
            processed,
            time_horizon=self.predict_action_frames,
            action_dim=self.action_dim,
        )
        action = action_outputs[0]  # (T,7) in [-1,1]



        

        self.add_action_tokens(orig_outputs.detach().cpu())
        print("Predicted action (normalized):", action)
        # Denormalize back to the real delta range
        action = self.unnormalize_action(action)  # (T,7)
        print("Unnormalized action:", action)

        action[..., 6] = (action[..., 6] > 0.5).astype(np.float32)

        return action  # (T,7) dx,dy,dz,droll,dpitch,dyaw,gripper(0/1)

    def unnormalize_action(self, action):
        """
        This must use the action_low/high (or q01/q99) that matches your FRANKA
        training data. The provided real pick-and-place settings are kept unchanged.
        """

        # 01-19
        # action_low = np.array([
        # -0.041074377947459925,
        # -0.02115786123026856,
        # -0.018932542600205205,
        # -0.042974891355319206,
        # -0.060854045088135356,
        # -0.04180403636932571,
        # -1e-10
        # ], dtype=np.float32)
        # action_high = np.array([
        # 0.03693350404228588,
        # 0.01641667504599731,
        # 0.013116346041143201,
        # 0.026680060439408987,
        # 0.060988134897771745,
        # 0.04234039681239539,
        # 0.99980000009996
        # ], dtype=np.float32)

        # 01-28-tidy_up only
        action_low = np.array([
        -0.06054988196215591,
        -0.029594094511228144,
        -0.027141910674417694,
        -0.06098617153039586,
        -0.05999057983091647,
        -0.0644833654585268,
        -1e-10
        ], dtype=np.float32)
        action_high = np.array([
        0.056466813299469995,
        0.019857303007381928,
        0.021288128847317733,
        0.04543325025394751,
        0.06498883743421667,
        0.0652449813307899,
        0.99980000009996
        ], dtype=np.float32)

        
        action = 0.5 * (action + 1) * (action_high - action_low) + action_low
        return action

def parse_crop_box(s: str):
    """
    Parse "left,top,right,bottom" -> (l,t,r,b) as int tuple.
    Return None if s is empty/None.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid crop_box '{s}', expect 'left,top,right,bottom'")
    l, t, r, b = parts
    if r <= l or b <= t:
        raise ValueError(f"Invalid crop_box '{s}', right/bottom must be > left/top")
    return (l, t, r, b)

def crop_then_resize(im: Image.Image, crop_box, size_hw):
    """
    im: PIL.Image RGB
    crop_box: (l,t,r,b) in pixel coords of original image, or None for no crop
    size_hw: (W,H) for final resize
    """
    if crop_box is not None:
        W0, H0 = im.size
        l, t, r, b = crop_box
        # clamp for safety
        l = max(0, min(l, W0))
        r = max(0, min(r, W0))
        t = max(0, min(t, H0))
        b = max(0, min(b, H0))
        if r > l and b > t:
            im = im.crop((l, t, r, b))
    return im.resize(size_hw, resample=Image.BICUBIC)
# -----------------------------
# 1) Load the model once when the service starts
# -----------------------------
def build_inferencer():

    #pick and place
    #emu_hub = os.environ.get("EMU_HUB", "/remote-home/jinminghao/structvla/logs/REAL_WORLD_FRANKA_TIDY_UP/checkpoint-2000")

    #tidy up 
    emu_hub = os.environ.get("EMU_HUB", "/remote-home/jinminghao/structvla/logs/REAL_WORLD_FRANKA_PICK_PLACE/checkpoint-2000")

    vq_hub = os.environ.get("VQ_HUB", "/remote-home/jinminghao/structvla/pretrain/Emu3-Base")
    vision_hub = os.environ.get("VISION_HUB", "/remote-home/jinminghao/structvla/pretrain/Emu3-VisionVQ")

    # Select the GPU via CUDA_VISIBLE_DEVICES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EmuVLAInferenceFranka(
        emu_hub=emu_hub,
        vq_hub=vq_hub,
        vision_hub=vision_hub,
        device=torch.device("cuda:0"),
        window_size=2,
        predict_action_frames=5,
        image_size=(256,144),
        image_size_wrist=(240, 135),  # (W,H) for resize
        crop_box_full="",     # e.g. "0,0,640,480"
        crop_box_wrist="",   # e.g. "50,20,590,430"
        use_action_history=True,
    )

    return model

inferencer = build_inferencer()



app = Flask(__name__)
@app.route('/api/inference', methods=['POST'])
def inference():

    query = request.files['json']

    with tempfile.NamedTemporaryFile(delete=False) as temp_query:
        query.save(temp_query.name)
        temp_query_path = temp_query.name

    input_query = json.load(open(temp_query_path))
    print(input_query)
    image = Image.open(request.files["main_images"].stream).convert("RGB")
    wrist = Image.open(request.files["wrist_images"].stream).convert("RGB")

    # ===== Save locally under real_experiment_img/<task_description>/main and /wrist =====
    save_root = os.path.join(os.getcwd(), "real_experiment_img", input_query['task_description'], input_query['exp_id'])
    main_dir = os.path.join(save_root, "main")
    wrist_dir = os.path.join(save_root, "wrist")
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(wrist_dir, exist_ok=True)

    # Filename: millisecond timestamp + thread id to reduce collisions under concurrency
    ts = int(time.time() * 1000)
    tid = threading.get_ident()
    fname = f"{ts}_{tid}.png"

    image.save(os.path.join(main_dir, fname))
    wrist.save(os.path.join(wrist_dir, fname))
    # ============================================================
    full_np = np.array(image)
    wrist_np = np.array(wrist)

    image_dict = {
        "full_image": full_np,
        "wrist_image": wrist_np,
    }
    answer = inferencer.step(image_dict, input_query['task_description'], input_query['exp_id'])
    if isinstance(answer, np.ndarray):
        answer = answer.tolist()
    # clean files

    os.remove(temp_query_path)
    print("Final Answer:", answer)
    return jsonify(answer)
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=5901)  #4090-118: 76 L20-115:5901
