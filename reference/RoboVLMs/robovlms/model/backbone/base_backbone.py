from einops import rearrange, repeat
import json
import os, sys, copy
import numpy as np
from typing import Optional, Tuple, List

import torch
from torch import nn

from robovlms.utils.model_utils import update_tokenizer
from robovlms.model.vlm_builder import build_vlm
from robovlms.model.policy_head.action_tokenizer import ActionTokenizer
from robovlms.data.vid_llava_constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from robovlms.model.text_encoder.clip_text_encoder import ClipTextFeatureEncoder


def initialize_param(model):
    with torch.no_grad():
        for m in model.children():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.fill_(0)
            else:
                initialize_param(m)


class BaseRoboVLM(nn.Module):
    def __init__(
        self,
        configs,
        train_setup_configs,
        act_encoder_configs=None,
        act_head_configs=None,
        fwd_head_configs=None,
        window_size=None,
        use_obs_queries=True,
        use_act_queries=True,
        use_hand_rgb=False,
        use_pixel_loss=True,
        use_mim_obs_loss=False,
        use_time_causal_attn=True,
        vision_masked_ratio=0.9,
        use_tube_mask=False,
        fwd_pred_next_n=1,
        use_vision_resampler=False,
        vision_resampler_configs=None,
        use_clip_norm=False,
        use_state=False,
        **kwargs,
    ):
        super().__init__()
        self.window_size = window_size
        self.use_obs_queries = use_obs_queries
        self.use_act_queries = use_act_queries
        self.use_hand_rgb = use_hand_rgb
        self.use_pixel_loss = use_pixel_loss
        self.use_mim_obs_loss = use_mim_obs_loss
        self.use_time_causal_attn = use_time_causal_attn
        self.use_clip_norm = use_clip_norm
        self.vision_masked_ratio = vision_masked_ratio
        self.use_tube_mask = use_tube_mask
        self.use_state = use_state
        self.fwd_pred_next_n = fwd_pred_next_n

        self.kwargs = kwargs
        self.configs = configs
        self.model_name = configs["model"]
        self.model_config = json.load(
            open(
                os.path.join(
                    self.configs["vlm"]["pretrained_model_name_or_path"], "config.json"
                ),
                "r",
            )
        )

        self.train_setup_configs = train_setup_configs
        self.act_encoder_configs = act_encoder_configs
        self.act_head_configs = act_head_configs
        self.fwd_head_configs = fwd_head_configs
        self.vision_resampler_configs = vision_resampler_configs

        self.tokenizer, self.backbone = self._init_backbone()
        # import pdb;pdb.set_trace()
        self.tokenizer = update_tokenizer(self.tokenizer, self.configs["tokenizer"])
        if self.train_setup_configs.get("reinit", False):
            initialize_param(self.backbone)
        self.act_head, self.fwd_head, self.clip_norm_head = self._init_heads()

        if self.act_head_configs is not None:
            self.action_space = self.act_head_configs.get("action_space", "continuous")
            if self.action_space == "discrete":
                self.action_tokenizer = ActionTokenizer(
                    self.tokenizer,
                    bins=self.act_head_configs["n_bin"],
                    min_action=self.act_head_configs["min_action"],
                    max_action=self.act_head_configs["max_action"],
                )

            if self.action_space == "continuous":
                self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
                self.action_token.requires_grad_(True)

        if self.fwd_head_configs is not None:
            self.image_latent_num = self.fwd_head_configs.get("image_latent_num", 8)
            self.pred_image = True
            self.pred_hand_image = self.fwd_head_configs.get("pred_hand_image", False)

            global_frame_num = self.fwd_head_configs.get("global_frame_num", -1)
            if global_frame_num != -1:
                predict_image_num = global_frame_num - 1
            else:
                predict_image_num = self.fwd_pred_next_n

            self.static_image_tokens = nn.Parameter(
                torch.zeros(self.image_latent_num * predict_image_num, self.hidden_size)
            )
            self.static_image_tokens.requires_grad_(True)
            if self.pred_hand_image:
                self.hand_image_tokens = nn.Parameter(
                    torch.zeros(
                        self.image_latent_num * predict_image_num, self.hidden_size
                    )
                )
                self.hand_image_tokens.requires_grad_(True)
        else:
            self.pred_image = False

        ### setup vision tower and configs

        self.use_vision_resampler = use_vision_resampler
        if self.use_vision_resampler:
            from robovlms.model.vision_encoder.vision_resampler import (
                PerceiverResampler,
            )

            self.vision_resampler = PerceiverResampler(dim=self.hidden_size)
        else:
            self.vision_resampler = None

        if self.use_state:
            # Embedding functions for states
            state_dim = 7
            self.embed_arm_state = torch.nn.Linear(state_dim - 1, self.hidden_size)
            self.embed_gripper_state = torch.nn.Embedding(
                2, self.hidden_size
            )  # one-hot gripper state
            self.embed_state = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)

        self._trainable_params_setup()

    def encode_state(self, state):
        arm_state_embeddings = self.embed_arm_state(state[..., :6])
        gripper_state_embeddings = self.embed_gripper_state(state[..., [-1]]).long()
        state_embeddings = torch.cat(
            (arm_state_embeddings, gripper_state_embeddings), dim=2
        )
        state_embeddings = self.embed_state(state_embeddings)  # (b, l, h)
        return state_embeddings

    def model_encode_images(self, images):
        raise NotImplementedError

    def encode_images(self, images, image_sizes=None):
        # input: images: list of b,c,h,w or b,t,c,h,w
        # output: image_features: b, t, n, d

        if images.ndim == 4:
            images = images.unsqueeze(1)

        bs, seq_len = images.shape[:2]

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.model_encode_images(concat_images)

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]

        else:
            image_features = self.model_encode_images(images)

        image_features = torch.stack(image_features, dim=0).view(
            bs, seq_len, -1, image_features[0].shape[-1]
        )

        if self.use_vision_resampler:
            ### downsample at token num dim: b, s, n, d -> b, s, v d
            # b T F v d -> b, T, n, d
            image_features = self.vision_resampler(
                image_features.unsqueeze(2)
            )  # downsample v_tok per image to n_tok
        # print(image_features.shape)
        return image_features

    def _init_backbone(self):
        tokenizer, model = build_vlm(self.configs["vlm"], self.configs["tokenizer"])
        if "Processor" in self.configs["tokenizer"]["type"]:
            self.processor = tokenizer
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = tokenizer
        return self.tokenizer, model

    def cat_multi_modal_input(
        self,
        input_embeds: torch.Tensor,
        multimodal_embeds: torch.Tensor = None,
        insert_idx: int = 0,
        masks: torch.Tensor = None,
    ):
        # concat multi-modal inputs
        if insert_idx >= 0:
            return torch.cat(
                (
                    input_embeds[:, :insert_idx],
                    multimodal_embeds,
                    input_embeds[:, insert_idx:],
                ),
                dim=1,
            )
        elif insert_idx == -1 and masks is not None:
            new_embed_list = []
            for mask, input_embed, multimodal_embed in zip(
                masks, input_embeds, multimodal_embeds
            ):
                # the concat index is up to mask first False index
                # concat_idx = (mask == False).nonzero()[0].item()
                indexs = (mask == False).nonzero()
                insert_idx = indexs[0].item() if len(indexs) > 0 else len(mask)
                new_embed = torch.cat(
                    (
                        input_embed[:insert_idx],
                        multimodal_embed,
                        input_embed[insert_idx:],
                    ),
                    dim=0,
                )
                new_embed_list.append(new_embed)
            return torch.stack(new_embed_list, dim=0)
        else:
            raise Exception(
                "insert_idx should be -1 or >= 0, and if you want to insert as last(-1), you should provide masks"
            )

    @property
    def hidden_size(self):
        raise NotImplementedError

    @property
    def word_embedding(self):
        raise NotImplementedError

    @property
    def vision_tower(self):
        raise NotImplementedError

    @property
    def text_tower(self):
        raise NotImplementedError

    @property
    def model(self):
        raise NotImplementedError

    @property
    def start_image_token_id(self):
        return None

    @property
    def end_image_token_id(self):
        return None

    @property
    def image_processor(self):
        import torchvision.transforms as T

        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        image_preprocess = T.Compose(
            [
                T.Resize(
                    (
                        self.configs.get("image_size", 224),
                        self.configs.get("image_size", 224),
                    ),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.Lambda(lambda img: img.convert("RGB")),
                T.ToTensor(),
                T.Normalize(clip_mean, clip_std),
            ]
        )
        return image_preprocess

    def merge_multi_modal_input(
        self,
        input_embeds: torch.Tensor,
        multimodal_feats: torch.Tensor = None,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        is_image=True,
        insert_idx=1,
        fill_zero=False,
    ):
        # if is_image, the vision_x needs to be processed by self.encode_images
        # otherwise merge
        bs = input_embeds.shape[0]

        if is_image:
            rgb_feats = self.encode_images(multimodal_feats)

            if self.start_image_token_id is not None:
                image_start_embed = (
                    self.word_embedding(self.start_image_token_id.to(self.model.device))
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(*rgb_feats.shape[:2], 1, 1)
                )

                if self.end_image_token_id is None:
                    end_image_token_id = self.start_image_token_id + 1
                else:
                    end_image_token_id = self.end_image_token_id
                image_end_embed = (
                    self.word_embedding(end_image_token_id.to(self.model.device))
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(*rgb_feats.shape[:2], 1, 1)
                )

                rgb_feats = torch.cat(
                    [image_start_embed, rgb_feats, image_end_embed], dim=2
                )

            rgb_feats = rearrange(
                rgb_feats, "b l n d -> b (l n) d"
            )  # flatten seq_len and n_tok_per_img dim

        else:
            rgb_feats = multimodal_feats

        added_seq_len = rgb_feats.shape[1]

        multimodal_embeds = torch.cat(
            [input_embeds[:, :insert_idx], rgb_feats, input_embeds[:, insert_idx:]],
            dim=1,
        )

        insert_mask = (
            torch.cat(
                [
                    torch.zeros(input_embeds[:, :insert_idx].shape[:2]),
                    torch.ones(rgb_feats.shape[:2]),
                    torch.zeros(input_embeds[:, insert_idx:].shape[:2]),
                ],
                dim=1,
            )
            .bool()
            .to(multimodal_embeds.device)
        )

        mutlimodal_labels = None
        if labels is not None:
            mutlimodal_labels = torch.full(
                (bs, added_seq_len), -100, dtype=labels.dtype, device=labels.device
            )
            mutlimodal_labels = self.cat_multi_modal_input(
                labels, mutlimodal_labels, insert_idx, attention_mask
            )
            if is_image:
                if self.start_image_token_id is not None:
                    mutlimodal_labels[:, 0] = self.start_image_token_id
                    mutlimodal_labels[
                        :, multimodal_feats.shape[1] + 1
                    ] = end_image_token_id

        multimodal_attention_mask = None
        if attention_mask is not None:
            val = False if fill_zero else True
            multimodal_attention_mask = torch.full(
                (bs, added_seq_len),
                val,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            multimodal_attention_mask = self.cat_multi_modal_input(
                attention_mask, multimodal_attention_mask, insert_idx, attention_mask
            )

        return (
            multimodal_embeds,
            mutlimodal_labels,
            multimodal_attention_mask,
            insert_mask,
        )

    def _init_heads(self):
        action_head = None
        if self.act_head_configs is not None:
            import robovlms.model.policy_head as action_heads

            _kwargs = copy.deepcopy(self.act_head_configs)
            _kwargs.update(
                dict(  # hidden_size=self.hidden_size,
                    tokenizer=self.tokenizer,
                    in_features=self.hidden_size,
                    fwd_pred_next_n=self.fwd_pred_next_n,
                    window_size=self.window_size,
                    n_bin=self.act_head_configs.get("n_bin", 256),
                    min_action=self.act_head_configs.get("min_action", -1),
                    max_action=self.act_head_configs.get("max_action", 1),
                )
            )
            _cls = getattr(action_heads, _kwargs.pop("type"))
            self.latent_num = self.act_head_configs.get("latent", 1)
            action_head = _cls(**_kwargs)

        fwd_decoder = None
        if self.fwd_head_configs is not None:
            import robovlms.model.forward_head as fwd_heads

            _kwargs = copy.deepcopy(self.fwd_head_configs)
            _kwargs.update(
                dict(
                    image_size=self.vision_tower.config.image_size,
                    patch_size=self.vision_tower.config.patch_size,
                    hidden_size=self.hidden_size,
                    chunk_size=1,
                )
            )
            _cls = getattr(fwd_heads, _kwargs.pop("type"))
            if self.use_mim_obs_loss:
                _kwargs["fwd_pred_next_n"] = 0
            fwd_decoder = _cls(**_kwargs)

        clip_norm_head = None
        if self.use_clip_norm:
            clip_norm_head = ClipTextFeatureEncoder(self.hidden_size)

        return action_head, fwd_decoder, clip_norm_head

    def _trainable_params_setup(self):
        model = self.model
        compute_dtype = (
            torch.float32
        )  # (torch.float16 if self.train_setup_configs['precision'] == 'fp16' else (torch.bfloat16 if self.train_setup_configs['precision'] == 'bf16' else torch.float32))

        model.config.use_cache = False

        if self.train_setup_configs["freeze_backbone"]:
            model.requires_grad_(False)
        else:
            if self.train_setup_configs.get("train_decoder_layers", -1) == -1:
                model.requires_grad_(True)
            else:
                model.requires_grad_(False)
                if hasattr(self.text_tower, "layers"):
                    for layer in self.text_tower.layers[
                        -self.train_setup_configs["train_decoder_layers"] :
                    ]:
                        layer.requires_grad_(True)
                elif hasattr(self.text_tower, "blocks"):
                    for layer in self.text_tower.blocks[
                        -self.train_setup_configs["train_decoder_layers"] :
                    ]:
                        layer.requires_grad_(True)

        if self.train_setup_configs.get("train_vision", False):
            self.vision_tower.requires_grad_(True)
        else:
            self.vision_tower.requires_grad_(False)

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            model.gradient_checkpointing = True
            model.training = True
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.word_embedding.register_forward_hook(make_inputs_require_grad)

        if self.train_setup_configs["lora_enable"]:
            from llava.train.train import find_all_linear_names
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.train_setup_configs["lora_r"],
                lora_alpha=self.train_setup_configs["lora_alpha"],
                target_modules=find_all_linear_names(model),
                lora_dropout=self.train_setup_configs["lora_dropout"],
                bias=self.train_setup_configs["lora_bias"],
                task_type="CAUSAL_LM",
            )
            print("Adding LoRA adapters...")
            self.model = get_peft_model(model, lora_config)
        # import pdb; pdb.set_trace()
        if self.train_setup_configs.get("train_text_embedding", False):
            self.word_embedding.requires_grad_(True)
        else:
            self.word_embedding.requires_grad_(False)

        if self.use_vision_resampler:
            if not self.train_setup_configs.get("freeze_resampler", False):
                self.vision_resampler.requires_grad_(True)
            else:
                self.vision_resampler.requires_grad_(False)

        if self.act_head is not None:
            self.act_head.requires_grad_(True)
        print({k for k, v in self.named_parameters() if v.requires_grad})

    def _forward_action_head(
        self,
        action_tokens: torch.Tensor,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        **kwargs,
    ):
        # action_tokens = get_target_modal_tokens(output_hs, self._action_mask(output_hs))
        action = self.act_head(
            action_tokens, actions=action_labels, action_masks=action_mask, **kwargs
        )

        action_loss = None
        if action_labels is not None:
            action, action_labels, action_mask = self.act_head.get_labels(
                action, action_labels, action_mask, tok_seq=action_tokens, **kwargs
            )
            action_loss = self.act_head.loss(action, action_labels, action_mask)

        return action, action_loss

    def _format_loss(self, loss):
        # for visualization and loss backward in pytorch lightning
        _loss = 0
        _keys = list(loss.keys())

        for k in _keys:
            if "loss" in k:
                _loss += loss[k]

        loss["loss"] = _loss
        return loss

    def forward_vl_task(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images=None,
        image_sizes: Optional[List[List[int]]] = None,
        **kwargs,
    ):
        loss = {}

        if inputs_embeds is None:
            (
                _,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                labels=labels,
                images=images,
            )

        output = self.model(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
        )
        self._update_loss(loss, {"loss_vl": output.loss}, "cotrain")

        return loss

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images=None,
        **kwargs,
    ):
        image_start_embed = None

        if self.start_image_token_id is not None:
            start_image_token_id = self.start_image_token_id
            if self.end_image_token_id is None:
                end_image_token_id = start_image_token_id + 1
            else:
                end_image_token_id = self.end_image_token_id

            image_start_embed = self.word_embedding(start_image_token_id).to(
                self.device
            )
            image_end_embed = self.word_embedding(end_image_token_id).to(self.device)

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)

            if image_start_embed is not None:
                image_start_embed = (
                    image_start_embed.unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(*image_features.shape[:2], 1, 1)
                )
                image_end_embed = (
                    image_end_embed.unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(*image_features.shape[:2], 1, 1)
                )
                image_features = torch.cat(
                    [image_start_embed, image_features, image_end_embed], dim=2
                )

            image_features = image_features.squeeze(
                1
            )  # squeeze over the seq_len dim (unsqueezed in encode_images)

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [_.squeeze(0) for _ in image_features]

        else:
            image_features = self.encode_images(images)
            if image_start_embed is not None:
                image_start_embed = (
                    image_start_embed.unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(*image_features.shape[:2], 1, 1)
                )
                image_end_embed = (
                    image_end_embed.unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(*image_features.shape[:2], 1, 1)
                )
                image_features = torch.cat(
                    [image_start_embed, image_features, image_end_embed], dim=2
                )
                image_features = image_features.squeeze(
                    1
                )  # squeeze over the seq_len dim (unsqueezed in encode_images)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.word_embedding(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.word_embedding(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [
                x.to(self.qwen_model.device) for x in cur_new_input_embeds
            ]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(
                torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
            )
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(
                    0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        new_labels = new_labels.long()

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    @staticmethod
    def _update_loss(loss, new_loss, suffix=None):
        """
        use new_loss to update loss.
            * if suffix is not None, the key from new_loss will be reformatted as: key|suffix
            * otherwise, if the key from new_loss is not in loss, it will be directly used: key
            * otherwise, the key from the new_loss will be reformatted as: key|index, where index is
                searched from 0->+inf so that key|index is not in loss.

        """

        def get_key(k, d):
            if suffix is not None:
                new_k = f"{k}_{suffix}"
                assert new_k not in d
                return new_k

            ind = 0
            while True:
                if ind == 0:
                    new_k = k
                else:
                    new_k = f"{k}_{ind}"
                if new_k not in d:
                    return new_k
                ind += 1

        for k in new_loss:
            new_k = get_key(k, loss)
            loss[new_k] = new_loss[k]

        return loss

    def forward_discrete(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        use_cached_vision_x: bool = False,  # TODO: Do we need this? If not we can remove it from here
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper=None,
        fwd_rgb_labels: torch.Tensor = None,
        fwd_hand_rgb_labels: torch.Tensor = None,
        fwd_mask: torch.Tensor = None,
        instr_and_action_ids=None,
        instr_and_action_labels=None,
        instr_and_action_mask=None,
        **kwargs,
    ):
        loss = {}
        assert vision_x is not None
        bs, window_size = vision_x.shape[:2]

        if instr_and_action_ids.ndim == 2:
            instr_and_action_ids = instr_and_action_ids.unsqueeze(1).repeat(
                1, window_size, 1
            )
            instr_and_action_labels = instr_and_action_labels.unsqueeze(1).repeat(
                1, window_size, 1
            )
            instr_and_action_mask = instr_and_action_mask.unsqueeze(1).repeat(
                1, window_size, 1
            )

        instr_and_action_ids = instr_and_action_ids.flatten(0, 1)
        instr_and_action_labels = instr_and_action_labels.flatten(0, 1)
        instr_and_action_mask = instr_and_action_mask.flatten(0, 1)

        input_embeds = self.word_embedding(instr_and_action_ids)
        vision_x = vision_x.flatten(0, 1)

        if vision_gripper is not None:
            vision_gripper = vision_gripper.flatten(0, 1)

        (
            multimodal_embeds,
            mutlimodal_labels,
            multimodal_attention_mask,
            _,
        ) = self.merge_multi_modal_input(
            input_embeds, vision_x, instr_and_action_labels, instr_and_action_mask
        )

        if vision_gripper is not None:
            (
                multimodal_embeds,
                mutlimodal_labels,
                multimodal_attention_mask,
                _,
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                vision_gripper,
                mutlimodal_labels,
                multimodal_attention_mask,
            )

        multimodal_embeds, mutlimodal_labels, multimodal_attention_mask = (
            rearrange(
                tensor,
                "(bs ws) seq_len ... -> bs (ws seq_len) ...",
                bs=bs,
                ws=window_size,
            )
            for tensor in (
                multimodal_embeds,
                mutlimodal_labels,
                multimodal_attention_mask,
            )
        )

        output = self.model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=multimodal_embeds,
            use_cache=use_cache,
        )

        output_hs = output.logits

        _, action_loss = self._forward_action_head(
            output_hs, mutlimodal_labels, multimodal_attention_mask
        )
        self._update_loss(loss, action_loss, "act")

        loss = self._format_loss(loss)

        return loss

    def forward_continuous(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        use_cached_vision_x: bool = False,  # TODO: Do we need this? If not we can remove it from here
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper=None,
        fwd_rgb_labels: torch.Tensor = None,
        fwd_hand_rgb_labels: torch.Tensor = None,
        fwd_mask: torch.Tensor = None,
        instr_and_action_ids=None,
        instr_and_action_labels=None,
        instr_and_action_mask=None,
        raw_text=None,
        rel_state=None,
        mode="train",
        **kwargs,
    ):
        loss = {}
        assert vision_x is not None
        bs, seq_len = vision_x.shape[:2]
        action_space = self.act_head_configs.get("action_space", "continuous")

        eos_offset = int(self.tokenizer.eos_token is not None)
        bos_offset = int(self.tokenizer.bos_token is not None)

        history_type = self.act_head_configs.get("history_type", "post")

        if history_type in ["post", "pre"]:
            vision_x = vision_x.reshape(bs * seq_len, *vision_x.shape[2:]).unsqueeze(1)
            # lang_x = lang_x.repeat(seq_len, 1)
            # attention_mask = attention_mask.repeat(seq_len, 1)
            lang_x = lang_x.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
            attention_mask = (
                attention_mask.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
            )
            if vision_gripper is not None:
                vision_gripper = vision_gripper.reshape(
                    bs * seq_len, *vision_gripper.shape[2:]
                ).unsqueeze(1)

        input_embeds = self.word_embedding(lang_x)
        # get <bos> & <eos> offset
        lang_size = (
            lang_x.shape[-1]
            - int(self.tokenizer.eos_token is not None)
            - int(self.tokenizer.bos_token is not None)
        )

        (
            multimodal_embeds,
            mutlimodal_labels,
            multimodal_attention_mask,
            _,
        ) = self.merge_multi_modal_input(
            input_embeds,
            vision_x,
            labels=None,
            attention_mask=attention_mask,
            insert_idx=bos_offset,
        )

        if vision_gripper is not None:
            (
                multimodal_embeds,
                mutlimodal_labels,
                multimodal_attention_mask,
                _,
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                vision_gripper,
                mutlimodal_labels,
                multimodal_attention_mask,
                insert_idx=bos_offset,
            )

        if rel_state is not None and self.use_state:
            insert_idx = multimodal_embeds.shape[1] - int(
                self.tokenizer.eos_token is not None
            )  # insert at last
            state_token = self.encode_state(rel_state)  # bs, seq_len, d
            state_token = state_token.reshape(
                bs * seq_len, state_token.shape[-1]
            ).unsqueeze(
                1
            )  # bs*seq_len, 1, d
            (
                multimodal_embeds,
                mutlimodal_labels,
                multimodal_attention_mask,
                action_token_mask,
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                state_token,
                mutlimodal_labels,
                multimodal_attention_mask,
                is_image=False,
                insert_idx=insert_idx,
                fill_zero=self.act_head_configs.get("fill_zero", False),
            )

        if action_space == "continuous":
            insert_idx = multimodal_embeds.shape[1] - int(
                self.tokenizer.eos_token is not None
            )  # insert at last

            action_tokens = repeat(
                self.action_token,
                "d -> b n d",
                b=multimodal_embeds.shape[0],
                n=self.latent_num,
            )
            (
                multimodal_embeds,
                mutlimodal_labels,
                multimodal_attention_mask,
                action_token_mask,
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                action_tokens,
                mutlimodal_labels,
                multimodal_attention_mask,
                is_image=False,
                insert_idx=insert_idx,
                fill_zero=self.act_head_configs.get("fill_zero", False),
            )

        if history_type == "pre":
            multimodal_embeds = rearrange(
                multimodal_embeds, "(b l) n d -> b (l n) d", l=seq_len
            )
            if multimodal_attention_mask is not None:
                multimodal_attention_mask = rearrange(
                    multimodal_attention_mask, "(b l) n -> b (l n)", l=seq_len
                )

        output = self.model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=multimodal_embeds,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        output_hs = output.hidden_states[-1].clone()
        if history_type == "pre":
            multimodal_embeds = rearrange(
                multimodal_embeds, "b (l n) d -> (b l) n d", l=seq_len
            )
            output_hs = rearrange(output_hs, "b (l n) d -> (b l) n d", l=seq_len)

        if history_type == "video":
            seq_len = 1

        if action_space == "continuous":
            # tmp_mask = torch.all(multimodal_embeds == self.action_token, dim=-1)
            action_hs = output_hs[action_token_mask].reshape(
                bs, seq_len, self.latent_num, -1
            )

        elif action_space == "down_sample":
            action_hs = output_hs
            token_src = self.act_head_configs.get("token_source", "all")

            if token_src == "text":
                # fetch the language tokens
                action_hs = action_hs[
                    :, -lang_size - eos_offset : action_hs.shape[1] - eos_offset
                ].reshape(bs, seq_len, lang_size, -1)
            elif token_src == "vision":
                action_hs = action_hs[:, bos_offset : -lang_size - eos_offset].reshape(
                    bs, seq_len, -1, action_hs.shape[-1]
                )
            elif token_src == "all":
                action_hs = action_hs.reshape(bs, seq_len, *action_hs.shape[1:])
            else:
                raise ValueError(f"Unsupported token source {token_src}")

        else:
            raise ValueError(f"Unsupported action space {action_space}")

        if history_type == "video" and action_hs.ndim == 4:
            action_hs = action_hs.squeeze(1)  # squeeze the seq_len dim

        if self.use_clip_norm and mode == "train":
            clip_loss = self.clip_norm_head(action_hs, raw_text)
            self._update_loss(loss, clip_loss, "clip")

        action_logits, action_loss = self._forward_action_head(
            action_hs, action_labels, action_mask
        )

        # cur = time.time()
        # print("predict action consumes {} sec".format(cur-st))
        # st = cur

        if mode == "train":
            self._update_loss(loss, action_loss, "act")
            loss = self._format_loss(loss)
        else:
            return action_logits

        return loss

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        use_cached_vision_x: bool = False,  # TODO: Do we need this? If not we can remove it from here
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper=None,
        fwd_rgb_labels: torch.Tensor = None,
        fwd_hand_rgb_labels: torch.Tensor = None,
        fwd_mask: torch.Tensor = None,
        instr_and_action_ids=None,
        instr_and_action_labels=None,
        instr_and_action_mask=None,
        raw_text=None,
        data_source=[],
        **kwargs,
    ):
        loss = {}
        if isinstance(data_source, list):
            data_source = "_".join(data_source)

        if "action" in data_source:
            tmp_loss = self.forward_action(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
                action_labels=action_labels,
                action_mask=action_mask,
                caption_labels=caption_labels,
                caption_mask=caption_mask,
                vision_gripper=vision_gripper,
                fwd_rgb_labels=fwd_rgb_labels,
                fwd_hand_rgb_labels=fwd_hand_rgb_labels,
                fwd_mask=fwd_mask,
                instr_and_action_ids=instr_and_action_ids,
                instr_and_action_labels=instr_and_action_labels,
                instr_and_action_mask=instr_and_action_mask,
                raw_text=raw_text,
            )
            loss = self._update_loss(loss, tmp_loss)

        if "vl_pretrain" in data_source:
            tmp_loss = self.forward_vl_task(
                input_ids=instr_and_action_ids,
                labels=instr_and_action_labels,
                attention_mask=instr_and_action_mask,
                images=vision_x,
            )
            loss = self._update_loss(loss, tmp_loss)

        return loss

    def forward_action(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        use_cached_vision_x: bool = False,  # TODO: Do we need this? If not we can remove it from here
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper=None,
        fwd_rgb_labels: torch.Tensor = None,
        fwd_hand_rgb_labels: torch.Tensor = None,
        fwd_mask: torch.Tensor = None,
        instr_and_action_ids=None,
        instr_and_action_labels=None,
        instr_and_action_mask=None,
        raw_text=None,
        rel_state=None,
        **kwargs,
    ):
        action_space = self.act_head_configs.get("action_space", "continuous")
        ### discard the latter visual observation is with_history is False
        ### while we can maintain the multi-step action (chunk) prediction

        if action_space == "discrete":
            return self.forward_discrete(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
                action_labels=action_labels,
                action_mask=action_mask,
                caption_labels=caption_labels,
                caption_mask=caption_mask,
                vision_gripper=vision_gripper,
                fwd_rgb_labels=fwd_rgb_labels,
                fwd_hand_rgb_labels=fwd_hand_rgb_labels,
                fwd_mask=fwd_mask,
                instr_and_action_ids=instr_and_action_ids,
                instr_and_action_labels=instr_and_action_labels,
                instr_and_action_mask=instr_and_action_mask,
            )
        else:
            return self.forward_continuous(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
                action_labels=action_labels,
                action_mask=action_mask,
                caption_labels=caption_labels,
                caption_mask=caption_mask,
                vision_gripper=vision_gripper,
                fwd_rgb_labels=fwd_rgb_labels,
                fwd_hand_rgb_labels=fwd_hand_rgb_labels,
                fwd_mask=fwd_mask,
                instr_and_action_ids=instr_and_action_ids,
                instr_and_action_labels=instr_and_action_labels,
                instr_and_action_mask=instr_and_action_mask,
                raw_text=raw_text,
            )

    def pred_action_discrete(
        self, instr_and_action_ids, vision_x, vision_gripper=None, attention_mask=None
    ):
        assert vision_x is not None
        input_embeds = self.word_embedding(instr_and_action_ids)

        (
            multimodal_embeds,
            _,
            multimodal_attention_mask,
            _,
        ) = self.merge_multi_modal_input(
            input_embeds, vision_x, attention_mask=attention_mask
        )

        if vision_gripper is not None:
            (
                multimodal_embeds,
                _,
                multimodal_attention_mask,
                _,
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                vision_gripper,
                attention_mask=multimodal_attention_mask,
            )

        action_dim = self.act_head_configs["action_dim"]

        generated_ids = []
        kv_cache = None
        self.fwd_pred_next_n = 1
        # import pdb; pdb.set_trace()
        for i in range(action_dim * self.fwd_pred_next_n):
            if kv_cache is None:
                output_hs = self.model(
                    inputs_embeds=multimodal_embeds,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
            else:
                output_hs = self.model(
                    inputs_embeds=multimodal_embeds[:, -1:],
                    past_key_values=kv_cache,
                    use_cache=True,
                )
            kv_cache = output_hs.past_key_values
            cur_id = output_hs.logits[:, -1].argmax(dim=-1)
            generated_ids.append(cur_id)
            cur_embed = self.word_embedding(cur_id)
            multimodal_embeds = torch.cat(
                [multimodal_embeds, cur_embed.unsqueeze(1)], dim=1
            )

        generated_ids = torch.cat(generated_ids, dim=0).reshape(
            self.fwd_pred_next_n, action_dim
        )

        predicted_action_ids = generated_ids[:, -action_dim:].cpu().numpy()
        discretized_actions = self.action_tokenizer.decode_token_ids_to_actions(
            predicted_action_ids
        )

        if isinstance(discretized_actions, list):
            discretized_actions = np.array(discretized_actions)

        discretized_actions[:, -1] = np.where(discretized_actions[:, -1] > 0, 1, -1)

        return discretized_actions

    def inference(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        use_cached_vision_x: bool = False,  # TODO: Do we need this? If not we can remove it from here
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper=None,
        **kwargs,
    ):
        prediction = {}

        assert vision_x is not None
        bs, seq_len = vision_x.shape[:2]
        action_space = self.act_head_configs.get("action_space", "continuous")
        if self.train_setup_configs["predict_action"]:
            if action_space == "discrete":
                action = self.pred_action_discrete(
                    lang_x, vision_x, vision_gripper, attention_mask
                )
                prediction["action"] = action

            else:
                prediction["action"] = self.forward_continuous(
                    vision_x,
                    lang_x,
                    attention_mask,
                    vision_gripper=vision_gripper,
                    mode="inference",
                )

        return prediction


def deep_update(d1, d2):
    # use d2 to update d1
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict)
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1


import json


def load_config(config_file):
    _config = json.load(open(config_file))
    config = {}
    if _config.get("parent", None):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config
