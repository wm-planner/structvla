from einops import rearrange, repeat
import copy
from typing import Tuple
import numpy as np

import torch
from torch import nn

from robovlms.utils.model_utils import build_tokenizer, get_target_modal_tokens
from robovlms.model.vision_encoder.vision_transformer import clip_vision_encoder
from robovlms.model.flamingo_builder import build_llm_flamingo, FLAMINGO_MODEL_CONFIGS
from robovlms.model.backbone.base_backbone import BaseRoboVLM, load_config, deep_update

try:
    from open_flamingo.src.flamingo_lm import FlamingoLMMixin
    from open_flamingo.src.utils import extend_instance
    from open_flamingo.src.factory import _infer_decoder_layers_attr_name
except:
    pass


class RoboFlamingo(BaseRoboVLM):
    @property
    def hidden_size(self):
        if hasattr(self.lang_config, "d_model"):
            return self.lang_config.d_model  # mpt uses d_model
        else:
            return self.lang_config.hidden_size

    @property
    def word_embedding(self):
        return self.model.get_input_embeddings()

    @property
    def text_tower(self):
        return self.model

    @property
    def vision_tower(self):
        return self.vision_encoder

    @property
    def model(self):
        return self.backbone

    def _init_llm(self):
        lang_encoder = build_llm_flamingo(self.configs["vlm"])
        lang_encoder_path = (
            self.configs["vlm"]
            if isinstance(self.configs["vlm"], str)
            else self.configs["vlm"]["pretrained_model_name_or_path"]
        )
        if "mpt-1b" in lang_encoder_path or "MPT1b" in lang_encoder_path:

            class EmbeddingFnMixin:
                def get_input_embeddings(self):
                    return self.transformer.wte

                def set_input_embeddings(self, new_embeddings):
                    self.transformer.wte = new_embeddings

            extend_instance(lang_encoder, EmbeddingFnMixin)

        extend_instance(lang_encoder, FlamingoLMMixin)

        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)

        lang_encoder.resize_token_embeddings(len(self.tokenizer))

        if hasattr(lang_encoder.config, "d_model"):
            hidden_size = lang_encoder.config.d_model  # mpt uses d_model
        else:
            hidden_size = lang_encoder.config.hidden_size

        self.lang_config = lang_encoder.config

        lang_encoder.init_flamingo(
            media_token_id=self.media_token_id,
            lang_hidden_size=hidden_size,
            gradient_checkpointing=False,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=FLAMINGO_MODEL_CONFIGS[
                self.configs["vlm"]["name"]
            ]["cross_attn_every_n_layers"],
        )

        self.num_transformer_params = sum(
            [p.numel() for p in lang_encoder.parameters()]
        )

        return lang_encoder

    def load_openflamingo_ckpt(self):
        checkpoint_path = FLAMINGO_MODEL_CONFIGS[self.configs["vlm"]["name"]][
            "openflamingo_checkpoint"
        ]
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        msg = self.load_state_dict(ckpt, strict=False)
        print(f"OpenFlamingo Checkpoint Loaded!")

        if self.configs["vlm"]["residual"]:
            self.model.clone_parameters()

    def _init_backbone(self):
        self.tokenizer = build_tokenizer(self.configs["tokenizer"])
        self.eoc_token_id = self.tokenizer.encode("<|endofchunk|>")[-1]
        self.media_token_id = self.tokenizer.encode("<image>")[-1]
        # Initialize vision encoder
        (
            self.vision_encoder,
            self.clip_preprocess,
            self.vis_dim,
        ) = self._init_vision_encoder()
        from open_flamingo.src.helpers import PerceiverResampler

        self.perceiver = PerceiverResampler(dim=self.vis_dim)

        self.model = self._init_llm()

        if self.train_setup_configs.get("load_vl_param", True):
            self.load_openflamingo_ckpt()
        return self.tokenizer, self.model

    def _init_vision_encoder(self):
        return clip_vision_encoder(
            self.configs["vision_encoder"]["clip_vision_encoder_path"],
            self.configs["vision_encoder"]["clip_vision_encoder_pretrained"],
        )

    def _trainable_params_setup(self):
        self.requires_grad_(False)
        if self.train_setup_configs["train_vision"]:
            self.vision_encoder.requires_grad_(True)
        if self.train_setup_configs["train_decoder_layers"] == -1:
            self.model.gated_cross_attn_layers.requires_grad_(True)
        else:
            assert self.train_setup_configs["train_decoder_layers"] <= len(
                self.model.gated_cross_attn_layers
            ), "train_decoder_layers should be less than the number of layers in the decoder"
            ix = self.train_setup_configs["train_decoder_layers"]
            for layer in self.model.gated_cross_attn_layers[-ix:]:
                layer.requires_grad_(True)

        if self.train_setup_configs["train_full_decoder"]:
            self.model.requires_grad_(True)
        if self.train_setup_configs["train_resampler"]:
            self.perceiver.requires_grad_(True)
        else:
            self.perceiver.requires_grad_(False)
        if self.train_setup_configs["train_text_embedding"]:
            self.model.get_input_embeddings().requires_grad_(True)
        else:
            self.model.get_input_embeddings().requires_grad_(False)

        self.act_head.requires_grad_(True)

    @staticmethod
    def _get_target_modal_tokens(tok_seq, tok_mask):
        index = tok_mask.nonzero(as_tuple=True)
        return tok_seq[index]

    def get_modal_tokens(self, tok_seq, tok_mask_dict, modal_name):
        assert modal_name in tok_mask_dict, f"{modal_name} not in token sequence"
        return self._get_target_modal_tokens(tok_seq, tok_mask_dict[modal_name])

    def _get_obs_embed(self, rgb):
        batch_size, seq_length, c, h, w = rgb.shape
        rgb = rgb.reshape(batch_size * seq_length, c, h, w)
        # print('rgb input shape', rgb.shape)
        patch_embeddings = (
            self.vision_encoder.visual(rgb)[1].unsqueeze(1).unsqueeze(1)
        )  # b*l, 1, 1, v, d
        # print('path_embedding shape after vit', patch_embeddings.shape)
        # patch_embeddings = patch_embeddings.view(batch_size, seq_length, *patch_embeddings.shape[1:])

        # patch_embeddings = patch_embeddings.unsqueeze(1).unsqueeze(1) # b*l, 1, 1, v, d
        patch_embeddings = self.perceiver(patch_embeddings)  # b*l, 1, n, d

        return patch_embeddings.reshape(
            batch_size, seq_length, *patch_embeddings.shape[-2:]
        )
        # return patch_embeddings

    def _encode_multi_vision_post_fusion(
        self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor = None
    ):
        vision_rgb = self._get_obs_embed(vision_rgb)
        if vision_gripper is not None:
            vision_gripper = self._get_obs_embed(vision_gripper)
            vision_rgb = torch.cat(
                [vision_rgb, vision_gripper], dim=2
            )  # reshapes to (b, T, 2*n, d)

        for layer in self.model._get_decoder_layers():
            layer.condition_vis_x(vision_rgb)

        return vision_rgb

    def cat_multi_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_ids: torch.Tensor = None,
        insert_idx: int = 0,
        attention_masks: torch.Tensor = None,
    ):
        bs, seq_len = input_ids.shape[:2]
        device = input_ids.device
        if insert_idx >= 0:
            return_ids = torch.cat(
                (input_ids[:, :insert_idx], multimodal_ids, input_ids[:, insert_idx:]),
                dim=1,
            )
            insert_masks = torch.cat(
                (
                    torch.zeros(bs, insert_idx),
                    torch.ones(multimodal_ids.shape),
                    torch.zeros(bs, seq_len - insert_idx),
                ),
                dim=1,
            )

        elif insert_idx == -1 and attention_masks is not None:
            new_id_list = []
            new_mask_list = []
            for mask, input_id, multimodal_id in zip(
                attention_masks, input_ids, multimodal_ids
            ):
                indexs = (mask == False).nonzero()
                insert_idx = indexs[0].item() if len(indexs) > 0 else len(mask)
                insert_idx -= self.eos_offset
                new_embed = torch.cat(
                    (input_id[:insert_idx], multimodal_id, input_id[insert_idx:]), dim=0
                )
                new_mask = torch.cat(
                    (
                        torch.zeros(insert_idx),
                        torch.ones(multimodal_id.shape),
                        torch.zeros(seq_len - insert_idx),
                    ),
                    dim=0,
                )
                new_id_list.append(new_embed)
                new_mask_list.append(new_mask)
            return_ids = torch.stack(new_id_list, dim=0)
            insert_masks = torch.stack(new_mask_list, dim=0)
        else:
            raise Exception(
                "insert_idx should be -1 or >= 0, and if you want to insert as last(-1), you should provide masks"
            )
        return_ids = return_ids.to(device)
        insert_masks = insert_masks.to(device).bool()

        return return_ids, insert_masks

    @property
    def eos_offset(self):
        return int(self.tokenizer.eos_token is not None)

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
        mode="train",
        **kwargs,
    ):
        loss = {}
        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )
        bs, seq_len = vision_x.shape[:2]
        action_space = self.act_head_configs.get("action_space", "continuous")
        if seq_len > 1:
            lang_x = lang_x.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
            attention_mask = (
                attention_mask.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
            )

            vision_x = vision_x.reshape(bs * seq_len, *vision_x.shape[2:]).unsqueeze(1)
            if vision_gripper is not None:
                vision_gripper = vision_gripper.reshape(
                    bs * seq_len, *vision_gripper.shape[2:]
                ).unsqueeze(1)

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.model.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_multi_vision_post_fusion(vision_x, vision_gripper)
        if action_space == "continuous":
            action_ids = torch.full(
                (bs * seq_len, self.latent_num), self.action_token_id
            ).to(lang_x.device)
            tmp_action_masks = torch.ones_like(action_ids)
            input_ids, action_ids_mask = self.cat_multi_input_ids(
                lang_x, action_ids, -1, attention_mask
            )
            attention_mask, _ = self.cat_multi_input_ids(
                attention_mask, tmp_action_masks, -1, attention_mask
            )
        else:
            input_ids = lang_x
        # print(lang_x.shape, attention_mask.shape)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask.bool(),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        output_hs = output.hidden_states[-1].clone()

        if self.train_setup_configs["predict_action"] and (
            action_labels is not None or mode == "inference"
        ):
            # output_hs = output.hidden_states[-1].clone()
            if action_space == "continuous":
                action_hs = output_hs[action_ids_mask].reshape(
                    bs, seq_len, self.latent_num, -1
                )
            elif action_space == "down_sample":
                action_hs = output_hs.reshape(bs, seq_len, *output_hs.shape[-2:])

            action_logits, action_loss = self._forward_action_head(
                action_hs, action_labels, action_mask
            )
            if mode == "train":
                self._update_loss(loss, action_loss, "act")
            else:
                return action_logits

        loss = self._format_loss(loss)

        return loss

    def forward_discrete(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_cached_vision_x: bool = False,
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
        mode="train",
        **kwargs,
    ):
        loss = {}
        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )
        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.model.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_multi_vision_post_fusion(vision_x, vision_gripper)

        if instr_and_action_ids.ndim == 2:
            instr_and_action_ids = instr_and_action_ids.unsqueeze(1)

        bs, window_size = instr_and_action_ids.shape[:2]
        instr_and_action_ids = instr_and_action_ids.flatten(0, 1)
        media_ids = torch.full((bs * window_size, 1), self.media_token_id).to(
            instr_and_action_ids.device
        )
        instr_and_action_ids, _ = self.cat_multi_input_ids(
            instr_and_action_ids, media_ids
        )

        # import pdb; pdb.set_trace()
        if mode != "train":
            action_dim = self.act_head_configs["action_dim"]
            action_ids = self.model.generate(
                input_ids=instr_and_action_ids, max_new_tokens=action_dim
            )
            action_ids = action_ids[0, -action_dim:].cpu().numpy()
            discretized_actions = self.action_tokenizer.decode_token_ids_to_actions(
                action_ids
            )
            action = np.array(discretized_actions)
            action[-1] = 1 if action[-1] > 0 else -1
            return action

        instr_and_action_labels = instr_and_action_labels.flatten(0, 1)
        instr_and_action_mask = instr_and_action_mask.flatten(0, 1)

        media_mask = torch.ones_like(media_ids)
        media_labels = torch.full_like(media_ids, -100)

        instr_and_action_labels, _ = self.cat_multi_input_ids(
            instr_and_action_labels, media_labels
        )
        instr_and_action_mask, _ = self.cat_multi_input_ids(
            instr_and_action_mask, media_mask
        )

        instr_and_action_ids, instr_and_action_labels, instr_and_action_mask = (
            rearrange(
                tensor,
                "(bs ws) seq_len ... -> bs (ws seq_len) ...",
                bs=bs,
                ws=window_size,
            )
            for tensor in (
                instr_and_action_ids,
                instr_and_action_labels,
                instr_and_action_mask,
            )
        )

        output = self.model(
            input_ids=instr_and_action_ids,
            attention_mask=instr_and_action_mask.bool(),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        if (
            self.train_setup_configs["predict_action"]
            and instr_and_action_labels is not None
        ):
            output_hs = output.logits
            action, action_loss = self._forward_action_head(
                output_hs, instr_and_action_labels
            )
            self._update_loss(loss, action_loss, "act")

        if self.train_setup_configs["predict_caption"] and caption_labels is not None:
            logits = output.logits.clone()
            text_selector = self._caption_mask()
            logits = get_target_modal_tokens(logits, text_selector)
            if caption_mask is None:
                caption_mask = attention_mask
            _, caption_loss = self._forward_caption(
                logits,
                caption_labels,
                caption_mask,
            )
            self._update_loss(loss, caption_loss, "cap")

        loss = self._format_loss(loss)

        return loss

    def pred_action_discrete(
        self,
        instr_and_action_ids,
        vision_x,
        vision_gripper=None,
        attention_mask=None,
        use_cached_vision_x=False,
    ):
        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )
        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.model.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_multi_vision_post_fusion(vision_x, vision_gripper)

        action_dim = self.act_head_configs["action_dim"]
        generated_ids = []

        for i in range(action_dim):
            output_hs = self.model(input_ids=instr_and_action_ids)
            kv_cache = output_hs.past_key_values
            cur_id = output_hs.logits[:, -1].argmax(dim=-1)
            generated_ids.append(cur_id)
            instr_and_action_ids = torch.cat(
                [instr_and_action_ids, cur_id.unsqueeze(0)], dim=1
            )

        # generated_ids = torch.cat(generated_ids, dim=0).unsqueeze(0)
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
        # discretized_actions[:, -1] = 1 if discretized_actions[:, -1] > 0 else -1

        return discretized_actions


if __name__ == "__main__":
    configs = load_config(
        "configs/finetune_flamingo_mpt_1b_ift_hist=8_act=10_lstm_calvin.json"
    )
    use_hand_rgb = False  # True
    model = RoboFlamingo(
        configs=configs,
        train_setup_configs=configs["train_setup"],
        fwd_head_configs=None,
        window_size=configs["window_size"],
        use_hand_rgb=use_hand_rgb,
        act_head_configs=configs["act_head"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
        use_state=True,
    )
    import pdb

    pdb.set_trace()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"LLaVA Model Parameters: {total_params / 1000000:.2f}M")
    bs, seq_len = 2, 2
    device = "cuda:0"
    # device = 'cpu'
    img_size = configs["image_size"]
    vision_x = torch.zeros(
        (bs, seq_len, 3, img_size, img_size), dtype=torch.float16
    ).to(device)
    vision_gripper = torch.zeros(
        (bs, seq_len, 3, img_size, img_size), dtype=torch.float16
    ).to(device)
    lang_x = torch.ones((bs, 10), dtype=torch.long).to(device) * 100
    attention_mask = torch.ones((bs, 10)).bool().to(device)
    action_lables = (
        torch.randn(bs, seq_len, configs["fwd_pred_next_n"], 6).to(device),
        torch.zeros(bs, seq_len, configs["fwd_pred_next_n"]).to(device),
    )
    model = model.to(device).to(torch.float16)
    test_res = model(
        vision_x,
        lang_x,
        attention_mask=attention_mask,
        position_ids=None,
        use_cached_vision_x=False,
        action_labels=action_lables,
        action_mask=None,
        caption_labels=None,
        caption_mask=None,
        past_key_values=None,
        use_cache=False,
        vision_gripper=vision_gripper,
        fwd_rgb_labels=None,
        fwd_hand_rgb_labels=None,
        fwd_mask=None,
        data_source=["calvin_action"],
    )

    print(test_res)
