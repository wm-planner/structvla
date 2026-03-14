import torch

from robovlms.model.backbone.base_backbone import BaseRoboVLM, load_config, deep_update


class RoboLLaVA(BaseRoboVLM):
    @property
    def hidden_size(self):
        if hasattr(self.model.config, "d_model"):
            return self.model.config.d_model  # mpt uses d_model
        else:
            return self.model.config.hidden_size

    @property
    def word_embedding(self):
        return self.text_tower.wte

    @property
    def text_tower(self):
        return self.model.transformer

    @property
    def vision_tower(self):
        return self.model.get_vision_tower()

    @property
    def model(self):
        return self.backbone

    def encode_images(self, images, image_sizes=None):
        if images.ndim == 4:
            images = images.unsqueeze(1)
        bs, seq_len = images.shape[:2]
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.model.encode_images(concat_images)

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(
                self.model_config, "mm_patch_merge_type", "flat"
            )
            image_aspect_ratio = getattr(
                self.model_config, "image_aspect_ratio", "square"
            )
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = (
                            width
                        ) = self.model.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == "anyres":
                            from llava.mm_utils import get_anyres_image_grid_shape

                            (
                                num_patch_width,
                                num_patch_height,
                            ) = get_anyres_image_grid_shape(
                                image_sizes[image_idx],
                                self.model_config.image_grid_pinpoints,
                                self.model.get_vision_tower().config.image_size,
                            )
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                        else:
                            raise NotImplementedError
                        if "unpad" in mm_patch_merge_type:
                            from llava.model.llava_arch import unpad_image

                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat(
                            (base_image_feature, image_feature), dim=0
                        )
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.model.image_newline[None].to(
                                        image_feature.device
                                    ),
                                ),
                                dim=0,
                            )
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {self.model_config.mm_patch_merge_type}"
                )
        else:
            image_features = self.model.encode_images(images)

        image_features = torch.stack(image_features, dim=0).view(
            bs, seq_len, -1, image_features[0].shape[-1]
        )

        if self.use_vision_resampler:
            ### downsample at token num dim: b, s, n, d -> b, s, v d
            # b T F v d -> b, T, n, d
            image_features = self.vision_resampler(
                image_features.unsqueeze(2)
            )  # downsample v_tok per image to n_tok

        return image_features


if __name__ == "__main__":
    configs = load_config(
        "configs/finetune_mpt-7b_cont-lstm-post_ful_ft_wd=0_hist=8_act=10_aug-shift_warmup=0.25-epoch_lr=2e-5_act-norm.json"
    )
    use_hand_rgb = False  # True
    model = RoboLLaVA(
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
