import torch

from robovlms.model.backbone.base_backbone import BaseRoboVLM, load_config, deep_update


class RoboQwen(BaseRoboVLM):
    @property
    def image_processor(self):
        return self.model.transformer.visual.image_transform

    @property
    def model(self):
        return self.backbone

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def word_embedding(self):
        return self.model.transformer.wte

    @property
    def text_tower(self):
        return self.model.transformer

    @property
    def vision_tower(self):
        return self.model.transformer.visual

    @property
    def model(self):
        return self.backbone

    @property
    def start_image_token_id(self):
        return torch.LongTensor([self.model.config.visual["image_start_id"]])

    @property
    def end_image_token_id(self):
        return torch.LongTensor([self.model.config.visual["image_start_id"] + 1])

    def encode_images(self, images):
        # input: images: list of b,c,h,w or b,t,c,h,w
        # output: image_features: list of bx[txn, d]
        if images.ndim == 4:
            images = images.unsqueeze(1)
        bs, seq_len = images.shape[:2]
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            images = torch.cat([image for image in images], dim=0)
        image_features = self.vision_tower(images)
        image_features = image_features.view(
            bs, seq_len, -1, image_features[0].shape[-1]
        )
        # import pdb; pdb.set_trace()
        if self.use_vision_resampler:
            ### downsample at token num dim: b, s, n, d -> b, s, v d
            # b T F v d -> b, T, n, d
            image_features = self.vision_resampler(
                image_features.unsqueeze(2)
            )  # downsample v_tok per image to n_tok

        return image_features


if __name__ == "__main__":
    configs = load_config(
        "configs/finetune_qwen-vl-7b_cont-lstm-post_full_ft_text_vision_wd=0_hist=8_act=10_aug-shift_act-norm.json"
    )
    use_hand_rgb = False  # True
    model = RoboQwen(
        configs=configs,
        train_setup_configs=configs["train_setup"],
        fwd_head_configs=None,
        window_size=configs["window_size"],
        use_hand_rgb=use_hand_rgb,
        act_head_configs=configs["act_head"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
        use_state=True,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Qwen Model Parameters: {total_params / 1000000:.2f}M")
    bs, seq_len = 2, 4
    device = "cuda:0"
    model = model.to(device)
    # device = 'cpu'
    img_size = 448
    fwd_next_n = configs["fwd_pred_next_n"]
    vision_x = torch.zeros(
        (bs, seq_len, 3, img_size, img_size), dtype=torch.float16
    ).to(device)
    vision_gripper = torch.zeros(
        (bs, seq_len, 3, img_size, img_size), dtype=torch.float16
    ).to(device)
    lang_x = torch.ones((bs, 10), dtype=torch.long).to(device) * 100
    attention_mask = torch.ones((bs, 10)).bool().to(device)
    action_lables = (
        torch.randn(bs, seq_len, fwd_next_n, 6).to(device),
        torch.zeros(bs, seq_len, fwd_next_n).to(device),
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
        data_source="action",
    )

    print(test_res)
