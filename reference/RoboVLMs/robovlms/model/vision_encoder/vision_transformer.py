from einops import rearrange

import torch
import torch.nn as nn

import open_clip


def clip_vision_encoder(clip_vision_encoder_path, clip_vision_encoder_pretrained):
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    vision_encoder.visual.output_tokens = True
    vis_dim = open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
        "width"
    ]
    return vision_encoder, image_processor, vis_dim


class VisionTransformer(nn.Module):
    def __init__(
        self, clip_vision_encoder_path, clip_vision_encoder_pretrained, resampler=None
    ):
        super().__init__()
        self.vision_encoder, _ = clip_vision_encoder(
            clip_vision_encoder_path, clip_vision_encoder_pretrained
        )
        self.perceiver = nn.Identity() if resampler is None else resampler

    def forward(self, vision_x):
        """
        vision_x: bs, seq_len, c, h, w
        """
        bs, seq_len = vision_x.shape[:2]
        vision_x = rearrange(vision_x, "b s c h w -> (b s) c h w")
        vision_x = self.vision_encoder(vision_x)  # (b s), v, d
        vision_x = vision_x.unsqueeze(1).unsqueeze(1)
        vision_x = self.perceiver(vision_x)
        pass
