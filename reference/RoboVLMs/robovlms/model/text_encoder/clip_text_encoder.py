import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import open_clip

from robovlms.model.vision_encoder.vision_transformer import clip_vision_encoder


class ClipTextFeatureEncoder(nn.Module):
    def __init__(
        self,
        in_features,
        down_sample="pooling",
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        **kwargs,
    ):
        super().__init__()

        self.model, self.img_preprocess, self.vis_dim = clip_vision_encoder(
            clip_vision_encoder_path, clip_vision_encoder_pretrained
        )
        self.text_dim = self.model.text_projection.shape[1]
        self.lang_fc = nn.Linear(in_features, self.text_dim)
        self.tokenizer = open_clip.get_tokenizer(clip_vision_encoder_path)
        self.down_sample = down_sample
        if self.down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        else:
            pass

    def loss(self, output_hs, clip_text_ids):
        clip_text_feats = self.model.encode_text(clip_text_ids)
        clip_text_feats = clip_text_feats / clip_text_feats.norm(dim=-1, keepdim=True)

        if output_hs.ndim > clip_text_feats.ndim:
            # clip_text_feats = clip_text_feats.unsqueeze(1) # unsqueeze on seq_len dim
            clip_text_feats = repeat(
                clip_text_feats, "b d -> b l d", l=output_hs.shape[1]
            )

        output_hs = self.lang_fc(output_hs)

        l1_loss = F.smooth_l1_loss(output_hs, clip_text_feats)

        return {"text_l1": l1_loss}

    def forward(self, tok_seq, raw_text):
        if self.down_sample == "pooling":
            bs, seq_len = tok_seq.shape[:2]
            tok_seq = rearrange(tok_seq, "b l n d-> (b l) n d")
            tok_seq = self.global_1d_pool(
                tok_seq.permute(0, 2, 1)
            )  # bs*seq_len, n_tok, tok_dim -> bs*seq_len, tok_dim
            tok_seq = rearrange(tok_seq, "(b l) d n -> b l (n d)", b=bs, l=seq_len)
        else:
            tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")

        clip_text_ids = self.tokenizer(raw_text).to(tok_seq.device)

        return self.loss(tok_seq, clip_text_ids)


if __name__ == "__main__":
    in_features = 4096
    clip_vision_encoder_path = "ViT-L-14"
    clip_vision_encoder_pretrained = "openai"
    model = ClipTextFeatureEncoder(
        in_features, clip_vision_encoder_path, clip_vision_encoder_pretrained
    )
    import open_clip

    tokenizer = open_clip.get_tokenizer(clip_vision_encoder_path)
    bs = 4
    output_hs = torch.randn((bs, in_features))
    text = ["rotate the blue block right"] * bs
    clip_text_ids = tokenizer(text)
    loss = model.loss(output_hs, clip_text_ids)
    print(loss)
