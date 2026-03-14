from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F


def lstm_decoder(
    in_features: int, hidden_size: int, num_layers: int, policy_rnn_dropout_p: float
) -> torch.nn.Module:
    return nn.LSTM(
        input_size=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=False,
        batch_first=True,
        dropout=policy_rnn_dropout_p,
    )


class MLPTanhHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPNohHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPSigmoidHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            # torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.mlp(x)


class BasePolicyHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        action_dim,
        action_space="continuous",
        down_sample="pooling",
        latent=1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim

        self.down_sample = down_sample
        self.latent = latent
        self.action_space = action_space

    @staticmethod
    def _get_target_modal_tokens(tok_seq, tok_mask):
        index = tok_mask.nonzero(as_tuple=True)
        return tok_seq[index]

    def get_modal_tokens(self, tok_seq, tok_mask_dict, modal_name):
        assert modal_name in tok_mask_dict, f"{modal_name} not in token sequence"
        return self._get_target_modal_tokens(tok_seq, tok_mask_dict[modal_name])

    def loss(self, pred_action, labels, attention_mask=None):
        """
        pred_action_logits: [bs, seq_len, chunck_size, 7], 1-6 refers to ee pose, 7 refers to gripper open/close
        lables: (pose gt [bs, seq_len, chunck_size, 6], gripper gt [bs, seq_len, chunck_size])
        attention_mask: [bs, seq_len, chunck_size]
        """
        if labels is None or labels[0] is None:
            return {"loss": None}

        if isinstance(pred_action, tuple) or isinstance(pred_action, list):
            if pred_action[0].ndim == pred_action[1].ndim:
                pred_action = torch.cat(pred_action, dim=-1)
            elif pred_action[0].ndim == pred_action[1].ndim + 1:
                pred_action = torch.cat(
                    [pred_action[0], pred_action[1].unsqueeze(-1)], dim=-1
                )
            else:
                raise ValueError("Can not solve the gripper action dim")
        if attention_mask is None:
            pose_loss = torch.nn.functional.huber_loss(pred_action[..., :6], labels[0])
            # pose_loss = torch.nn.functional.mse_loss(pred_action[..., :6], labels[0])
            gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_action[..., -1], labels[1]
            )
        else:
            pose_loss = torch.nn.functional.huber_loss(
                pred_action[..., :6], labels[0], reduction="none"
            )
            # pose_loss = torch.nn.functional.mse_loss(pred_action[..., :6], labels[0], reduction='none')
            attention_mask = attention_mask.bool()
            pose_loss = pose_loss[attention_mask].mean()
            # gripper_loss = torch.nn.functional.binary_cross_entropy(pred_action[..., -1], labels[1], reduction='none')
            gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_action[..., -1], labels[1], reduction="none"
            )
            gripper_loss = gripper_loss[attention_mask].mean()

        gripper_action_preds = (F.sigmoid(pred_action[..., -1]) > 0.5).float()
        acc_gripper_act = torch.eq(gripper_action_preds, labels[1]).float()
        if attention_mask is None:
            acc_gripper_act = acc_gripper_act.mean()
        else:
            # acc_gripper_act = (acc_gripper_act * attention_mask).sum() / attention_mask.sum()
            acc_gripper_act = acc_gripper_act[attention_mask].mean()

        return {
            "loss_arm": pose_loss,
            "loss_gripper": gripper_loss,
            "acc_gripper": acc_gripper_act.item(),
        }

    def get_labels(self, pred_actions, labels, action_masks, **kwargs):
        return pred_actions, labels, action_masks


class DiscreteDecoder(BasePolicyHead):
    def __init__(
        self,
        hidden_size,
        action_dim,
        action_space="continuous",
        down_sample="pooling",
        latent=1,
        cont_token_nun=1,
        n_bin=256,
        min_action=-1,
        max_action=1,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__(
            hidden_size, action_dim, action_space, down_sample, latent, **kwargs
        )
        self.cont_token_num = cont_token_nun
        self.n_bin = n_bin
        self.min_action = min_action
        self.max_action = max_action

        from robovlms.model.policy_head.action_tokenizer import ActionTokenizer

        self.action_tokenizer = ActionTokenizer(
            tokenizer,
            bins=self.n_bin,
            min_action=self.min_action,
            max_action=self.max_action,
        )

    def process_token_sequence(self, tok_seq):
        bs, seq_len = tok_seq.shape[:2]

        if self.action_space == "continuous":
            # flatten the cont_token_num and token_dim dimension
            tok_seq = tok_seq.reshape(bs, seq_len, -1)

        elif self.action_space == "down_sample":
            # swap the latent token and token_dim dimension
            tok_seq = tok_seq.permute(0, 1, 3, 2)

        elif self.action_space == "discrete":
            pass
        else:
            raise ValueError(f"Unsupported action space {self.action_space}")

        return tok_seq

    def forward(self, tok_seq, **kwargs):
        return tok_seq

    def loss(self, pred_action_logits, labels, attention_mask=None):
        """pred_action should be logits for discrete actions"""

        shift_logits = pred_action_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        # shift for auto-regressive
        pred_action_logits = shift_logits
        labels = shift_labels
        # import pdb; pdb.set_trace()
        # TODO forget to consider the last token prediction
        mask = torch.logical_and(
            labels > self.action_tokenizer.action_token_begin_idx,
            labels < self.action_tokenizer.action_token_end_idx,
        )
        pred_action = pred_action_logits.argmax(dim=-1)
        # print(pred_action[mask].view(-1, 7))
        correct_preds = torch.logical_and((pred_action == labels), mask)
        ### get the suffix token for predicted action and label
        correct_preds_cut = torch.masked_select(correct_preds, mask).reshape(-1, 7)
        # action_accuracy = correct_preds.sum().float() / mask.sum().float()
        # arm_acc = correct_preds[...,:6].sum().float() / mask[...,:6].sum().float()
        # gripper_acc = correct_preds[...,-1].sum().float() / mask[...,-1].sum().float()
        arm_acc = (
            correct_preds_cut[:, :6].sum().float() / correct_preds_cut[:, :6].numel()
        )
        gripper_acc = (
            correct_preds_cut[:, -1].sum().float() / correct_preds_cut[:, -1].numel()
        )

        # Compute L1 Loss on Predicted (Continuous) Actions
        # TODO Note that the l1 loss can't add to trainable loss because argmax detach the gradient graph
        continuous_actions_pred = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(
                pred_action[mask].cpu().numpy()
            )
        ).to(shift_logits.device)
        continuous_actions_gt = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(
                labels[mask].cpu().numpy()
            )
        ).to(shift_logits.device)
        action_l1_loss = torch.nn.functional.l1_loss(
            continuous_actions_pred, continuous_actions_gt
        )

        return {
            "loss_arm": loss,
            "action_l1": action_l1_loss,
            "acc_arm": arm_acc,
            "acc_gripper": gripper_acc,
        }


def initialize_param(model):
    with torch.no_grad():
        for m in model.children():
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias"):
                    m.bias.fill_(0)
            else:
                initialize_param(m)


class FCDecoder(BasePolicyHead):
    def __init__(
        self,
        in_features,
        hidden_size,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        **kwargs,
    ):
        super(FCDecoder, self).__init__(hidden_size, action_dim, **kwargs)
        self.down_sample = down_sample
        self.latent = latent
        self.fwd_pred_next_n = fwd_pred_next_n
        self.actions = MLPTanhHead(
            self.hidden_size * latent, fwd_pred_next_n * (self.action_dim - 1)
        )
        self.gripper = MLPSigmoidHead(self.hidden_size * latent, fwd_pred_next_n)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features * latent, in_features * latent // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features * latent // 2, hidden_size * latent),
        )
        if self.down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
        elif self.down_sample == "resampler":
            pass
        elif self.down_sample == "none":
            pass
        else:
            raise NotImplementedError
        initialize_param(self)

    def forward(self, tok_seq, **kwargs):
        if len(tok_seq.shape) == 4:
            bs, seq_len, n_tok, tok_dim = tok_seq.shape
            tok_seq = rearrange(
                tok_seq, "b l n d-> (b l) n d"
            )  # reduce the seq_len dim
        elif tok_seq.dim() == 3:
            bs, n_tok, tok_dim = tok_seq.shape
            seq_len = None
        else:
            assert len(tok_seq.shape) == 2
            bs, tok_dim = tok_seq.shape
            seq_len = None
            n_tok = None
            tok_seq = tok_seq.unsqueeze(1)

        # here tok_seq is (bs*seq_len, n_tok, tok_dim)
        if self.down_sample == "pooling":
            tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1))
            tok_seq = rearrange(tok_seq, "b d n -> b (n d)")
        elif self.down_sample == "resampler":
            raise NotImplementedError
        elif self.down_sample == "none":
            tok_seq = rearrange(tok_seq, "b n d -> b (n d)")
        else:
            raise NotImplementedError

        tok_seq = self.mlp(tok_seq)
        actions = self.actions(tok_seq)
        gripper = self.gripper(tok_seq)
        if seq_len is not None:
            # input is 4-dim
            actions = rearrange(
                actions,
                "(b l) (n d) -> b l n d",
                b=bs,
                l=seq_len,
                n=self.fwd_pred_next_n,
            )
            gripper = rearrange(
                gripper,
                "(b l) (n d) -> b l n d",
                b=bs,
                l=seq_len,
                n=self.fwd_pred_next_n,
            )
        elif n_tok is not None:
            # input is 3-dim
            actions = rearrange(
                actions, "b (n d) -> b n d", b=bs, n=self.fwd_pred_next_n
            )
            gripper = rearrange(
                gripper, "b (n d) -> b n d", b=bs, n=self.fwd_pred_next_n
            )

        return actions, gripper


class LSTMDecoder(BasePolicyHead):
    def __init__(
        self,
        in_features,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        window_size,
        hidden_size=1024,
        num_layers=4,
        policy_rnn_dropout_p=0.0,
        **kwargs,
    ):
        super(LSTMDecoder, self).__init__(in_features, action_dim, **kwargs)
        self.down_sample = down_sample
        self.latent = latent
        self.window_size = window_size
        self.history_len = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.history_memory = []
        self.hidden_size = hidden_size
        # TODO if there is needed for in_features*latents
        self.rnn = lstm_decoder(
            in_features * latent, hidden_size * latent, num_layers, policy_rnn_dropout_p
        )
        self.actions = MLPTanhHead(
            self.hidden_size * latent, fwd_pred_next_n * (self.action_dim - 1)
        )
        self.gripper = MLPSigmoidHead(self.hidden_size * latent, fwd_pred_next_n)
        self.hidden_state = None
        if self.down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
        elif self.down_sample == "resampler":
            raise NotImplementedError
        elif self.down_sample == "none":
            pass
        else:
            raise NotImplementedError
        initialize_param(self)

    def reset(self):
        self.hidden_state = None
        self.history_memory = []

    def forward(self, tok_seq, h_0=None, **kwargs):
        # import pdb; pdb.set_trace()
        """
        [bs, window_size, latent num, feature_dim]
        """
        if self.down_sample == "pooling":
            bs, seq_len = tok_seq.shape[:2]
            tok_seq = rearrange(tok_seq, "b l n d-> (b l) n d")
            tok_seq = self.global_1d_pool(
                tok_seq.permute(0, 2, 1)
            )  # bs*seq_len, n_tok, tok_dim -> bs*seq_len, tok_dim
            tok_seq = rearrange(tok_seq, "(b l) d n -> b l (n d)", b=bs, l=seq_len)
        elif self.down_sample == "resampler":
            raise NotImplementedError
        elif self.down_sample == "none":
            tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")
        else:
            raise NotImplementedError

        if tok_seq.shape[1] == 1:
            self.history_memory.append(tok_seq)
            if len(self.history_memory) <= self.history_len:
                # print('cur hist_mem len: {}'.format(len(self.history_memory)))
                x, h_n = self.rnn(tok_seq, self.hidden_state)
                self.hidden_state = h_n
                x = x[:, -1].unsqueeze(1)
                self.rnn_out = x.squeeze(1)
            else:
                # the hidden state need to be refreshed based on the history window
                # print('hist_mem exceeded, refresh hidden state')
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                self.hidden_state = None
                x, h_n = self.rnn(hist_feature, self.hidden_state)
                x = x[:, -1].unsqueeze(1)
        else:
            self.hidden_state = h_0
            x, h_n = self.rnn(tok_seq, self.hidden_state)
            self.hidden_state = h_n

        # self.hidden_state = h_0
        # x, h_n = self.rnn(tok_seq, self.hidden_state)
        # self.hidden_state = h_n
        actions = self.actions(x)
        gripper = self.gripper(x)

        actions = rearrange(actions, "b l (n d) -> b l n d", n=self.fwd_pred_next_n)
        gripper = rearrange(gripper, "b l (n d) -> b l n d", n=self.fwd_pred_next_n)

        return actions, gripper


class GPTDecoder(BasePolicyHead):
    def __init__(
        self,
        in_features,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        window_size,
        hidden_size=1024,
        **kwargs,
    ):
        super(GPTDecoder, self).__init__(in_features, action_dim, **kwargs)
        self.down_sample = down_sample
        self.latent = latent

        self.window_size = window_size
        self.history_len = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.history_memory = []
        self.hidden_size = hidden_size

        # self.history_len = 1
        # self.window_size = 1

        from robovlms.model.policy_head.trajectory_gpt2 import get_gpt_model

        self.gpt = get_gpt_model(self.hidden_size * latent, window_size)

        if hidden_size != in_features:
            self.fc = nn.Linear(in_features, hidden_size)
        else:
            self.fc = nn.Identity()

        self.actions = MLPTanhHead(
            self.hidden_size * latent, fwd_pred_next_n * (self.action_dim - 1)
        )
        self.gripper = MLPSigmoidHead(self.hidden_size * latent, fwd_pred_next_n)

        if self.down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
        elif self.down_sample == "resampler":
            raise NotImplementedError
        elif self.down_sample == "none":
            pass
        else:
            raise NotImplementedError

        # initialize_param(self)

    def forward(self, tok_seq, **kwargs):
        """
        [bs, window_size, latent num, feature_dim]
        """
        # import pdb; pdb.set_trace()
        if self.down_sample == "pooling":
            bs, seq_len = tok_seq.shape[:2]
            tok_seq = rearrange(tok_seq, "b l n d-> (b l) n d")
            tok_seq = self.global_1d_pool(
                tok_seq.permute(0, 2, 1)
            )  # bs*seq_len, n_tok, tok_dim -> bs*seq_len, tok_dim
            tok_seq = rearrange(tok_seq, "(b l) d n -> b l (n d)", b=bs, l=seq_len)
        elif self.down_sample == "resampler":
            raise NotImplementedError
        elif self.down_sample == "none":
            tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")
        else:
            raise NotImplementedError

        time_step = None
        attention_mask = None
        tok_seq = self.fc(tok_seq)

        if tok_seq.shape[1] == 1:
            self.history_memory.append(tok_seq)

            if len(self.history_memory) <= self.history_len:
                hist_feature = torch.cat(self.history_memory, dim=1)
                x = self.gpt(hist_feature, time_step, attention_mask)
                x = x[:, -1].unsqueeze(1)

            else:
                # the hidden state need to be refreshed based on the history window
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                x = self.gpt(hist_feature, time_step, attention_mask)
                x = x[:, -1].unsqueeze(1)

        else:
            x = self.gpt(tok_seq, time_step, attention_mask)

        actions = self.actions(x)
        gripper = self.gripper(x)

        actions = rearrange(actions, "b l (n d) -> b l n d", n=self.fwd_pred_next_n)
        gripper = rearrange(gripper, "b l (n d) -> b l n d", n=self.fwd_pred_next_n)

        return actions, gripper

    def get_pattern_name(self):
        return "gpt_{}_".format(
            self.hidden_size,
        )


if __name__ == "__main__":
    net = FCDecoder(
        in_features=1024,
        hidden_size=1024,
        action_dim=7,
        down_sample="pooling",
        latent=1,
        fwd_pred_next_n=2,
        window_size=12,
    )
    import pdb

    pdb.set_trace()
    net = LSTMDecoder(
        in_features=1024,
        action_dim=7,
        down_sample="pooling",
        latent=1,
        fwd_pred_next_n=2,
        window_size=12,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    bs = 5
    window_size = 12
    text_len = 8
    tokens = torch.randn(bs, window_size, text_len, 1024)
    # actions, gripper = net(tokens)
    # pred_action_logitss = torch.cat([actions, gripper.unsqueeze(-1)], dim=-1)
    labels = (torch.randn(bs, window_size, 2, 6), torch.ones(bs, window_size, 2))
    att_mask = torch.ones(bs, window_size, 2)
    for i in range(10000):
        actions, gripper = net(tokens)
        pred_action_logitss = torch.cat([actions, gripper.unsqueeze(-1)], dim=-1)
        optimizer.zero_grad()
        loss = net.loss(pred_action_logitss, labels, att_mask)

        loss_arm = loss["loss_arm"]
        loss_gripper = loss["loss_gripper"]
        acc_gripper = loss["acc_gripper"]
        loss_act = loss_arm + 0.01 * loss_gripper
        loss_act.backward()
        optimizer.step()
        print(
            "iter: {}, loss: {} gripper: {} acc: {}".format(
                i, loss_act.item(), loss_gripper.item(), acc_gripper
            )
        )
    print(loss)
    pass
