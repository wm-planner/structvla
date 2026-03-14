from torch import nn


class LinearActionEncoder(nn.Module):
    def __init__(self, c_dim, d_dim, **kwargs):
        super().__init__()
        self.hidden_size = kwargs.get("hidden_size")
        self.c_dim = c_dim
        self.d_dim = d_dim

        # action prediction
        self.act_mlps = nn.ModuleList(
            [
                nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
            ]
        )
        self.arm_mlp = nn.Linear(c_dim, self.hidden_size // 2)  # arm action
        self.gripper_mlp = nn.Linear(
            d_dim, self.hidden_size // 2
        )  # gripper action (binary)

    def forward(self, action, **kwargs):
        """

        action: B x L x (c_dim + d_dim)

        Return:
            action_embed: B x L x h_act
            attn_mask: B x L
            loss (Optional): the additional loss from action encoder
        """
        c_action = action[..., : self.c_dim]
        d_action = action[..., self.c_dim :]
        assert c_action.shape[-1] == self.c_dim and d_action.shape[-1] == self.d_dim
        c_embed = self.arm_mlp(c_action)
        d_embed = self.gripper_mlp(d_action)
        action_embed = c_embed + d_embed
        for mlp in self.act_mlps:
            action_embed = mlp(action_embed)
        return action_embed
