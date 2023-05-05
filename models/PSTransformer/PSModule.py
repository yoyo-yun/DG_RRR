# PSModule takes inputs as sequence representation (bs, None, dim) and user/product embedding initialized with zeros (bs, dim)
import torch
from torch import nn


class PSModule(nn.Module):
    def __init__(self, config, out_dim=None):
        super(PSModule, self).__init__()
        self.user_dim = config.usr_dim
        self.item_dim = config.prd_dim
        self.hidden_states = config.hidden_states
        self.projection_scale = nn.Linear(self.user_dim+self.item_dim, self.hidden_states if out_dim is None else out_dim)
        self.projection_bias = nn.Linear(self.user_dim+self.item_dim, self.hidden_states if out_dim is None else out_dim)

    def forward(self, hidden_states, user_hidden_states, item_hidden_states):
        # hidden_states (bs, seq, dim)
        # user or item states (bs, 1, dim)
        fused_information = torch.cat((user_hidden_states, item_hidden_states), dim=-1)  # (bs, 1, dim)
        inject_scale, inject_bias = self.projection_scale(fused_information), self.projection_bias(fused_information)

        assert len(hidden_states.size()) in [2, 3], "wrong hidden_state dim"
        if len(hidden_states.size()) == 2:
            fused_hidden_states = torch.mul(hidden_states, 1 + inject_scale.squeeze(1)).add(inject_bias.squeeze(1))
            # fused_hidden_states = torch.mul(hidden_states, 1 + inject_scale.squeeze(1))
            # fused_hidden_states = hidden_states.add(inject_bias.squeeze(1))
        else:
            fused_hidden_states = torch.mul(hidden_states, 1 + inject_scale).add(inject_bias)
            # fused_hidden_states = torch.mul(hidden_states, 1 + inject_scale)
            # fused_hidden_states = hidden_states.add(inject_bias)
        return fused_hidden_states