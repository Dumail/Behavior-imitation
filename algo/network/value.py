import torch
from torch import nn

from .utils import build_mlp


class StateFunction(nn.Module):
    """价值函数网络"""

    def __init__(self, state_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(), cnn=False):
        super().__init__()

        self.cnn = cnn
        self.state_dim = state_shape[0] * state_shape[1] * state_shape[2] if cnn else state_shape[0]
        self.net = build_mlp(
            input_dim=self.state_dim,
            output_dim=1,  # 仅输出状态价值
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

        if self.cnn:
            self.cnn_model = nn.Conv2d(3, 3, (3, 3), padding=(1, 1))

    def forward(self, states):
        if self.cnn:
            states = self.cnn_model(states)
            states = states.reshape((-1, self.state_dim))
        return self.net(states)


class StateActionFunction(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))


class TwinnedStateActionFunction(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net1 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.net2 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        xs = torch.cat([states, actions], dim=-1)
        return self.net1(xs), self.net2(xs)

    def q1(self, states, actions):
        return self.net1(torch.cat([states, actions], dim=-1))
