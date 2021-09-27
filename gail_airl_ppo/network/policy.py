import torch
from torch import nn

from .utils import build_mlp, reparameterize, evaluate_lop_pi


class StateIndependentPolicy(nn.Module):
    # 策略网络模型 输入状态， 输出各行为概率
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        if isinstance(action_shape,int):
            action_dim = action_shape
        else:
            action_dim = action_shape[0]

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_dim,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_dim)) # 行为分布的对数标准差

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):
    # 策略网络模型,输入状态，输出行为概率的均值和方差
    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        # 构造多层感知机
        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))
