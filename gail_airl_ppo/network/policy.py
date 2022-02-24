import torch
from torch import nn

from .utils import build_mlp, reparameterize, evaluate_lop_pi


# def cnn_wrapper(func):
#     def temp_fun(self, *args):
#         if self.cnn:
#             if isinstance(args, tuple):
#                 temp_states = args[0]
#                 temp_states = torch.reshape(self.cnn_model(temp_states), (-1, self.state_shape[0]))
#                 return func(self, temp_states, *args[1:])
#             else:
#                 temp_states = args
#                 temp_states = torch.reshape(self.cnn_model(temp_states), (-1, self.state_shape[0]))
#                 return func(self, temp_states)
#         else:
#             return func(self, *args)
#
#     return temp_fun


class MaskNet(nn.Module):
    # 用于棋盘的网络
    def __init__(self, *args):
        super(MaskNet, self).__init__()
        self.net = build_mlp(*args)
        self.state_dim = args[0]
        self.action_dim = args[1]
        self.cnn_model = nn.Conv2d(3, 3, (3, 3), padding=(1, 1))
        self.out_activation = nn.Softmax()

    def forward(self, x):
        # masks = x[:, 2].detach().reshape(-1, self.action_dim - 2)
        # masks = torch.cat((masks, torch.zeros(masks.shape[0], 2)), -1)
        temp_states = torch.reshape(self.cnn_model(x), (-1, self.state_dim))
        temp_states = self.net(temp_states)
        temp_states = self.out_activation(temp_states)
        # temp_states = temp_states * masks  # 只在合法动作中选择
        return temp_states


class StateIndependentPolicy(nn.Module):
    # 策略网络模型 输入状态， 输出各行为概率
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(), discrete=False, cnn=False):
        super().__init__()

        self.cnn = cnn
        # if isinstance(action_shape, int):
        #     action_dim = action_shape
        # else:
        self.action_dim = action_shape[0]
        self.state_shape = state_shape

        # 使用cnn作爲棋盤解析器  
        if self.cnn:
            self.net = MaskNet(
                state_shape[0] * state_shape[1] * state_shape[2],
                self.action_dim,
                hidden_units,
                hidden_activation,
                nn.Softmax() if discrete and not cnn else None
            )
        else:
            self.net = build_mlp(
                input_dim=state_shape[0],
                output_dim=self.action_dim,
                hidden_units=hidden_units,
                hidden_activation=hidden_activation,
                output_activation=nn.Softmax() if discrete and not cnn else None
            )
        self.log_stds = nn.Parameter(torch.zeros(1, self.action_dim))  # 行为分布的对数标准差

        # if self.cnn:
        #     self.cnn_model = nn.Conv2d(2, 3, (3, 3), padding=(1, 1))

    def forward(self, states):
        action = torch.tanh(self.net(states))
        if self.cnn:
            masks = states[:, 2].detach().reshape(-1, self.action_dim - 2)
            masks = torch.cat((masks, torch.zeros(masks.shape[0], 2, device=torch.device('cuda'))), -1)
            action = masks * action
        return action

    def sample(self, states):
        action, log_pi = reparameterize(self.net(states), self.log_stds)
        if self.cnn:
            masks = states[:, 2].detach().reshape(-1, self.action_dim - 2)
            masks = torch.cat((masks, torch.zeros(masks.shape[0], 2, device=torch.device('cuda'))), -1)
            action = masks * action
        return action, log_pi

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):
    # 策略网络模型,输入状态，输出行为概率的均值和方差
    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True), cnn=False):
        super().__init__()
        # self.cnn = cnn

        # 构造多层感知机
        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

        # if self.cnn:
        #     self.cnn_model = nn.Conv2d(3, 3, (3, 3), padding=(1, 1))

    # @cnn_wrapper
    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    # @cnn_wrapper
    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))
