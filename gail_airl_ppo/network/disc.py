import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp


class GAILDiscrim(nn.Module):
    """
    GAIL 鉴别器
    """

    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],  # 鉴别对象为行为状态对
            output_dim=1,  # 输出概率 范围无限制
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))

    def calculate_reward(self, states, actions):
        """
        计算GAIL的奖励
        PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        """
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, actions))


class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True), cnn=False):
        super().__init__()

        self.input_dim = state_shape[0] * state_shape[1] * state_shape[2] if cnn else state_shape[0]

        self.cnn = cnn
        self.g = build_mlp(
            input_dim=self.input_dim,
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=self.input_dim,
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        if self.cnn:
            self.cnn_model = nn.Conv2d(3, 3, (3, 3), padding=(1, 1))

        self.gamma = gamma

    def f(self, states, dones, next_states):
        # f(s,a,s') = g(s,a) + y h(s') -h(s)  f(s,a)=-E(s,a):负能量函数
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        if self.cnn:
            states = self.cnn_model(states)
            states = states.reshape((-1, self.input_dim))
            next_states = self.cnn_model(next_states)
            next_states = next_states.reshape((-1, self.input_dim))

        # 鉴别器输出为sigmoid(f - log_pi)=exp(f)/(exp(f)+pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)  # -log(1/(1+exp(-log d))) =-log(1/1-d)) = log(1-d) - log(d)
