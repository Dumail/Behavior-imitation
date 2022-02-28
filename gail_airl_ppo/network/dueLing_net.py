#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 21:44
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : dueLing_net.py
# @Software: PyCharm

import numpy as np
import torch
from torch import nn


class DuelingNet(nn.Module):
    """
    Dueling 网络模型，输出分为优势函数和价值函数两部分
    """

    def __init__(self, state_shape, action_shape, device='cpu'):
        super(DuelingNet, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 3, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.extractor = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )

        self.extractor_replace = nn.Sequential(
            nn.Linear(3 * 8 * 8, 64),
            nn.ReLU(inplace=True)
        )

        self.state_shape = state_shape
        self.device = device
        # 公共网络
        self.common_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        # 价值函数网络
        self.value_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        # 优势函数网络
        self.advantage_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, np.prod(action_shape))
        )

    def q_value(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
        if len(x.shape) >= 4:
            # 输入为棋盘需要cnn
            temp = self.cnn_layer(x)
            temp = temp.reshape((-1, np.prod(self.state_shape)))
            temp = self.extractor(temp)

            # temp = x.reshape((-1,np.prod(self.state_shape)))
            # temp = self.extractor_replace(temp)
        else:
            temp = x
        common = self.common_layer(temp)
        value = self.value_layer(common)
        advantage = self.advantage_layer(common)
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_value

    def forward(self, x):
        q_value = self.q_value(x)
        q_value = nn.functional.softmax(q_value)
        if len(x.shape) >= 4:
            masks = x[:, 2].detach().reshape(-1, 64)
            if masks.any() > 0:
                expand_mask = torch.zeros(masks.shape[0], 2).to(self.device)
            else:
                expand_mask = torch.ones(masks.shape[0], 2).to(self.device)
            masks = torch.cat((masks, expand_mask), -1)
            q_value = masks * q_value
        return q_value

    def evaluate_log_pi(self, states, actions):
        q_value = self(states)
        # actions.argmax(1).unsqueeze_(1)
        pi = q_value.gather(1, actions)
        log_pi = torch.log(pi + 1e-6).sum(dim=-1, keepdim=True)
        return log_pi
