#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/10/28 下午1:53
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : super_test.py
# @Software: PyCharm
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.network.dueLing_net import DuelingNet

state_shape = np.array([3, 8, 8])
action_shape = np.array([66])
device = torch.device('cuda')

expert_buffer = "./buffers/568507529aab81be75ae4316_size2400_train.pth"
test_expert_buffer = "./buffers/568507529aab81be75ae4316_size800_test.pth"
expert_buffer = SerializedBuffer(path=expert_buffer, device=device)
test_expert_buffer = SerializedBuffer(path=test_expert_buffer, device=device)

net = DuelingNet(state_shape, action_shape, device).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()


def actions_similarity(buffer):
    """计算行为相似度"""
    states = buffer.states
    actions = buffer.actions
    same_num = 0  # 相同的动作数量

    pred_actions = net(states).argmax(dim=1)
    true_actions = actions.argmax(dim=1)
    same_num = (pred_actions == true_actions).sum()

    return (same_num / len(states)).item()


def train(epoch, eval_freq=10):
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epoch // 9) + 1)
    state_exp = expert_buffer.states
    action_exp = expert_buffer.actions
    state_test = test_expert_buffer.states
    action_test = test_expert_buffer.actions
    train_data = TensorDataset(torch.cat([state_exp, state_test], dim=0), torch.cat([action_exp, action_test]))

    data_loader = DataLoader(train_data, 64)
    for e in range(epoch):
        scheduler.step(e)
        for states, actions in data_loader:
            q_value = net.q_value(states)
            loss = loss_fn(q_value, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if e % eval_freq == 0:
            sim = actions_similarity(expert_buffer)
            print("train:", sim)
            test_sim = actions_similarity(test_expert_buffer)
            print("test:,", test_sim)


if __name__ == '__main__':
    train(10000)
