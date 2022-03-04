#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/9/26 上午10:49
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : rand_algo.py
# @Software: PyCharm

import numpy as np
import torch

from algo.algo.base import Algorithm


class ReversiAlgo(Algorithm):
    """黑白棋智能体算法,
    默认随机执行动作"""

    def __init__(self, env, state_shape, action_shape, color, device='cpu', seed=0, gamma=0.99, actor='greedy'):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        self.env = env
        self.color = color
        if actor == 'random':
            self.actor = RandomActor(env)
        elif actor == 'greedy':
            self.actor = GreedyActor(env, self.color)
        else:
            self.actor = actor

    def is_update(self, step):
        pass

    def update(self):
        pass

    def save_models(self, save_dir):
        pass


class ListAlgorithm:
    """按行为序列依次执行"""

    def __init__(self, env, actions):
        self.env = env
        self.actions = actions  # 行为序列
        self.step = 0  # 当前步数

    def policy(self, state):
        if self.step >= len(self.actions):
            return torch.tensor([-1])
        action = torch.tensor([self.actions[self.step]])
        self.step += 1
        return action

    def sample(self, state=None):
        return self.policy(state)

    def __call__(self, state=None):
        return self.policy(state)


class RandomActor:
    def __init__(self, env):
        self.env = env

    def policy(self, state):
        return torch.tensor([self.env.sample_action()])

    def sample(self, state=None):
        return self.policy(state)

    def __call__(self, state=None):
        return self.policy(state)


class GreedyActor:
    def __init__(self, env, color):
        self.env = env
        self.color = color

    def policy(self, state):
        state = torch.squeeze(state).numpy()
        possible_actions = self.env.get_possible_actions(state, self.color)
        out_action = torch.zeros((1, self.env.action_space.n))  # 填充为one-hot编码的动作
        # 获取合法行为，没有合法行为时：
        if possible_actions == [state.shape[-1] ** 2 + 1]:
            out_action[0][possible_actions[0]] = 1
            return out_action  # torch.tensor([possible_actions[0]])
        # print(possible_actions)
        rewards = np.zeros(len(possible_actions))
        for i, action in enumerate(possible_actions):
            state_temp = state.copy()
            state_temp = self.env.make_place(state_temp, action, self.color)  # 模拟每个合法行为
            rewards[i] = np.sum(state_temp[self.color]) - np.sum(state[self.color])  # 计算每个行为的结果，即吃子数
        action_index = possible_actions[np.argmax(rewards)]
        out_action[0][action_index] = 1
        # return action_index
        return torch.tensor([action_index])

    def sample(self, state=None):
        return self.policy(state)

    def __call__(self, state=None):
        return self.policy(state)
