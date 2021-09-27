#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/9/26 上午10:49
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : rand_algo.py
# @Software: PyCharm
import random

import numpy as np

from gail_airl_ppo.algo.base import Algorithm
import torch


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
        if possible_actions == [state.shape[-1] ** 2 + 1]:
            return torch.tensor([possible_actions[0]])
        # print(possible_actions)
        rewards = np.zeros(len(possible_actions))
        for i, action in enumerate(possible_actions):
            state_temp = state.copy()
            state_temp = self.env.make_place(state_temp, action, self.color)
            rewards[i] = np.sum(state_temp[self.color]) - np.sum(state[self.color])
        return torch.tensor([possible_actions[np.argmax(rewards)]])

    def sample(self, state=None):
        return self.policy(state)

    def __call__(self, state=None):
        return self.policy(state)


class ReversiAlgo(Algorithm):
    def __init__(self, env, state_shape, action_shape, color, device='cpu', seed=0, gamma=0.99, actor='random'):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        self.env = env
        self.color = color
        if actor == 'random':
            self.actor = RandomActor(env)
        elif actor == 'greedy':
            self.actor = GreedyActor(env, self.color)

    def is_update(self, step):
        pass

    def update(self):
        pass

    def save_models(self, save_dir):
        pass
