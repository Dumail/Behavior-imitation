#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/13 23:45
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : memory.py
# @Software: PyCharm
from typing import Dict

import numpy as np


class Memory(object):
    """智能体经验池"""

    def __init__(self, state_shape, size: int, batch_size: int = 32):
        """
        构造函数
        :param state_shape: 状态空间的形状
        :param size: 经验池的容量大小
        :param batch_size: 采样大小
        """
        self.state_shape = state_shape
        self.states_buffer = np.zeros([size, np.prod(state_shape)], dtype=np.float32)
        self.next_states_buffer = np.zeros([size, np.prod(state_shape)], dtype=np.float32)
        self.actions_buffer = np.zeros([size], dtype=np.float32)
        self.rewards_buffer = np.zeros([size], dtype=np.float32)
        self.done_buffer = np.zeros([size], dtype=np.float32)
        self.max_size = size
        self.batch_size = batch_size
        self.current_mem = 0  # 当前记忆指针
        self.size = 0  # 当前经验池大小

    def store(self, state, action: np.ndarray, reward: float, next_state, done: bool):
        """
        将一次经验放入经验池
        :param state:状态
        :param action:行为
        :param reward: 奖励
        :param next_state: 下一状态
        :param done: episode是否结束
        """
        self.states_buffer[self.current_mem] = state.flatten()
        self.actions_buffer[self.current_mem] = action
        self.rewards_buffer[self.current_mem] = reward
        self.next_states_buffer[self.current_mem] = next_state.flatten()
        self.done_buffer[self.current_mem] = done
        self.current_mem = (self.current_mem + 1) % self.max_size  # 循环大小
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        """
        采样
        :return: 字典形式的经验簇
        """
        index = np.random.choice(self.size, size=self.batch_size, replace=False)  # 随机选择多个索引
        states = self.states_buffer[index]
        next_states = self.next_states_buffer[index]
        if (not isinstance(self.state_shape, np.int32)) and (not isinstance(self.state_shape, np.int64)):
            print(type(self.state_shape))
            ates = states.reshape(self.state_shape[0], -1)
            next_states = next_states.reshape(self.state_shape[0], -1)
        return dict(obs=states, next_obs=next_states,
                    actions=self.actions_buffer[index],
                    rewards=self.rewards_buffer[index],
                    done=self.done_buffer[index])

    def __len__(self):
        return self.size
