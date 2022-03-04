#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 14:14
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : prioritized_memory.py
# @Software: PyCharm
import random
from typing import List

import numpy as np

from algo.memory import Memory
from algo.sum_segment_tree import SumSegmentTree, MinSegmentTree


class PrioritizedMemory(Memory):
    """
    采用SumTree实现的优先回放经验池
    """

    def __init__(self, state_shape, size: int, batch_size: int = 32, alpha: float = 0.6):
        """
        构造函数
        :param state_shape: 状态空间的形状
        :param size: 经验池的容量大小
        :param batch_size: 采样大小
        """

        super(PrioritizedMemory, self).__init__(state_shape, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        tree_size = 1
        while tree_size < self.max_size:
            tree_size *= 2

        self.sum_tree = SumSegmentTree(tree_size)
        self.min_tree = MinSegmentTree(tree_size)

    def store(self, state, action: int, reward: float, next_state, done: bool):
        """
        存储经验到经验池
        :param state:当前状态 
        :param action: 行为
        :param reward: 奖励
        :param next_state: 下一状态 
        :param done: 是否结束episode
        """

        super(PrioritizedMemory, self).store(state, action, reward, next_state, done)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def _sample_prioritized(self):
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            index = self.sum_tree.find(upperbound)
            indices.append(index)

        return indices

    def update_priorities(self, indices: List[int], priorities):
        # assert len(indices) == len(priorities)

        for index, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= index < len(self)

            self.sum_tree[index] = priority ** self.alpha
            self.min_tree[index] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _calculate_weight(self, index: int, beta: float):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        p_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight

    def sample(self, beta: float = 0.4):
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_prioritized()
        states = self.states_buffer[indices]
        next_states = self.next_states_buffer[indices]
        actions = self.actions_buffer[indices]
        rewards = self.rewards_buffer[indices]
        done = self.done_buffer[indices]

        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        return dict(obs=states, next_obs=next_states,
                    actions=actions,
                    rewards=rewards,
                    done=done,
                    weights=weights,
                    indices=indices)
