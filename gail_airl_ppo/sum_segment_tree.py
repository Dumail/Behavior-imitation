#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/17 17:33
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : sum_segment_tree.py
# @Software: PyCharm
import operator

from gail_airl_ppo.segment_tree import SegmentTree


class SumSegmentTree(SegmentTree):
    """求和线段树"""

    def __init__(self, size):
        super(SumSegmentTree, self).__init__(size=size, operation=operator.add, element=0.0)

    def sum(self, start=0, end=None):
        """区间求和"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        index = 1
        while index < self.size:
            if self.value[2 * index] > prefixsum:
                index = 2 * index
            else:
                prefixsum -= self.value[2 * index]
                index = 2 * index + 1
        return index - self.size


class MinSegmentTree(SegmentTree):
    """最小线段树"""

    def __init__(self, size):
        super(MinSegmentTree, self).__init__(size, min, float('inf'))

    def min(self, start=0, end=None):
        """区间求和"""
        return super(MinSegmentTree, self).reduce(start, end)
