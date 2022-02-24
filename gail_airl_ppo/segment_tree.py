#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 14:26
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : segment_tree.py
# @Software: PyCharm

class SegmentTree(object):
    """
    静态数组实现线段树
    """

    def __init__(self, size, operation, element):
        """
        线段树初始化
        :param size: 存储空间大小
        :param operation: 内部结点操作
        :param element: 初始化数据元素
        """
        assert size > 0 and size & (size - 1) == 0, "The size must be positive and a power of 2."
        self.size = size
        self.value = [element for _ in range(2 * size)]  # 需要的存储空间大小为容量的两倍-1
        self.operation = operation

    def reduce(self, start=0, end=None):
        """
        将操作映射到连续的子序列上
        :param start: 子序列开始位置
        :param end: 子序列结束位置
        :return: 操作结果
        """
        if end is None:
            end = self.size
        if end < 0:
            end += self.size
        end -= 1
        return self._reduce(start, end, 1, 0, self.size - 1)

    def _reduce(self, start, end, node, node_start, node_end):
        """
        二分查找法
        :param start: 查找区间开始位置
        :param end: 查找区间结束位置
        :param node: 当前查找位置
        :param node_start: 子区间开始位置
        :param node_end: 子区间结束位置
        :return: 对子区间进行操作的结果
        """
        if start == node_start and end == node_end:
            return self.value[node]  # 区间与结点相等则直接返回
        mid = (node_start + node_end) // 2
        if end <= mid:
            # 去左子树查找
            return self._reduce(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                # 去右子树查找
                return self._reduce(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                # 区间被分为两部分
                return self.operation(
                    self._reduce(start, mid, 2 * node, node_start, mid),
                    self._reduce(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def __setitem__(self, index, value):
        """修改结点值，回溯树"""
        index += self.size
        self.value[index] = value
        index //= 2
        while index >= 1:
            self.value[index] = self.operation(self.value[2 * index], self.value[2 * index + 1])
            index //= 2

    def __getitem__(self, index):
        """获取结点值"""
        assert 0 <= index < self.size
        return self.value[self.size + index]
