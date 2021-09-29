import os
import numpy as np
import torch


class SerializedBuffer:
    """序列化缓冲区"""

    def __init__(self, path, device):
        if path is not None:
            # 载入数据
            tmp = torch.load(path)
            self.buffer_size = self._n = tmp['state'].size(0)
            self.device = device

            self.states = tmp['state'].clone().to(self.device)
            self.actions = tmp['action'].clone().to(self.device)
            self.rewards = tmp['reward'].clone().to(self.device)
            self.dones = tmp['done'].clone().to(self.device)
            self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        """
        在缓冲区中采样数据
        :param batch_size:每次采样的数据大小
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):
    """
    轨迹缓冲区 循环队列方式存储
    :param buffer_size: 缓冲区总大小
    :param state_shape: 状态空间纬度
    :param action_shape: 行为空间纬度
    """

    def __init__(self, buffer_size, state_shape, action_shape, device, discrete=False):
        super().__init__(None, device)
        self._n = 0  # kkkk
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device
        self.discrete = discrete  # 动作空间是否离散

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

        if not discrete:
            self.actions = torch.empty(
                (buffer_size, *action_shape), dtype=torch.float, device=device)
        else:
            self.actions = torch.empty(
                (buffer_size, 1), dtype=torch.float, device=device)

        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        if not self.discrete:
            self.actions[self._p].copy_(torch.from_numpy(action))
        else:
            self.actions[self._p] = float(action)

        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


class RolloutBuffer:
    """
    随机采样缓冲区 循环队列
    :param mix: 空间大小倍率
    """

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1, discrete=False):
        self._n = 0  # 队列实际数据数
        self._p = 0  # 队列头位置
        self.mix = mix
        self.buffer_size = buffer_size  # 缓冲区大小
        self.total_size = mix * buffer_size  # 实际总大小
        self.discrete = discrete

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        if not discrete:
            self.actions = torch.empty((self.total_size, *action_shape), dtype=torch.float, device=device)
        else:
            self.actions = torch.empty((self.total_size, 1), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        if not self.discrete:
            self.actions[self._p].copy_(torch.from_numpy(action))
        else:
            self.actions[self._p] = float(action)
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        """获取一系列连续数据"""
        assert self._p % self.buffer_size == 0  # 刚好搜集满给定n个缓冲区大小的数据 on-policy？
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)  # 获取给定缓冲区大小的数据
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        """从缓冲区中采样数据"""
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
