#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 下午4:32
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : d3qn.py
# @Software: PyCharm
import random

import numpy as np
import torch.nn.functional
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

from gail_airl_ppo.algo.base import Algorithm
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.memory import Memory
from gail_airl_ppo.network.dueLing_net import DuelingNet
from gail_airl_ppo.prioritized_memory import PrioritizedMemory


# class ThreeLayersNet(nn.Module):
#     """
#     三层神经网络模型
#     """
#     def __init__(self, state_shape, action_shape, device):
#         super(ThreeLayersNet, self).__init__()
#         self.device = device
#         self.model = nn.Sequential(
#             nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
#             nn.Linear(128, 64), nn.ReLU(inplace=True),
#             nn.Linear(128, 64), nn.ReLU(inplace=True),
#             nn.Linear(64, np.prod(action_shape))
#         )
#
#     def forward(self, x) -> torch.Tensor:
#         if not isinstance(x, torch.Tensor):
#             x = torch.tensor(x, dtype=torch.float)
#         return self.model(x)


class D3QN(Algorithm):
    def __init__(self, state_shape, action_shape, action_space, device, seed,
                 memory_size=2 ** 10, batch_size=256, update_pred=500, epsilon_decay=1 / 500, max_epsilon=1.0,
                 min_epsilon=0.1, gamma=0.97, alpha=0.2, beta=0.6, prior_eps=1e-6, expert_buffer=None,
                 test_expert_buffer=None):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        self.action_space = action_space
        self.actor = self.create_net()
        self.loss_func = torch.nn.functional.mse_loss
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.update_pred = update_pred
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.target_actor = self.create_net()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()

        self.beta = beta
        self.prior_eps = prior_eps
        self.buffer = PrioritizedMemory(self.state_shape, memory_size, batch_size, alpha)
        # self.buffer = Memory(np.prod(self.state_shape), memory_size, batch_size)

        # expert_buffer = "./buffers/568507529aab81be75ae4316_size2400_train.pth"
        # test_expert_buffer = "./buffers/568507529aab81be75ae4316_size800_test.pth"

        if expert_buffer is not None:
            self.expert_buffer = SerializedBuffer(
                # 读取示例数据
                path=expert_buffer,
                device=device
            )

        if test_expert_buffer is not None:
            self.test_expert_buffer = SerializedBuffer(
                path=test_expert_buffer,
                device=device
            )
        # self.sum_action = [1, 0]

    def create_net(self):
        return DuelingNet(self.state_shape, self.action_shape, self.device).to(self.device)
        # return ThreeLayersNet(self.state_shape, self.action_shape, self.device).to(self.device)

    def is_update(self, step):
        if step % self.update_pred == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
        if step % 100 == 0:
            self.epsilon = max(self.min_epsilon,
                               self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
        return len(self.buffer) >= self.batch_size  # and step % 100 == 0

    def explore(self, state):
        if self.epsilon > np.random.random():
            # 对于黑白棋只能在合法范围内随机
            if len(state.shape) >= 3:
                possible_actions = np.where(state[2].flatten() != 0)[0].tolist()
                action = np.int64(random.choice(possible_actions))
            else:
                action = self.action_space.sample()
        else:
            with torch.no_grad():
                if len(state.shape) >= 3:
                    action = self.actor(torch.tensor(state, dtype=torch.float).unsqueeze_(0).to(self.device)).argmax()
                else:
                    action = self.actor(torch.tensor(state, dtype=torch.float).to(self.device)).argmax()
            action = np.int64(action.detach().cpu().numpy())
        return action

    def exploit(self, state):
        with torch.no_grad():
            if len(state.shape) >= 3:
                action = self.actor(torch.tensor(state, dtype=torch.float).unsqueeze_(0).to(self.device)).argmax();
            else:
                action = self.actor(torch.tensor(state, dtype=torch.float).to(self.device)).argmax()
        action = np.int64(action.detach().cpu().numpy())
        return action

    def compute_next_q_value(self, next_states: torch.FloatTensor):
        """
        计算下一状态的最大q值表
        :param next_states:
        :return:
        """
        return self.target_actor(next_states).max(dim=1, keepdim=True)[0].detach()  # 下一状态中最大的q值

    def compute_loss(self, samples) -> torch.Tensor:
        """
        通过样本计算网络的损失值
        :param samples: 采样得到的样本
        :return: 损失值
        """
        # 获取从采样中得到的各类数据并装换为Tensor
        states, actions, rewards, done, next_states = self.discompose_sample(samples)

        q_value = self.actor(states).gather(1, actions)  # 状态行为对的当前价值
        next_q_value = self.compute_next_q_value(next_states)
        target = (rewards + self.gamma * next_q_value * (1 - done)).to(self.device)  # 计算TD目标
        loss = self.loss_func(q_value, target)
        return loss

    def step(self, env, state, t, step):
        """
                与环境交互并存储轨迹数据
                """
        t += 1  # 时间步

        action = self.explore(state)

        possible_actions = np.where(state[2].flatten() != 0)[0].tolist()

        if not isinstance(action, np.int64):
            print("action is not a int:", action)
            print(type(action))
        # temp_action = action if not self.discrete else np.argmax(action)
        # if step>10000:
        #     env.render()
        next_state, reward, done, _ = env.step(action)
        self.buffer.store(state, action, reward, next_state, done)
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, writer):
        samples = self.buffer.sample()
        loss = self.compute_loss(samples)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(loss.item())
        return loss
        # states_exp, actions_exp, _, _, _ = self.expert_buffer.sample(self.batch_size)
        # states_test, actions_test, _, _, _ = self.test_expert_buffer.sample(32)
        # self.supervised_update_actor(states_exp, actions_exp, writer)
        # self.supervised_update_actor(states_test, actions_test, writer)

    def update_actor(self, states, actions, rewards, dones, next_states, writer):
        q_value = self.actor(states).gather(1, actions)  # 状态行为对的当前价值
        next_q_value = self.compute_next_q_value(next_states)
        target = (rewards + self.gamma * next_q_value * (1 - dones)).to(self.device)  # 计算TD目标
        loss = self.loss_func(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        writer.add_scalar('loss/actor', loss.item(), self.learning_steps)

    def supervised_update_actor(self, epochs, writer, eval_freq=None, fix=True):
        """监督学习的方式进行模仿"""
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        if fix:
            for i in self.actor.cnn_layer.parameters():
                i.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor.parameters()))
        loss_fn = nn.CrossEntropyLoss()

        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs // 9) + 1)
        state_exp = self.expert_buffer.states
        action_exp = self.expert_buffer.actions
        # state_test = self.test_expert_buffer.states
        # action_test = self.test_expert_buffer.actions
        # train_data = TensorDataset(torch.cat([state_exp, state_test], dim=0), torch.cat([action_exp, action_test]))
        # train_data = TensorDataset(state_exp, action_exp)
        # data_loader = DataLoader(train_data, 2048)

        # print("Lenght:",len(state_exp))
        for epoch in range(epochs):
            # scheduler.step(epoch)
            # for states, actions in data_loader:
            q_value = self.actor.q_value(state_exp)
            loss = loss_fn(q_value, action_exp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/actor', loss.item(), epoch)
            if eval_freq is not None and epoch % eval_freq == 0:
                print("Epoch:", epoch)
                print("Train actor similarity:", self.actions_similarity(self.expert_buffer))
                print("Test actions similarity:", self.actions_similarity(self.test_expert_buffer))

    def save_models(self, save_dir):
        torch.save(self.actor.state_dict(), save_dir)

    def discompose_sample(self, samples):
        states = torch.FloatTensor(samples["obs"].reshape(-1, *self.state_shape)).to(self.device)
        next_states = torch.FloatTensor(samples["next_obs"].reshape(-1, *self.state_shape)).to(self.device)
        actions = torch.LongTensor(samples["actions"].reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(samples["rewards"].reshape(-1, 1)).to(self.device)
        done = torch.LongTensor(samples["done"].reshape(-1, 1)).to(self.device)
        return states, actions, rewards, done, next_states

    def actions_similarity(self, expert_buffer):
        """计算行为相似度"""
        states = expert_buffer.states
        actions = expert_buffer.actions
        same_num = 0  # 相同的动作数量

        pred_actions = self.actor(states).argmax(dim=1)
        true_actions = actions.argmax(dim=1)
        same_num = (pred_actions == true_actions).sum()

        return (same_num / len(states)).item()
