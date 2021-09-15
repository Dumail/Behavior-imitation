#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 下午7:23
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : mairl.py
# @Software: PyCharm
import os
import random
from collections import OrderedDict
from copy import deepcopy
import torch
import gym
from tensorboardX import SummaryWriter

from gail_airl_ppo.algo import AIRL
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.env import make_env


class Task:
    def __init__(self, env: gym.Env, expert_buffer):
        self.env = env
        self.expert_buffer = expert_buffer


class MAIRL:
    def __init__(self, tasks: [Task], state_shape, action_shape, device, seed=0):
        self.device = device
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.seed = seed
        self.model = AIRL(None, state_shape, action_shape, device, seed)  # 元模型

        self.tasks = tasks
        self.inner_steps_size = 0.5e-4
        self.inner_epochs = 10
        self.outer_step_size0 = 0.5

        # 日志设置
        self.summary_dir = os.path.join("logs", 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)

    def task_sample(self, num_tasks):
        """采样n个任务"""
        indices = random.sample(range(len(self.tasks)), num_tasks)
        return [self.tasks[i] for i in indices]

    def train_subtask(self, task, epoch=0):
        """
        训练子任务
        :param task: 待训练的子任务
        :param epoch: 训练轮数
        """
        epoch = self.inner_epochs if epoch == 0 else epoch

        self.model.set_expert_buffer(task.expert_buffer)  # 设置模型数据为子任务数据
        env = task.env  # 环境为子任务环境

        # 以子任务学习率训练子任务模型
        # self.model.train(self.inner_epochs, l_rate=self.inner_steps_size, batch_size=batch_size)

        t = 0  # 每个回合的时间步
        # step = 0
        state = env.reset()
        # for e in range(self.inner_epochs):
        for step in range(1, epoch * 1000 + 1):
            # while True:
            # 模型与环境交互更新状态和时间步
            state, t = self.model.step(env, state, t, step)
            # 在适当的时候更新模型
            if self.model.is_update(step):
                self.model.update(self.writer)
                # if t == 0:  # 回合结束
                #     break

    def meta_train(self, epoch=100, num_tasks=3):
        """
        元模型训练
        :param epoch: 训练轮数
        :param num_tasks: 每次采样的子任务数
        """
        for e in range(epoch):
            print("epoch:", e)
            weights_before = [deepcopy(self.model.disc.state_dict()),
                              deepcopy(self.model.actor.state_dict()), deepcopy(self.model.critic.state_dict())]
            tasks = self.task_sample(num_tasks)
            for t in tasks:
                self.train_subtask(t)
                # 子任务训练之后的网络权重
                weights_after = [self.model.disc.state_dict(), self.model.actor.state_dict(),
                                 self.model.critic.state_dict()]

                # 元模型学习率线性衰减
                outer_step_size = self.outer_step_size0 * (1 - e / epoch)
                # 更新元模型
                self.update_params(weights_before, weights_after, outer_step_size)

                self.evaluate(t.env)

    def evaluate(self, env_test, num_eval_episodes=10):
        """
        评估模型
        """
        mean_return = 0.0  # 评估时的平均回报
        for _ in range(num_eval_episodes):
            state = env_test.reset()
            episode_return = 0.0
            done = False
            while not done:
                action = self.model.exploit(state)
                state, reward, done, _ = env_test.step(action)
                episode_return += reward

            mean_return += episode_return / num_eval_episodes

        print(f'Return: {mean_return:<5.1f}   ')

    def update_params(self, weights_before, weights_after, l_rate):
        """更新元模型参数"""
        disc_weights_before, actor_weights_before, critic_weights_before = weights_before[0], weights_before[1], \
                                                                           weights_before[2]
        disc_weights_after, actor_weights_after, critic_weights_after = weights_after[0], weights_after[1], \
                                                                        weights_after[2]
        self.model.disc.load_state_dict(OrderedDict(
            {name: disc_weights_before[name] + (disc_weights_after[name] - disc_weights_before[name]) * l_rate
             for name in disc_weights_before}))
        self.model.actor.load_state_dict(OrderedDict(
            {name: actor_weights_before[name] + (actor_weights_after[name] - actor_weights_before[name]) * l_rate
             for name in actor_weights_before}))
        self.model.critic.load_state_dict(OrderedDict(
            {name: critic_weights_before[name] + (critic_weights_after[name] - critic_weights_before[name]) * l_rate
             for name in critic_weights_before}))


if __name__ == '__main__':
    device = torch.device('cpu')
    env = make_env("InvertedPendulumMuJoCoEnv-v0")
    buffer_expert = SerializedBuffer("../../buffers/InvertedPendulumMuJoCoEnv-v0/size1000_std0.0_prand0.0.pth", device)
    tasks = [Task(env, buffer_expert) for _ in range(5)]

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    mairl = MAIRL(tasks, state_shape, action_shape, device)
    mairl.meta_train()
