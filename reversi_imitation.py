#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/9/27 下午2:54
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : reversi_imitation.py
# @Software: PyCharm
import os
from datetime import datetime
from copy import deepcopy

import numpy as np

from gail_airl_ppo.algo import AIRL
from gail_airl_ppo.algo.airl_dqn import AIRL_DQN
from gail_airl_ppo.algo.reversi_algo import ReversiAlgo
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.trainer import Trainer
from reversi_env import ReversiEnv
import torch


def train_imitation():
    """训练reversi的模仿智能体"""
    # buffer = './buffers/reversi/size3600_std0_prand0.pth'  # 示例数据文件
    buffer_train = './buffers/568507529aab81be75ae4316_size800_test.pth'
    buffer_test = './buffers/568507529aab81be75ae4316_size2400_train.pth'
    device = torch.device('cuda')
    seed = 0
    # rollout_length = 3600
    num_steps = 24000  # 总共训练步数
    eval_interval = 600  # 评估间隔

    env = ReversiEnv()  # reversi训练环境
    env_test = ReversiEnv()  # reversi测试环境
    buffer_expert = SerializedBuffer(
        # 读取示例数据
        path=buffer_train,
        device=device
    )
    test_buffer_expert = SerializedBuffer(
        path=buffer_test,
        device=device
    )

    # state_shape = observation_space.shape[0]*

    # 对抗模仿算法
    # algo = AIRL(
    #     buffer_expert=buffer_expert,
    #     state_shape=env.observation_space.shape,
    #     action_shape=[env.action_space.n],  # 形状都是可迭代对象
    #     device=device,
    #     seed=seed,
    #     rollout_length=rollout_length,
    #     discrete=True,  # 动作空间离散
    #     cnn=True  # 使用cnn处理棋盘
    # )
    algo = AIRL_DQN(
        expert_buffer=buffer_expert,
        test_expert_buffer=test_buffer_expert,
        state_shape=env.observation_space.shape,
        action_shape=[env.action_space.n],  # 形状都是可迭代对象
        action_space=env.action_space,
        device=device,
        seed=seed,
        cnn=True  # 使用cnn处理棋盘
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', 'Reversi', 'AIRL_DQN', f'seed0-{time}')

    trainer = Trainer(  # 训练器
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=num_steps,
        eval_interval=eval_interval,
        seed=seed
    )
    # trainer.actions_similarity()
    print("Actions similarity:", algo.actions_similarity(algo.expert_buffer))
    trainer.train()


if __name__ == '__main__':
    train_imitation()
