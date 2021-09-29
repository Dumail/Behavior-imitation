#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/9/27 下午2:54
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : reversi_imitation.py
# @Software: PyCharm
import os
from datetime import datetime

import numpy as np

from gail_airl_ppo.algo import AIRL
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.trainer import Trainer
from reversi_env import ReversiEnv
import torch


def train_imitation():
    buffer = './buffers/reversi/size3600_std0_prand0.pth'
    device = torch.device('cpu')
    seed = 0
    rollout_length = 1200
    num_steps = 12000
    eval_interval = 1200

    env = ReversiEnv()
    env_test = ReversiEnv()
    buffer_expert = SerializedBuffer(
        path=buffer,
        device=device
    )

    algo = AIRL(
        buffer_expert=buffer_expert,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.n,
        device=device,
        seed=seed,
        rollout_length=rollout_length,
        discrete=True
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', 'reversi', 'AIRL', f'seed0-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=num_steps,
        eval_interval=eval_interval,
        seed=seed, discrete=True
    )
    trainer.train()


if __name__ == '__main__':
    train_imitation()
