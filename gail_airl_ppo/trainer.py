import os
from time import time, sleep
from datetime import timedelta

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from gail_airl_ppo.env import make_env


class Trainer:
    """
    模型训练器
    :param env: 搜集数据的环境
    :param env_test: 评估的环境
    :param algo: 待训练的算法模型
    :param log_dir: 日志文件目录
    :param seed
    :param num_steps:训练的总步数
    :param eval_interval:评估的频率
    :param num_eval_episodes: 每次评估执行的回合数
    """

    def __init__(self, env, env_test, algo, log_dir, seed=0, num_steps=10 ** 5,
                 eval_interval=10 ** 3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2 ** 31 - seed)

        self.algo = algo
        self.log_dir = log_dir

        # 日志设置
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        self.start_time = time()  # 开始训练的时间戳
        t = 0  # 每个回合的时间步

        state = self.env.reset()

        for step in range(1, self.num_steps + 1):

            # lam = lambda f: 1 - f / self.num_steps
            # torch.optim.lr_scheduler.LambdaLR(self.algo.optim_actor, lr_lambda=lam)
            # torch.optim.lr_scheduler.LambdaLR(self.algo.optim_critic, lr_lambda=lam)

            # print(self.algo.optim_critic)

            # 模型与环境交货更新状态和时间步
            state, t = self.algo.step(self.env, state, t, step)

            # 在适当的时候更新模型
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # 在适当的适合评估模型并存储
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))

        # 等待日志写入完成
        sleep(10)

    def render(self, env_id):
        env = make_env(env_id)
        env.render(mode='human')
        state = env.reset()
        episode_return = 0.0
        done = False

        while not done:
            # action = self.algo.exploit(state)
            action = np.array(
                [-0.73017883, 0.02064807, -0.50737125, -0.74813616, 0.8141342, 0.5373625, 0.3600479, -0.7246281])
            # print("pa",action)
            # action = env.action_space.sample()
            # print("sa",action)
            state, reward, done, _ = env.step(action)
            episode_return += reward

        print("return:", episode_return)

    def evaluate(self, step):
        mean_return = 0.0  # 评估时的平均回报

        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            episode_return = 0.0
            done = False

            while not done:
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    @property
    def time(self):
        """计算当前时间与开始时间点的差"""
        return str(timedelta(seconds=int(time() - self.start_time)))
