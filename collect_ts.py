#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 下午5:14
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : collect_ts.py
# @Software: PyCharm
import gym
import tianshou as ts
from tianshou.utils.net.common import Net
import torch

device = torch.device("cpu")

if __name__ == '__main__':
    # 创建环境
    env = gym.make("CartPole-v0")
    # 环境并行化
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make("CartPole-v0") for _ in range(10)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make("CartPole-v0") for _ in range(10)])

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape, hidden_sizes=[128, 128], device=device).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    policy = ts.policy.DQNPolicy(net, optim)

    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10),
                                        exploration_noise=True)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=10000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    )
    print(f'Finished training! Use {result["duration"]}')

    torch.save(policy.state_dict())

    policy.eval()  # 测试模型
    buffer = ts.data.ReplayBuffer(200000)
    train_collector.reset()
    train_collector.collect(n_episode=10)
