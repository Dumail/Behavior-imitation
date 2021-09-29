import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SAC, PPO
from gail_airl_ppo.trainer import Trainer

if __name__ == '__main__':
    num_steps = 15 * 2048  # 总步数
    eval_interval = 2048  # 多少步评估一次
    env_id = 'Pendulum-v0'  # 环境名
    seed = 65535  # 随机种子
    device = torch.device("cpu")

    # 构造训练和测试环境
    env = make_env(env_id)
    env_test = make_env(env_id)

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape or [env.action_space.n]
    # if env.action_space.shape:
    #     diverse = False
    # else:
    #     diverse = True

    # 使用PPO算法 连续状态，连续行为
    algo = PPO(state_shape=state_shape, action_shape=action_shape, device=device,
               seed=seed, gamma=0.995)  # , rollout_length=2048,
    # mix_buffer=20, lr_actor=3e-4, lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
    # epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.01, max_grad_norm=0.5)

    # 日志文件夹
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', env_id, 'sac', f'seed{seed}-{time}')

    # RL 训练器
    trainer = Trainer(env=env, env_test=env_test, algo=algo, log_dir=log_dir, num_steps=num_steps,
                      eval_interval=eval_interval, seed=seed)
    trainer.train()
