import os
import argparse
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert
from gail_airl_ppo.utils import collect_demo


def run(args):
    env = make_env(args.env_id)  # 构造环境

    # 采用SAC算法的专家智能体
    algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.weight  # 载入模型文件
    )

    # 搜集轨迹并存入缓冲区
    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )

    # 序列化缓冲区
    buffer.save(os.path.join(
        'buffers',
        args.env_id,
        f'size{args.buffer_size}_std{args.std}_prand{args.p_rand}.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, default="weights/InvertedPendulum-v2.pth")  # 训练好的模型文件
    p.add_argument('--env_id', type=str, default='InvertedPendulumMuJoCoEnv-v0')  # 环境名
    p.add_argument('--buffer_size', type=int, default=1000)  # 存储步数
    p.add_argument('--std', type=float, default=0.0)  # 加到行为上的高斯噪声的标准差
    p.add_argument('--p_rand', type=float, default=0.0)  # 模型随机执行行为的概率
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
