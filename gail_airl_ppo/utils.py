from tianshou.data import ReplayBuffer
from tqdm import tqdm
import numpy as np
import torch

from .buffer import Buffer


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0, discrete=False):
    """
    智能体与环境交互生成轨迹数据
    :param env: 环境
    :param algo: 智能体
    :param device: 计算设备
    :param std: 添加到行为的高斯噪声标准差
    :param p_rand: 随机执行行为的概率
    :param seed 随机种子
    """
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 轨迹缓冲区
    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape or [env.action_space.n],
        device=device,
        discrete=discrete
    )

    total_return = 0.0  # 累计回报
    num_episodes = 0  # 交互回合数

    state = env.reset()
    t = 0  # 每个回合的时间步
    episode_return = 0.0  # 每个回合的总回报

    for _ in tqdm(range(1, buffer_size + 1)):
        # 持续交互知道充满缓冲区
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()  # 随机行为
        else:
            action = algo.exploit(state)  # 利用模型得到行为分布
            if std != 0:
                action = add_random_noise(action, std)  # 为行为分布添加噪声

        next_state, reward, done, _ = env.step(action)
        # mask = False if t == env._max_episode_steps else done  # 到达交互最大次数或交互结束
        mask = done  # 到达交互最大次数或交互结束

        if discrete:
            index = np.argmax(action)
            action.fill(0)
            action[index] = 1
        buffer.append(state, action, reward, mask, next_state)  # 添加数据到缓冲区
        episode_return += reward  # 累计每回合奖励作为回报 无折扣

        if done:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0
            # print("Black score:", env.black_score, "White score:", env.white_score)

        state = next_state

    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer


def buffer_trans(replay_buffer: ReplayBuffer, device='cpu', discrete=False, action_shape=None):
    """
    tianshou buffer to buffer
    """
    obs = replay_buffer.obs
    acts = replay_buffer.act
    if discrete:
        temp_acts = np.zeros((len(acts), action_shape))
        for i in range(len(acts)):
            temp_acts[i][acts[i]] = 1
        acts = temp_acts

    rew = replay_buffer.rew
    done = replay_buffer.done
    obs_next = replay_buffer.obs_next

    buffer = Buffer(buffer_size=len(replay_buffer),
                    state_shape=obs[0].shape,
                    action_shape=acts[0].shape,
                    device=device)
    for i in range(len(replay_buffer)):
        buffer.append(obs[i], np.array(acts[i]), rew[i], done[i], obs_next[i])
    return buffer
