import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from reversi_env import ReversiEnv
from .base import Algorithm
from gail_airl_ppo.buffer import RolloutBuffer
from gail_airl_ppo.network import StateIndependentPolicy, StateFunction
from ..utils import get_possible_action


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    """
    计算Generalized Advantage Estimation
    """
    # 计算TD误差 \delta = r+(1-d)*\lambda*V(s')+V(s)
    deltas = rewards + gamma * next_values * (1 - dones) - values

    gaes = torch.empty_like(rewards)

    # 从后往前递归计算GAE A_t = \sum_{t=0}^{T-t} ((1-d)*\gamma*\lambda)^l * \delta_{t+l}
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0, discrete=False):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        if discrete:
            self.color = ReversiEnv.BLACK

        # 采样缓冲区
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer,
            discrete=discrete
        )

        if discrete:
            state_shape = [np.prod(state_shape)]
        # 策略网络
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh(),
            cnn=True if discrete else False
        ).to(device)

        # 价值网络
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh(),
            cnn=True if discrete else False
        ).to(device)

        # 正交初始化
        # for m in self.actor.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.orthogonal(m.weight)
        #
        # for m in self.critic.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.orthogonal(m.weight)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0  # PPO的学习步数
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps  # PPP-clip 限制范围 1-eps ~ 1+eps
        self.lambd = lambd  # TD-lambda
        self.coef_ent = coef_ent  # 策略熵系数
        self.max_grad_norm = max_grad_norm  # 最大梯度norm

        self.old_value = 0
        self.discrete = discrete

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        """
        与环境交互并存储轨迹数据
        """
        t += 1  # 时间步

        action, log_pi = self.explore(state)
        if self.discrete:
            action = get_possible_action(env, action, state, self.color)

        next_state, reward, done, _ = env.step(action)
        # mask = False if t == env._max_episode_steps else done
        mask = done

        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, writer):
        """从缓冲区得到数据并更新网络"""
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states, writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1

            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer):

        loss_critic = (self.critic(states) - targets).pow_(2).mean()  # 价值误差
        # values = self.critic(states)
        # clip_value = self.old_value + torch.clamp(values - self.old_value, -self.clip_eps, self.clip_eps)
        # value_max = torch.max(((values - targets) ** 2), ((clip_value - targets) ** 2))
        # loss_critic = value_max.mean()
        # self.old_value = values.detach()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)  # 梯度裁剪
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)  # 当前的策略的log p_pi(a|s)
        entropy = -log_pis.mean()  # 策略熵

        ratios = (log_pis - log_pis_old).exp_()  # 策略改进程度 p_pi(a|s)/p_pi'(a|s)
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        # max -(p_pi(a|s)/p_pi'(s|a) * A_pi'(s|a), clip(p_pi(a|s)/p_pi'(s|a),1-eps,1+eps) * A_pi'(s|a))
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)  # policy_loss + coefficient * entropy
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir):
        pass
