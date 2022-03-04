import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from algo.network import GAILDiscrim
from .ppo import PPO


class GAIL(PPO):
    """
    生成对抗模仿学习算法
    :param buffer_expert: 专家经验缓冲区
    :param state_shape
    :param action_shape
    :param device
    :param seed
    :param gamma: 折扣率
    :param rollout_length: 采样步数
    :param mix_buffer:
    :param batch_size
    :param lr_actor: 策略网络的学习率
    :param lr_critic: 价值网络的学习率
    :param lr_disc: 鉴别器的学习率
    :param units_actor: 策略网络的隐藏层设置
    :param units_critic: 价值网络的隐藏层设置
    :param units_disc: 鉴别器网络的隐藏层设置
    :param epoch_ppo: PPO算法的更新迭代次数
    :param epoch_disc: 鉴别器的更新迭代次数
    :param clip_eps:
    :param lam:
    :param coef_ent:
    :param max_grad_norm:
    """

    def __init__(self, buffer_expert, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=50000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(100, 100), epoch_ppo=50, epoch_disc=10,
                 clip_eps=0.2, lam=0.97, coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lam, coef_ent, max_grad_norm
        )

        # 专家轨迹缓冲区
        self.buffer_exp = buffer_expert

        #  鉴别器
        self.disc = GAILDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_disc,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.learning_steps_disc = 0  # 鉴别器的更新次数
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)  # 鉴别器优化器
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer):
        """
        更新模型
        """
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            # 鉴别器的更新速度远大于生成器

            self.learning_steps_disc += 1  # 鉴别器学习次数增加

            # 从当前策略生成的轨迹缓冲区中采样状态和行为
            states, actions = self.buffer.sample(self.batch_size)[:2]
            # 从专家轨迹中采样同样的大小
            states_expert, actions_expert = self.buffer_exp.sample(self.batch_size)[:2]

            # 更新鉴别器
            self.update_disc(states, actions, states_expert, actions_expert, writer)

        # GAIL 并没有用到采集的奖励信号
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # 计算估计奖励
        rewards = self.disc.calculate_reward(states, actions)

        # 使用估计奖励更新PPO策略
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc(self, states, actions, states_exp, actions_exp, writer):
        """
        使用一批数据更新鉴别器
        """

        # 鉴别器的输出范围是(-inf, inf),不是[0, 1].
        logits_pi = self.disc(states, actions)
        logits_exp = self.disc(states_exp, actions_exp)

        # 鉴别器目标是最大化 E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()  # 生成数据的鉴别器损失
        loss_exp = -F.logsigmoid(logits_exp).mean()  # 专家数据到鉴别器损失
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            # 当鉴别器更新次数达到鉴别器每次训练的轮数时，记录损失值和准确率
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
