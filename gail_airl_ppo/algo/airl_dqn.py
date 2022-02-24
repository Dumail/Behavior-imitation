import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from gail_airl_ppo.network import AIRLDiscrim
from .d3qn import D3QN


class AIRL_DQN(D3QN):
    def __init__(self, state_shape, action_shape, device, seed, action_space,
                 gamma=0.97, batch_size=256, lr_disc=1e-6, units_disc_r=(64, 64), units_disc_v=(64, 64),
                 epoch_disc=20, cnn=True, expert_buffer=None, test_expert_buffer=None):
        super().__init__(state_shape, action_shape, action_space=action_space, device=device, seed=seed, gamma=gamma,
                         expert_buffer=expert_buffer, test_expert_buffer=test_expert_buffer)

        # 鉴别器 D(s,s')
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True),
            cnn=cnn
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def set_expert_buffer(self, expert_buffer, test_expert_buffer):
        self.expert_buffer = expert_buffer
        self.test_expert_buffer = test_expert_buffer

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            samples = self.buffer.sample()
            states, actions, _, dones, next_states = self.discompose_sample(samples)

            with torch.no_grad():
                log_pis = self.actor.evaluate_log_pi(states, actions)

            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, dones_exp, next_states_exp = \
                self.expert_buffer.sample(self.batch_size)
            # 计算专家行为的对数概率
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(states_exp, actions_exp.argmax(1).unsqueeze_(1))
            # Update discriminator.
            self.update_disc(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, writer
            )

        # We don't use reward signals here,
        samples = self.buffer.sample()
        states, actions, _, dones, next_states = self.discompose_sample(samples)
        with torch.no_grad():
            log_pis = self.actor.evaluate_log_pi(states, actions)

        states_exp, actions_exp, _, _, _ = self.expert_buffer.sample(self.batch_size)
        # sim = self.actions_similarity(states_exp, actions_exp)  # 当前策略与专家的行为相似度
        # if sim>0.2:
        #     print(sim)
        # Calculate rewards.
        rewards = self.disc.calculate_reward(states, dones, log_pis, next_states)

        # Update actor using estimated rewards.
        # before_params = deepcopy(self.actor.state_dict())
        self.update_actor(states, actions, rewards, dones, next_states, writer)
        # update_sim = self.actions_similarity(self.expert_buffer)  # 更新后的相似度
        # if update_sim < sim - 0.02:
        #     # 相似度减小的不更新
        #     self.actor.load_state_dict(before_params)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # sigmoid(logits) = D(s,a,s')
        # 鉴别器最大化 E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar('loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
