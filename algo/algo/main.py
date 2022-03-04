#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 下午7:23
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : mairl.py
# @Software: PyCharm
import os
import random
from collections import OrderedDict
from copy import deepcopy

import matplotlib.ticker as mticker
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from algo.algo.compare import Supervised
from algo.algo.irl_dqn import IRL_DQN
from algo.buffer import SerializedBuffer
from bot import get_random_start, test_actor
from reversi_env import ReversiEnv

BOT_NAMES = ['CAB', 'SHA', 'RAM', 'RVB', 'LEB', 'RVT']
BASELINES = [17.12, 19.47, 21.13, 17.54, 18.39, 19.84]


class Task:
    def __init__(self, expert_buffer, test_expert_buffer):
        self.expert_buffer = expert_buffer
        self.test_expert_buffer = test_expert_buffer


class MIRL:
    def __init__(self, env, tasks: [Task], device, seed=0):
        self.env = env
        self.device = device
        self.action_shape = [env.action_space.n]
        self.state_shape = env.observation_space.shape
        self.action_space = env.action_space
        self.seed = seed
        self.model = IRL_DQN(state_shape=state_shape, action_shape=self.action_shape, device=device,
                             seed=seed, action_space=self.action_space, cnn=True)  # 元模型

        self.tasks = tasks
        self.inner_steps_size = 0.5e-3
        self.inner_epochs = 10
        self.outer_step_size0 = 0.5

        # 日志设置
        self.summary_dir = os.path.join("logs", 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)

    def task_sample(self, num_tasks):
        """采样n个任务"""
        indices = random.sample(range(len(self.tasks)), num_tasks)
        return [self.tasks[i] for i in indices]

    def train_subtask(self, task, epoch=0):
        """
        训练子任务
        :param task: 待训练的子任务
        :param epoch: 训练轮数
        """
        epoch = self.inner_epochs if epoch == 0 else epoch

        # return
        # 以子任务学习率训练子任务模型
        # self.model.train(self.inner_epochs, l_rate=self.inner_steps_size, batch_size=batch_size)
        t = 0  # 每个回合的时间步
        # step = 0
        # for e in range(self.inner_epochs):
        #     self.model.update(self.writer)
        state = self.env.reset().copy()
        for step in range(1, epoch * 60 + 1):
            # while True:
            # 模型与环境交互更新状态和时间步
            state, t = self.model.step(self.env, state, t, step)
            state = state.copy()
            # 在适当的时候更新模型
            if self.model.is_update(step):
                self.model.update(self.writer)
                if t == 0:  # 回合结束
                    break
        self.model.direct_update_actor(epoch, self.writer, fix=True)  # Test

    def meta_train(self, epoch=300, num_tasks=3, save_actor_file="../../weights/mairl_actor.pth",
                   save_disc_file="../../weights/mairl_disc.pth"):
        """
        元模型训练
        :param epoch: 训练轮数
        :param num_tasks: 每次采样的子任务数
        """
        for e in range(epoch):
            print("epoch:", e)
            self.env.opponent = deepcopy(self.model)  # 自对弈更新
            weights_before = [deepcopy(self.model.disc.state_dict()),
                              deepcopy(self.model.actor.state_dict())]
            tasks = self.task_sample(num_tasks)
            for t in tasks:
                print("Train action similarity: ", self.actions_similarity(t.expert_buffer) * 100,
                      "%")  # 计算元模型在该示例上的相似度
                print("Test action similarity: ", self.actions_similarity(t.test_expert_buffer) * 100,
                      "%")  # 计算元模型在该示例上的相似度
                self.model.set_expert_buffer(t.expert_buffer, t.test_expert_buffer)  # 设置模型数据为子任务数据
                self.train_subtask(t)
                # 子任务训练之后的网络权重
                weights_after = [self.model.disc.state_dict(), self.model.actor.state_dict()]
                # 元模型学习率线性衰减
                outer_step_size = self.outer_step_size0 * (1 - e / epoch)
                # 更新元模型
                self.update_params(weights_before, weights_after, outer_step_size)
                # self.evaluate(self.env)

            self.save(save_actor_file, save_disc_file)

    def train(self, task, epoch=10, number=5, train_number=0, save=True, is_out_list=False):
        # 固定CNN层不训练
        for i in self.model.actor.cnn_layer.parameters():
            i.requires_grad = False
        self.model.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.actor.parameters()))

        if train_number == 0:
            self.model.set_expert_buffer(task.expert_buffer, task.test_expert_buffer)  # 设置模型数据为子任务数据
        else:
            self.model.set_expert_buffer(task.expert_buffer.reduce(train_number), task.test_expert_buffer)

        # print(self.model.expert_buffer.states.shape)
        # for g in self.model.optimizer.param_groups:
        #     g['lr'] = 1e-3
        sim_list = []
        max_sim = 0
        for i in range(epoch):
            print(i)
            # self.train_subtask(task, epoch // 50)
            self.train_subtask(task)
            # for j in range(9):
            #     states_exp, actions_exp = self.model.expert_buffer.states[j*240:(j+1)*240], self.model.expert_buffer.actions[j*240:(j+1)*240]
            #     self.model.supervised_update_actor(states_exp, actions_exp, self.writer)
            # self.evaluate(self.env)
            # print("Train action similarity: ", self.actions_similarity(task.expert_buffer) * 100, "%")  # 计算元模型在该示例上的相似度
            test_sim = self.actions_similarity(task.test_expert_buffer) * 100
            # print("Test action similarity: ", test_sim, "%")  # 计算元模型在该示例上的相似度
            sim_list.append(test_sim)
            if test_sim > max_sim:
                max_sim = test_sim
                if save:
                    self.save("../../weights/actor" + str(number) + ".pth", "../../weights/disc" + str(number) + ".pth")
        print("max_sim: ", max_sim)
        if save:
            with open("sim.txt", 'a') as f:
                f.write(" ".join(["number:", str(number), ", max sim:", str(max_sim)]) + "\n")
        if is_out_list:
            return sim_list
        else:
            return max_sim

    def actions_similarity(self, expert_buffer):
        """计算行为相似度"""
        states = expert_buffer.states
        actions = expert_buffer.actions
        same_num = 0  # 相同的动作数量

        pred_actions = self.model.actor(states).argmax(dim=1)
        true_actions = actions.argmax(dim=1)
        same_num = (pred_actions == true_actions).sum()

        return (same_num / len(states)).item()

        # for i in range(len(states)):
        #     # print(np.where(states[i][2].flatten() != 0))
        #     pred_action = self.model.actor(states[i].unsqueeze_(0))
        #     pred_action = pred_action.argmax().item()
        #     true_action = torch.argmax(actions[i]).item()
        #     # print(pred_action, true_action)
        #     if pred_action == true_action:
        #         same_num += 1
        # return same_num / len(states)

    def evaluate(self, env_test, num_eval_episodes=10):
        """
        评估模型
        """
        mean_return = 0.0  # 评估时的平均回报
        for _ in range(num_eval_episodes):
            state = env_test.reset()
            episode_return = 0.0
            done = False
            while not done:
                action = self.model.exploit(state)
                state, reward, done, _ = env_test.step(action)
                episode_return += reward

            mean_return += episode_return / num_eval_episodes

        print(f'Return: {mean_return:<5.1f}   ')

    def update_params(self, weights_before, weights_after, l_rate):
        """更新元模型参数"""
        disc_weights_before, actor_weights_before = weights_before[0], weights_before[1]
        disc_weights_after, actor_weights_after = weights_after[0], weights_after[1]
        self.model.disc.load_state_dict(OrderedDict(
            {name: disc_weights_before[name] + (disc_weights_after[name] - disc_weights_before[name]) * l_rate
             for name in disc_weights_before}))
        self.model.actor.load_state_dict(OrderedDict(
            {name: actor_weights_before[name] + (actor_weights_after[name] - actor_weights_before[name]) * l_rate * 0.5
             for name in actor_weights_before}))

    def save(self, path_actor, path_disc):
        torch.save(self.model.actor.state_dict(), path_actor)
        torch.save(self.model.disc.state_dict(), path_disc)

    def load(self, path_actor, path_disc):
        self.model.actor.load_state_dict(torch.load(path_actor))
        self.model.disc.load_state_dict(torch.load(path_disc))

    def compare(self, number=10):
        win = 0
        for i in range(number):
            start_state = get_random_start()  # 随机开局
            result = test_actor(self.model.actor, start_state, self.device)  # 与该bot比赛获取结果
            print("game result:", result)
            if result == 1:
                win += 1
        return win / number


def bot_plot(buffer_expert, buffer_expert_test, epoch=50, bot_number=5):
    numbers = [10, 50, 100, 150, 200, 250, 300]
    mairl_sims = []

    for num in numbers:
        mairl.load("../../weights/mairl_actor_" + str(bot_number) + ".pth",
                   "../../weights/mairl_disc_" + str(bot_number) + ".pth")
        max_sim = mairl.train(Task(buffer_expert, buffer_expert_test), epoch, bot_number, num, save=False)
        mairl_sims.append(max_sim)

    supervised_sims = []
    for num in numbers:
        supervised = Supervised(buffer_expert, buffer_expert_test, device=mairl.device)
        max_sim = supervised.train(9000 + epoch * 10, number=num, start_eval=0)
        # max_sim = supervised.train(epoch * 10, number=num)
        supervised_sims.append(max_sim)

    with open("plot.txt", 'a') as f:
        f.write("Bot" + BOT_NAMES[bot_number] + ",".join(map(str, mairl_sims)) + ";" + ",".join(
            map(str, supervised_sims)) + "\n")

    plot(bot_number, mairl_sims, supervised_sims)


def epoch_plot(buffer_expert, buffer_expert_test, max_epoch, number=300, bot_number=1):
    mairl.load("../../weights/mairl_actor_" + str(bot_number) + ".pth",
               "../../weights/mairl_disc_" + str(bot_number) + ".pth")
    sims = mairl.train(Task(buffer_expert, buffer_expert_test), max_epoch, bot_number, number, save=False,
                       is_out_list=True)

    supervised = Supervised(buffer_expert, buffer_expert_test, device=mairl.device)
    super_sims = supervised.train(max_epoch * 10, number=number, start_eval=0, is_out_list=True)
    # super_sims = supervised.train(max_epoch * 10, number=number, is_out_list=True)

    x = list(range(0, max_epoch * 10, 10))
    # x_index = [str(i * 10) for i in x]
    plt.plot(x, sims, "b-", label="our method")
    plt.plot(x, super_sims, "g-", label="supervised method")

    plt.axhline(y=BASELINES[bot_number], ls=':', c="red", label="baseline")
    plt.legend(shadow=True, fontsize=12)
    plt.xlabel("Number of iterations", fontsize=15)
    plt.ylabel("Actions similarity", fontsize=15)
    plt.yticks(fontsize=15)
    # plt.xticks(x, x_index, fontsize=15)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d %%'))
    plt.title("Bot " + BOT_NAMES[bot_number], fontsize=17)
    plt.tight_layout()
    plt.savefig(BOT_NAMES[bot_number] + "_epochs.eps")
    plt.show()


def plot(bot_number, mairl_sims, supervised_sims):
    x = [1, 2, 3, 4, 5, 6, 7]
    x_index = ['10', '50', '100', '150', '200', '250', '300']
    plt.plot(x, mairl_sims, "b-d", label='our method')
    plt.plot(x, supervised_sims, 'k-s', label='supervised method')

    plt.axhline(y=BASELINES[bot_number], ls=':', c="red", label="baseline")

    plt.legend(shadow=True, fontsize=12)
    plt.xlabel("Number of gameplay records", fontsize=15)
    plt.ylabel("Actions similarity", fontsize=15)
    plt.yticks(fontsize=15)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d %%'))
    plt.xticks(x, x_index, fontsize=15)
    plt.title("Bot " + BOT_NAMES[bot_number], fontsize=17)
    plt.tight_layout()
    plt.savefig(BOT_NAMES[bot_number] + "sim.eps")
    plt.show()


def sims_test():
    for num in range(6):
        train_tasks = [tasks[i] for i in range(6) if i != num]
        assert len(train_tasks) == 5
        save_actor_file = "../../weights/mairl_actor_" + str(num) + ".pth"
        save_disc_file = "../../weights/mairl_disc_" + str(num) + ".pth"
        mairl_test = MIRL(env, tasks, device)
        mairl_test.meta_train(300, 3, save_actor_file, save_disc_file)
        mairl_test.load(save_actor_file, save_disc_file)
        mairl_test.train(tasks[num], 500, num + 1, train_number=500)


if __name__ == '__main__':
    device = torch.device('cuda')
    env = ReversiEnv()
    state_shape = env.observation_space.shape
    action_shape = [env.action_space.n]

    # -------------------载入数据-------------------
    bot_ids = ["5ac5d4c6611cfd778a04676f", "5bd0358b0681335cc1f44e28", "5c8b8a97836522250220bf5e",
               "5cda1c4bd2337e01c79b64ec", "5af87c9a14f1110f6826fef4", "5e74541be952081b8838a26a"]
    buffer_expert1 = SerializedBuffer("../../buffers/" + bot_ids[0] + "_size2400_train.pth", device)
    buffer_expert2 = SerializedBuffer("../../buffers/" + bot_ids[1] + "_size2400_train.pth", device)
    buffer_expert3 = SerializedBuffer("../../buffers/" + bot_ids[2] + "_size2400_train.pth", device)
    buffer_expert4 = SerializedBuffer("../../buffers/" + bot_ids[3] + "_size2400_train.pth", device)
    buffer_expert5 = SerializedBuffer("../../buffers/" + bot_ids[4] + "_size2400_train.pth", device)
    buffer_expert6 = SerializedBuffer("../../buffers/" + bot_ids[5] + "_size2400_train.pth", device)
    buffer_expert1_test = SerializedBuffer("../../buffers/" + bot_ids[0] + "_size800_test.pth", device)
    buffer_expert2_test = SerializedBuffer("../../buffers/" + bot_ids[1] + "_size800_test.pth", device)
    buffer_expert3_test = SerializedBuffer("../../buffers/" + bot_ids[2] + "_size800_test.pth", device)
    buffer_expert4_test = SerializedBuffer("../../buffers/" + bot_ids[3] + "_size800_test.pth", device)
    buffer_expert5_test = SerializedBuffer("../../buffers/" + bot_ids[4] + "_size800_test.pth", device)
    buffer_expert6_test = SerializedBuffer("../../buffers/" + bot_ids[5] + "_size800_test.pth", device)
    tasks = [Task(buffer_expert1, buffer_expert1_test), Task(buffer_expert2, buffer_expert2_test),
             Task(buffer_expert3, buffer_expert3_test), Task(buffer_expert4, buffer_expert4_test),
             Task(buffer_expert5, buffer_expert5_test), Task(buffer_expert6, buffer_expert6_test)]
    # tasks = [Task(buffer_expert1) for _ in range(5)]

    # -----------------模型训练------------------
    sims_test()
    mairl = MIRL(env, tasks, device)
    # mairl.meta_train(epoch=300)  # 训练元模型
    # 载入元模型文件并分别训练
    # mairl.load("../../weights/mairl_actor.pth", "../../weights/mairl_disc.pth")
    # mairl.train(Task(buffer_expert6, buffer_expert6_test), 500, 6)
    # mairl.load("../../weights/mairl_actor.pth", "../../weights/mairl_disc.pth")
    # mairl.train(Task(buffer_expert5, buffer_expert5_test), 500, 5)
    # mairl.load("../../weights/mairl_actor.pth", "../../weights/mairl_disc.pth")
    # mairl.train(Task(buffer_expert4, buffer_expert4_test), 500, 4)
    # mairl.load("../../weights/mairl_actor.pth", "../../weights/mairl_disc.pth")
    # mairl.train(Task(buffer_expert3, buffer_expert3_test), 500, 3)
    # mairl.load("../../weights/mairl_actor.pth", "../../weights/mairl_disc.pth")
    # mairl.train(Task(buffer_expert2, buffer_expert2_test), 500, 2)
    # mairl.load("../../weights/mairl_actor.pth", "../../weights/mairl_disc.pth")
    # mairl.train(Task(buffer_expert1, buffer_expert1_test), 500, 1)

    bot_plot(buffer_expert3, buffer_expert3_test, 500, 2)
    bot_plot(buffer_expert1, buffer_expert1_test, 500, 0)
    bot_plot(buffer_expert2, buffer_expert2_test, 500, 1)
    bot_plot(buffer_expert4, buffer_expert4_test, 500, 3)
    bot_plot(buffer_expert5, buffer_expert5_test, 500, 4)
    bot_plot(buffer_expert6, buffer_expert6_test, 500, 5)

    epoch_plot(buffer_expert1, buffer_expert1_test, 500, 250, 0)
    epoch_plot(buffer_expert2, buffer_expert2_test, 500, 250, 1)
    epoch_plot(buffer_expert3, buffer_expert3_test, 500, 250, 2)
    epoch_plot(buffer_expert4, buffer_expert4_test, 500, 250, 3)
    epoch_plot(buffer_expert5, buffer_expert5_test, 500, 250, 4)
    epoch_plot(buffer_expert6, buffer_expert6_test, 500, 250, 5)

    # plot_file = open("plot.txt", 'r')
    # lines = plot_file.readlines()[-6:]
    # for line in lines:
    #     name = line[3:6]
    #     mairl_sims_str, supervised_sims_str = line[6:].split(";")
    #     mairl_sims = list(map(float, mairl_sims_str.split(',')))
    #     supervised_sims = list(map(float, supervised_sims_str.split(',')))
    #     bot_number = BOT_NAMES.index(name)
    #     plot(bot_number, mairl_sims, supervised_sims)

    # mairl.load("../../weights/actor5.pth", "../../weights/disc5.pth")
    # print("bot5 win rate:",mairl.compare())
    # mairl.load("../../weights/actor4.pth", "../../weights/disc4.pth")
    # print("bot4 win rate:",mairl.compare())
    # mairl.load("../../weights/actor3.pth", "../../weights/disc3.pth")
    # print("bot3 win rate:",mairl.compare())
    # mairl.load("../../weights/actor2.pth", "../../weights/disc2.pth")
    # print("bot2 win rate:",mairl.compare())
    # mairl.load("../../weights/actor1.pth", "../../weights/disc1.pth")
    # print("bot1 win rate:",mairl.compare())
