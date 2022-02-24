#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2022/2/18 下午3:03
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : compare.py
# @Software: PyCharm
from gail_airl_ppo.buffer import SerializedBuffer
from torch import nn
import torch


class Supervised:
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(3 * 8 * 8, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 66),
                nn.Softmax()
            )

        def forward(self, x):
            x = x.reshape(-1, 3 * 8 * 8)
            return self.model(x)

    # class CONV(nn.Module):
    #     def __init__(self, input_chl, output_dim):
    #         self.model = nn.Sequential(
    #             nn.Conv2d(input_chl, )
    #         )

    def __init__(self, train_dataset: SerializedBuffer, test_dataset: SerializedBuffer,
                 device=torch.device("cpu")):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.net = self.MLP().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.02)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, epoch, eval_freq=None, number=-1):
        if number != -1:
            train_dataset = self.train_dataset.reduce(number)
        else:
            train_dataset = self.train_dataset
        state_exp = train_dataset.states.to(self.device)
        action_exp = train_dataset.actions.to(self.device)

        print("Lenght:", len(state_exp))
        max_metric = 0.0
        for e in range(epoch):
            q_value = self.net(state_exp)
            loss = self.loss_fn(q_value, action_exp)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metric = self.evaluate()
            if max_metric < metric:
                max_metric = metric
            if eval_freq is not None and e % eval_freq == 0:
                print("Epoch:", e)
                print("Train action similarity:", self.actions_similarity(state_exp, action_exp), "%")
                print("Test action similarity:", metric, "%")
        print("Max test action similarity:", max_metric, "%")
        return max_metric

    def evaluate(self):
        state_exp = self.test_dataset.states
        action_exp = self.test_dataset.actions
        metric = self.actions_similarity(state_exp, action_exp)
        return metric

    def actions_similarity(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)
        same_num = 0  # 相同的动作数量
        pred_actions = self.net(states).argmax(dim=1)
        true_actions = actions.argmax(dim=1)
        same_num = (pred_actions == true_actions).sum()
        return (same_num / len(states)).item() * 100


if __name__ == '__main__':
    device = torch.device('cuda')
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

    supervised1 = Supervised(buffer_expert1, buffer_expert1_test, device)
    supervised1.train(300, number=500)
    supervised2 = Supervised(buffer_expert2, buffer_expert2_test, device)
    supervised2.train(300, number=500)
    supervised3 = Supervised(buffer_expert3, buffer_expert3_test, device)
    supervised3.train(300, number=500)
    supervised4 = Supervised(buffer_expert4, buffer_expert4_test, device)
    supervised4.train(300, number=500)
    supervised5 = Supervised(buffer_expert5, buffer_expert5_test, device)
    supervised5.train(300, number=500)
    supervised6 = Supervised(buffer_expert6, buffer_expert6_test, device)
    supervised6.train(300, number=500)
