#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/10/18 下午7:21
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : data_process.py
# @Software: PyCharm
import itertools
import json
import multiprocessing
import os
import pickle
import random

import numpy as np

from algo.algo.reversi_algo import ListAlgorithm, ReversiAlgo
from algo.buffer import Buffer
from reversi_env import ReversiEnv


def json_obj_process(json_obj, bot_id):
    """处理json对象, 提取相应的bot数据"""
    scores = json_obj['scores']
    player1 = json_obj['players'][0]
    player2 = json_obj['players'][1]
    # print(player1, player2)

    if player1['type'] == 'bot' and player1['bot'] == bot_id:
        first = True
    elif player2['type'] == 'bot' and player2['bot'] == bot_id:
        # first = False
        return None, None  # 只获取先手的数据
    else:
        return None, None

    log = json_obj['log']
    # print(log)
    moves = []
    for i in range(len(log) // 4 - 1):
        player1_move = log[i * 4 + 1]['0']['response']
        move_x, move_y = player1_move['x'], player1_move['y']
        moves.append([move_x, move_y])

        player2_move = log[i * 4 + 3]['1']['response']
        move_x, move_y = player2_move['x'], player2_move['y']
        if len(moves) != 0:
            moves.append([move_x, move_y])

    return first, moves
    # return moves


def json_file_process(file_name, bot_id):
    """
    处理json文件提取每个json对象
    """
    print("Processing json file:", file_name)
    bot_data = []
    # json_objs = []
    with open(file_name, "r") as f:
        for l in f:
            # json_objs.append(json.loads(l))
            json_obj = json.loads(l)
            first, moves = json_obj_process(json_obj, bot_id)
            if moves is not None and len(moves) > 30:
                bot_data.append([first, moves])
    return bot_data


def get_bot_data(dirs, bot_id):
    """
    对于文件夹内的每个文件提取相应的bot数据
    """
    file_names = os.listdir(dirs)
    temp_bot_data = []
    pool = multiprocessing.Pool(processes=4)
    for filename in file_names:
        temp_bot_data.append(pool.apply_async(json_file_process, args=('data/' + filename, bot_id)))
        # bot_data += json_file_process('data/' + filename,bot_id)

    pool.close()
    pool.join()
    temp_bot_data = [d.get() for d in temp_bot_data]
    bot_data = list(itertools.chain(*temp_bot_data))
    print(len(bot_data))

    random.shuffle(bot_data)  # 打乱数据

    # 分别返回训练数据和测试数据
    return bot_data[:2400], bot_data[2400:3000]


def save_to_buffer(data_file):
    """
    储存数据到buffer
    """
    with open(data_file, 'rb') as f:
        pick_data = pickle.load(f)
    # for d in pick_data:
    #     first = pick_data[0]
    #     moves = pick_data[1]


def coordinate_to_action(coords):
    """坐标行为列表转换为行为列表, 8x8"""
    actions = []
    for c in coords:
        a = c[0] * 8 + c[1]
        actions.append(a)
    return actions


def actions_to_buffer(buffer, player_first, actions):
    """将玩家记录转换为buffer"""
    player_actions = actions[0::2] if player_first else actions[1::2]
    agent_actions = actions[1::2] if player_first else actions[0::2]
    # 将坐标转换为动作位置
    player_actions = coordinate_to_action(player_actions)
    agent_actions = coordinate_to_action(agent_actions)

    pair_len = len(player_actions)  # the length of state-action pair

    # 用户智能体，按照动作列表执行
    agent_actor = ListAlgorithm(None, agent_actions)
    reversi_env = ReversiEnv(player_color='black', opponent=agent_actor) if player_first else ReversiEnv(
        player_color='white', opponent=agent_actor)
    state_shape = reversi_env.observation_space.shape
    action_shape = reversi_env.action_space.shape or reversi_env.action_space.n
    player = ReversiAlgo(reversi_env, state_shape, action_shape, ReversiEnv.BLACK,
                         actor=ListAlgorithm(reversi_env, player_actions))

    # buffer = Buffer(
    #     buffer_size=1000,
    #     state_shape=state_shape,
    #     action_shape=[action_shape],
    #     device='cpu',
    #     discrete=True
    # )
    sum_random_sim = 0.0  # 总共的随机相似度
    random_sim = 0.0  # 这场对局的随机相似度

    obs = reversi_env.state.copy()
    temp_action = np.zeros(64 + 2)
    # for i in range(len(player_actions)):
    while True:
        # print("possible_actions:", reversi_env.possible_actions)
        sum_random_sim += 1 / len(reversi_env.possible_actions)  # 所有可行行为数的倒数
        action = player.exploit(obs)
        # print("selected action:", action)
        # reversi_env.render()
        if action < 0:  # -1 表示动作已执行完成
            random_sim = sum_random_sim / len(player_actions)
            # 最后一步
            done = True
            if player_first:
                # 玩家先手则玩家分数为黑棋分数
                reward = 1 if reversi_env.black_score > reversi_env.white_score else -1
            else:
                reward = 0 if reversi_env.black_score > reversi_env.white_score else -1
            break

        next_obs, reward, done, _ = reversi_env.step(action)

        temp_action.fill(0)
        temp_action[action] = 1

        buffer.append(obs, temp_action, reward, done, next_obs)  # 添加数据到缓冲区
        obs = next_obs.copy()

    return pair_len, random_sim


def get_sim_and_buffer(data_file):
    state_shape = (3, 8, 8)
    action_shape = 66
    buffer = Buffer(
        buffer_size=5000,
        state_shape=state_shape,
        action_shape=[action_shape],
        device='cpu',
        discrete=True
    )

    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        # print(data)

    pair_len = 0
    sum_sim = 0  # 随机相似度之和
    for d in data:
        player_first, moves = d[0], d[1]
        temp_pair_len, random_sim = actions_to_buffer(buffer, player_first, moves)
        sum_sim += random_sim
        pair_len += temp_pair_len

    return pair_len, sum_sim / len(data), buffer


def get_results_file(filename, bot_id, opponent_id):
    """获取双方对战结果"""
    temp_sum = 0
    temp_win = 0
    print(filename)
    with open(filename, "r") as f:
        for l in f:
            json_obj = json.loads(l)
            scores = json_obj['scores']
            player1 = json_obj['players'][0]
            player2 = json_obj['players'][1]

            if player1['type'] == 'bot' and player2['type'] == 'bot':
                if (player1['bot'] == bot_id and player2['bot'] == opponent_id) or (
                        player2['bot'] == bot_id and player1['bot'] == opponent_id):
                    temp_sum += 1
                    if scores[0] == 2:
                        temp_win += 1
    return temp_sum, temp_win


def get_results_dir(dirs, bot_id, opponent_id):
    file_names = os.listdir(dirs)
    pool = multiprocessing.Pool(processes=8)
    data = []
    for file_name in file_names:
        data.append(pool.apply_async(get_results_file, args=(dirs + '/' + file_name, bot_id, opponent_id)))
    pool.close()
    pool.join()

    sum_matches = sum([d.get()[0] for d in data])
    bot_win_matches = sum([d.get()[1] for d in data])

    return sum_matches, bot_win_matches


if __name__ == '__main__':
    # # "60404a8181fb3b738e80c6d1",
    # bot_id = "5e74541be952081b8838a26a"
    bot_ids = ["5c0e0b4be0008d16c8471a9f", "5cc0375d35f461309c280504", "5c20862abcf3f64c76e40316",
               "5ac5d4c6611cfd778a04676f", "568507529aab81be75ae4316", "5e74541be952081b8838a26a"]
    # # bot_ids = ["5c0e0b4be0008d16c8471a9f"]
    for bot_id in bot_ids:
        print("Processing bot:", bot_id)
        train_data, test_data = get_bot_data('data', bot_id)
        print("Get", len(train_data) + len(test_data), "matches.")
        f_train = open('bots_first/' + bot_id + '_train.pkl', 'wb')
        pickle.dump(train_data, f_train)  # 存储训练数据
        f_train.close()
        f_test = open('bots_first/' + bot_id + '_test.pkl', 'wb')
        pickle.dump(test_data, f_test)  # 存储测试数据
        f_test.close()

        file_name_train = 'bots_first/' + bot_id + '_train.pkl'
        pair_len_train, sim_train, buffer_train = get_sim_and_buffer(file_name_train)
        buffer_train.save('buffers/' + bot_id + '_size2400_train.pth')

        file_name_test = 'bots_first/' + bot_id + '_test.pkl'
        pair_len_test, sim_test, buffer_test = get_sim_and_buffer(file_name_test)
        buffer_test.save('buffers/' + bot_id + '_size800_test.pth')

        with open("bot_info.txt", 'a') as f:
            f.write(" ".join(
                ["bot_id:", str(bot_id), ", pair length:", str(pair_len_train + pair_len_test), ", random similarity:",
                 str((sim_train + sim_test) / 2 * 100), '%']) + '\n')

    # bot_ids = ["5c0e0b4be0008d16c8471a9f", "5cc0375d35f461309c280504", "5c20862abcf3f64c76e40316",
    #            "5ac5d4c6611cfd778a04676f", "568507529aab81be75ae4316"]
    # opponent_id = "5c1f3d22bcf3f64c76e2fd95"
    #
    # for bot_id in bot_ids:
    #     sum_, win = get_results_dir('data', bot_id, opponent_id)
    #     print(sum_, win)
    #     with open("bot_mathes.txt", 'a') as f:
    #         f.write(" ".join(
    #             ["bot_id:", bot_id, ", opponent_id:", opponent_id, ", total matches:", str(sum_), ", bot win:",
    #              str(win)]) + "\n")
