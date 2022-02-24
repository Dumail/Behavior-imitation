#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/10/27 下午5:54
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : bot.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
'''
update:
    估值函数为权值表+行动力+稳定子
    对于较顶层节点使用一步预搜索获得估值排序后的前maxN个最优操作，限制搜索宽度并尽量提前剪枝

test:
    潜在行动力无助于提升性能

'''

import json
import random

import numpy
import torch

from gail_airl_ppo.algo.reversi_algo import ReversiAlgo
from reversi_env import ReversiEnv

DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 方向向量


# 放置棋子，计算新局面
def place(board, x, y, color):
    if x < 0:
        return False
    board[x][y] = color
    valid = False
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid


# 查询合法落子
def getmoves(board, color):
    moves = []
    ValidBoardList = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                newBoard = board.copy()
                if place(newBoard, i, j, color):
                    moves.append((i, j))
                    ValidBoardList.append(newBoard)
    return moves, ValidBoardList


def getbound(board, color):
    bound1 = 0
    bound2 = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                flag1 = 0
                flag2 = 0
                for d in range(8):
                    ii = i + DIR[d][0]
                    jj = j + DIR[d][1]
                    if 0 <= ii and ii < 8 and 0 <= jj and jj < 8:
                        if flag1 == 0 and board[ii][jj] == -color:
                            flag1 = 1
                            bound1 += 1
                        if flag2 == 0 and board[ii][jj] == color:
                            flag2 = 1
                            bound2 += 1
    return bound1, bound2


def getstable(board, color):
    stable = [0, 0, 0]
    # 角, 边, 八个方向都无空格
    cind1 = [0, 0, 7, 7]
    cind2 = [0, 7, 7, 0]
    inc1 = [0, 1, 0, -1]
    inc2 = [1, 0, -1, 0]
    stop = [0, 0, 0, 0]
    for i in range(4):
        if board[cind1[i]][cind2[i]] == color:
            stop[i] = 1
            stable[0] += 1
            for j in range(1, 7):
                if board[cind1[i] + inc1[i] * j][cind2[i] + inc2[i] * j] != color:
                    break
                else:
                    stop[i] = j + 1
                    stable[1] += 1
    for i in range(4):
        if board[cind1[i]][cind2[i]] == color:
            for j in range(1, 7 - stop[i - 1]):
                if board[cind1[i] - inc1[i - 1] * j][cind2[i] - inc2[i - 1] * j] != color:
                    break
                else:
                    stable[1] += 1
    colfull = numpy.zeros((8, 8), dtype=numpy.int64)
    colfull[:, numpy.sum(abs(board), axis=0) == 8] = True
    rowfull = numpy.zeros((8, 8), dtype=numpy.int64)
    rowfull[numpy.sum(abs(board), axis=1) == 8, :] = True
    diag1full = numpy.zeros((8, 8), dtype=numpy.int64)
    for i in range(15):
        diagsum = 0
        if i <= 7:
            sind1 = i
            sind2 = 0
            jrange = i + 1
        else:
            sind1 = 7
            sind2 = i - 7
            jrange = 15 - i
        for j in range(jrange):
            diagsum += abs(board[sind1 - j][sind2 + j])
        if diagsum == jrange:
            for k in range(jrange):
                diag1full[sind1 - j][sind2 + j] = True
    diag2full = numpy.zeros((8, 8), dtype=numpy.int64)
    for i in range(15):
        diagsum = 0
        if i <= 7:
            sind1 = i
            sind2 = 7
            jrange = i + 1
        else:
            sind1 = 7
            sind2 = 14 - i
            jrange = 15 - i
        for j in range(jrange):
            diagsum += abs(board[sind1 - j][sind2 - j])
        if diagsum == jrange:
            for k in range(jrange):
                diag2full[sind1 - j][sind2 - j] = True
    stable[2] = sum(
        sum(numpy.logical_and(numpy.logical_and(numpy.logical_and(colfull, rowfull), diag1full), diag2full)))
    return stable


Vmap = numpy.array([[500, -25, 10, 5, 5, 10, -25, 500],
                    [-25, -45, 1, 1, 1, 1, -45, -25],
                    [10, 1, 3, 2, 2, 3, 1, 10],
                    [5, 1, 2, 1, 1, 2, 1, 5],
                    [5, 1, 2, 1, 1, 2, 1, 5],
                    [10, 1, 3, 2, 2, 3, 1, 10],
                    [-25, -45, 1, 1, 1, 1, -45, -25],
                    [500, -25, 10, 5, 5, 10, -25, 500]])


def mapweightsum(board, mycolor):
    return sum(sum(board * Vmap)) * mycolor


def evaluation(moves, board, mycolor):
    moves_, ValidBoardList_ = getmoves(board, -mycolor)
    stable = getstable(board, mycolor)
    value = mapweightsum(board, mycolor) + 15 * (len(moves) - len(moves_)) + 10 * sum(stable)
    return int(value)


def onestepplace(board, mycolor):
    stage = sum(sum(abs(board)))
    if stage <= 9:
        depth = 5
    elif stage >= 50:
        depth = 6
    else:
        depth = 4
    value, bestmove = alphabetav2(board, depth, -10000, 10000, mycolor, mycolor, depth)
    return bestmove


def alphabetav2(board, depth, alpha, beta, actcolor, mycolor, maxdepth):
    moves, ValidBoardList = getmoves(board, actcolor)
    if len(moves) == 0:
        return evaluation(moves, board, mycolor), (-1, -1)
    if depth == 0:
        return evaluation(moves, board, mycolor), []

    if depth == maxdepth:
        for i in range(len(moves)):
            if Vmap[moves[i][0]][moves[i][1]] == Vmap[0][0] and actcolor == mycolor:
                return 1000, moves[i]

    # 对于较顶层节点使用一步预搜索获得估值排序后的前maxN个最优操作，限制搜索宽度并尽量提前剪枝
    if depth >= 4:
        Vmoves = []
        for i in range(len(ValidBoardList)):
            value, bestmove = alphabetav2(ValidBoardList[i], 1, -10000, 10000, -actcolor, mycolor, maxdepth)
            Vmoves.append(value)
        ind = numpy.argsort(Vmoves)
        maxN = 6
        moves = [moves[i] for i in ind[0:maxN]]
        ValidBoardList = [ValidBoardList[i] for i in ind[0:maxN]]

    bestmove = []
    bestscore = -10000
    for i in range(len(ValidBoardList)):
        score, childmove = alphabetav2(ValidBoardList[i], depth - 1, -beta, -max(alpha, bestscore), -actcolor, mycolor,
                                       maxdepth)
        score = -score
        if score > bestscore:
            bestscore = score
            bestmove = moves[i]
            if bestscore > beta:
                return bestscore, bestmove
    return bestscore, bestmove


# 处理输入，还原棋盘
def initBoard():
    fullInput = json.loads(input())
    requests = fullInput["requests"]
    responses = fullInput["responses"]
    board = numpy.zeros((8, 8), dtype=numpy.int)
    board[3][4] = board[4][3] = 1  # 白
    board[3][3] = board[4][4] = -1  # 黑
    myColor = 1
    if requests[0]["x"] >= 0:
        myColor = -1
        place(board, requests[0]["x"], requests[0]["y"], -myColor)
    turn = len(responses)
    for i in range(turn):
        place(board, responses[i]["x"], responses[i]["y"], myColor)
        place(board, requests[i + 1]["x"], requests[i + 1]["y"], -myColor)
    return board, myColor


class BotActor:
    def __init__(self, env):
        self.env = env

    def policy(self, state):
        state = state.detach().cpu().numpy()[0]
        player_board, opponent_board = state[0], state[1]
        myColor = 1  # 白色
        # opponent_board = -opponent_board
        player_board = -player_board
        board = player_board + opponent_board
        # board, myColor = initBoard()
        x, y = onestepplace(board, myColor)
        # print("bot:", x, y)
        return torch.tensor([x * 8 + y])

    def sample(self, state=None):
        return self.policy(state)

    def __call__(self, state=None):
        return self.policy(state)


def get_random_start():
    """生成随机起始状态"""
    reversi_env = ReversiEnv()
    state_shape = reversi_env.observation_space.shape
    action_shape = reversi_env.action_space.shape or reversi_env.action_space.n
    algo = ReversiAlgo(reversi_env, state_shape, action_shape, ReversiEnv.BLACK, actor="random")
    obs = reversi_env.reset()
    epoch = random.randint(0, 15)
    for i in range(epoch):
        action = algo.exploit(obs)
        obs, rew, done, _ = reversi_env.step(action)
    return obs


def test_actor(actor, start_state, device=torch.device("cpu")):
    # 测试黑白棋环境
    bot = BotActor(None)
    reversi_env = ReversiEnv(player_color='black', opponent=bot)
    state_shape = reversi_env.observation_space.shape
    action_shape = reversi_env.action_space.shape or reversi_env.action_space.n
    algo = ReversiAlgo(reversi_env, state_shape, action_shape, ReversiEnv.BLACK, actor=actor, device=device)

    if start_state is not None:
        reversi_env.set_start_state(start_state)
        obs = reversi_env.state
    else:
        obs = reversi_env.reset()
    done = False
    while not done:
        # reversi_env.render()
        action = algo.exploit(obs)
        # print("player:", action // 8, action % 8)
        # print("player",action.argmax())
        obs, rew, done, _ = reversi_env.step(action)

    if reversi_env.black_score > reversi_env.white_score:
        return 1
    elif reversi_env.black_score < reversi_env.white_score:
        return -1
    else:
        return 0


if __name__ == '__main__':
    start_state = get_random_start()  # 随机开局
    print("result:", test_actor('greedy', start_state))  # 与该bot比赛获取结果
