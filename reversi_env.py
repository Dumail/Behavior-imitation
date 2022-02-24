"""
Game of Reversi
"""
import os
import random

from six import StringIO
import sys
import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding

import torch

from gail_airl_ppo.algo.reversi_algo import ReversiAlgo
from gail_airl_ppo.utils import collect_demo


def reversi_state_wrapper(state):
    """
    黑白棋状态装饰器
    :param state: 原始状态，表示为 3*board_size*board_size 的np数组
    :return: 表示为 2*board_size*board_size 的np 数组
    """
    board_size = state.shape[1]
    new_state = np.zeros(board_size ** 2 * 2)
    new_state[:board_size ** 2] = np.reshape(state[0, :, :], board_size ** 2)
    new_state[board_size ** 2:] = np.reshape(state[1, :, :], board_size ** 2)
    return new_state


class ReversiEnv(gym.Env):
    """
    Reversi environment. Play against a fixed opponent.
    """
    BLACK = 0
    WHITE = 1
    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, player_color='black', opponent=None, observation_type='numpy3c',
                 illegal_place_mode='lose',
                 board_size=8):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent agent
            observation_type: State encoding
            illegal_place_mode: What to do when the agent makes an illegal place. Choices: 'raise' or 'lose'
            board_size: size of the Reversi board
        """
        self._max_episode_steps = 30
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size
        self.state = np.zeros((3, self.board_size, self.board_size))

        colormap = {
            'black': ReversiEnv.BLACK,
            'white': ReversiEnv.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error("player_color must be 'black' or 'white', not {}".format(player_color))

        self.opponent = opponent
        if self.opponent:
            self.opponent = ReversiAlgo(self, self.state.shape, self.board_size ** 2 + 2,
                                        color=[ReversiEnv.BLACK, ReversiEnv.WHITE][player_color == 'black'],
                                        actor=self.opponent)
        else:
            self.opponent = ReversiAlgo(self, self.state.shape, self.board_size ** 2 + 2,
                                        color=[ReversiEnv.BLACK, ReversiEnv.WHITE][player_color == 'black'])

        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        assert illegal_place_mode in ['lose', 'raise']
        self.illegal_place_mode = illegal_place_mode

        if self.observation_type != 'numpy3c':
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        self.seed()

        observation = self.reset()
        # One action for each board position and resign and pass
        self.action_space = spaces.Discrete(self.board_size ** 2 + 2)
        self.observation_space = spaces.Box(np.zeros(observation.shape), np.ones(observation.shape), dtype=np.float64)

    def sample_action(self):
        if len(self.possible_actions) == 0:
            action = self.board_size ** 2 + 1
        else:
            action = random.choice(self.possible_actions)
        return action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # # Update the random policy if needed
        # if isinstance(self.opponent, str):
        #     if self.opponent == 'random':
        #         self.opponent_policy = make_random_policy(self.np_random)
        #     else:
        #         raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
        # else:
        #     self.opponent_policy = self.opponent

        return [seed]

    def set_start_state(self, state):
        self.state = state
        self.to_play = ReversiEnv.BLACK
        self.possible_actions = ReversiEnv.get_possible_actions(self.state, self.to_play)

    def reset(self):
        # init board setting
        self.state = np.zeros((3, self.board_size, self.board_size))
        centerL = int(self.board_size / 2 - 1)
        centerR = int(self.board_size / 2)
        self.state[2, :, :] = 0
        # self.state[2, (centerL):(centerR + 1), (centerL):(centerR + 1)] = 0
        self.state[0, centerR, centerL] = 1
        self.state[0, centerL, centerR] = 1
        self.state[1, centerL, centerL] = 1
        self.state[1, centerR, centerR] = 1
        self.to_play = ReversiEnv.BLACK
        self.possible_actions = ReversiEnv.get_possible_actions(self.state, self.to_play)
        # 合法行为直接放入状态
        for p_action in self.possible_actions:
            if p_action <= 64:
                x, y = ReversiEnv.action_to_coordinate(self.state, p_action)
                self.state[2, x, y] = 1

        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            a = self.opponent.exploit(self.state)
            ReversiEnv.make_place(self.state, a, ReversiEnv.BLACK)
            self.to_play = ReversiEnv.WHITE
            self.possible_actions = ReversiEnv.get_possible_actions(self.state, self.to_play)
            # 合法行为直接放入状态
            self.state[2, :, :] = 0
            for p_action in self.possible_actions:
                if p_action <= 64:
                    x, y = ReversiEnv.action_to_coordinate(self.state, p_action)
                    self.state[2, x, y] = 1
        return self.state

    def step(self, action, show_action=False):
        raw_action = action
        if not isinstance(action, np.int64):
            action = np.argmax(action)
        assert self.to_play == self.player_color
        # If already terminal, then don't do anything
        if self.done:
            return self.state, 0., True, {'state': self.state}
        if ReversiEnv.pass_place(self.board_size, action):
            pass
        elif ReversiEnv.resign_place(self.board_size, action):
            return self.state, -1, True, {'state': self.state}
        elif not ReversiEnv.valid_place(self.state, action, self.player_color):
            if self.illegal_place_mode == 'raise':
                print(raw_action)
                print(action)
                print(self.possible_actions)
                raise
            elif self.illegal_place_mode == 'lose':
                # Automatic loss on illegal place
                print(raw_action)
                print(action)
                print(self.possible_actions)
                self.done = True
                return self.state, -1., True, {'state': self.state}
            else:
                raise error.Error('Unsupported illegal place action: {}'.format(self.illegal_place_mode))
        else:
            ReversiEnv.make_place(self.state, action, self.player_color)

        if show_action:
            print("Player", ['black', 'white'][self.player_color == ReversiEnv.WHITE], " action:",
                  self.action_to_coordinate(self.state, action, to_view=True))

        # Opponent play
        # a = self.opponent.policy(self.state, 1 - self.player_color)
        self.possible_actions = ReversiEnv.get_possible_actions(self.state, 1 - self.player_color)
        # 合法行为直接放入状态
        self.state[2, :, :] = 0
        for p_action in self.possible_actions:
            if p_action <= 64:
                x, y = ReversiEnv.action_to_coordinate(self.state, p_action)
                self.state[2, x, y] = 1

        a = self.opponent.exploit(self.state)

        if not isinstance(a, np.int64):
            a = np.argmax(a)
        # Making place if there are places left
        if a is not None and a >= 0:
            if ReversiEnv.pass_place(self.board_size, a):
                pass
            elif ReversiEnv.resign_place(self.board_size, a):
                return self.state, 1, True, {'state': self.state}
            elif not ReversiEnv.valid_place(self.state, a, 1 - self.player_color):
                if self.illegal_place_mode == 'raise':
                    raise
                elif self.illegal_place_mode == 'lose':
                    # Automatic loss on illegal place
                    print(raw_action)
                    print(action)
                    print(self.possible_actions)
                    self.done = True
                    return self.state, 1., True, {'state': self.state}
                else:
                    raise error.Error('Unsupported illegal place action: {}'.format(self.illegal_place_mode))
            else:
                ReversiEnv.make_place(self.state, a, 1 - self.player_color)

        if show_action:
            print("Opponent ", ['black', 'white'][self.opponent.color == ReversiEnv.WHITE], " action:",
                  self.action_to_coordinate(self.state, a, to_view=True))

        self.possible_actions = ReversiEnv.get_possible_actions(self.state, self.player_color)
        # 合法行为直接放入状态
        self.state[2, :, :] = 0
        for p_action in self.possible_actions:
            if p_action <= 64:
                x, y = ReversiEnv.action_to_coordinate(self.state, p_action)
                self.state[2, x, y] = 1

        no_complete_finish = ReversiEnv.pass_place(self.board_size, a) and ReversiEnv.pass_place(self.board_size,
                                                                                                 action)
        reward = ReversiEnv.game_finished(self.state, no_complete_finish)
        if self.player_color == ReversiEnv.WHITE:
            reward = - reward
        self.done = reward != 0
        return self.state, reward, self.done, {'state': self.state}

    # def _reset_opponent(self):
    #     if self.opponent == 'random':
    #         self.opponent_policy = random_policy
    #     else:
    #         raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))

    def render(self, mode='human', close=False):
        if close:
            return
        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write(' ' * 7)
        for j in range(board.shape[1]):
            outfile.write(' ' + str(j + 1) + '  | ')
        outfile.write('\n')
        outfile.write(' ' * 5)
        outfile.write('-' * (board.shape[1] * 6 - 1))
        outfile.write('\n')
        for i in range(board.shape[1]):
            outfile.write(' ' + str(i + 1) + '  |')
            for j in range(board.shape[1]):
                if board[2, i, j] == 1:
                    outfile.write('  O  ')
                elif board[0, i, j] == 1:
                    outfile.write('  B  ')
                elif board[1, i, j] == 1:
                    outfile.write('  W  ')
                else:
                    outfile.write('  _  ')
                outfile.write('|')
            outfile.write('\n')
            outfile.write(' ')
            outfile.write('-' * (board.shape[1] * 7 - 1))
            outfile.write('\n')
        outfile.write('\n')

        if mode != 'human':
            return outfile

    # @staticmethod
    # def pass_place(board_size, action):
    #     return action == board_size ** 2

    @staticmethod
    def resign_place(board_size, action):
        return action == board_size ** 2

    @staticmethod
    def pass_place(board_size, action):
        return action == board_size ** 2 + 1

    @staticmethod
    def get_possible_actions(board, player_color):
        actions = []
        d = board.shape[-1]
        opponent_color = 1 - player_color
        for pos_x in range(d):
            for pos_y in range(d):
                if (board[0, pos_x, pos_y] == 1 or board[1, pos_x, pos_y] == 1):
                    continue
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if (dx == 0 and dy == 0):
                            continue
                        nx = pos_x + dx
                        ny = pos_y + dy
                        n = 0
                        if (nx not in range(d) or ny not in range(d)):
                            continue
                        while (board[opponent_color, nx, ny] == 1):
                            tmp_nx = nx + dx
                            tmp_ny = ny + dy
                            if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                                break
                            n += 1
                            nx += dx
                            ny += dy
                        if (n > 0 and board[player_color, nx, ny] == 1):
                            actions.append(pos_x * d + pos_y)
        if len(actions) == 0:
            actions = [d ** 2 + 1]
        return list(set(actions))

    @staticmethod
    def valid_reverse_opponent(board, coords, player_color):
        '''
        check whether there is any reversible places
        '''
        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx == 0 and dy == 0):
                    continue
                nx = pos_x + dx
                ny = pos_y + dy
                n = 0
                if (nx not in range(d) or ny not in range(d)):
                    continue
                while (board[opponent_color, nx, ny] == 1):
                    tmp_nx = nx + dx
                    tmp_ny = ny + dy
                    if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                        break
                    n += 1
                    nx += dx
                    ny += dy
                if (n > 0 and board[player_color, nx, ny] == 1):
                    return True
        return False

    @staticmethod
    def valid_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)
        # check whether there is any empty places
        if board[2, coords[0], coords[1]] == 1:
            # check whether there is any reversible places
            if ReversiEnv.valid_reverse_opponent(board, coords, player_color):
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def make_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)

        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx == 0 and dy == 0):
                    continue
                nx = pos_x + dx
                ny = pos_y + dy
                n = 0
                if (nx not in range(d) or ny not in range(d)):
                    continue
                while (board[opponent_color, nx, ny] == 1):
                    tmp_nx = nx + dx
                    tmp_ny = ny + dy
                    if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                        break
                    n += 1
                    nx += dx
                    ny += dy
                if (n > 0 and board[player_color, nx, ny] == 1):
                    nx = pos_x + dx
                    ny = pos_y + dy
                    while (board[opponent_color, nx, ny] == 1):
                        board[2, nx, ny] = 0
                        board[player_color, nx, ny] = 1
                        board[opponent_color, nx, ny] = 0
                        nx += dx
                        ny += dy
                    board[2, pos_x, pos_y] = 0
                    board[player_color, pos_x, pos_y] = 1
                    board[opponent_color, pos_x, pos_y] = 0
        return board

    @staticmethod
    def coordinate_to_action(board, coords):
        return coords[0] * board.shape[-1] + coords[1]

    @staticmethod
    def action_to_coordinate(board, action, to_view=False):
        if to_view:
            return action // board.shape[-1] + 1, action % board.shape[-1] + 1
        return action // board.shape[-1], action % board.shape[-1]

    @staticmethod
    def game_finished(board, no_complete_finish=False):
        # Returns 1 if player 1 wins, -1 if player 2 wins and 0 otherwise
        d = board.shape[-1]

        player_score_x, player_score_y = np.where(board[0, :, :] == 1)
        player_score = len(player_score_x)
        opponent_score_x, opponent_score_y = np.where(board[1, :, :] == 1)
        opponent_score = len(opponent_score_x)
        if player_score == 0:
            return -1
        elif opponent_score == 0:
            return 1
        else:
            free_x, free_y = np.where(board[2, :, :] == 1)
            if free_x.size == 0:
                if player_score > (d ** 2) / 2:
                    return 1
                elif player_score == (d ** 2) / 2:
                    return 1
                else:
                    return -1
            elif no_complete_finish:
                if player_score > opponent_score:
                    return 1
                elif player_score < opponent_score:
                    return -1
            else:
                return 0
        return 0

    @property
    def black_score(self):
        return len(np.where(self.state[0, :, :] == 1)[0])

    @property
    def white_score(self):
        return len(np.where(self.state[1, :, :] == 1)[0])


def test_envs():
    # 测试黑白棋环境
    reversi_env = ReversiEnv()
    state_shape = reversi_env.observation_space.shape
    action_shape = reversi_env.action_space.shape or reversi_env.action_space.n
    # 贪心智能体
    random_algo = ReversiAlgo(reversi_env, state_shape, action_shape, ReversiEnv.BLACK, actor="random")

    obs = reversi_env.reset()
    done = False
    while not done:
        reversi_env.render()
        action = random_algo.exploit(obs)
        print(action)
        obs, rew, done, _ = reversi_env.step(action)

    print("Black score:", reversi_env.black_score, "White score:", reversi_env.white_score)


def test_collect():
    """测试黑白棋环境buffer搜集"""
    env = ReversiEnv()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape or env.action_space.n
    algo = ReversiAlgo(env, state_shape, action_shape, ReversiEnv.BLACK, actor="greedy")

    # 搜集轨迹并存入缓冲区
    buffer = collect_demo(env=env, algo=algo, buffer_size=5000, device=torch.device("cpu"), std=0, p_rand=0,
                          discrete=True)

    # 序列化缓冲区
    buffer.save(os.path.join('buffers', 'reversi', f'size3600_std0_prand0.pth'))


if __name__ == '__main__':
    test_collect()
