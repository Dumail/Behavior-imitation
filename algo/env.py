import gym
import numpy as np
from gym.spaces import Discrete

gym.logger.set_level(40)


def make_env(env_id):
    env = gym.make(env_id)
    return NormalizedEnv(env)


class NormalizedEnv(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.discrete = isinstance(env.action_space, Discrete)

        if self.discrete:
            self.scale = 1
        else:
            self.scale = env.action_space.high
            self.action_space.high /= self.scale
            self.action_space.low /= self.scale

    def step(self, action):
        if self.discrete:
            action = np.argmax(action)
        return self.env.step(action * self.scale)
