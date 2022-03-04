import os
from abc import ABC, abstractmethod

import numpy as np
import torch


# def discrete_wrapper(func):
#     def temp_fun(self, *args, **kwargs):
#         result = func(self, *args, **kwargs)
#         if not isinstance(result, tuple):
#             return np.argmax(result)
#         else:
#             temp_action = np.argmax(result[0])
#             return temp_action, result[1]
#
#     return temp_fun


class Algorithm(ABC):

    def __init__(self, state_shape, action_shape, device, seed, gamma):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    # @discrete_wrapper
    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze_(0))
        act = action.cpu().numpy()[0]
        log_pi = log_pi.item()
        return act, log_pi

    # @discrete_wrapper
    def exploit(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
