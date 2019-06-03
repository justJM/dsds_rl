import random
import numpy as np
from collections import deque

class ReplayMemory():
    '''
    Replay memory to store states, actions, rewards, dones for batch sampling
    '''
    def __init__(self, capacity):
        '''
        :param capacity: replay memory capacity
        '''
        self.buffer = deque(maxlen=capacity)

    # e_t = (s_t,, a_t, r_t+1, s_t+1)
    def add(self, state, action, reward, done, next_state):
        '''
        :param state: current state, atari_wrappers.LazyFrames object
        :param action: action
        :param reward: reward for the action
        :param done: "done" flag is True when the episode finished
        :param next_state: next state, atari_wrappers.LazyFrames object
        '''
        experience = (state, action, reward, done, next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        '''
        Samples the data from the buffer of a desired size

        :param batch_size: sample batch size
        :return: batch of (states, actions, rewards, dones, next states).
                 all are numpy arrays. states and next states have shape of
                 (batch_size, frames, width, height), where frames = 4.
                 actions, rewards and dones have shape of (batch_size,)
        '''
        if self.count() < batch_size:
            batch = random.sample(self.buffer, self.count())
        else:
            batch = random.sample(self.buffer, batch_size)

        state_batch = np.array([np.array(experience[0]) for experience in batch])
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        done_batch = np.array([experience[3] for experience in batch])
        next_state_batch = np.array([np.array(experience[4]) for experience in batch])

        return state_batch, action_batch, reward_batch, done_batch, next_state_batch

    def count(self):
        return len(self.buffer)
