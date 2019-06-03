import os
import sys
sys.path.append('../')
import datetime

import time
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from tensorboardX import SummaryWriter
from gym.wrappers import Monitor

from utils.atari_wrappers import wrap_dqn
from dqn.replay_memory import ReplayMemory

use_gpu = torch.cuda.is_available()
print('Use GPU: {}'.format(use_gpu))


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        '''
        Forward propogation

        :param inputs: images. expected sshape is (batch_size, frames, width, height)
        '''
        out = self.relu(self.conv1(inputs))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))

        out = out.view(out.size(0), -1)
        out = self.relu(self.fc4(out))
        out = self.fc5(out)

        return out



class PongAgent:
    def __init__(self, mode=None):
        self.env = wrap_dqn(gym.make('PongDeterministic-v4'))
        if mode == 'test':
            self.env = Monitor(self.env, './video', force=True, video_callable=lambda episode_id: True)
        self.num_actions = self.env.action_space.n

        self.dqn = DQN(self.num_actions)
        self.target_dqn = DQN(self.num_actions)

        if use_gpu:
            self.dqn.cuda()
            self.target_dqn.cuda()

        self.buffer = ReplayMemory(1000)

        self.gamma = 0.99

        self.mse_loss = nn.MSELoss()
        self.optim = optim.RMSprop(self.dqn.parameters(), lr=0.01)

        self.out_dir = './model'
        self.writer = SummaryWriter()

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)


    def to_var(self, x):
        x_var = Variable(x)
        if use_gpu:
            x_var = x_var.cuda()
        return x_var


    def predict_q_values(self, states):
        states = self.to_var(torch.from_numpy(states).float())
        actions = self.dqn(states)
        return actions


    def predict_q_target_values(self, states):
        states = self.to_var(torch.from_numpy(states).float())
        actions = self.target_dqn(states)
        return actions


    def select_action(self, state, epsilon):
        choice = np.random.choice([0, 1], p=(epsilon, (1 - epsilon)))

        if choice == 0:
            return np.random.choice(range(self.num_actions))
        else:
            state = np.expand_dims(state, 0)
            actions = self.predict_q_values(state)
            return np.argmax(actions.data.cpu().numpy())


    def update(self, predicts, targets, actions):
        targets = self.to_var(torch.unsqueeze(torch.from_numpy(targets).float(), -1))
        actions = self.to_var(torch.unsqueeze(torch.from_numpy(actions).long(), -1))

        affected_values = torch.gather(predicts, 1, actions)
        loss = self.mse_loss(affected_values, targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


    def get_epsilon(self, total_steps, max_epsilon_steps, epsilon_start, epsilon_final):
        return max(epsilon_final, epsilon_start - total_steps / max_epsilon_steps)


    def sync_target_network(self):
        primary_params = list(self.dqn.parameters())
        target_params = list(self.target_dqn.parameters())
        for i in range(0, len(primary_params)):
            target_params[i].data[:] = primary_params[i].data[:]


    def calculate_q_targets(self, next_states, rewards, dones):
        dones_mask = (dones == 1)

        predicted_q_target_values = self.predict_q_target_values(next_states)

        next_max_q_values = np.max(predicted_q_target_values.data.cpu().numpy(), axis=1)
        next_max_q_values[dones_mask] = 0 # no next max Q values if the game is over
        q_targets = rewards + self.gamma * next_max_q_values

        return q_targets


    def save_final_model(self):
        filename = '{}/final_model.pth'.format(self.out_dir)
        torch.save(self.dqn.state_dict(), filename)


    def save_model_during_training(self, episode):
        filename = '{}/current_model_{}.pth'.format(self.out_dir, episode)
        torch.save(self.dqn.state_dict(), filename)


    def load_model(self, filename):
        self.dqn.load_state_dict(torch.load(filename))
        self.sync_target_network()


    def play(self, episodes):
        for i in range(1, episodes + 1):
            done = False
            state = self.env.reset()
            while not done:
                action = self.select_action(state, 0) # force to choose an action from the network
                state, reward, done, _ = self.env.step(action)
                # self.env.render()


    def close_env(self):
        self.env.close()


    def train(self, replay_buffer_fill_len, batch_size, episodes,
              max_epsilon_steps, epsilon_start, epsilon_final, sync_target_net_freq):
        start_time = time.time()
        print('Start training at: '+ time.asctime(time.localtime(start_time)))

        total_steps = 0
        running_episode_reward = 0

        # populate replay memory
        print('Populating replay buffer... ')
        print('\n')
        state = self.env.reset()
        for i in range(replay_buffer_fill_len):
            action = self.select_action(state, 1) # force to choose a random action
            next_state, reward, done, _ = self.env.step(action)

            self.buffer.add(state, action, reward, done, next_state)

            state = next_state
            if done:
                self.env.reset()

        print('replay buffer populated with {} transitions, start training...'.format(self.buffer.count()))
        print('\n')

        # main loop - iterate over episodes
        for i in range(1, episodes + 1):
            # reset the environment
            done = False
            state = self.env.reset()

            # reset spisode reward and length
            episode_reward = 0
            episode_length = 0

            # play until it is possible
            while not done:
                # synchronize target network with estimation network in required frequence
                if (total_steps % sync_target_net_freq) == 0:
                    self.sync_target_network()

                # calculate epsilon and select greedy action
                epsilon = self.get_epsilon(total_steps, max_epsilon_steps, epsilon_start, epsilon_final)
                action = self.select_action(state, epsilon)

                # execute action in the environment
                next_state, reward, done, _ = self.env.step(action)

                # store transition in replay memory
                self.buffer.add(state, action, reward, done, next_state)

                # sample random minibatch of transitions
                s_batch, a_batch, r_batch, d_batch, next_s_batch = self.buffer.sample(batch_size)

                # predict Q value using the estimation network
                predicted_values = self.predict_q_values(s_batch)

                # estimate Q value using the target network
                q_targets = self.calculate_q_targets(next_s_batch, r_batch, d_batch)

                # update weights in the estimation network
                self.update(predicted_values, q_targets, a_batch)

                # set the state for the next action selction and update counters and reward
                state = next_state
                total_steps += 1
                episode_length += 1
                episode_reward += reward
                self.writer.add_scalar('data/reward', reward, total_steps)
                self.writer.add_scalar('data/epsilon', epsilon, total_steps)

            running_episode_reward = running_episode_reward * 0.9 + 0.1 * episode_reward
            self.writer.add_scalar('data/episode_reward', episode_reward, i)
            self.writer.add_scalar('data/running_episode_reward', running_episode_reward, i)


            if (i % 30) == 0:
                print('global step: {}'.format(total_steps))
                print('episode: {}'.format(i))
                print('running reward: {}'.format(round(running_episode_reward, 2)))
                print('current epsilon: {}'.format(round(epsilon, 2)))
                print('episode_length: {}'.format(episode_length))
                print('episode reward: {}'.format(episode_reward))
                curr_time = time.time()
                print('current time: ' + time.asctime(time.localtime(curr_time)))
                print('running for: ' + str(datetime.timedelta(seconds=curr_time - start_time)))
                print('saving model after {} episodes...'.format(i))
                print('\n')
                self.save_model_during_training(i)

        print('Finish training at: '+ time.asctime(time.localtime(start_time)))
