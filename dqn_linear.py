import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple


class DQN(nn.Module):

    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, lr):
        """
        Initializes Deep Q-Network.
        :param input_dims: the dimensions of the input
        :param fc1_dims: the dimensions of output of first layer
        :param fc2_dims: the dimensions of the output of the second layer
        :param n_actions: number of actions in action space
        :param lr: learning rate
        """
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Forward for Q-Network, called with one element to determine next action or with a batch
        for optimization.
        :param state: the state
        :return: a tensor representing the output of the last layer (raw, no activation)
        """
        self.eval()
        # x = F.relu(self.bn1(self.fc1(state)))
        # x = F.relu(self.bn2(self.fc2(x)))
        # TODO: Why is it slower training-wise without batch normalization?
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size,
                 n_actions, max_mem_size=100000, eps_end=.01, eps_dec=5e-4):
        """
        Initialization of the Agent.
        :param gamma: discount factor
        :param epsilon: initial probability to explore rather than exploit
        :param lr: the learning rate
        :param input_dims: the dimensions of the inputs
        :param batch_size: size of each batch
        :param n_actions: number of actions in action space
        :param max_mem_size: the maximum size of memory to store
        :param eps_end: the ending probability to explore
        :param eps_dec: linear decrement of epsilon
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DQN(input_dims=input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions, lr=self.lr)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        """
        Store a transition in memory.
        :param state: the current state
        :param action: action taken
        :param reward: reward gained
        :param state_: the next state
        :param done: flag indicating whether terminal state or not
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        """
        Choosing an action given an observation.
        :param observation: the current observation of the state
        :return: greedy action with probability of epsilon, random action otherwise
        """
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).float().to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        """
        Method for agent to learn.
        """
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)

        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min







