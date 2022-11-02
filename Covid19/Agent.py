import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import random

from QNetwork import QNetwork
from Buffer import Buffer

BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-4  # for soft update of target parameters
LR = 5e-4  # learning rate


class Agent:
    """
    Class of an agent
    """

    def __init__(self, dataset: pd.DataFrame, seed: int = 0) -> None:
        """
        Function to initialize object of class

        :param dataset: input dataset
        :param seed: random seed
        :type dataset: pd.Dataframe
        :type seed: int
        :return: None
        """

        self.size = len(dataset.index)
        self.state_dim = len([column for column in dataset.columns if column.endswith('_stat_control')]) + \
                         len([column for column in dataset.columns if column.endswith('_dinam_fact')]) + \
                         len([column for column in dataset.columns if column.endswith('_stat_fact')])
        self.action_dim = len([column for column in dataset.columns if column.endswith('_dinam_control')])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.qnetwork_local = QNetwork(self.state_dim, self.action_dim, seed).to(self.device)
        self.qnetwork_target = QNetwork(self.state_dim, self.action_dim, seed).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = Buffer(dataset)
        self.t_step = 0

    def step(self, index: str, t_point: str, state: np.array, action: np.array,
             next_step: np.array, reward: float, done: int) -> None:
        """
        Function of agent step

        :param index: index of episode
        :param t_point: t_point of episode
        :param state: state
        :param action: action
        :param next_step: next_step
        :param reward: reward
        :param done: done
        :type index: int
        :type t_point: int
        :type state: np.array
        :type action: np.array
        :type next_step: np.array
        :type reward: float
        :type done: int
        :return: None
        """

        self.memory.add(index, t_point, state, action, next_step, reward, done)

        if len(self.memory) > BATCH_SIZE:
            experience = self.memory.sample()
            self.learn(experience, GAMMA)

    def action(self, state: np.array, eps: float = 0) -> int or np.array:
        """
        Function that returns action for given state

        :param state: state
        :param eps: epsilon
        :type state: np.array
        :type eps: float
        :return: action for given state
        :rtype: int or np.array
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))

    def learn(self, experiences: tuple, gamma: float) -> None:
        """
        Function than update value parameters

        :param experiences: tuple of buffer element params
        :param gamma: discount factor
        :type experiences: tuple
        :type gamma: float
        :return: None
        """

        indexes, states, actions, next_states, rewards, dones = experiences

        criterion = torch.nn.MSELoss()

        self.qnetwork_local.train()

        self.qnetwork_target.eval()

        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model: QNetwork, target_model: QNetwork, tau: float) -> None:
        """
        Function of soft update model parameters

        :param local_model: weights will be copied from
        :param target_model: weights will be copied to
        :param tau: interpolation parameter
        :type local_model: QNetwork
        :type target_model: QNetwork
        :type tau: float
        :return: None
        """

        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
