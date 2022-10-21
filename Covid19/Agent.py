import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from QNetwork import QNetwork
from Buffer import Buffer


class Agent:
    """
    Class of an agent
    """
    def __init__(self, dataset: pd.DataFrame, seed: int) -> None:
        """
        Function to initialize object of class

        :param dataset: input dataset
        :param seed: random seed
        :type dataset: pd.Dataframe
        :return: None
        """

        self.size = len(dataset.index)
        self.state_dim = len([column for column in dataset.columns if column.endswith('_stat_control')]) + \
                         len([column for column in dataset.columns if column.endswith('_dinam_fact')]) + \
                         len([column for column in dataset.columns if column.endswith('_stat_fact')])
        self.action_dim = len([column for column in dataset.columns if column.endswith('_dinam_control')])
        self.seed = seed

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.qnetwork_local = QNetwork(self.state_dim, self.action_dim, seed).to(self.device)
        self.qnetwork_target = QNetwork(self.state_dim, self.action_dim, seed).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters())

        self.memory = Buffer(dataset)
        self.t_step = 0

    def action(self, state: np.array) -> np.array:
        """
        Function returns acting for given state

        :param state: input state
        :type state: np.array
        :return: acting for given state
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences: tuple, gamma: float) -> None:
        """
        Function that updates value parameters using given batch of experience tuples

        :param experiences: tuple of (state, action, next_state, reward, done)
        :param gamma: discount factor
        :type experiences: tuple
        :type gamma: float
        :return: None
        """

        states, actions, rewards, next_state, dones = experiences

        criterion = torch.nn.MSELoss()

        self.qnetwork_local.train()

        self.qnetwork_target.eval()

        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model: QNetwork, target_model: QNetwork, tau: float = 0.001) -> None:
        """
        Function that soft update model parameters

        :param local_model: weights will be copied from
        :param target_model: weights will be copied to
        :param tau: interpolation parameter
        :type local_model: QNetwork
        :type target_model: QNetwork
        :type tau: float
        :return: None
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

