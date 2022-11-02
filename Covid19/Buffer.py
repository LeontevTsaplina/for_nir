import numpy as np
import pandas as pd
import torch
from collections import deque


class Buffer:
    """
    Class to load dataset in buffer
    """

    def __init__(self, dataset: pd.DataFrame) -> None:
        """
        Function to initialize object of class

        :param dataset: input dataset
        :type dataset: pd.DataFrame
        :return: None
        """

        self.size = len(dataset.index)
        self.batch_size = 64
        self.stat_control_dim = len([column for column in dataset.columns if column.endswith('_stat_control')])
        self.dinam_fact_dim = len([column for column in dataset.columns if column.endswith('_dinam_fact')])
        self.stat_fact_dim = len([column for column in dataset.columns if column.endswith('_stat_fact')])
        self.state_dim = self.stat_fact_dim + self.dinam_fact_dim + self.stat_control_dim
        self.action_dim = len([column for column in dataset.columns if column.endswith('_dinam_control')])

        dataset.fillna(dataset.mean(numeric_only=True), inplace=True)
        self.df = dataset

        self.dinam_fact_means = dataset[
            [column for column in dataset.columns if column.endswith('_dinam_fact')]].mean().to_list()
        self.stat_fact_means = dataset[
            [column for column in dataset.columns if column.endswith('_stat_fact')]].mean().to_list()

        self.indexes = [("", "") for _ in range(self.size)]
        self.state = np.zeros((self.size, self.state_dim))
        self.action = np.zeros((self.size, self.action_dim))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.size, 1))
        self.done = np.zeros((self.size, 1))

        self.point = 0
        self.current_size = 0

        self.memory = deque(maxlen=self.size)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add(self, index: str, t_point: str, state: np.array, action: np.array,
            next_state: np.array, reward: float, done: int) -> None:
        """
        Function to add element of buffer

        :param index: index of episode
        :param t_point: t_point of episode
        :param state: state in current object
        :param action: action in current object
        :param next_state: state in object in time of t+1
        :param reward: reward in current object
        :param done: done element in current object
        :type index: str
        :type t_point: str
        :type state: np.array
        :type action: np.array
        :type next_state: np.array
        :type reward: float
        :type done: int
        :return: None
        """

        self.indexes[self.point] = (index, t_point)
        self.state[self.point] = state
        self.action[self.point] = action
        self.next_state[self.point] = next_state
        self.reward[self.point] = reward
        self.done[self.point] = done

        self.point = (self.point + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

        self.memory.append((state, action, next_state, reward, done))

    def load_dataset(self) -> object:
        """
        Function to load dataset in buffer

        :return: loaded object of Buffer class
        :rtype: object
        """

        for index, row in self.df.iterrows():
            next_step = row if row['end_epizode'] == 1 else \
                self.df[(self.df.index == index) & (
                        self.df.t_point == 't_' + str(int(row.t_point.split('_')[1]) + 1))].iloc(0)[0]

            self.add(index, row.t_point, np.array([row[column] for column in self.df.columns if column.endswith('_stat_control')] + \
                                     [row[column] for column in self.df.columns if column.endswith('_dinam_fact')] + \
                                     [row[column] for column in self.df.columns if column.endswith('_stat_fact')]),
                     np.array([row[column] for column in self.df.columns if column.endswith('_dinam_control')]),
                     np.array([next_step[column] for column in self.df.columns if column.endswith('_stat_control')] + \
                              [next_step[column] for column in self.df.columns if column.endswith('_dinam_fact')] + \
                              [next_step[column] for column in self.df.columns if column.endswith('_stat_fact')]),
                     100 + (row.current_process_duration - row.long_observation_tar) - next_step.outcome_tar * 100,
                     row['end_epizode'])

        return self

    def sample(self) -> tuple:
        """
        Function that returns random sample a batch of experiences

        :return: random sample a batch of experiences
        :rtype: tuple
        """

        ind = np.random.choice(range(self.current_size), size=self.batch_size, replace=False)

        return (
            np.array(self.indexes)[ind],
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

    def __len__(self) -> int:
        """
        Function that returns the current size of internal memory

        :return: the current size of internal memory
        :rtype: int
        """

        return len(self.memory)
