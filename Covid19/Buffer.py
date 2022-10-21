import numpy as np
import pandas as pd


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
        self.state_dim = len([column for column in dataset.columns if column.endswith('_stat_control')]) + \
                         len([column for column in dataset.columns if column.endswith('_dinam_fact')]) + \
                         len([column for column in dataset.columns if column.endswith('_stat_fact')])
        self.action_dim = len([column for column in dataset.columns if column.endswith('_dinam_control')])
        self.point = 0

        self.state = np.zeros((self.size, self.state_dim))
        self.action = np.zeros((self.size, self.action_dim))
        self.next_state = np.zeros((self.size, self.state_dim))
        self.reward = np.zeros((self.size, 1))
        self.done = np.zeros((self.size, 1))

    def add(self, state: np.array, action: np.array, next_state: np.array, reward: float, done: int) -> None:
        """
        Function to add element of buffer

        :param state: state in current object
        :param action: action in current object
        :param next_state: state in object in time of t+1
        :param reward: reward in current object
        :param done: done element in current object
        :type state: np.array
        :type action: np.array
        :type next_state: np.array
        :type reward: float
        :type done: int
        :return: None
        """

        self.state[self.point] = state
        self.action[self.point] = action
        self.next_state[self.point] = next_state
        self.reward[self.point] = reward
        self.done[self.point] = done

        self.point = (self.point + 1) % self.size

    def load_dataset(self, dataset: pd.DataFrame) -> object:
        """
        Function to load dataset in buffer

        :param dataset: input dataset
        :type dataset: pd.DataFrame
        :return: loaded object of Buffer class
        """

        dataset.fillna(dataset.mean(numeric_only=True), inplace=True)

        for index, row in dataset.iterrows():

            next_step = row if row['end_epizode'] == 1 else \
                dataset[(dataset.index == index) & (dataset.t_point == 't_' + str(int(row.t_point.split('_')[1]) + 1))].iloc(0)[0]

            self.add(np.array([row[column] for column in dataset.columns if column.endswith('_stat_control')] + \
                              [row[column] for column in dataset.columns if column.endswith('_dinam_fact')] + \
                              [row[column] for column in dataset.columns if column.endswith('_stat_fact')]),
                     np.array([row[column] for column in dataset.columns if column.endswith('_dinam_control')]),
                     np.array([next_step[column] for column in dataset.columns if column.endswith('_stat_control')] + \
                              [next_step[column] for column in dataset.columns if column.endswith('_dinam_fact')] + \
                              [next_step[column] for column in dataset.columns if column.endswith('_stat_fact')]),
                     -(next_step.current_process_duration - row.current_process_duration) - next_step.outcome_tar * 100,
                     row['end_epizode'])

        return self
