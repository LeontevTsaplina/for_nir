import gym
import numpy as np
from Buffer import Buffer


class Environment(gym.Env):
    """
    Class of environment
    """
    def __init__(self, buffer: Buffer) -> None:
        """
        Function to initialize object of class

        :param buffer: buffer
        :return: None
        """

        self.size = len(buffer)
        self.state_dim = buffer.state_dim
        self.action_dim = buffer.action_dim

        self.current_state = [0 for _ in range(buffer.stat_control_dim)] + buffer.dinam_fact_means + buffer.stat_fact_means

        self.action_space = gym.spaces.Discrete(self.action_dim)

        self.observation_space = gym.spaces.Box(low=0, high=self.size, shape=(1,), dtype=np.float32)

        self.buffer = buffer

    def reset(self, curr_episode_index: str) -> np.array:
        """
        Function that returns first observation of an episode

        :param curr_episode_index: index of episode
        :type curr_episode_index: str
        :return: first observation of an episode
        :rtype: np.array
        """

        self.current_state = self.buffer.state[self.buffer.indexes.index((curr_episode_index, "t_0"))]

        return np.array([self.current_state]).astype(np.float32)

    def step(self, curr_episode_index: str, current_episode_t_step: int) -> tuple:
        """
        Function that parameters of observation

        :param curr_episode_index: index of episode
        :param current_episode_t_step: t_step of episode
        :type curr_episode_index: str
        :type current_episode_t_step: int
        :return: parameters of observation
        :rtype: tuple
        """

        next_state = self.buffer.next_state[self.buffer.indexes.index((curr_episode_index, "t_" + str(current_episode_t_step)))]
        reward = self.buffer.reward[self.buffer.indexes.index((curr_episode_index, "t_" + str(current_episode_t_step)))]
        done = self.buffer.done[self.buffer.indexes.index((curr_episode_index, "t_" + str(current_episode_t_step)))]

        return next_state, reward, done


