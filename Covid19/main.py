import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from Buffer import Buffer
from Environment import Environment
from Agent import Agent


# Input your file_name, imports in Files path
file_name = ''
dataset_path = f'Files/{file_name}'

dataset = pd.read_pickle(dataset_path)

episodes = list(set(dataset.index))

buffer = Buffer(dataset)
buffer.load_dataset()

agent = Agent(dataset)
env = Environment(buffer)


def dqn(max_t: int = 1000, eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.996) -> tuple:
    """
    Function of Deep Q-Learning

    :param max_t: maximum number of timesteps per episode
    :param eps_start: starting value of epsilon, for epsilon-greedy action selection
    :param eps_end: minimum value of epsilon
    :param eps_decay: mutiplicative factor (per episode) for decreasing epsilon
    :type max_t: int
    :type eps_start: float
    :type eps_end: float
    :type eps_decay: float
    :return: tuple of scores
    :rtype: tuple
    """

    scores = []
    scores_mean = []
    best_avg = 0
    scores_window = deque(maxlen=max_t)
    eps = eps_start
    for i_episode in range(len(episodes)):
        curr_episode_index = episodes[i_episode]
        state = env.reset(curr_episode_index)
        score = 0
        for t in range(max_t):
            action = agent.action(state, eps)
            next_state, reward, done = env.step(curr_episode_index, t)
            agent.step(curr_episode_index, "t_" + str(t), state, action, next_state, reward, done)

            state = next_state
            score += reward
            if done:
                break
            scores_window.append(score)
            scores.append(score)
            scores_mean.append(np.mean(scores_window))
            eps = max(eps * eps_decay, eps_end)
            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode + 1, np.mean(scores_window)), end="")

            if np.mean(scores_window) >= 320.0 and best_avg < np.mean(scores_window):
                best_avg = np.mean(scores_window)
                print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode + 1,
                                                                                            np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break

    return scores, scores_mean


if __name__ == '__main__':

    scores, scores_mean = dqn()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, linewidth=2, label='scores')
    plt.plot(np.arange(len(scores)), scores_mean, linewidth=4, label='mean scores')
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Epsiode #')
    plt.show()

    agent_test = Agent(dataset)
    agent_test.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    state = buffer.state[2]
    action = agent_test.action(state)

    print("\n\n\n\n")
    print(f"Treatment is: {[column for column in dataset.columns if column.endswith('_dinam_control')][action]}")
