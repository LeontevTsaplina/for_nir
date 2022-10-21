import pandas as pd

from Buffer import Buffer


def dqn():
    """
    Function of DQN learning
    """
    pass


if __name__ == '__main__':

    # Input your file_name, imports in Files path
    file_name = ''
    dataset_path = f'Files/{file_name}'

    dataset = pd.read_pickle(dataset_path)

    buffer = Buffer(dataset)

    buffer.load_dataset(dataset)

    # Learning

