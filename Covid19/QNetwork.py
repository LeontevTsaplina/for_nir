import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Class of a model architecture
    """

    def __init__(self, state_size: int, action_size: int, seed: int, fc1_unit: int = 64, fc2_unit: int = 64) -> None:
        """
        Function to initialize object of class

        :param state_size: state size of Model
        :param action_size: action size of Model
        :param seed: torch seed param
        :param fc1_unit: hidden layer units
        :param fc2_unit: hidden layer units
        """

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)

    def forward(self, q: F.relu) -> F.relu:
        """
        Function to forward NN

        :param q: state
        :return: result of forward by NN
        """
        q = F.relu(self.fc1(q))
        q = F.relu(self.fc2(q))
        return self.fc3(q)


