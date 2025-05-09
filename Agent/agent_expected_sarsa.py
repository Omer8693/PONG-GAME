import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    A simple fully connected Q-network for value-based reinforcement learning.
    Takes a state as input and outputs Q-values for each possible action.
    """

    def __init__(self, obs_dim, act_dim):
        super(QNetwork, self).__init__()
        # Two hidden layers with 128 neurons each
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # Output layer produces one Q-value per action
        self.out = nn.Linear(128, act_dim)

    def forward(self, x):
        """
        Standard forward pass through the Q-network.
        Returns Q-values for all actions given a state.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def get_q_values(self, state):
        """
        Returns Q-values for a single state without tracking gradients.
        Useful for action selection (e.g., in epsilon-greedy policy).
        """
        with torch.no_grad():
            q_values = self.forward(state)
        return q_values
