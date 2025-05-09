import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class A2CNetwork(nn.Module):
    """
    Advantage Actor-Critic (A2C) network with shared layers.
    Contains separate heads for the policy (actor) and value (critic).
    """

    def __init__(self, obs_dim, act_dim):
        super(A2CNetwork, self).__init__()
        # Shared feature extraction layers
        self.shared_fc1 = nn.Linear(obs_dim, 128)
        self.shared_fc2 = nn.Linear(128, 128)

        # Actor head (outputs action logits)
        self.policy_head = nn.Linear(128, act_dim)

        # Critic head (outputs scalar value estimate)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass through shared layers.
        """
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        return x

    def get_action(self, state):
        """
        Samples an action from the policy and returns:
        - the action,
        - its log-probability,
        - the entropy of the policy.
        """
        x = self.forward(state)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def get_value(self, state):
        """
        Returns the state value estimated by the critic.
        """
        x = self.forward(state)
        return self.value_head(x)

    def evaluate_actions(self, states, actions):
        """
        Computes:
        - log-probabilities of the given actions,
        - state values,
        - policy entropy
        for a batch of states and actions.
        Useful during policy gradient updates.
        """
        x = self.forward(states)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_head(x).squeeze(-1)  # Squeeze to shape (batch,)

        return log_probs, values, entropy
