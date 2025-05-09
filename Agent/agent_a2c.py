import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class A2CNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(A2CNetwork, self).__init__()
        self.shared_fc1 = nn.Linear(obs_dim, 128)
        self.shared_fc2 = nn.Linear(128, 128)

        self.policy_head = nn.Linear(128, act_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        return x

    def get_action(self, state):
        x = self.forward(state)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def get_value(self, state):
        x = self.forward(state)
        return self.value_head(x)

    def evaluate_actions(self, states, actions):
        x = self.forward(states)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_head(x).squeeze(-1)
        return log_probs, values, entropy
