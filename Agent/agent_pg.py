import torch
import torch.nn as nn
import torch.distributions as distributions

class PolicyNetwork(nn.Module):
    """
    A neural network for policy approximation using both actor and critic heads.
    Used in REINFORCE and compatible with A2C and other policy gradient methods.
    """

    def __init__(self, obs_dim, act_dim):
        super(PolicyNetwork, self).__init__()

        # Shared fully connected layers
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Output layer for policy (action probabilities)
        self.policy_head = nn.Linear(128, act_dim)
        
        # Output layer for state-value function
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        """
        Shared forward pass through the common layers.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def act(self, state):
        """
        Samples an action using the policy distribution.
        Returns both the action and its log probability.
        """
        x = self.forward(state)
        policy_logits = self.policy_head(x)
        dist = distributions.Categorical(logits=policy_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def get_action(self, state):
        """
        Alias for act(), added for compatibility with other modules like A2C or SARSA.
        """
        return self.act(state)

    def value(self, state):
        """
        Returns the scalar value estimate for the given state.
        """
        x = self.forward(state)
        value = self.value_head(x)
        return value.item()

    def dist(self, state):
        """
        Returns the categorical distribution over actions for entropy/loss calculations.
        """
        x = self.forward(state)
        policy_logits = self.policy_head(x)
        return distributions.Categorical(logits=policy_logits)
