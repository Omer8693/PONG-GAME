import torch
import torch.nn as nn
import torch.distributions as distributions

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNetwork, self).__init__()
        # Ortak katmanlar
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Politika (aksiyon olasılıkları) için çıkış katmanı
        self.policy_head = nn.Linear(128, act_dim)
        
        # Değer fonksiyonu için çıkış katmanı
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def act(self, state):
        x = self.forward(state)
        policy_logits = self.policy_head(x)
        dist = distributions.Categorical(logits=policy_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def get_action(self, state):
        # A2C ve SARSA ile ortak kullanım için alias fonksiyon
        return self.act(state)

    def value(self, state):
        x = self.forward(state)
        value = self.value_head(x)
        return value.item()

    def dist(self, state):
        x = self.forward(state)
        policy_logits = self.policy_head(x)
        return distributions.Categorical(logits=policy_logits)
