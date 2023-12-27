
import torch

from torch import nn


class PolicyModel(nn.Module):
    def __init__(self, log_std=0., hidden=64):
        super().__init__()
        self.hidden = hidden

        self.action_log_std = nn.Parameter(torch.ones(1, 1) * log_std)

        self.policy_model = nn.Sequential(
            nn.Linear(4, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 1),
            nn.Tanh()
        )

        self.value_model = nn.Sequential(
            nn.Linear(4, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 1)
        )

    def get_policy(self, x):
        means = self.policy_model(x.float())
        action_log_std = self.action_log_std.expand_as(means)
        var = torch.exp(action_log_std)
        return means, var

    def get_value(self, x):
        out = self.value_model(x.float())
        return out

    def forward(self, x):
        policy = self.get_policy(x)
        value = self.get_value(x)

        return policy, value
