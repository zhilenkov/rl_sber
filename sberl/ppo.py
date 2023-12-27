
from torch import nn
from torch.optim import Optimizer

from sberl.abstract import Agent

import torch


class PPO:
    def __init__(self,
                 agent: Agent,
                 optimizer: Optimizer,
                 clip_range: float = 0.2,
                 value_loss_weight: float = 0.45,
                 max_grad_norm: float = 0.5):
        self.agent = agent
        self.optimizer = optimizer
        self.clip_range = clip_range
        self.value_loss_weight = value_loss_weight
        # Note that we don't need entropy regularization for this env.
        self.max_grad_norm = max_grad_norm

    def policy_loss(self, trajectory, act):
        """ Computes and returns policy loss on a given trajectory. """
        fixed_log_probs = torch.tensor(trajectory["log_probs"])
        advantages = torch.tensor(trajectory["advantages"])
        obs = torch.tensor(trajectory["observations"])
        rds = torch.tensor(trajectory["rewards"])
        dns = torch.tensor(trajectory["resets"])

        dist, values = self.agent.choose_action(obs, training=True).values()

        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()
        return policy_surr

    def value_loss(self, trajectory, act, optim_value_iternum=1):
        """ Computes and returns value loss on a given trajectory. """
        for _ in range(optim_value_iternum):
            values_pred = torch.tensor(trajectory["values"])
            returns = torch.tensor(trajectory["value_targets"])
            value_loss = (values_pred - returns).pow(2).mean()
        return value_loss

    def loss(self, trajectory):
        act = self.agent.choose_action(trajectory["observations"], training=True)
        policy_loss = self.policy_loss(trajectory, act)
        value_loss = self.value_loss(trajectory, act)

        return policy_loss + self.value_loss_weight * value_loss

    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step. """
        self.optimizer.zero_grad()
        loss = self.loss(trajectory)

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
