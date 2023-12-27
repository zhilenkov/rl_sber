from tqdm import tqdm
from dataclasses import dataclass

import matplotlib.pyplot as plt

import torch

from sberl.abstract import AbstractConfig, Agent
from sberl.ppo import PPO
from sberl.environment import EnvRunner, Summaries, Normalize
from sberl.utils import AsArray, GAE

import numpy as np


class TrajectorySampler:
    """ Samples mini batches from trajectory for a number of epochs. """

    def __init__(self, runner, num_epochs, num_mini_batches, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.transforms = transforms or []
        self.minibatch_count = 0
        self.epoch_count = 0
        self.trajectory = None

    def shuffle_trajectory(self):
        """ Shuffles all elements in trajectory.

        Should be called at the beginning of each epoch.
        """
        trajectory_len = self.trajectory["observations"].shape[0]

        permutation = np.random.permutation(trajectory_len)
        for key, value in self.trajectory.items():
            if key != 'state':
                self.trajectory[key] = value[permutation]

    def get_next(self):
        """ Returns next minibatch.  """
        if not self.trajectory:
            self.trajectory = self.runner.get_next()

        if self.minibatch_count == self.num_mini_batches:
            self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count += 1

        if self.epoch_count == self.num_epochs:
            self.trajectory = self.runner.get_next()

            self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count = 0

        trajectory_len = self.trajectory["observations"].shape[0]

        batch_size = trajectory_len // self.num_mini_batches

        minibatch = {}
        for key, value in self.trajectory.items():
            if key != 'state':
                minibatch[key] = value[self.minibatch_count * batch_size: (self.minibatch_count + 1) * batch_size]

        self.minibatch_count += 1

        for transform in self.transforms:
            transform(minibatch)

        return minibatch


class NormalizeAdvantages:
    """ Normalizes advantages to have zero mean and variance 1. """

    def __call__(self, trajectory):
        adv = trajectory['advantages']
        normal_adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-7)
        return normal_adv


def make_ppo_runner(env, policy, num_runner_steps=1024,
                    gamma=0.99, lambda_=0.95,
                    num_epochs=10, num_mini_batches=32):
    """ Creates runner for PPO algorithm. """
    runner_transforms = [AsArray(),
                         GAE(policy, gamma=gamma, lambda_=lambda_)]
    runner = EnvRunner(env, policy, num_runner_steps,
                       transforms=runner_transforms)

    sampler_transforms = [NormalizeAdvantages()]
    sampler = TrajectorySampler(runner, num_epochs=num_epochs,
                                num_mini_batches=num_mini_batches,
                                transforms=sampler_transforms)
    return sampler


@dataclass
class Trainer:
    cfg: AbstractConfig
    env: Summaries | Normalize
    agent: Agent

    def train(self):
        cfg = self.cfg
        env = self.env

        agent = self.agent

        runner = make_ppo_runner(
            env,
            agent,
            cfg.num_steps,
            cfg.gamma,
            cfg.alpha,
            cfg.num_epochs_train,
            cfg.num_mini_batches,
        )

        optimizer = torch.optim.Adam(agent.model.parameters(), lr=cfg.lr, eps=cfg.eps)
        epochs = cfg.num_epochs

        lr_mult = lambda epoch_num: (1 - (epoch_num / epochs))
        lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_mult)

        ppo = PPO(agent, optimizer)

        for epoch in tqdm(range(epochs)):
            trajectory = runner.get_next()

            if (epoch + 1) % cfg.plot_frequency == 0:
                rewards = np.array(env.episode_rewards)

                if rewards.size > 0 and cfg.plot_results:
                    plt.plot(rewards[:, 0], rewards[:, 1], label="episode rewards", c="cornflowerblue")
                    plt.title("RewardingFunction")
                    plt.xlabel("Total steps")
                    plt.ylabel("RewardingFunction")
                    plt.grid()
                    plt.savefig(cfg.figure_path)
                    # plt.show()

            if (epoch + 1) % cfg.save_model_frequency == 0:
                model_name = f"model_{epoch + 1}.pth"
                torch.save(agent.model.state_dict(), cfg.models_path / model_name)

            ppo.step(trajectory)
            lr_schedule.step()
