"""
    Raw python script outside the package to run environment with any policy.
"""


import time

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

# import gymnasium as gym

import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class PathConfig:
    def __init__(self):

        self.project = Path(__file__).parent.parent
        self.docs = self.project / "docs"
        self.reports = self.project / "reports"
        self.reports_included = self.reports / "included"
        self.reports_excluded = self.reports / "excluded"
        self.models = self.project / "models"
        self.package = self.project / "sberl"


class InvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            model_path=str(Path(__file__).parent / "configuration" / "model.xml"),
            frame_skip=2,
            observation_space=observation_space,
            **kwargs
        )
        self.last_ob = None

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all())

        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {}

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        qpos[1] = 3.14  # Set the pole to be facing down
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    @staticmethod
    def get_reward(angle, pos: float | None = None):

        angle -= np.pi

        def hinge(v: float):
            if abs(v) < 1:
                return v
            return v ** 2

        def simple(v: float):
            if abs(v) - np.pi < 0.2:
                return 5
            if abs(v) - np.pi < 1:
                return 1
            return 0

        if pos is None:
            # reward = -hinge(abs(angle) - np.pi)
            # tmp = -hinge(abs(np.pi) - np.pi)
            # return -hinge(abs(angle) - np.pi)
            return simple(angle)

        # return -hinge(abs(angle) - np.pi) - abs(pos)
        return simple(angle) - abs(pos)


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, hidden, device):
        super(ActorCritic, self).__init__()
        self.hidden = hidden
        self.device = device

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, self.hidden),
                            nn.Tanh(),
                            nn.Linear(self.hidden, self.hidden),
                            nn.Tanh(),
                            nn.Linear(self.hidden, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, self.hidden),
                            nn.Tanh(),
                            nn.Linear(self.hidden, self.hidden),
                            nn.Tanh(),
                            nn.Linear(self.hidden, action_dim),
                            nn.Softmax(dim=-1)
                        )

            # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 1)
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var) #.unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = torch.clamp(dist.sample(), min=-3., max=3.)
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, hidden, device, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space

        self.device = device

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init, hidden, device
        ).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init, hidden, device
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):

        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            # Copy new weights into old policy
            self.policy_old.load_state_dict(self.policy.state_dict())

            # clear buffer
            self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def play(random=True, num_steps=500, human=True):
    device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"
    env_name = "InvertedPendulum"
    has_continuous_action_space = True
    max_ep_len = 1000
    action_std = 1e-5

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    # env_name = "RoboschoolWalker2d-v1"
    # has_continuous_action_space = True
    # max_ep_len = 1000           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    total_test_episodes = 10  # total num of testing episodes

    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic
    num_hidden_units = 64

    #####################################################

    env = InvertedPendulumEnv(render_mode="human")

    # state space dimension
    state_dim = env.observation_space.shape[0]

    action_dim = 1

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    num_hidden_units, device, action_std)

    random_seed = 0
    run_num_pretrained = 19

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("------------------")

    state = env.reset_model()
    # policy = Box(-3., 3., shape=(1,), dtype=np.float32)
    rewards = []
    for _ in range(num_steps):
        # if human:
        #     action = np.array([float(input())])
        # else:
        #     action = policy.sample()
        action = ppo_agent.select_action(state)
        state, reward, done, _, _ = env.step(action)
        pass

    pass


def main():
    play(num_steps=500, human=False)


if __name__ == "__main__":
    main()
