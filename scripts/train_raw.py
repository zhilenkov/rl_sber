
from typing import Optional

from pathlib import Path
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from tqdm import tqdm

from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt

import torch

from torch import nn

from torch.distributions.normal import Normal

from collections import defaultdict

import numpy as np


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
        self.environment_name = "InvertedPendulumEnv"

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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        return self.reset_model()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


def normalize_angle(angle):
    return angle % (2 * np.pi)


def get_reward(angle, pos=0., pendulum_momentum=1.):
    angle -= np.pi

    if abs(pendulum_momentum) < 1e-4:
        pendulum_momentum = 1e-4

    if abs(abs(angle) - np.pi) < np.pi / 90 and abs(pendulum_momentum) < 0.5:
        return 1500

    if abs(abs(angle) - np.pi) < np.pi / 30:
        reward = 30. + 2 * abs(angle) ** 3 + 30 / ((abs(pendulum_momentum) / 10) ** 1.5 + 1e-6)
        reward -= abs(pendulum_momentum) * 2 - 5 * abs(pos)
        return min(reward, 1000)

    if abs(angle) > 3 * np.pi / 4.:
        reward = 10. + 3 * abs(angle) ** 2
        reward -= abs(pendulum_momentum) / 2. - 2 * abs(pos)
        return reward

    if abs(angle) > np.pi / 1.8:
        return 3. + abs(angle) * 2 - abs(pos)

    return -1. - abs(pendulum_momentum) / 5.


class Summaries(gym.Wrapper):
    """ Wrapper to write summaries. """

    def __init__(self, env):
        super().__init__(env)
        self.episode_counter = 0
        self.current_step_var = 0

        self.episode_rewards = []
        self.episode_lens = []

        self.current_reward = 0
        self.current_len = 0

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if hasattr(self.env, "environment_name"):  # Experimental
            obs[1] = normalize_angle(obs[1])
            rew = get_reward(obs[1], obs[0], obs[3]) / 100

        self.current_reward += rew
        self.current_len += 1
        self.current_step_var += 1

        self.episode_rewards.append((self.current_step_var, self.current_reward))

        if terminated or truncated:
            self.episode_lens.append((self.current_step_var, self.current_len))

        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        self.episode_counter += 1

        self.current_reward = 0
        self.current_len = 0

        obs = self.env.reset(**kwargs)

        if hasattr(self.env, "environment_name"):
            obs[1] = normalize_angle(obs[1])  # Experimental

        return obs


class RunningMeanVar:
    """ Computes running mean and variance.

    Args:
      eps (float): a small constant used to initialize mean to zero and
        variance to 1.
      shape tuple(int): shape of the statistics.
    """

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    def __init__(self, eps=1e-4, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = eps

    def update(self, batch):
        """ Updates the running statistics given a batch of samples. """
        if not batch.shape[1:] == self.mean.shape:
            raise ValueError(f"batch has invalid shape: {batch.shape}, "
                             f"expected shape {(None,) + self.mean.shape}")
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """ Updates the running statistics given their new values on new data. """
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count,
                                       batch_mean, batch_var, batch_count):
    """ Updates running mean statistics given a new batch. """
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    new_var = (
            var * (count / tot_count)
            + batch_var * (batch_count / tot_count)
            + np.square(delta) * (count * batch_count / tot_count ** 2))
    new_count = tot_count

    return new_mean, new_var, new_count


class Normalize(gym.Wrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    # pylint: disable=too-many-arguments

    def __init__(self, env, obs=True, ret=True,
                 clipobs=10., cliprew=10., gamma=0.99, eps=1e-8):
        super().__init__(env)
        self.obs_rmv = (RunningMeanVar(shape=self.observation_space.shape)
                        if obs else None)
        self.ret_rmv = RunningMeanVar(shape=()) if ret else None
        self.clipob = clipobs
        self.cliprew = cliprew
        self.ret = np.zeros(getattr(self.env.unwrapped, "nenvs", 1))
        self.gamma = gamma
        self.eps = eps

    def observation(self, obs):
        """ Preprocesses a given observation. """
        if not self.obs_rmv:
            return obs
        rmv_batch = (np.expand_dims(obs, 0)
                     if not hasattr(self.env.unwrapped, "nenvs")
                     else obs)
        self.obs_rmv.update(rmv_batch)
        obs = (obs - self.obs_rmv.mean) / np.sqrt(self.obs_rmv.var + self.eps)
        obs = np.clip(obs, -self.clipob, self.clipob)
        return obs

    def step(self, action):
        obs, rews, terminated, truncated, info = self.env.step(action)
        self.ret = self.ret * self.gamma + rews
        obs = self.observation(obs)
        if self.ret_rmv:
            self.ret_rmv.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rmv.var + self.eps),
                           -self.cliprew, self.cliprew)
        self.ret[terminated] = 0.
        return obs, rews, terminated, truncated, info

    def reset(self, **kwargs):
        self.ret = np.zeros(getattr(self.env.unwrapped, "nenvs", 1))
        obs = self.env.reset(**kwargs)
        return self.observation(obs)


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
        action_logstd = self.action_log_std.expand_as(means)
        var = torch.exp(action_logstd)
        return means, var

    def get_value(self, x):
        out = self.value_model(x.float())
        return out

    def forward(self, x):
        policy = self.get_policy(x)
        value = self.get_value(x)

        return policy, value


class Policy:
    def __init__(self, model):
        self.model = model

    def act(self, inputs, training=False):
        inputs = torch.tensor(inputs)
        if inputs.ndim < 2:
            inputs = inputs.unsqueeze(0)

        batch_size = inputs.shape[0]

        means, covar_matrix = self.model.get_policy(inputs)
        normal_distr = Normal(means, covar_matrix)

        actions = torch.clamp(normal_distr.sample(), -3., 3.)
        log_probs = normal_distr.log_prob(actions)

        values = self.model.get_value(inputs)

        if not training:
            return {'actions': actions.cpu().numpy().tolist()[0],
                    'log_probs': log_probs[0].detach().cpu().numpy(),
                    'values': values[0].detach().cpu().numpy()}
        else:
            return {'distribution': normal_distr, 'values': values}


class AsArray:
    """
    Converts lists of interactions to ndarray.
    """

    def __call__(self, trajectory):
        # Modify trajectory inplace.
        for k, v in filter(lambda kv: kv[0] != "state", trajectory.items()):
            trajectory[k] = np.asarray(v)


class EnvRunner:
    """ Reinforcement learning runner in an environment with given policy """

    def __init__(self, env, policy, nsteps, transforms=None, step_var=None):
        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.transforms = transforms or []
        self.step_var = step_var if step_var is not None else 0
        self.state = {"latest_observation": self.env.reset()}

    @property
    def nenvs(self):
        """ Returns number of batched envs or `None` if env is not batched """
        return getattr(self.env.unwrapped, "nenvs", None)

    def reset(self, **kwargs):
        """ Resets env and runner states. """
        self.state["latest_observation"] = self.env.reset(**kwargs)
        self.policy.reset()

    def get_next(self):
        """ Runs the agent in the environment.  """
        trajectory = defaultdict(list, {"actions": []})
        observations = []
        rewards = []
        resets = []
        self.state["env_steps"] = self.nsteps

        for i in range(self.nsteps):
            observations.append(self.state["latest_observation"])
            act = self.policy.choose_action(self.state["latest_observation"])
            if "actions" not in act:
                raise ValueError("result of policy.act must contain 'actions' "
                                 f"but has keys {list(act.keys())}")
            for key, val in act.items():
                trajectory[key].append(val)

            obs, rew, terminated, truncated, _ = self.env.step(trajectory["actions"][-1])
            done = np.logical_or(terminated, truncated)
            if i == self.nsteps - 1:
                done = True
            self.state["latest_observation"] = obs
            rewards.append(rew)
            resets.append(done)
            self.step_var += self.nenvs or 1

            # Only reset if the env is not batched. Batched envs should
            # auto-reset.
            if not self.nenvs and np.all(done):
                self.state["env_steps"] = i + 1
                self.state["latest_observation"] = self.env.reset()

        trajectory.update(
            observations=observations,
            rewards=rewards,
            resets=resets)
        trajectory["state"] = self.state

        for transform in self.transforms:
            transform(trajectory)
        return trajectory


class DummyPolicy:
    def act(self, inputs, training=False):
        assert not training
        return {"actions": Box(-3., 3., (1,)).sample(), "values": np.nan}


def gae(rewards, values, episode_ends, gamma, lam, policy_val=.241):
    """Compute generalized advantage estimate.
        rewards: a list of rewards at each step.
        values: the value estimate of the state at each step.
        episode_ends: an array of the same shape as rewards, with a 1 if the
            episode ended at that step and a 0 otherwise.
        gamma: the discount factor.
        lam: the GAE lambda parameter.
    """
    # Invert episode_ends to have 0 if the episode ended and 1 otherwise
    episode_ends = (episode_ends * -1) + 1

    T = rewards.shape[0]
    gae_step = 0.
    advantages = np.zeros(shape=rewards.shape)

    prev_value = policy_val.detach().numpy()

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * prev_value * episode_ends[t] - values[t]
        gae_step = delta + gamma * lam * episode_ends[t] * gae_step
        advantages[t] = gae_step
        prev_value = values[t]

    return advantages


class GAE:
    """ Generalized Advantage Estimator. """

    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, trajectory):
        gamma = self.gamma
        lam = self.lambda_

        rewards = trajectory["rewards"]
        values = trajectory["values"]
        resets = trajectory["resets"]

        latest_state = torch.tensor(trajectory["state"]["latest_observation"])

        policy_val = self.policy.model.get_value(latest_state)

        advantages = gae(rewards, values, resets, gamma, lam, policy_val)

        trajectory["advantages"] = advantages
        trajectory["value_targets"] = advantages.reshape(-1, 1) + trajectory["values"]


class TrajectorySampler:
    """ Samples minibatches from trajectory for a number of epochs. """

    def __init__(self, runner, num_epochs, num_minibatches, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
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

        if self.minibatch_count == self.num_minibatches:
            self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count += 1

        if self.epoch_count == self.num_epochs:
            self.trajectory = self.runner.get_next()

            self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count = 0

        trajectory_len = self.trajectory["observations"].shape[0]

        batch_size = trajectory_len // self.num_minibatches

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
                    num_epochs=10, num_minibatches=32):
    """ Creates runner for PPO algorithm. """
    runner_transforms = [AsArray(),
                         GAE(policy, gamma=gamma, lambda_=lambda_)]
    runner = EnvRunner(env, policy, num_runner_steps,
                       transforms=runner_transforms)

    sampler_transforms = [NormalizeAdvantages()]
    sampler = TrajectorySampler(runner, num_epochs=num_epochs,
                                num_minibatches=num_minibatches,
                                transforms=sampler_transforms)
    return sampler


class PPO:
    def __init__(self, policy, optimizer,
                 cliprange=0.2,
                 value_loss_coef=0.45,
                 max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        # Note that we don't need entropy regularization for this env.
        self.max_grad_norm = max_grad_norm

    def policy_loss(self, trajectory, act):
        """ Computes and returns policy loss on a given trajectory. """
        fixed_log_probs = torch.tensor(trajectory["log_probs"])
        advantages = torch.tensor(trajectory["advantages"])
        obs = torch.tensor(trajectory["observations"])
        rds = torch.tensor(trajectory["rewards"])
        dns = torch.tensor(trajectory["resets"])

        dist, values = self.policy.choose_action(obs, training=True).values()

        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange) * advantages
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
        act = self.policy.choose_action(trajectory["observations"], training=True)
        policy_loss = self.policy_loss(trajectory, act)
        value_loss = self.value_loss(trajectory, act)

        return policy_loss + self.value_loss_coef * value_loss

    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step. """
        self.optimizer.zero_grad()
        loss = self.loss(trajectory)

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)

        self.optimizer.step()


@dataclass
class Config:
    actions_std: float = 0.
    hidden = 64
    epochs: int = 50 * 10 ** 5
    num_runner_steps: int = 8192
    gamma: float = 0.99
    lambda_: float = 0.95
    num_epochs: int = 20
    num_mini_batches: int = 512


def main():
    cfg = Config()
    env = Normalize(Summaries(InvertedPendulumEnv(render_mode="rgb_array")))
    # env.reset()

    print("observation space: ", env.observation_space,
          "\nobservations:", env.reset())
    print("action space: ", env.action_space,
          "\naction_sample: ", env.action_space.sample())

    runner = EnvRunner(env, DummyPolicy(), 3,
                       transforms=[AsArray()])
    trajectory = runner.get_next()

    tmp = {k: v.shape for k, v in trajectory.items() if k != "state"}

    model = PolicyModel(cfg.actions_std, hidden=cfg.hidden)

    policy = Policy(model)

    runner = make_ppo_runner(
        env,
        policy,
        cfg.num_runner_steps,
        cfg.gamma,
        cfg.lambda_,
        cfg.num_epochs,
        cfg.num_mini_batches,
    )

    optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-3, eps=1e-5)
    epochs = cfg.epochs

    lr_mult = lambda epoch: (1 - (epoch / epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_mult)

    ppo = PPO(policy, optimizer)

    for epoch in tqdm(range(epochs)):
        trajectory = runner.get_next()

        if (epoch + 1) % 10000 == 0:
            rewards = np.array(env.episode_rewards)
            # print(trajectory["actions"][:10].T, lr_mult(epoch))

            if rewards.size > 0:
                plt.plot(rewards[:, 0], rewards[:, 1], label="episode rewards", c="cornflowerblue")
                plt.title("RewardingFunction")
                plt.xlabel("Total steps")
                plt.ylabel("RewardingFunction")
                plt.grid()
                plt.savefig("tmp.png")
                # plt.show()

        if (epoch + 1) % 50_001 == 0:
            torch.save(model.state_dict(), f"practical_rl_models/model_{epoch + 1}.pth")

        ppo.step(trajectory)
        sched.step()

    torch.save(model.state_dict(), "practical_rl_models/model.pth")


if __name__ == "__main__":
    main()
