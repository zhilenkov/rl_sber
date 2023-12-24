from typing import Optional
from pathlib import Path
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from torch import nn
from torch.distributions.normal import Normal

import gymnasium as gym
import numpy as np
import torch


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


class PolicyModel(nn. Module):
    def __init__(self, log_std=0., hidden=64):
        super().__init__()
        self.hidden = hidden

        # self.log_std = -0.5 * np.ones(1, dtype=np.float32)
        # print(self.log_std)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(self.log_std))
        # print(self.log_std)

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
        # var = torch.exp(self.log_std)
        # print(var)
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
        # inputs = inputs.cuda()

        batch_size = inputs.shape[0]

        means, covar_matrix = self.model.get_policy(inputs)
        # print(covar_matrix.shape)
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


def main():
    env = Normalize(Summaries(InvertedPendulumEnv(render_mode="human")))
    obs = env.reset()

    model = PolicyModel(hidden=64)
    model.load_state_dict(torch.load(str(Path(__file__).parent / "practical_rl_models" / "model_1900000.pth")))
    model.eval()

    policy = Policy(model)

    rewards = []
    states = []

    for i in range(2000):
        action = policy.act(obs)["actions"]
        obs, r, d, _, _ = env.step(action)
        rewards.append(r)
        states.append(obs)


if __name__ == "__main__":
    main()
