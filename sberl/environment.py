
"""
    Given MuJoCo environment. Rewards are non-informative. Should be changed.
"""


from typing import Optional, SupportsFloat, Any
from collections import defaultdict
from gymnasium import utils
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from sberl.utils import normalize_angle, update_mean_var_count_from_moments
from sberl.abstract import Agent

import numpy as np
import gymnasium as gym


class InvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, model_path, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        model_path = str(model_path)
        MujocoEnv.__init__(
            self,
            model_path=model_path,
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


class RewardingAdjustedAngle(gym.Wrapper):

    def __init__(self, env, reward):
        super().__init__(env)
        self.reward = reward

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        obs[1] = normalize_angle(obs[1])
        rew = self.reward(
            angle=obs[1],
            pos=obs[0],
            car_momentum=obs[2],
            pendulum_momentum=obs[3]
        )

        return obs, rew, terminated, truncated, info

    def reset(
        self, **kwargs
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs = self.env.reset(**kwargs)
        obs[1] = normalize_angle(obs[1])
        return obs


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


class EnvRunner:
    """ Reinforcement learning runner in an environment with given policy """

    def __init__(self, env, agent: Agent, nsteps, transforms=None, step_var=None):
        self.env = env
        self.agent = agent
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
        self.agent.reset()

    def get_next(self):
        """ Runs the agent in the environment.  """
        trajectory = defaultdict(list, {"actions": []})
        observations = []
        rewards = []
        resets = []
        self.state["env_steps"] = self.nsteps

        for i in range(self.nsteps):
            observations.append(self.state["latest_observation"])
            act = self.agent.choose_action(self.state["latest_observation"])
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
