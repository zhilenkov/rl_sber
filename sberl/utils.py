
import numpy as np
import torch

from gymnasium.spaces import Box


class AsArray:
    """
    Converts lists of interactions to ndarray.
    """

    def __call__(self, trajectory):
        # Modify trajectory inplace.
        for k, v in filter(lambda kv: kv[0] != "state", trajectory.items()):
            trajectory[k] = np.asarray(v)


class DummyPolicy:
    @staticmethod
    def choose_action(inputs, training=False):
        assert not training
        return {"actions": Box(-3., 3., (1,)).sample(), "values": np.nan}


def normalize_angle(angle):
    return angle % (2 * np.pi)


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

