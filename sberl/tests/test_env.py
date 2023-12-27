from pathlib import Path
from sberl.environment import InvertedPendulumEnv, RewardingAdjustedAngle, Normalize, Summaries
from sberl.abstract import Environment
from sberl.tests.test_mocks import get_reward

import numpy as np


def test_env_creation():
    env_xml_config_path = Path(__file__).parent / "testing_data" / "model.xml"
    env = InvertedPendulumEnv(render_mode="rgb_array", model_path=env_xml_config_path)
    assert isinstance(env, InvertedPendulumEnv)
    assert isinstance(env, Environment), "Env should follow environment protocol"


def test_rewarding_adjusted_env():
    env_xml_config_path = Path(__file__).parent / "testing_data" / "model.xml"
    env = RewardingAdjustedAngle(
        InvertedPendulumEnv(
            render_mode="rgb_array",
            model_path=env_xml_config_path
        ),
        get_reward
    )
    actions = np.random.random(size=(10_000, )) * 6. - 3.
    actions = np.array(actions).reshape(-1, 1)

    state = env.reset()

    for action in actions:
        state, reward, terminated, truncated, info = env.step(action)

        assert 0. <= state[1] <= 2. * np.pi


def test_full_wrapped_env():
    env_xml_config_path = Path(__file__).parent / "testing_data" / "model.xml"
    env = Normalize(
        Summaries(
            RewardingAdjustedAngle(
                InvertedPendulumEnv(
                    render_mode="rgb_array",
                    model_path=env_xml_config_path
                ),
                get_reward
            )
        )
    )

    actions = np.random.random(size=(10_000,)) * 6. - 3.
    actions = np.array(actions).reshape(-1, 1)

    state = env.reset()

    for action in actions:
        state, reward, terminated, truncated, info = env.step(action)

        assert -10. <= state[1] <= 10.
