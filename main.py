
from dataclasses import dataclass
from pathlib import Path

from sberl.environment import InvertedPendulumEnv, RewardingAdjustedAngle, Summaries, Normalize
from sberl.trainer import Trainer
from sberl.agent import AgentPPO
from sberl.model import PolicyModel
from sberl.recorder import Recorder

import numpy as np



@dataclass
class Config:
    project_path: Path = Path(__file__).parent
    db_path: Path = project_path / "data" / "experiments.db"
    num_steps: int = 2048
    gamma: float = 0.99
    alpha: float = 0.95
    num_epochs: int = 2_000
    num_epochs_train: int = 5
    num_mini_batches: int = 8
    lr: float = 1e-4
    eps: float = 1e-5
    models_path: Path = project_path / "models"
    plot_results: bool = True
    plot_frequency: int = 1_000
    figure_path: Path = project_path / "reports" / "excluded" / "pic_report.png"
    save_model_frequency: int = 1_000
    xml_pendulum_config: Path = project_path / "pendulum_xml_config" / "model.xml"
    action_log_std: float = 0.
    hidden_units: int = 64


def get_reward(angle, pos=0., pendulum_momentum=0., car_momentum=0.):
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

    return -1. - abs(pendulum_momentum) / 4.


def main():

    cfg = Config()

    rec = Recorder(cfg, get_reward)
    rec.record()

    env = Normalize(
        Summaries(
            RewardingAdjustedAngle(
                InvertedPendulumEnv(
                    render_mode="rgb_array",
                    model_path=cfg.xml_pendulum_config
                ),
                reward=get_reward
            )
        )
    )

    model = PolicyModel(log_std=cfg.action_log_std, hidden=cfg.hidden_units)
    agent = AgentPPO(model)

    trainer = Trainer(cfg, env, agent)

    trainer.train()


if __name__ == "__main__":
    main()
