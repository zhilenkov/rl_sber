
from pathlib import Path
from dataclasses import dataclass


def get_reward(angle, pos=0., pendulum_momentum=0., car_momentum=0.):
    return 1.


@dataclass
class Config:
    project_path = Path(__file__).parent.parent.parent
    db_path = project_path / "sberl" / "tests" / "testing_data" / "test.db"
    num_steps: int = 2048
    gamma: float = 0.99
    alpha: float = 0.95
    num_epochs: int = 100_000
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
