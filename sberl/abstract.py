from typing import Protocol, Any, runtime_checkable
from dataclasses import dataclass
from abc import abstractmethod
from pathlib import Path


@runtime_checkable
class Environment(Protocol):
    """
        Base environment protocol
    """

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def step(self, action):
        ...

    @abstractmethod
    def render(self):
        ...

    @abstractmethod
    def close(self):
        ...


class Agent(Protocol):

    """
        Agent protocol.
    """
    model: Any

    @abstractmethod
    def choose_action(self, state, training=True):
        ...


@dataclass
class AbstractConfig(Protocol):
    project_path: Path
    db_path: Path
    num_steps: int
    gamma: float
    alpha: float
    num_epochs: int
    num_epochs_train: int
    num_mini_batches: int
    lr: float
    eps: float
    models_path: Path
    plot_results: bool
    plot_frequency: int
    figure_path: Path
    save_model_frequency: int
