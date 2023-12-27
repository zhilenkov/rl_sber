
from dataclasses import dataclass
from sberl.abstract import Environment, Agent, AbstractConfig


@dataclass
class InvertedPendulumMDP:
    cfg: AbstractConfig
    env: Environment
    agent: Agent

    def run(self, save=False):

        s = self.env.reset()

        for _ in range(self.cfg.num_steps):
            action = self.agent.choose_action(s)
            s, r, d, t, info = self.env.step(action)

