
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sberl.recorder import Recorder
from sberl.abstract import AbstractConfig

from sberl.tests.test_mocks import get_reward, Config


def test_recorder_create_record_and_record():
    cfg = Config()
    rec = Recorder(cfg, get_reward)

    rec.record()

    if cfg.db_path.exists():
        cfg.db_path.unlink()

    assert True
