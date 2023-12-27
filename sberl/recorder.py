

from dataclasses import dataclass
from typing import Callable

from sberl.abstract import AbstractConfig

import pandas as pd

import sqlite3
import pickle
import dataclasses
import pathlib

import marshal


@dataclass
class Recorder:
    cfg: AbstractConfig
    rewarding_function: Callable
    query_create_configs_table: str = """
            CREATE TABLE IF NOT EXISTS 
                experiment_configs (
                    id integer primary key AUTOINCREMENT, 
                    project_path varchar(200),
                    num_steps int,
                    gamma real,
                    alpha real,
                    num_epochs integer,
                    num_epochs_train integer,
                    num_mini_batches integer,
                    lr real,
                    eps real,
                    models_path varchar(200),
                    plot_results boolean,
                    plot_frequency integer,
                    figure_path varchar(200),
                    save_model_frequency integer,
                    xml_pendulum_config varchar(200),
                    action_log_std real,
                    hidden_units integer
                );
        """
    query_create_rewards_table = """
            CREATE TABLE IF NOT EXISTS 
                rewarding_functions (
                    id integer primary key AUTOINCREMENT, 
                    rewarding_function blob
                );
        """

    def serialize_rewards(self):
        # pdata = pickle.dumps(self.rewarding_function, pickle.HIGHEST_PROTOCOL)
        # return sqlite3.Binary(pdata)
        m_func = marshal.dumps(self.rewarding_function.__code__)
        return m_func

    def create_record(self):
        record = dataclasses.asdict(self.cfg)
        for k, v in record.items():
            if isinstance(record[k], pathlib.PosixPath):
                record[k] = str(record[k])
        return record

    def record(self):
        with sqlite3.connect(self.cfg.db_path) as connection:
            cursor = connection.cursor()
            record = pd.DataFrame(self.create_record(), index=[0])
            record.to_sql(name="experiment_configs", con=connection, if_exists="append", index=False)
            reward = self.serialize_rewards()
            cursor.execute(self.query_create_rewards_table)
            cursor.execute(
                "insert into rewarding_functions(rewarding_function) values (?)",
                [sqlite3.Binary(reward)]
            )
