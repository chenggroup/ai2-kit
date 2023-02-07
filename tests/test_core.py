from pydantic import BaseModel
from ai2_kit.core.executor import BaseExecutorConfig, ExecutorManager, Slrum, Lsf, SshConnector, HpcExecutor
from ai2_kit.core.util import load_yaml_file
from typing import Dict
from unittest import TestCase
from pathlib import Path


class SimpleConfig(BaseModel):
    executors: Dict[str, BaseExecutorConfig]


class TestConfig(TestCase):

    def setUp(self):
        self.config_file = Path(__file__).parent / 'all-in-one.yaml'

    def test_load_config(self):
        data = load_yaml_file(self.config_file)
        config = SimpleConfig.parse_obj(data)

        self.assertIsNone(config.executors['slurm-hpc'].queue_system.lsf)
        self.assertIsNotNone(config.executors['slurm-hpc'].queue_system.slurm)

        self.assertIsNone(config.executors['lsf-hpc'].queue_system.slurm)
        self.assertIsNotNone(config.executors['lsf-hpc'].queue_system.lsf)
        self.assertIsNotNone(config.executors['lsf-hpc'].ssh.gateway)


class TestExecutor(TestCase):

    def setUp(self):
        self.config_file = Path(__file__).parent / 'all-in-one.yaml'

    def test_executor_manager(self):
        data = load_yaml_file(self.config_file)
        config = SimpleConfig.parse_obj(data)

        executor_manager = ExecutorManager(config.executors)

        executor: HpcExecutor = executor_manager.get_executor('slurm-hpc')  # type: ignore
        self.assertIsInstance(executor, HpcExecutor)
        self.assertIsInstance(executor.queue_system, Slrum)
        self.assertIsInstance(executor.connector, SshConnector)

        executor: HpcExecutor = executor_manager.get_executor('lsf-hpc')  # type: ignore
        self.assertIsInstance(executor, HpcExecutor)
        self.assertIsInstance(executor.queue_system, Lsf)
        self.assertIsInstance(executor.connector, SshConnector)
