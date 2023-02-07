from .queue_system import QueueSystemConfig, BaseQueueSystem, Slrum, Lsf
from .job import JobState
from .artifact import Artifact
from .connector import SshConfig, BaseConnector, SshConnector, LocalConnector

from pydantic import BaseModel
from typing import Optional, Dict, List, TypeVar, Callable
from abc import ABC, abstractmethod
from invoke import Result
import os
import shlex
import base64
import cloudpickle


class BaseExecutorConfig(BaseModel):
    ssh: Optional[SshConfig]
    queue_system: QueueSystemConfig
    work_dir: str
    python_cmd: str = 'python'

FnType = TypeVar('FnType', bound=Callable)

class Executor(ABC):

    name: str

    @abstractmethod
    def mkdir(self, path: str):
        ...

    @abstractmethod
    def run_python_script(self, script: str, python_cmd=None):
        ...

    @abstractmethod
    def run_python_fn(self, fn: FnType, python_cmd=None) -> FnType:
        ...

    @abstractmethod
    def get_full_path(self, path: str) -> str:
        ...

    @abstractmethod
    def dump_text(self, text: str, path: str):
        ...

    @abstractmethod
    def load_text(self, path: str) -> str:
        ...

    @abstractmethod
    def glob(self, pattern: str) -> List[str]:
        ...

    @abstractmethod
    def run(self, script: str, **kwargs) -> Result:
        ...

    @abstractmethod
    def submit(self, script: str, **kwargs) -> JobState:
        ...

    @abstractmethod
    def upload(self, from_artifact: Artifact, to_dir: str) -> Artifact:
        ...

    @abstractmethod
    def download(self, from_artifact: Artifact, to_dir: str) -> Artifact:
        ...

    @abstractmethod
    def sym_link(self, from_artifact: Artifact, to_dir: str) -> Artifact:
        ...

    @abstractmethod
    def unpack_artifact(self, artifact: Artifact) -> List[str]:
        ...


class HpcExecutor(Executor):

    @classmethod
    def from_config(cls, config: BaseExecutorConfig, name: str):
        if config.ssh:
            connector = SshConnector.from_config(config.ssh)
        else:
            connector = LocalConnector()
        queue_system = None
        if config.queue_system.slurm:
            queue_system = Slrum()
            queue_system.config = config.queue_system.slurm
        elif config.queue_system.lsf:
            queue_system = Lsf()
            queue_system.config = config.queue_system.lsf
        if queue_system is None:
            raise ValueError('Queue system config is missing!')
        queue_system.connector = connector
        return cls(connector, queue_system, config.work_dir, config.python_cmd, name)

    @property
    def is_local(self):
        return isinstance(self.connector, LocalConnector)

    def __init__(self, connector: BaseConnector, queue_system: BaseQueueSystem, work_dir: str, python_cmd: str, name: str):
        self.name = name
        self.connector = connector
        self.queue_system = queue_system
        self.work_dir = work_dir
        self.python_cmd = python_cmd

    def get_full_path(self, path: str):
        return os.path.join(self.work_dir, path)

    def mkdir(self, path: str):
        return self.connector.run('mkdir -p {}'.format(shlex.quote(path)))

    def dump_text(self, text: str, path: str):
        return self.connector.dump_text(text, path)

    # TODO: handle error properly
    def load_text(self, path: str) -> str:
        return self.connector.run('cat {}'.format(shlex.quote(path)), hide=True).stdout

    def glob(self, pattern: str):
        return self.connector.glob(pattern)

    def run(self, script: str, **kwargs):
        return self.connector.run(script, **kwargs)

    def run_python_script(self, script: str, python_cmd=None):
        if python_cmd is None:
            python_cmd = self.python_cmd
        return self.connector.run('{} -c {}'.format(python_cmd, shlex.quote(script)))

    def run_python_fn(self, fn: FnType, python_cmd=None) -> FnType:
        def remote_fn(*args, **kwargs):
            script = fn_to_script(lambda: fn(*args, **kwargs), delimiter='@')
            ret = self.run_python_script(script=script, python_cmd=python_cmd)
            _, r = ret.stdout.rsplit('@')
            return cloudpickle.loads(base64.b64decode(r))
        return remote_fn  # type: ignore

    def submit(self, script: str, cwd: str, **kwargs):
        return self.queue_system.submit(script, cwd=cwd, **kwargs)

    def unpack_artifact(self, artifact: Artifact) -> List[str]:
        if artifact.glob is None:
            return [artifact.url]
        pattern = os.path.join(artifact.url, artifact.glob)
        return self.glob(pattern)

    def upload(self, from_artifact: Artifact, to_dir: str) -> Artifact:
        dest_path = self.connector.upload(from_artifact.url, to_dir)
        return Artifact(
            executor=self.name,
            url=dest_path,
            attrs=from_artifact.attrs,
        ) # type: ignore

    def download(self, from_artifact: Artifact, to_dir: str) -> Artifact:
        dest_path = self.connector.download(from_artifact.url, to_dir)
        return Artifact(
            url=dest_path,
            attrs=from_artifact.attrs,
        ) # type: ignore

    def sym_link(self, from_artifact: Artifact, to_dir: str) -> Artifact:
        dest_path = self.connector.sym_link(from_artifact.url, to_dir)
        return Artifact(
            executor=from_artifact.executor,
            url=dest_path,
            attrs=from_artifact.attrs,
        ) # type: ignore


def create_executor(config: BaseExecutorConfig, name: str) -> Executor:
    if config.queue_system is not None:
        return HpcExecutor.from_config(config, name)
    raise RuntimeError('The executor configuration is not supported!')


class ExecutorManager:
    def __init__(self, executor_configs: Dict[str, BaseExecutorConfig]):
        self._executor_configs = executor_configs
        self._executors: Dict[str, Executor] = dict()

    def get_executor(self, name: str):
        config = self._executor_configs.get(name)
        if config is None:
            raise ValueError(
                'Executor with name {} is not found!'.format(name))

        if name not in self._executors:
            executor = create_executor(config, name)
            self._executors[name] = executor

        return self._executors[name]


def fn_to_script(fn: Callable, delimiter='@'):
    dumped_fn = base64.b64encode(cloudpickle.dumps(fn, protocol=cloudpickle.DEFAULT_PROTOCOL))
    script = [
        f'''import base64,cloudpickle as cp''',
        f'''r=cp.loads(base64.b64decode({repr(dumped_fn)}))()''',
        f'''print({repr(delimiter)}+base64.b64encode(cp.dumps(r, protocol=cp.DEFAULT_PROTOCOL)).decode('ascii'))''',
    ]
    return ';'.join(script)
