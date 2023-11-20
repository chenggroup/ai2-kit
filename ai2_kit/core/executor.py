from .queue_system import QueueSystemConfig, BaseQueueSystem, Slurm, Lsf, PBS
from .job import JobFuture
from .artifact import Artifact
from .connector import SshConfig, BaseConnector, SshConnector, LocalConnector
from .util import s_uuid
from .log import get_logger

logger = get_logger(__name__)


from pydantic import BaseModel
from typing import Optional, Dict, List, TypeVar, Callable, Mapping, Union
from abc import ABC, abstractmethod
from invoke import Result
import os
import shlex
import base64
import bz2
import cloudpickle


class BaseExecutorConfig(BaseModel):
    ssh: Optional[SshConfig]
    queue_system: QueueSystemConfig
    work_dir: str
    python_cmd: str = 'python'

ExecutorMap = Mapping[str, BaseExecutorConfig]

FnType = TypeVar('FnType', bound=Callable)

class Executor(ABC):

    name: str
    work_dir: str
    tmp_dir: str
    python_cmd: str

    def init(self):
        ...

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
    def submit(self, script: str, **kwargs) -> JobFuture:
        ...

    @abstractmethod
    def upload(self, from_artifact: Artifact, to_dir: str) -> Artifact:
        ...

    @abstractmethod
    def download(self, from_artifact: Artifact, to_dir: str) -> Artifact:
        ...

    @abstractmethod
    def resolve_artifact(self, artifact: Artifact) -> List[str]:
        ...

    def setup_workspace(self, workspace_dir: str, dirs: List[str]):
        paths = [os.path.join(workspace_dir, dir) for dir in dirs]
        for path in paths :
            self.mkdir(path)
            logger.info('create path: %s', path)
        return paths

class HpcExecutor(Executor):

    @classmethod
    def from_config(cls, config: Union[dict, BaseExecutorConfig], name: str = ''):
        if isinstance(config, dict):
            config = BaseExecutorConfig.parse_obj(config)
        if config.ssh:
            connector = SshConnector.from_config(config.ssh)
        else:
            connector = LocalConnector()
        queue_system = None
        if config.queue_system.slurm:
            queue_system = Slurm()
            queue_system.config = config.queue_system.slurm
        elif config.queue_system.lsf:
            queue_system = Lsf()
            queue_system.config = config.queue_system.lsf
        elif config.queue_system.pbs:
            queue_system = PBS()
            queue_system.config = config.queue_system.pbs
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
        self.tmp_dir = os.path.join(self.work_dir, '.tmp')  # TODO: make it configurable

    def init(self):
        # if work_dir is relative path, it will be relative to user home
        if not os.path.isabs(self.work_dir):
            user_home = self.run('echo $HOME', hide=True).stdout.strip()
            self.work_dir = os.path.normpath(os.path.join(user_home, self.work_dir))

        self.mkdir(self.work_dir)
        self.mkdir(self.tmp_dir)

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

    def run_python_script(self, script: str, python_cmd=None, cwd=None):
        if python_cmd is None:
            python_cmd = self.python_cmd
        if cwd is None:
            cwd = self.work_dir
        cd_cwd = f'cd {shlex.quote(cwd)}  &&'
        script_len = len(script)
        logger.info('the size of generated python script is %s', script_len)
        if script_len < 100_000: # ssh connection will be closed of the size of command is too large
            return self.connector.run(f'{cd_cwd} {python_cmd} -c {shlex.quote(script)}', hide=True)
        else:
            script_path = os.path.join(self.tmp_dir, f'run_python_script_{s_uuid()}.py')
            self.dump_text(script, script_path)
            ret = self.connector.run(f'{cd_cwd} {python_cmd} {shlex.quote(script_path)}', hide=True)
            self.connector.run(f'rm {shlex.quote(script_path)}')
            return ret

    def run_python_fn(self, fn: FnType, python_cmd=None, cwd=None) -> FnType:
        def remote_fn(*args, **kwargs):
            script = fn_to_script(lambda: fn(*args, **kwargs), delimiter='@')
            ret = self.run_python_script(script=script, python_cmd=python_cmd, cwd=None)
            _, r = ret.stdout.rsplit('@')
            return cloudpickle.loads(bz2.decompress(base64.b64decode(r)))
        return remote_fn  # type: ignore

    def submit(self, script: str, cwd: str, **kwargs):
        return self.queue_system.submit(script, cwd=cwd, **kwargs)

    def resolve_artifact(self, artifact: Artifact) -> List[str]:
        if artifact.includes is None:
            return [artifact.url]
        pattern = os.path.join(artifact.url, artifact.includes)
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


def create_executor(config: BaseExecutorConfig, name: str) -> Executor:
    if config.queue_system is not None:
        return HpcExecutor.from_config(config, name)
    raise RuntimeError('The executor configuration is not supported!')


class ExecutorManager:
    def __init__(self, executor_configs: Mapping[str, BaseExecutorConfig]):
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
    dumped_fn = base64.b64encode(bz2.compress(cloudpickle.dumps(fn, protocol=cloudpickle.DEFAULT_PROTOCOL), 5))
    script = [
        f'''import base64,bz2,sys,cloudpickle as cp''',
        f'''r=cp.loads(bz2.decompress(base64.b64decode({repr(dumped_fn)})))()''',
        f'''sys.stdout.flush()''',  # ensure all output is printed
        f'''print({repr(delimiter)}+base64.b64encode(bz2.compress(cp.dumps(r, protocol=cp.DEFAULT_PROTOCOL),5)).decode('ascii'))''',
        f'''sys.stdout.flush()''',  # ensure all output is printed
    ]
    return ';'.join(script)
