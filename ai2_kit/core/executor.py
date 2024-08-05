from typing import Optional, Dict, List, TypeVar, Callable, Mapping, Union
from abc import ABC, abstractmethod
from invoke import Result
import cloudpickle as cp
import tempfile
import tarfile
import os
import shlex
import base64
import bz2

from .queue_system import QueueSystemConfig, BaseQueueSystem, Slurm, Lsf, PBS
from .job import JobFuture
from .artifact import Artifact
from .connector import SshConfig, BaseConnector, SshConnector, LocalConnector
from .util import s_uuid
from .log import get_logger
from .pydantic import BaseModel


logger = get_logger(__name__)

class BaseExecutorConfig(BaseModel):
    ssh: Optional[SshConfig] = None
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
    def run_python_fn(self, fn: FnType, python_cmd=None, cwd=None) -> FnType:
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
    def upload(self, from_path: str, to_dir: str) -> str:
        ...

    @abstractmethod
    def download(self, from_path: str, to_dir: str) -> str:
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
        self.tmp_dir = os.path.join(self.work_dir, '.tmp')
        self.python_pkgs_dir = os.path.join(self.tmp_dir, 'python_pkgs')

    def init(self):
        # if work_dir is relative path, it will be relative to user home
        if not os.path.isabs(self.work_dir):
            user_home = self.run('echo $HOME', hide=True).stdout.strip()
            self.work_dir = os.path.normpath(os.path.join(user_home, self.work_dir))

        self.mkdir(self.work_dir)
        self.mkdir(self.tmp_dir)
        self.mkdir(self.python_pkgs_dir)
        self.upload_python_pkg('ai2_kit')

    def upload_python_pkg(self, pkg: str):
        """
        upload python package to remote server
        """
        pkg_path = os.path.dirname(__import__(pkg).__file__)
        with tempfile.NamedTemporaryFile(suffix='.tar.gz') as fp:
            with tarfile.open(fp.name, 'w:gz') as tar_fp:
                tar_fp.add(pkg_path, arcname=os.path.basename(pkg_path), filter=_filter_pyc_files)
            fp.flush()
            self.upload(fp.name, self.python_pkgs_dir)
            file_name = os.path.basename(fp.name)
        self.run(f'cd {shlex.quote(self.python_pkgs_dir)} && tar -xf {shlex.quote(file_name)}')
        logger.info('add python package: %s', pkg_path)

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
        base_cmd = f'cd {shlex.quote(cwd)} && PYTHONPATH={shlex.quote(self.python_pkgs_dir)} '
        script_len = len(script)
        logger.info('the size of generated python script is %s', script_len)
        if script_len < 100_000: # ssh connection will be closed of the size of command is too large
            return self.connector.run(f'{base_cmd} {python_cmd} -c {shlex.quote(script)}', hide=True)
        else:
            script_path = os.path.join(self.tmp_dir, f'run_python_script_{s_uuid()}.py')
            self.dump_text(script, script_path)
            ret = self.connector.run(f'{base_cmd} {python_cmd} {shlex.quote(script_path)}', hide=True)
            self.connector.run(f'rm {shlex.quote(script_path)}')
            return ret

    def run_python_fn(self, fn: FnType, python_cmd=None, cwd=None) -> FnType:
        def remote_fn(*args, **kwargs):
            script = fn_to_script(fn, args, kwargs, delimiter='@')
            ret = self.run_python_script(script=script, python_cmd=python_cmd, cwd=cwd)
            _, r = ret.stdout.rsplit('@', 1)
            return cp.loads(bz2.decompress(base64.b64decode(r)))
        return remote_fn  # type: ignore

    def submit(self, script: str, cwd: str, **kwargs):
        return self.queue_system.submit(script, cwd=cwd, **kwargs)

    def resolve_artifact(self, artifact: Artifact) -> List[str]:
        if artifact.includes is None:
            return [artifact.url]
        pattern = os.path.join(artifact.url, artifact.includes)
        return self.glob(pattern)

    def upload(self, from_path: str, to_dir: str):
        return self.connector.upload(from_path, to_dir)

    def download(self, from_path: str, to_dir: str):
        return self.connector.download(from_path, to_dir)


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


def fn_to_script(fn: Callable, args, kwargs, delimiter='@'):
    script = [
        f'''import base64,bz2,sys,cloudpickle as cp''',
        f'''fn,args,kwargs={pickle_converts((fn, args, kwargs))}''',
        'r=fn(*args, **kwargs)',
        f'''sys.stdout.flush()''',  # ensure all output is printed
        f'''print({repr(delimiter)}+base64.b64encode(bz2.compress(cp.dumps(r, protocol=cp.DEFAULT_PROTOCOL),5)).decode('ascii'))''',
        f'''sys.stdout.flush()''',  # ensure all output is printed
    ]
    return ';'.join(script)


def pickle_converts(obj, pickle_module='cp', bz2_module='bz2', base64_module='base64'):
    """
    convert an object to its pickle string form
    """
    obj_pkl = cp.dumps(obj, protocol=cp.DEFAULT_PROTOCOL)
    compress_level = 5 if len(obj_pkl) > 4096 else 1
    compressed = bz2.compress(obj_pkl, compress_level)
    obj_b64 = base64.b64encode(compressed).decode('ascii')
    return f'{pickle_module}.loads({bz2_module}.decompress({base64_module}.b64decode({repr(obj_b64)})))'

def _filter_pyc_files(tarinfo):
    if tarinfo.name.endswith('.pyc') or tarinfo.name.endswith('__pycache__'):
        return None
    return tarinfo
