from fabric import Connection, Result
from .pydantic import BaseModel
from typing import Optional, List
from abc import ABC, abstractmethod
from io import StringIO
import shlex
import invoke
import os
import stat
import json
import glob


class SshConfig(BaseModel):
    host: str
    gateway: Optional['SshConfig'] = None


class BaseConnector(ABC):

    @abstractmethod
    def dump_text(self, text: str, path: str):
        ...

    @abstractmethod
    def glob(self, pattern: str) -> List[str]:
        ...

    @abstractmethod
    def run(self, script: str, **kwargs) -> Result:
        ...

    @abstractmethod
    def upload(self, from_path: str, to_dir: str) -> str:
        ...

    @abstractmethod
    def download(self, from_path: str, to_dir: str) -> str:
        ...

    @abstractmethod
    def sym_link(self, from_path: str, to_dir: str) -> str:
        ...

class SshConnector(BaseConnector):

    @classmethod
    def from_config(cls, config: SshConfig):
        connection = Connection(host=config.host)
        next_config = config.gateway
        next_connection = connection
        while next_config is not None:
            next_connection.gateway = Connection(host=next_config.host)
            next_connection = next_connection.gateway
            next_config = next_config.gateway
        return cls(connection)

    def __init__(self, connection: Connection):
        self._connection = connection

    def dump_text(self, text: str, path: str):
        f = StringIO(text)
        self.put(f, path)

    def glob(self, pattern: str):
        python_script = 'from glob import glob; from json import dumps; print(dumps(glob({})))'.format(repr(pattern))
        cmd = 'python -c {}'.format(shlex.quote(python_script))
        result = self.run(cmd, hide=True)
        return json.loads(result.stdout)

    def run(self, script, **kwargs) -> Result:
        return self._connection.run(script, **kwargs)

    def sym_link(self, from_path: str, to_dir: str) -> str:
        self.mkdir(to_dir)
        basename = safe_basename(from_path)
        to_path = os.path.join(to_dir, basename)
        self.run(get_ln_cmd(from_path, to_path))
        return to_path

    def upload(self, from_path: str, to_dir: str) -> str:
        self.mkdir(to_dir)
        to_path = os.path.join(to_dir, safe_basename(from_path))
        if os.path.isdir(from_path):
            self.put_dir(from_path, to_path)
        else:
            self.put(from_path, to_path)
        return to_path

    def download(self, from_path: str, to_dir: str) -> str:
        sftp = self.get_sftp()
        os.makedirs(to_dir, exist_ok=True)
        to_path = os.path.join(to_dir, safe_basename(from_path))
        if stat.S_ISDIR(sftp.lstat(from_path).st_mode):  # type: ignore
            self.get_dir(from_path, to_path)
        else:
            sftp.get(from_path, to_path)
        return to_path

    def put(self, *args, **kwargs):
        return self._connection.put(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self._connection.get(*args, **kwargs)

    def mkdir(self, dir_path: str):
        self._connection.run('mkdir -p {}'.format(shlex.quote(dir_path)))

    def put_dir(self, from_dir: str, to_dir: str):
        self.mkdir(to_dir)
        for item in os.listdir(from_dir):
            from_path = os.path.join(from_dir, item)
            to_path = os.path.join(to_dir, item)
            if os.path.isdir(from_path):
                self.put_dir(from_path, to_path)
            else:
                self.put(from_path, to_path)

    def get_dir(self, from_dir: str, to_dir: str):
        sftp = self.get_sftp()
        os.makedirs(to_dir, exist_ok=True)

        for item in sftp.listdir_attr(from_dir):
            from_path = os.path.join(from_dir, item.filename)
            to_path = os.path.join(to_dir, item.filename)
            if stat.S_ISDIR(item.st_mode):  # type: ignore
                self.get_dir(from_path, to_path)
            else:
                sftp.get(from_path, to_path)

    def get_sftp(self):
        return self._connection.sftp()


class LocalConnector(BaseConnector):

    def dump_text(self, text: str, path: str):
        with open(path, 'w') as f:
            f.write(text)

    def glob(self, pattern: str):
        return glob.glob(pattern)

    def run(self, script, **kwargs):
        return invoke.run(script, **kwargs)

    def sym_link(self, from_path: str, to_dir: str) -> str:
        os.makedirs(to_dir, exist_ok=True)
        basename = safe_basename(from_path)
        to_path = os.path.join(to_dir, basename)
        self.run(get_ln_cmd(from_path, to_path))
        return to_path

    def upload(self, from_path: str, to_dir: str) -> str:
        os.makedirs(to_dir, exist_ok=True)
        os.system(f'cp -r {from_path} {to_dir}')
        return os.path.join(to_dir, safe_basename(from_path))

    def download(self, from_path: str, to_dir: str) -> str:
        os.makedirs(to_dir, exist_ok=True)
        os.system(f'cp -r {from_path} {to_dir}')
        return os.path.join(to_dir, safe_basename(from_path))


def get_ln_cmd(from_path: str, to_path: str):
    """
    The reason to `rm -d` to_path is to workaround the limit of ln.
    `ln` command cannot override existed directory,
    so we need to ensure to_path is not existed.
    Here we use -d option instead of -rf to avoid remove directory with content.
    The error of `rm -d` is suppressed as it will fail when to_path is file.
    `-T` option of `ln` is used to avoid some unexpected result.
    """
    to_path = os.path.normpath(to_path)
    return 'rm -d {to_path} || true && ln -sfT {from_path} {to_path}'.format(
        from_path=shlex.quote(from_path),
        to_path=shlex.quote(to_path)
    )

def safe_basename(path: str, default=''):
    """
    Ensure return valid file name as basename
    """
    basename = os.path.basename(path)
    if basename in ('/', '.', '..', ''):
        return default
    return basename
