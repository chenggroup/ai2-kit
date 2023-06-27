from ruamel.yaml import YAML
from pathlib import Path
from typing import Tuple, List, TypeVar
from dataclasses import field
from cp2k_input_tools.parser import CP2KInputParserSimplified

import shortuuid
import hashlib
import base64
import copy
import os
import io

from .log import get_logger

logger = get_logger(__name__)

EMPTY = object()

def default_mutable_field(obj):
    return field(default_factory=lambda: copy.copy(obj))

def __merge_dict():
    """cloudpickle compatible: https://stackoverflow.com/questions/75292769"""
    def merge_dict(lo: dict, ro: dict, path=None):
        """
        merge two dict, the left dict will be overridden

        this method won't merge list
        """
        if path is None:
            path = []

        for key, value in ro.items():
            if key in lo:
                current_path = path + [ str(key) ]
                if isinstance(lo[key], dict) and isinstance(value, dict):
                    merge_dict(lo[key], value, current_path)
                else:
                    print('.'.join(current_path) + ' has been overridden')
                    lo[key] = value
            else:
                lo[key] = value
        return lo
    return merge_dict
merge_dict = __merge_dict()


# TODO: support http(s) url
def load_yaml_file(path: Path):
    yaml = YAML(typ='safe')
    JoinTag.register(yaml)
    ReadTag.register(yaml)
    return yaml.load(path)


def load_yaml_files(*paths: Tuple[Path]):
    d = {}
    for path in paths:
        print('load yaml file: ', path)
        d = merge_dict(d, load_yaml_file(Path(path)))  # type: ignore
    return d


def s_uuid():
    """short uuid"""
    return shortuuid.uuid()


# TODO: this should be moved out from core module
def parse_cp2k_input(text: str):
    parser = CP2KInputParserSimplified(key_trafo=str.upper)
    return parser.parse(io.StringIO(text))


def dict_nested_get(d: dict, keys: List[str], default=EMPTY):
    """get value from nested dict"""
    for key in keys:
        if key not in d and default is not EMPTY:
            return default
        d = d[key]
    return d


def dict_nested_set(d: dict, keys: List[str], value):
    """set value to nested dict"""
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def sort_unique_str_list(l: List[str]) -> List[str]:
    """remove duplicate str and sort"""
    return list(sorted(set(l)))


T = TypeVar('T')
def flatten(l: List[List[T]]) -> List[T]:
    return [item for sublist in l for item in sublist]


def format_env_string(s: str) -> str:
    return s.format(**os.environ)


def split_list(l: List[T], n: int) -> List[List[T]]:
    """split list into n chunks"""
    # ref: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(l), n)
    return [l[i*k+min(i, m) : (i+1)*k+min(i+1, m)] for i in range(n)]


def short_hash(s: str) -> str:
    """short hash string"""
    digest = hashlib.sha1(s.encode('utf-8')).digest()
    # use urlsafe encode to avoid '/' in the string, as it will cause problem in file path
    return base64.urlsafe_b64encode(digest).decode('utf-8')[:-2]

async def to_awaitable(value: T) -> T:
    return value


class JoinTag:
    """a tag to join strings in a list"""

    yaml_tag = u'!join'

    @classmethod
    def from_yaml(cls, constructor, node):
        seq = constructor.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    @classmethod
    def to_yaml(cls, dumper, data):
        # do nothing
        return dumper.represent_sequence(cls.yaml_tag, data)

    @classmethod
    def register(cls, yaml: YAML):
        yaml.register_class(cls)


class ReadTag:
    """a tag to read string from file"""

    yaml_tag = u'!read'

    @classmethod
    def from_yaml(cls, constructor, node):
        seq = constructor.construct_sequence(node)
        path = os.path.join(*seq)
        with open(path, 'r') as f:
            return f.read()

    @classmethod
    def to_yaml(cls, dumper, data):
        # do nothing
        return dumper.represent_sequence(cls.yaml_tag, data)

    @classmethod
    def register(cls, yaml: YAML):
        yaml.register_class(cls)
