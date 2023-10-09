from ruamel.yaml import YAML, ScalarNode, SequenceNode
from pathlib import Path
from typing import Tuple, List, TypeVar, Union
from dataclasses import field
from itertools import zip_longest

import shortuuid
import hashlib
import base64
import copy
import os
import random
import json
import glob

from .log import get_logger

logger = get_logger(__name__)

EMPTY = object()


def default_mutable_field(obj):
    return field(default_factory=lambda: copy.copy(obj))


def get_yaml():
    yaml = YAML(typ='safe')
    JoinTag.register(yaml)
    LoadTextTag.register(yaml)
    LoadYamlTag.register(yaml)
    return yaml


def load_yaml_file(path: Union[Path, str]):
    if isinstance(path, str):
        path = Path(path)
    yaml = get_yaml()
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


def sort_unique_str_list(l: List[str]) -> List[str]:
    """remove duplicate str and sort"""
    return list(sorted(set(l)))


T = TypeVar('T')


def flatten(l: List[List[T]]) -> List[T]:
    return [item for sublist in l for item in sublist]


def format_env_string(s: str) -> str:
    return s.format(**os.environ)


def list_split(l: List[T], n: int) -> List[List[T]]:
    """split list into n chunks"""
    # ref: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(l), n)
    return [l[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


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
        ...

    @classmethod
    def register(cls, yaml: YAML):
        yaml.register_class(cls)

class LoadTextTag:
    """a tag to read string from file"""

    yaml_tag = u'!load_text'

    @classmethod
    def from_yaml(cls, constructor, node):
        path = _yaml_get_path_node(node, constructor)
        with open(path, 'r') as f:
            return f.read()

    @classmethod
    def to_yaml(cls, dumper, data):
        ...

    @classmethod
    def register(cls, yaml: YAML):
        yaml.register_class(cls)


class LoadYamlTag:
    """a tag to read string from file"""

    yaml_tag = u'!load_yaml'

    @classmethod
    def from_yaml(cls, constructor, node):
        path = _yaml_get_path_node(node, constructor)
        yaml = get_yaml()
        with open(path, 'r') as f:
            return yaml.load(f)

    @classmethod
    def to_yaml(cls, dumper, data):
        ...

    @classmethod
    def register(cls, yaml: YAML):
        yaml.register_class(cls)


def _yaml_get_path_node(node, constructor):
    if isinstance(node, ScalarNode):
        return constructor.construct_scalar(node)
    elif isinstance(node, SequenceNode):
        seq = constructor.construct_sequence(node)
        return os.path.join(*seq)
    else:
        raise ValueError(f'Unknown node type {type(node)}')


def __export_remote_functions():
    """cloudpickle compatible: https://stackoverflow.com/questions/75292769"""

    def merge_dict(lo: dict, ro: dict, path=None, ignore_none=True):
        """
        Merge two dict, the left dict will be overridden.
        Note: list will be replaced instead of merged.
        """
        if path is None:
            path = []
        for key, value in ro.items():
            if ignore_none and value is None:
                continue
            if key in lo:
                current_path = path + [str(key)]
                if isinstance(lo[key], dict) and isinstance(value, dict):
                    merge_dict(lo[key], value, path=current_path, ignore_none=ignore_none)
                else:
                    print('.'.join(current_path) + ' has been overridden')
                    lo[key] = value
            else:
                lo[key] = value
        return lo

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

    def list_even_sample(l, size):
        if size <= 0 or size > len(l):
            return l
        # calculate the sample interval
        interval = len(l) / size
        return [l[int(i * interval)] for i in range(size)]

    def list_random_sample(l, size, seed = None):
        if seed is None:
            seed = len(l)
        random.seed(seed)
        return random.sample(l, size)

    def list_sample(l, size, method='even', **kwargs):
        if method == 'even':
            return list_even_sample(l, size)
        elif method == 'random':
            return list_random_sample(l, size, **kwargs)
        elif method == 'truncate':
            return l[:size]
        else:
            raise ValueError(f'Unknown sample method {method}')

    def flat_evenly(list_of_lists):
        """
        flat a list of lists and ensure the output result distributed evenly
        >>> flat_evenly([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        [1, 4, 7, 2, 5, 8, 3, 6, 9]
        Ref: https://stackoverflow.com/questions/76751171/how-to-flat-a-list-of-lists-and-ensure-the-output-result-distributed-evenly-in-p
        """
        return [e for tup in zip_longest(*list_of_lists) for e in tup if e is not None]


    def dump_json(obj, path: str):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, default=default)


    def dump_text(text: str, path: str, **kwargs):
        with open(path, 'w', **kwargs) as f:
            f.write(text)


    def flush_stdio():
        import sys
        sys.stdout.flush()
        sys.stderr.flush()


    def ensure_dir(path: str):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)


    def expand_globs(paths: List[str]) -> List[str]:
        paths = flatten([glob.glob(path) for path in paths])
        return sort_unique_str_list(paths)


    # export functions
    return (
        merge_dict,
        dict_nested_get,
        dict_nested_set,
        list_even_sample,
        list_random_sample,
        list_sample,
        flat_evenly,
        dump_json,
        dump_text,
        flush_stdio,
        ensure_dir,
        expand_globs,
    )


(
    merge_dict,
    dict_nested_get,
    dict_nested_set,
    list_even_sample,
    list_random_sample,
    list_sample,
    flat_evenly,
    dump_json,
    dump_text,
    flush_stdio,
    ensure_dir,
    expand_globs,
) = __export_remote_functions()
