from ruamel.yaml import YAML
from pathlib import Path
from typing import Tuple
import shortuuid
import copy
from dataclasses import field

def default_mutable_field(obj):
    return field(default_factory=lambda: copy.copy(obj))

def __merge_dict():
    """
    merge two dict, the left dict will be overridden
    this method won't merge list

    cloudpickle compatible: https://stackoverflow.com/questions/75292769
    """
    def merge_dict(lo: dict, ro: dict, path=None):
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
    return yaml.load(path)


def load_yaml_files(*paths: Tuple[Path]):
    d = {}
    for path in paths:
        print('load yaml file: ', path)
        d = merge_dict(d, load_yaml_file(Path(path)))
    return d


def tmpfile_name():
    return shortuuid.uuid()
