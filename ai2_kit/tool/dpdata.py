from ai2_kit.core.util import ensure_dir
from ai2_kit.core.log import get_logger

import dpdata
import glob

logger = get_logger(__name__)

class DpdataHelper:

    def __init__(self, label: bool = True):
        self._systems = []
        self._label = label

    def read(self, *file_path_or_glob: str,**kwargs):
        kwargs.setdefault('fmt', 'deepmd/npy')
        files = []
        for file_path in file_path_or_glob:
            files += sorted(glob.glob(file_path, recursive=True)) if '*' in file_path else [file_path]

        if len(files) == 0:
            raise FileNotFoundError(f'No file found for {file_path_or_glob}')
        for file in files:
            self._read(file, **kwargs)
        return self

    def filter(self, lambda_expr: str):
        fn = eval(lambda_expr)
        def _fn(system):
            return fn(system.data)
        self._systems = list(filter(_fn, self._systems))
        return self

    @property
    def merge_write(self):
        logger.warn('merge_write is deprecated, use write instead')
        return self.write

    def write(self, out_path: str, fmt='deepmd/npy', merge: bool = True):
        ensure_dir(out_path)
        if len(self._systems) == 0:
            raise ValueError('No data to merge')

        if merge:
            systems = dpdata.MultiSystems(self._systems[0])
        else:
            if self._label:
                systems = dpdata.LabeledSystem(self._systems[0])
            else:
                systems = dpdata.System(self._systems[0])

        for system in self._systems[1:]:
            systems.append(system)

        if fmt == 'deepmd/npy':
            systems.to_deepmd_npy(out_path)  # type: ignore
        elif fmt == 'deepmd/raw':
            systems.to_deepmd_raw(out_path)  # type: ignore
        elif fmt == 'deepmd/hdf5':
            systems.to_deepmd_hdf5(out_path)  # type: ignore
        else:
            raise ValueError(f'Unknown fmt {fmt}')

    def _read(self, file: str, **kwargs):
        if self._label:
            self._systems.append(dpdata.LabeledSystem(file, **kwargs))
        else:
            self._systems.append(dpdata.System(file, **kwargs))
