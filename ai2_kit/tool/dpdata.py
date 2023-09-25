from ai2_kit.core.util import ensure_dir

import dpdata
import glob

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

    def merge_write(self, out_path: str, fmt='deepmd/npy'):
        ensure_dir(out_path)
        if len(self._systems) == 0:
            raise ValueError('No data to merge')

        multi_systems = dpdata.MultiSystems(self._systems[0])
        for system in self._systems[1:]:
            multi_systems.append(system)

        if fmt == 'deepmd/npy':
            multi_systems.to_deepmd_npy(out_path)  # type: ignore
        elif fmt == 'deepmd/raw':
            multi_systems.to_deepmd_raw(out_path)  # type: ignore
        elif fmt == 'deepmd/hdf5':
            multi_systems.to_deepmd_hdf5(out_path)  # type: ignore
        else:
            raise ValueError(f'Unknown fmt {fmt}')

    def _read(self, file: str, **kwargs):
        if self._label:
            self._systems.append(dpdata.LabeledSystem(file, **kwargs))
        else:
            self._systems.append(dpdata.System(file, **kwargs))
