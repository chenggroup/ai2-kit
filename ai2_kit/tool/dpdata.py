from ai2_kit.core.util import ensure_dir, expand_globs
from ai2_kit.core.log import get_logger

import numpy as np

import dpdata
from dpdata.data_type import Axis, DataType


def __export_remote():
    def register_data_types():
        DATA_TYPES = [
            DataType("fparam", np.ndarray, (Axis.NFRAMES, -1), required=False),
        ]
        dpdata.System.register_data_type(*DATA_TYPES)
        dpdata.LabeledSystem.register_data_type(*DATA_TYPES)

    def set_fparam(system, fparam):
        nframes = system.get_nframes()
        system.data['fparam'] = np.tile(fparam, (nframes, 1))
        return system

    return (
        register_data_types,
        set_fparam,
    )
(
    register_data_types,
    set_fparam,
) = __export_remote()


logger = get_logger(__name__)
register_data_types()


class DpdataHelper:

    def __init__(self, label: bool = True):
        """
        label: if True, read data with labels (force, energy, etc), else read data without labels,
          use --nolabel to disable reading labels
        """
        self._systems = []
        self._label = label

    def read(self, *file_path_or_glob: str, **kwargs):
        """
        read data from multiple paths, support glob pattern
        default format is deepmd/npy

        :param file_path_or_glob: path or glob pattern to find data files
        :param kwargs: arguments to pass to dpdata.System / dpdata.LabeledSystem
        """
        kwargs.setdefault('fmt', 'deepmd/npy')
        files = expand_globs(file_path_or_glob)
        if len(files) == 0:
            raise FileNotFoundError(f'No file found for {file_path_or_glob}')
        for file in files:
            self._read(file, **kwargs)
        return self

    def filter(self, lambda_expr: str):
        """
        filter data with lambda expression

        :param lambda_expr: lambda expression to filter data
        """
        fn = eval(lambda_expr)
        self._systems = [ system for system in self._systems if fn(system.data)]
        return self

    @property
    def merge_write(self):
        logger.warn('merge_write is deprecated, use write instead')
        return self.write

    def write(self, out_path: str, fmt='deepmd/npy', merge: bool = True):
        """
        write data to specific path, support deepmd/npy, deepmd/raw, deepmd/hdf5 formats
        :param out_path: path to write data
        :param fmt: format to write, default is deepmd/npy
        :param merge: if True, merge all data use dpdata.MultiSystems, else write data without merging
        """
        ensure_dir(out_path)
        if len(self._systems) == 0:
            raise ValueError('No data to merge')
        if merge:
            systems = dpdata.MultiSystems(self._systems[0])
        else:
            systems = self._systems[0]

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

    def set_fparam(self, fparam):
        """
        Set fparam for all systems

        :param fparam: fparam to set, should be a scalar or vector
        """
        for system in self._systems:
            set_fparam(system, fparam)
        return self

    def _read(self, file: str, **kwargs):
        if self._label:
            self._systems.extend(dpdata.LabeledSystem(file, **kwargs))
        else:
            self._systems.extend(dpdata.System(file, **kwargs))

