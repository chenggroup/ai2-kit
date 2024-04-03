from ai2_kit.core.util import ensure_dir, expand_globs
from ai2_kit.feat.spectrum.viber import dpdata_read_cp2k_viber_data

import numpy as np

import dpdata
from dpdata.data_type import Axis, DataType

from ai2_kit.core.log import get_logger
logger = get_logger(__name__)


def register_data_types():
    if getattr(dpdata, '__registed__', False):
        return

    DATA_TYPES = [
        DataType("fparam", np.ndarray, (Axis.NFRAMES, -1), required=False),  # type: ignore
        DataType("aparam", np.ndarray, (Axis.NFRAMES, Axis.NATOMS, -1), required=False), # type: ignore
        DataType("efield", np.ndarray, (Axis.NFRAMES, Axis.NATOMS, 3), required=False), # type: ignore
        DataType("ext_efield", np.ndarray, (Axis.NFRAMES, 3), required=False), # type: ignore
        DataType("atomic_dipole", np.ndarray, (Axis.NFRAMES, -1), required=False), # type: ignore
        DataType("atomic_polarizability", np.ndarray, (Axis.NFRAMES, -1), required=False), # type: ignore
    ]
    dpdata.System.register_data_type(*DATA_TYPES) # type: ignore
    dpdata.LabeledSystem.register_data_type(*DATA_TYPES) # type: ignore
    dpdata.__registed__ = True  # type: ignore

register_data_types()


def set_fparam(system, fparam):
    nframes = system.get_nframes()
    system.data['fparam'] = np.tile(fparam, (nframes, 1))
    return system


class DpdataHelper:

    def __init__(self):
        self._systems = []

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

        :param fparam: fparam to set, should be a scalar or vector, e.g. 1.0 or [1.0, 2.0]
        """
        for system in self._systems:
            set_fparam(system, fparam)
        return self

    def _read(self, data_path: str, **kwargs):
        fmt = kwargs.get('fmt')
        assert fmt is not None, 'fmt is required'

        if fmt == 'cp2k/viber':
            kwargs.pop('fmt')
            try:
                system = dpdata_read_cp2k_viber_data(data_path, **kwargs)
            except:
                logger.warn(f'Fail to read {data_path}')
                return
        else:
            system = dpdata.LabeledSystem(data_path, **kwargs)

        fparam = kwargs.get('fparam', None)
        if fparam is not None:
            set_fparam(system, fparam)
            
        self._systems.extend(system)  # type: ignore
