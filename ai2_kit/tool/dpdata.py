from ai2_kit.core.util import ensure_dir
from ai2_kit.core.log import get_logger

import numpy as np
import dpdata
import glob
import os

logger = get_logger(__name__)

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
        files = _expand_paths(*file_path_or_glob)
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

    def write_fparams(self, *file_path_or_glob: str, fparams):
        """
        write fparam.npy to dataset, only support deepmd/npy format.

        This command will search all *.npy files under the specific path
        and create a fparam.npy file next to it.

        :param file_path_or_glob: path or glob pattern to find *.npy files
        :param fparams: fparams to write, can be a single value or a list of values
        """
        write_fparams(*file_path_or_glob, fparams=fparams)


    def _read(self, file: str, **kwargs):
        if self._label:
            self._systems.extend(dpdata.LabeledSystem(file, **kwargs))
        else:
            self._systems.extend(dpdata.System(file, **kwargs))


def __export_remote_fn():

    def write_fparams(*file_path_or_glob: str, fparams):
        # search for box.npy files
        paths = _expand_paths(*[os.path.join(path, '**/box.npy')
                                for path in file_path_or_glob])
        if len(paths) == 0:
            raise FileNotFoundError(f'No deepmd/npy datasets found in {file_path_or_glob}')

        for path in paths:
            box_arr = np.load(path)
            logger.debug(f'box.npy shape: {box_arr.shape}')
            fparam_arr = np.tile(fparams, (len(box_arr), 1))
            logger.debug(f'fparam.npy shape: {fparam_arr.shape}')
            fparam_file = os.path.join(os.path.dirname(path), 'fparam.npy')
            np.save(fparam_file, fparam_arr)

    return (write_fparams, )

(write_fparams, ) = __export_remote_fn()


def _expand_paths(*file_path_or_glob: str):
    files = []
    for file_path in file_path_or_glob:
        files += sorted(glob.glob(file_path, recursive=True)) if '*' in file_path else [file_path]
    return files
