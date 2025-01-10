from ai2_kit.feat.spectrum.viber import dpdata_read_cp2k_viber_data
from ai2_kit.domain.dplr import dpdata_read_cp2k_dplr_data, dplr_v3_to_v2
from ai2_kit.core.util import (
    ensure_dir,
    expand_globs,
    list_sample,
    SAMPLE_METHOD,
    slice_from_str,
)
from ai2_kit.core.log import get_logger

import os
import glob

from typing import Optional
from dpdata.data_type import Axis, DataType
import numpy as np
import dpdata


logger = get_logger(__name__)


def register_data_types():
    if getattr(dpdata, "__registed__", False):
        return

    DATA_TYPES = [
        DataType("fparam", np.ndarray, (Axis.NFRAMES, -1), required=False),  # type: ignore
        DataType("aparam", np.ndarray, (Axis.NFRAMES, Axis.NATOMS, -1), required=False),  # type: ignore
        DataType("efield", np.ndarray, (Axis.NFRAMES, Axis.NATOMS, 3), required=False),  # type: ignore
        DataType("ext_efield", np.ndarray, (Axis.NFRAMES, 3), required=False),  # type: ignore
        DataType("atomic_dipole", np.ndarray, (Axis.NFRAMES, Axis.NATOMS, 3), required=False),  # type: ignore
        DataType("atomic_polarizability", np.ndarray, (Axis.NFRAMES, Axis.NATOMS, 9), required=False),  # type: ignore
        DataType("wannier_spread", np.ndarray, (Axis.NFRAMES, Axis.NATOMS, 4), required=False),  # type: ignore
    ]
    dpdata.System.register_data_type(*DATA_TYPES)  # type: ignore
    dpdata.LabeledSystem.register_data_type(*DATA_TYPES)  # type: ignore
    dpdata.__registed__ = True  # type: ignore


register_data_types()


class DpdataTool:

    def __init__(self, verbose=False, systems: Optional[list] = None):
        self._systems = [] if systems is None else systems
        self._verbose = verbose

    def read(self, *file_path_or_glob: str, **kwargs):
        """
        read data from multiple paths, support glob pattern
        default format is deepmd/npy

        :param file_path_or_glob: path or glob pattern to locate data path
        :param fmt: format to read, default is deepmd/npy
        :param label: default is True, use dpdata.LabeledSystem if True, else use dpdata.System
        :param kwargs: arguments to pass to dpdata.System or dpdata.LabeledSystem
        """
        systems = read(*file_path_or_glob, **kwargs)
        self._systems.extend(systems)
        return self

    def filter(self, lambda_expr: str):
        """
        filter data with lambda expression

        :param lambda_expr: lambda expression to filter data
        """
        fn = eval(lambda_expr)
        self._systems = [system for system in self._systems if fn(system.data)]
        return self

    def slice(self, expr: str):
        """
        slice systems by python slice expression, for example
        `10:`, `:10`, `::2`, etc

        :param start: start index
        :param stop: stop index
        :param step: step
        """
        s = slice_from_str(expr)
        self._systems = self._systems[s]
        return self

    def sample(self, size: int, method: SAMPLE_METHOD = "even", **kwargs):
        """
        sample data

        :param size: size of sample, if size is larger than data size, return all data
        :param method: method to sample, can be 'even', 'random', 'truncate', default is 'even'
        :param seed: seed for random sample, only used when method is 'random'

        Note that by default the seed is length of input list,
        if you want to generate different sample each time, you should set random seed manually
        """
        self._systems = list_sample(self._systems, size, method, **kwargs)
        return self

    def size(self):
        """
        size of loaded data
        """
        print(len(self._systems))
        return self

    def write(
        self,
        out_path: str,
        fmt="deepmd/npy",
        merge: bool = True,
        v2: bool = False,
        sel_symbol: Optional[list] = None,
    ):
        """
        write data to specific path, support deepmd/npy, deepmd/raw, deepmd/hdf5 formats
        :param out_path: path to write data
        :param fmt: format to write, default is deepmd/npy
        :param merge: if True, merge all data use dpdata.MultiSystems, else write data without merging
        :param v2: if True, write data in v2 format, else write data in v3 format
        :param sel_symbol: the selected symbols of atom, for example, ["O", "K", "F"] means only write data of O, K, F
        """
        ensure_dir(out_path)
        if len(self._systems) == 0:
            raise ValueError("No data to merge")
        if merge:
            systems = dpdata.MultiSystems(self._systems[0])
        else:
            systems = self._systems[0]

        for system in self._systems[1:]:
            systems.append(system)

        if fmt == "deepmd/npy":
            systems.to_deepmd_npy(out_path)  # type: ignore
            if v2:
                assert (
                    sel_symbol is not None
                ), "sel_symbol must be provided when v2 is True"
                dplr_v3_to_v2(out_path, sel_symbol)

        elif fmt == "deepmd/raw":
            systems.to_deepmd_raw(out_path)  # type: ignore
        elif fmt == "deepmd/hdf5":
            systems.to_deepmd_hdf5(out_path)  # type: ignore
        else:
            raise ValueError(f"Unknown fmt {fmt}")

    def set_fparam(self, fparam):
        """
        Set fparam for all systems

        :param fparam: fparam to set, should be a scalar or vector, e.g. 1.0 or [1.0, 2.0]
        """
        for system in self._systems:
            set_fparam(system, fparam)
        return self

    def eval(self, dp_model: str):
        """
        Use deepmd model to label energy, force and viral

        :param dp_model: path to deepmd frozen model
        """
        from deepmd.infer import DeepPot

        systems = dpdata.System()
        systems.extend(self._systems)  # merge systems to one

        pot = DeepPot(dp_model, auto_batch_size=True)  # type: ignore

        # remap atypes to pot's type
        atom_names = systems.get_atom_names()
        target_atom_names = pot.get_type_map()
        mapping = {
            i: target_atom_names.index(name) for i, name in enumerate(atom_names)
        }
        vectorized_mapping = np.vectorize(mapping.get)

        atypes = vectorized_mapping(systems.data["atom_types"])
        coords = systems.data["coords"]
        cells = None if systems.nopbc else systems.data["cells"]

        e, f, v = pot.eval(coords=coords, cells=cells, atom_types=atypes)  # type: ignore

        n_atoms = systems.get_natoms()
        n_frames = systems.get_nframes()

        e = e.reshape((n_frames,))
        f = f.reshape((n_frames, n_atoms, 3))
        v = v.reshape((n_frames, 3, 3))

        data = {**systems.data, "energies": e, "forces": f, "virials": v}
        # replace system files
        self._systems = []
        self._systems.extend(dpdata.LabeledSystem.from_dict({"data": data}))  # type: ignore
        return self

    def to_ase(self):
        """
        Convert dpdata format to ase format, and use ase tool to handle
        """
        from .ase import AseTool

        atoms_list = []
        for sys in self._systems:
            atoms_list.extend(sys.to_ase_structure())
        return AseTool(atoms_arr=atoms_list)

    def _verbose_log(self, msg, **kwargs):
        if self._verbose:
            logger.info(msg, **kwargs)


def set_fparam(system, fparam):
    nframes = system.get_nframes()
    system.data["fparam"] = np.tile(fparam, (nframes, 1))
    return system


def read(*file_path_or_glob: str, **kwargs):
    """
    read data from multiple paths, support glob pattern
    default format is deepmd/npy

    :param file_path_or_glob: path or glob pattern to locate data path
    :param fmt: format to read, default is deepmd/npy
    :param label: default is True, use dpdata.LabeledSystem if True, else use dpdata.System
    :parse ignore_error: if True, ignore error when read data, default is False
    :param kwargs: arguments to pass to dpdata.System or dpdata.LabeledSystem
    """
    kwargs.setdefault("fmt", "deepmd/npy")
    ignore_error = kwargs.pop("ignore_error", False)
    files = expand_globs(file_path_or_glob)
    if len(files) == 0:
        raise FileNotFoundError(f"No file found in {file_path_or_glob}")
    systems = []
    for file in files:
        try:
            system = _read(file, **kwargs)
            if system is not None and len(system) > 0:
                systems.extend(system)
            else:
                logger.warning(f'Ignore invalid system from {file}: {system}')
        except Exception:
            if not ignore_error:
                raise
            logger.exception(f"Fail to process file {file}, ignore and continue")
    return systems


def _read(data_path: str, **kwargs):
    # pop custom arguments or else it will be passed to dpdata.System and raise error
    fmt = kwargs.pop("fmt", "deepmd/npy")
    fparam = kwargs.pop("fparam", None)
    label = kwargs.pop("label", True)

    if fmt == "cp2k/viber":
        system = dpdata_read_cp2k_viber_data(data_path, **kwargs)
    elif fmt == "cp2k/dplr":
        system = dpdata_read_cp2k_dplr_data(data_path, **kwargs)
    else:
        system = (
            dpdata.LabeledSystem(data_path, fmt=fmt, **kwargs)
            if label else
            dpdata.System(data_path, fmt=fmt, **kwargs)
        )

    if fparam is not None:
        set_fparam(system, fparam)

    return system
