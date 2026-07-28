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
import random

from pathlib import Path
from typing import List, Optional, Tuple
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
        DataType("atomic_weight", np.ndarray, (Axis.NFRAMES, Axis.NATOMS, 1), required=False),  # type: ignore
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
        :param recursive: if True, walk each given directory recursively and read every dataset found in it
        :param file_name: the file name to search for when recursive is True,
            if not given it is inferred from fmt, wildcard is supported, e.g. `*OUTCAR`
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

        a decimal value is treated as a fraction of the data size, for example
        `:0.9` selects the first 90% of the data, `0.9:` and `-0.1:` the last 10%

        :param expr: the slice expression
        """
        s = slice_from_str(expr, len(self._systems))
        self._systems = self._systems[s]
        return self

    def shuffle(self, seed=None):
        """
        shuffle systems in random order

        :param seed: seed for random shuffle, set it to get reproducible result
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._systems)
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


# fmt -> (file name to search for, whether the parent dir is the data path)
# a format whose reader takes a directory is marked by the file that identifies
# such a directory, and the parent dir of the match is what gets read
FMT_FILE_NAME = {
    "deepmd/npy": ("type.raw", True),
    "deepmd/raw": ("type.raw", True),
    "deepmd/comp": ("type.raw", True),
    "deepmd/hdf5": ("*.hdf5", False),
    "vasp/outcar": ("OUTCAR", False),
    "vasp/xml": ("vasprun.xml", False),
    "vasp/poscar": ("POSCAR", False),
    "vasp/contcar": ("CONTCAR", False),
    "cp2k/output": ("cp2k.out", False),
    "cp2k/aimd_output": ("cp2k.out", True),
    "lammps/dump": ("*.dump", False),
    "lammps/lmp": ("*.lmp", False),
    "abacus/stru": ("STRU", False),
    "xyz": ("*.xyz", False),
}


def _resolve_file_name(fmt: str, file_name: Optional[str], kwargs: dict) -> Tuple[str, bool]:
    """
    resolve the file name to search for in recursive mode, and whether the data path
    is the parent dir of the match instead of the match itself

    :param fmt: format to read
    :param file_name: file name given by user, if None it is inferred from fmt
    :param kwargs: the remaining arguments to pass to the reader, used by cp2k formats
    """
    # cp2k/dplr and cp2k/viber read a directory whose output file name is an argument
    if fmt == "cp2k/dplr":
        default = (kwargs.get("cp2k_output", "output"), True)
    elif fmt == "cp2k/viber":
        default = (kwargs.get("output_file", "output"), True)
    else:
        default = FMT_FILE_NAME.get(fmt)  # type: ignore

    if file_name is not None:
        return file_name, False if default is None else default[1]
    if default is None:
        raise ValueError(
            f"Cannot infer file_name for fmt {fmt}, please specify it explicitly, "
            f"e.g. --file_name OUTCAR. Known formats are: "
            f"{', '.join(sorted([*FMT_FILE_NAME, 'cp2k/dplr', 'cp2k/viber']))}"
        )
    return default


def _expand_recursive(roots: List[str], file_name: str, use_parent: bool) -> List[str]:
    """
    walk each directory in roots recursively and collect all path matching file_name,
    a root that is not a directory is kept as it is

    :param roots: list of paths to walk
    :param file_name: file name to search for, wildcard is supported
    :param use_parent: if True, collect the parent dir of the match instead of the match
    """
    paths = []
    for root in roots:
        if not os.path.isdir(root):
            result = [root]
        else:
            result = sorted(
                str(p.parent if use_parent else p) for p in Path(root).rglob(file_name)
            )
            if len(result) == 0:
                logger.warning(f'No {file_name} found in {root}')
        for p in result:
            if p not in paths:
                paths.append(p)
            else:
                logger.warning(f'path {p} already exists in the list')
    return paths


def read(*file_path_or_glob: str, **kwargs):
    """
    read data from multiple paths, support glob pattern
    default format is deepmd/npy

    :param file_path_or_glob: path or glob pattern to locate data path
    :param fmt: format to read, default is deepmd/npy
    :param label: default is True, use dpdata.LabeledSystem if True, else use dpdata.System
    :parse ignore_error: if True, ignore error when read data, default is False
    :param recursive: if True, walk each given directory recursively and read every dataset found in it
    :param file_name: the file name to search for when recursive is True,
        if not given it is inferred from fmt, wildcard is supported, e.g. `*OUTCAR`
    :param kwargs: arguments to pass to dpdata.System or dpdata.LabeledSystem
    """
    kwargs.setdefault("fmt", "deepmd/npy")
    ignore_error = kwargs.pop("ignore_error", False)
    recursive = kwargs.pop("recursive", False)
    file_name = kwargs.pop("file_name", None)
    files = expand_globs(file_path_or_glob)
    if recursive:
        file_name, use_parent = _resolve_file_name(kwargs["fmt"], file_name, kwargs)
        files = _expand_recursive(files, file_name, use_parent)
    elif file_name is not None:
        logger.warning('file_name is ignored as recursive is not set')
    if len(files) == 0:
        hint = f' (recursive search of {file_name})' if recursive else ''
        raise FileNotFoundError(f"No file found in {file_path_or_glob}{hint}")
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
