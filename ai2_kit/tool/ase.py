from ai2_kit.core.util import ensure_dir, expand_globs, slice_from_str, SAMPLE_METHOD, list_sample
from ai2_kit.core.log import get_logger
from ai2_kit.domain.cp2k import dump_coord_n_cell

from typing import List, Union, Optional
from ase import Atoms
import ase.io


logger = get_logger(__name__)


class AseTool:

    def __init__(self, atoms_arr: Optional[List[Atoms]] = None):
        self._atoms_arr: List[Atoms] = [] if atoms_arr is None else atoms_arr

    def read(self, *file_path_or_glob: str, nat_sort=False, **kwargs):
        """
        read atoms from file, support multiple files and glob pattern

        :param file_path_or_glob: path or glob pattern to locate data path
        :param nat_sort: sort files by natural order, default is False
        :param kwargs: other arguments for ase.io.read
        """

        files = expand_globs(file_path_or_glob, nature_sort=nat_sort)
        if len(files) == 0:
            raise FileNotFoundError(f'No file found for {file_path_or_glob}')
        for file in files:
            self._read(file, **kwargs)
        return self

    def set_cell(self, cell, scale_atoms=False, apply_constraint=True):
        for atoms in self._atoms_arr:
            atoms.set_cell(cell, scale_atoms=scale_atoms, apply_constraint=apply_constraint)
        return self

    def set_pbc(self, pbc):
        for atoms in self._atoms_arr:
            atoms.set_pbc(pbc)
        return self

    def set_by_ref(self, ref_file: str, **kwargs):
        kwargs.setdefault('index', 0)
        ref_atoms = ase.io.read(ref_file, **kwargs)
        assert isinstance(ref_atoms, Atoms), 'Only support single frame reference'
        for atoms in self._atoms_arr:
            try:
                atoms.set_cell(ref_atoms.get_cell())
            except ValueError:
                pass
            try:
                atoms.set_pbc(ref_atoms.get_pbc())
            except ValueError:
                pass
        return self

    def size(self):
        """
        size of loaded data
        """
        return len(self._atoms_arr)

    def slice(self, expr: str):
        """
        slice systems by python slice expression, for example
        `10:`, `:10`, `::2`, etc

        :param start: start index
        :param stop: stop index
        :param step: step
        """
        s = slice_from_str(expr)
        self._atoms_arr = self._atoms_arr[s]
        return self

    def sample(self, size: int, method: SAMPLE_METHOD='even', **kwargs):
        """
        sample data

        :param size: size of sample, if size is larger than data size, return all data
        :param method: method to sample, can be 'even', 'random', 'truncate', default is 'even'
        :param seed: seed for random sample, only used when method is 'random'

        Note that by default the seed is length of input list,
        if you want to generate different sample each time, you should set random seed manually
        """
        self._atoms_arr= list_sample(self._atoms_arr, size, method, **kwargs)
        return self

    def delete_atoms(self, id: Union[int, List[int]], start_id=0):
        """
        delete atoms by id or list of id

        :param id: id or list of id
        :param start_id: the start id of first item, for example, in LAMMPS the id of first item is 1, in ASE it is 0
        """

        ids = [id] if isinstance(id, int) else id
        for atoms in self._atoms_arr:
            for i in sorted(ids, reverse=True):
                assert i >= start_id, f'Invalid id {i}'
                del atoms[i - start_id]
        return self

    def write(self, filename: str, slice=None, chain=False, **kwargs):
        """
        write atoms to file
        :param filename: the filename to write
        :param slice: slice expression to select data
        :param chain: if True, return self, useful for chain operation
        :param kwargs: other arguments for ase.io.write
        """
        atoms_arr = self._atoms_arr
        if slice is not None:
            atoms_arr = atoms_arr[slice_from_str(slice)]

        ensure_dir(filename.format(i=0))
        self._write(filename, atoms_arr, **kwargs)
        if chain:
            return self

    def write_frames(self, filename: str, slice=None, chain=False, **kwargs):
        """
        write each frame to a separate file, useful to write to format only support single frame, POSCAR for example

        :param filename: the filename template, use {i} to represent the index, for example, 'frame_{i}.xyz'
        :param slice: slice expression to select data
        :param chain: if True, return self, useful for chain operation
        :param kwargs: other arguments for ase.io.write
        """
        atoms_arr = self._atoms_arr
        if slice is not None:
            atoms_arr = atoms_arr[slice_from_str(slice)]

        for i, atoms in enumerate(atoms_arr):
            _filename = filename.format(i=i)
            ensure_dir(_filename)
            self._write(_filename, atoms, **kwargs)
        if chain:
            return self

    @property
    def write_each_frame(self):
        logger.warning('write_each_frame has been deprecated, use write_frames instead')
        return self.write_frames

    def write_dplr_lammps_data(self, filename: str,
                               type_map: List[str], sel_type: List[int],
                               sys_charge_map: List[float], model_charge_map: List[float]):

        """
        write atoms to LAMMPS data file for DPLR
        the naming convention of params follows Deepmd-Kit's

        about dplr: https://docs.deepmodeling.com/projects/deepmd/en/master/model/dplr.html
        about fitting tensor: https://docs.deepmodeling.com/projects/deepmd/en/master/model/train-fitting-tensor.html


        :param filename: the filename of LAMMPS data file, use {i} to represent the index, for example, 'frame_{i}.lammps.data'
        :param type_map: the type map of atom type, for example, [O,H]
        :param sel_type: the selected type of atom, for example, [0] means atom type 0, aka O is selected
        :param sys_charge_map: the charge map of atom in system, for example, [6, 1]
        :param model_charge_map: the charge map of atom in model, for example, [-8]
        """

        from ai2_kit.domain.dplr import dump_dplr_lammps_data
        ensure_dir(filename.format(i=0))
        for i, atoms in enumerate(self._atoms_arr):
            with open(filename.format(i=i), 'w') as f:
                dump_dplr_lammps_data(f, atoms=atoms, type_map=type_map, sel_type=sel_type,
                                      sys_charge_map=sys_charge_map, model_charge_map=model_charge_map)

    def to_dpdata(self, labeled=False):
        """
        convert to dpdata format and use dpdata tool to handle

        :param labeled: if True, use dpdata.LabeledSystem, else use dpdata.System
        """
        from .dpdata import dpdata, DpdataTool
        System = dpdata.LabeledSystem if labeled else dpdata.System
        systems = [System(atom, fmt='ase/structure') for atom in self._atoms_arr]
        return DpdataTool(systems=systems)

    def _read(self, filename: str, **kwargs):
        kwargs.setdefault('index', ':')
        data = ase.io.read(filename, **kwargs)
        if not isinstance(data, list):
            data = [data]
        self._atoms_arr += data

    def _write(self, filename: str, atoms_list, **kwargs):
        assert len(atoms_list) > 0, 'No atoms to write'
        fmt = kwargs.get('format', None)
        if fmt == 'lammps-dump-text':
            return self._write_lammps_dump_text(filename, atoms_list, **kwargs)
        if fmt == 'cp2k-inc' or (fmt is None and filename.endswith('.inc')):
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            if len(atoms_list) != 1:
                raise ValueError('cp2k-inc only support single frame')
            return self._write_cp2k_inc(filename, atoms_list[0])
        return ase.io.write(filename, atoms_list, **kwargs)

    def _write_cp2k_inc(self, filename: str, atoms: Atoms):
        with open(filename, 'w') as f:
            dump_coord_n_cell(f, atoms)

    def _write_lammps_dump_text(self, filename: str, atoms_list, **kwargs):
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        type_map = kwargs.get('type_map')
        lines = []
        for i, atoms in enumerate(atoms_list):
            lines += [
                "ITEM: TIMESTEP", str(i),
                "ITEM: NUMBER OF ATOMS", str(len(atoms)),
            ]
            # It is kind of hacky to set boundary and cell in this way.
            # TODO: refactor it in the future.
            pbc = atoms.get_pbc()
            boundary =  ' '.join(['pp' if b else 'ff' for b in pbc])
            cell = atoms.get_cell()
            lines += [
                f"ITEM: BOX BOUNDS xy xz yz {boundary}",
                "%.16e %.16e %.16e" % (0, cell[0][0], 0),
                "%.16e %.16e %.16e" % (0, cell[1][1], 0),
                "%.16e %.16e %.16e" % (0, cell[2][2], 0),
                # TODO: support more properties, forces for example
                "ITEM: ATOMS id type x y z",
            ]

            for j, atom in enumerate(atoms):
                pos = atom.position
                no = atom.number if type_map is None else type_map.index(atom.symbol) + 1
                lines.append("%3d %2s %10.5f %10.5f %10.5f" % (j+1, no , pos[0], pos[1], pos[2]))

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
