from collections import namedtuple
from typing import List, Tuple
from ase import Atoms

import ase.io
import os

import pandas as pd

from ai2_kit.core.util import expand_globs, slice_from_str
from ai2_kit.core.log import get_logger

logger = get_logger(__name__)


MdArray = List[Tuple[pd.DataFrame, str]]


class ModelDevi:
    """
    A tool to analyze the deviation of model from model deviation file of deepmd-kit
    """

    def __init__(self):
        self._items:List[dict] = []
        self._stats = {}

    def read(self, *dir_or_glob: str, traj_file: str, md_file='model_devi.out', ignore_error=False, **kwargs):
        """
        read model deviation from file, support multiple files and glob pattern

        :param dir_or_glob: path or glob pattern to locate data path
        :param traj_file: trajectory file name to read, relative to data path, e.g dump.lammpstrj
        :param md_file: model deviation file name to read, default is model_devi.out
        :param ignore_error: ignore error when reading files
        :param kwargs: other arguments for ase.io.read
        """
        kwargs['index'] = ':'
        dirs = expand_globs(dir_or_glob)
        if len(dirs) == 0:
            raise FileNotFoundError(f'No file found for {dir_or_glob}')
        for data_dir in dirs:
            try:
                traj_file_path = os.path.join(data_dir, traj_file)
                atoms: Atoms = ase.io.read(traj_file_path, **kwargs)  # type: ignore
                md_file_path = os.path.join(data_dir, md_file)
                with open(md_file_path, 'r') as f:
                    f.seek(1)  # skip the leading '#'
                    md_df = pd.read_csv(f, delim_whitespace=True, header=0)
                assert len(atoms) == len(md_df), 'The length of atoms and model deviation should be the same'
            except Exception as e:
                if ignore_error:
                    logger.exception(f'Error when reading {data_dir}: {e}')
                    continue
                raise e
            self._items.append({
                'atoms': atoms,
                'md_df': md_df,
                'dir': data_dir,
                'md_file': md_file_path,
                'traj_file': traj_file_path,
            })
        return self


    def grade(self, lo: float, hi: float, col: str = 'max_devi_f'):
        """
        Grade atoms based on the deviation of model into 3 levels: good, decent, poor

        the grade is based on the column of max_devi_f by default,
        if the value is below lo, the level is good,
        if the value is above hi, the level is poor,
        otherwise, the level is decent

        :param lo: the lower bound of decent level, below this value is good
        :param hi: the upper bound of decent level, above this value is poor
        :param col: the column of model deviation to grade, default is max_devi_f
        """
        for item in self._items:
            df = item['md_df']
            if col not in df.columns:
                raise ValueError(f"Unknown model deviation column: {col}")
            good = df[col] < lo
            decent = (df[col] >= lo) & (df[col] <= hi)
            poor = df[col] > hi
            self._stats[item['md_file']] = {
                'g': good.sum(),
                'd': decent.sum(),
                'p': poor.sum(),
                'all': len(df),
            }
            item['good'] = good
            item['decent'] = decent
            item['poor'] = poor
        return self

    def slice(self, expr: str):
        """
        Slice the atoms based on the expression

        :param expr: the expression to slice the atoms, e.g. '0:10'
        """
        _slice = slice_from_str(expr)
        for item in self._items:
            item['md_df'] = item['md_df'].iloc[_slice]
            item['atoms'] = item['atoms'][_slice]
            if 'good' in item:
                item['good'] = item['good'].iloc[_slice]
            if 'decent' in item:
                item['decent'] = item['decent'].iloc[_slice]
            if 'poor' in item:
                item['poor'] = item['poor'].iloc[_slice]
        return self

    def dump_stats(self, out_file: str = '', fmt='tsv'):
        """
        Dump the statistics of grading

        :param out_file: the file path to write the statistics
        :param fmt: the format of table, default is tsv
        """
        from tabulate import tabulate

        headers = ['file', 'total', 'good', 'decent', 'poor', 'good%', 'decent%', 'poor%']
        table = []
        for file, stats in self._stats.items():
            total = stats['all']
            g = stats['g']
            d = stats['d']
            p = stats['p']
            g_pct = '{:.2%}'.format(g / total)
            d_pct = '{:.2%}'.format(d / total)
            p_pct = '{:.2%}'.format(p / total)
            table.append([file, total, g, d, p, g_pct, d_pct, p_pct])
        stats_report = tabulate(table, headers=headers, tablefmt=fmt)
        if out_file:
            with open(out_file, 'w') as f:
                f.write(stats_report)
        else:
            logger.info(f'model deviation statistics:\n{stats_report}')
        return self

    def write(self, file_path: str, inplace=False, level='decent',
              ignore_error=False, **kwargs):
        """
        Write atoms to file based on the level of grading

        :param file_path: the file path to write
        :param inplace: if True, write to the original data directory
        :param level: the level of grading to write, default is decent
        :param ignore_error: ignore error when writing files
        :param kwargs: other arguments for ase.io.write
        """
        atoms_arr = []
        for item in self._items:
            data_dir = item['dir']
            out_file = os.path.join(data_dir, file_path)
            atoms = item['atoms']
            sel = item[level]
            atoms = [atoms[i] for i in sel.index[sel]]
            if inplace:
                try: # write in place
                    ase.io.write(out_file, atoms, **kwargs)
                except Exception as e:
                    if ignore_error:
                        logger.exception(f'Error when writing {out_file}: {e}')
                    else:
                        raise e
            else:
                atoms_arr += atoms
        if not inplace:
            ase.io.write(file_path, atoms_arr, **kwargs)
        return self

    def done(self):
        return None
