from typing import List, Tuple
from ase import Atoms

import pandas as pd

from ai2_kit.core.util import expand_globs
from ai2_kit.core.log import get_logger

logger = get_logger(__name__)

MdArray = List[Tuple[pd.DataFrame, str]]


class ModelDevi:
    """
    A tool to analyze the deviation of model from model deviation file of deepmd-kit
    """

    @staticmethod
    def md_arr_load(*files: str) -> MdArray:
        """
        Load model deviation files
        """
        md_arr = []
        for file in expand_globs(files):
            with open(file, 'r') as f:
                f.seek(1)  # skip the leading '#'
                df = pd.read_csv(f, delim_whitespace=True)
                md_arr.append((df, file))
        return md_arr

    def __init__(self, atoms_arr: List[Atoms], md_arr: MdArray):
        self._md_df = pd.concat([df for df, _ in md_arr])
        self._atoms_arr = atoms_arr
        self._md_arr = md_arr
        if len(atoms_arr) != len(self._md_df):
            raise ValueError("The size of atoms and model deviation records should be the same")
        self._stats = {}
        self._grade = {}

    def grade(self, lo: float, hi: float, col: str = 'max_devi_f'):
        """
        Grade atoms based on the deviation of model: the good, the bad and the ugly
        the grade is based on the column of max_devi_f by default,
        if the value is below lo, the level is good,
        if the value is above hi, the level is ugly,
        otherwise, the level is bad

        :param lo: the lower bound of good level
        :param hi: the upper bound of ugly level
        :param col: the column of model deviation to grade, default is max_devi_f
        """
        if col not in self._md_df.columns:
            raise ValueError(f"Unknown model deviation column: {col}")
        for df, file in self._md_arr:
            good = df[col] < lo
            bad = (df[col] >= lo) & (df[col] <= hi)
            ugly = df[col] > hi
            self._stats[file] = {
                'g': good.sum(),
                'b': bad.sum(),
                'u': ugly.sum(),
                'all': len(df),
            }
        self._grade['good'] = self._md_df[col] < lo
        self._grade['bad'] = (self._md_df[col] >= lo) & (self._md_df[col] <= hi)
        self._grade['ugly'] = self._md_df[col] > hi
        return self

    def dump_stats(self, out_file: str = '', fmt='tsv'):
        """
        Dump the statistics of grading
        """
        from tabulate import tabulate

        headers = ['file', 'total', 'good', 'bad', 'ugly', 'good%', 'bad%', 'ugly%']
        table = []

        overall = {
            'all': len(self._md_df),
            'g': self._grade['good'].sum(),
            'b': self._grade['bad'].sum(),
            'u': self._grade['ugly'].sum(),
        }
        for file, stats in [*self._stats.items(), ('', overall)]:
            total = stats['all']
            g = stats['g']
            b = stats['b']
            u = stats['u']
            g_pct = '{:.2%}'.format(g / total)
            b_pct = '{:.2%}'.format(b / total)
            u_pct = '{:.2%}'.format(u / total)
            table.append([file, total, g, b, u, g_pct, b_pct, u_pct])
        stats_report = tabulate(table, headers=headers, tablefmt=fmt)
        if out_file:
            with open(out_file, 'w') as f:
                f.write(stats_report)
        else:
            logger.info(f'model deviation statistics:\n{stats_report}')
        return self

    def to_ase(self, level):
        """
        Hand over the atoms to ase tool

        :param level: the grade level to hand over, valid values are good, bad, ugly
        """
        if level not in self._grade:
            raise ValueError(f"Unknown grade level: {level}")
        atoms_arr = [self._atoms_arr[i] for i in self._grade[level].index]
        from .ase import AseTool
        return AseTool(atoms_arr)
