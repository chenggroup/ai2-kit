from ai2_kit.core.queue_system import inject_cmd_to_script
from ai2_kit.core.util import dict_remove_dot_keys, expand_globs, num_text_split, nat_sort, slice_from_str
from ai2_kit.domain.dplr import dump_dplr_lammps_data
from ai2_kit.domain.lammps import get_types_template_vars, get_ensemble
from unittest import TestCase
from pathlib import Path

data_dir = Path(__file__).parent / 'data-sample'

SLURM_SCRIPT_HEADER = """\
#!/bin/bash

#SBATCH -N 1
#SBATCH --partition cpu
"""

MASS_MAP_SECTION = """\
variable    H                equal 1
variable    O                equal 2
variable    _X_0             equal 3
variable    H_mol            equal 4
variable    H_null           equal 5
variable    O_null           equal 6

mass ${H} 1.0
mass ${O} 16.0
mass ${_X_0} 1.0
mass ${H_mol} 1.0
mass ${H_null} 1.0
mass ${O_null} 16.0
""".strip()

DPFF_GROUPS = """\
group real_atom    type 1 2 4 5 6
group virtual_atom type 3
neigh_modify    every 10 delay 0 check no exclude group real_atom virtual_atom
""".strip()


class TestUtil(TestCase):
    def test_dict_remove_dot_keys(self):
        d = {
            'a': 1,
            '.b': 2,
            'c': {
                '.d': 4,
                'e': 5,

            }
        }
        expect = {
            'a': 1,
            'c': {
                'e': 5,
            }
        }
        dict_remove_dot_keys(d)
        self.assertEqual(d, expect)

    def test_inject_cmd_to_script(self):
        cmd = "echo $SLUMR_JOB_ID > hello.running"
        in_script = '\n'.join([
            SLURM_SCRIPT_HEADER,
            'echo hello',
        ])
        expect_out = '\n'.join([
            SLURM_SCRIPT_HEADER,
            cmd,
            'echo hello',
        ])
        out_script = inject_cmd_to_script(in_script, cmd)
        self.assertEqual(out_script, expect_out)

    def test_dump_dplr_lammps_data(self):
        import io
        import ase.io

        atoms = ase.io.read(data_dir / 'h2o.xyz', index=0)
        fp = io.StringIO()
        setattr(fp, 'name', 'lmp.data')
        dump_dplr_lammps_data(fp, atoms, type_map = ['H', 'O'], sel_type=[1],  # type: ignore
                              sys_charge_map=[0.0, 0.843], model_charge_map=[-1])
        fp.seek(0)
        with open(data_dir / 'h2o.lammps.data', 'r+' ) as f:
            self.assertMultiLineEqual(fp.read(), f.read())

    def test_get_type_template_vars(self):
        type_map = ['H', 'O']
        mass_map = [1., 16.]
        type_alias = {
            'H': ['H_mol', 'H_null'],
            'O': ['O_null'],
        }
        sel_type = [1]
        ret = get_types_template_vars(type_map, mass_map, type_alias, sel_type,[], ['H_null', 'O_null'])
        self.assertEqual(ret['SPECORDER'], 'H O H H O')
        self.assertEqual(ret['FEP_INI_SPECORDER'], 'H O H H O')
        self.assertEqual(ret['FEP_FIN_SPECORDER'], 'H O H NULL NULL')
        self.assertEqual(ret['DPFF_REAL_ATOM'], '1 2 4 5 6')
        self.assertEqual(ret['DPFF_VIRTUAL_ATOM'], '3')
        self.assertEqual(ret['DPFF_GROUPS'], DPFF_GROUPS)
        self.assertEqual(ret['DPLR_TYPE_ASSOCIATION'], '2 3')
        self.assertEqual(ret['MASS_MAP'], MASS_MAP_SECTION)

    def test_get_ensemble(self):
        self.assertEqual(get_ensemble('npt'), 'fix 1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}')
        self.assertEqual(get_ensemble('npt', 'real_atom'), 'fix 1 real_atom npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}')
        self.assertEqual(get_ensemble('npt', '{DEFAULT_GROUP}'), 'fix 1 {DEFAULT_GROUP} npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}')
        self.assertTrue(get_ensemble('csvr').startswith('fix 1 all nve\nfix 2 all temp/csvr ${TEMP} ${TEMP} ${TIME_CONST}'))

    def test_num_text_split(self):
        cases = [
            ('2 ft 3 in', (2, ' ft ', 3, ' in')),
            ('2ft 3in', (2, 'ft ', 3, 'in')),
            ('1.traj', (1, '.traj')),
            ('v1.2.3', ('v', 1, '.', 2, '.', 3)),
        ]
        for s, expect in cases:
            self.assertEqual(num_text_split(s), expect)

    def test_nat_sort(self):
        cases = [
            (['1', '10', '2'], ['1', '2', '10']),
            (['a1', 'a10', 'a2'], ['a1', 'a2', 'a10']),
        ]
        for s, expect in cases:
            self.assertListEqual(nat_sort(s), expect)

    def test_slice_from_str(self):
        # integer expression, the length should not affect the result
        int_cases = [
            ('1:10', slice(1, 10)),
            (':10', slice(None, 10)),
            ('10:', slice(10, None)),
            ('::2', slice(None, None, 2)),
            ('-10:', slice(-10, None)),
            (':', slice(None, None)),
        ]
        for expr, expect in int_cases:
            self.assertEqual(slice_from_str(expr), expect)
            self.assertEqual(slice_from_str(expr, 1000), expect)

        # fractional expression
        frac_cases = [
            (':0.9', 1000, slice(None, 900)),
            ('0.9:', 1000, slice(900, None)),
            ('-0.1:', 1000, slice(-100, None)),
            ('0.1:0.9', 1000, slice(100, 900)),
            (':.9', 1000, slice(None, 900)),
            (':0.9', 10, slice(None, 9)),
            ('0.9:', 10, slice(9, None)),
            ('0.95:', 1001, slice(951, None)),  # round instead of truncate
            (':0.5:2', 10, slice(None, 5, 2)),
        ]
        for expr, length, expect in frac_cases:
            self.assertEqual(slice_from_str(expr, length), expect)

        # `:f` and `f:` should partition the data exactly
        data = list(range(1001))
        head = data[slice_from_str(':0.9', len(data))]
        tail = data[slice_from_str('0.9:', len(data))]
        self.assertListEqual(head + tail, data)
        self.assertListEqual(tail, data[slice_from_str('-0.1:', len(data))])

        # invalid expressions
        with self.assertRaises(TypeError):
            slice_from_str(':0.9')  # length is unknown
        with self.assertRaises(ValueError):
            slice_from_str('1:2:3:4', 1000)
        with self.assertRaises(ValueError):
            slice_from_str('a:b', 1000)

    def test_expand_globs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / 'root'
            for data_dir in ('iter-0/a', 'iter-1/b'):
                (root / data_dir / 'set.000').mkdir(parents=True)
                (root / data_dir / 'type.raw').touch()
            (root / 'plain.txt').touch()

            # note: patterns must be built as string, as pathlib drops the `.` of `/./`
            def expand(*patterns: str, **kwargs):
                return expand_globs([f'{root}/{p}' for p in patterns], **kwargs)

            # plain glob is not affected
            self.assertListEqual(expand('*/*/type.raw'), [
                str(root / 'iter-0/a/type.raw'),
                str(root / 'iter-1/b/type.raw'),
            ])
            # `/./` navigates from the match to its parent dir
            self.assertListEqual(expand('**/type.raw/./..'), [
                str(root / 'iter-0/a'),
                str(root / 'iter-1/b'),
            ])
            # the suffix is joined literally and the result is normalized
            self.assertListEqual(expand('**/set.000/./../..'), [
                str(root / 'iter-0'),
                str(root / 'iter-1'),
            ])
            # multiple patterns are merged, and a literal `/./` in a plain path still works
            self.assertListEqual(expand('./plain.txt', '*/*/type.raw/./..'), [
                str(root / 'plain.txt'),
                str(root / 'iter-0/a'),
                str(root / 'iter-1/b'),
            ])
            # the suffix is joined without checking existence, that is up to the caller
            self.assertListEqual(expand('**/type.raw/./../missing'), [
                str(root / 'iter-0/a/missing'),
                str(root / 'iter-1/b/missing'),
            ])
            # a head matching nothing still yields nothing
            self.assertListEqual(expand('no-such-file'), [])
            self.assertListEqual(expand('no-such-dir/./..'), [])
            with self.assertRaises(FileNotFoundError):
                expand('no-such-file', raise_invalid=True)
