from ai2_kit.core.queue_system import inject_cmd_to_script
from ai2_kit.core.util import dict_remove_dot_keys
from ai2_kit.domain.dpff import dump_dplr_lammps_data
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
        with open(data_dir / 'h2o.lammps.data', 'r') as f:
            # f.write(fp.getvalue())
            self.assertEqual(fp.getvalue(), f.read())

    def test_get_type_template_vars(self):

        type_map = ['H', 'O']
        mass_map = [1., 16.]
        type_alias = {
            'H': ['H_mol', 'H_null'],
            'O': ['O_null'],
        }
        sel_type = [1]
        ret = get_types_template_vars(type_map, mass_map, type_alias, sel_type)
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
