import ase.io
import fire
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from ai2_kit.core.log import get_logger
from ai2_kit.core.util import merge_dict, wait_for_change
from ai2_kit import res
from ai2_kit.domain.cp2k import dump_coord_n_cell
from typing import Optional, Literal
from ase import Atoms, Atom
from string import Template
import asyncio
import re
import os
import json


logger = get_logger(__name__)

DEEPMD_DEFAULT_TEMPLATE = os.path.join(res.DIR_PATH, 'catalysis/deepmd.json')
MLP_TRAINING_TEMPLATE = os.path.join(res.DIR_PATH, 'catalysis/mlp-training.yml')

CP2K_DEFAULT_TEMPLATE = os.path.join(res.DIR_PATH, 'catalysis/cp2k.inp')

CP2K_SCF_SEMICONDUCTOR = """\
        # CONFIGURATION FOR SEMICONDUCTOR
        &SCF
            SCF_GUESS RESTART
            EPS_SCF 3e-07
            MAX_SCF 50
            &OUTER_SCF
                EPS_SCF 3e-07
                MAX_SCF 20
            &END OUTER_SCF
            &OT
                MINIMIZER CG
                PRECONDITIONER FULL_SINGLE_INVERSE
                ENERGY_GAP 0.1
            &END OT
        &END SCF
        # END CONFIGURATION FOR SEMICONDUCTOR"""

CP2K_SCF_METAL = """\
        # CONFIGURATION FOR METAL
        &SCF
            SCF_GUESS RESTART
            EPS_SCF 3e-07
            MAX_SCF 500
            ADDED_MOS 500
            CHOLESKY INVERSE
            &SMEAR
                METHOD FERMI_DIRAC
                ELECTRONIC_TEMPERATURE [K] 300
            &END SMEAR
            &DIAGONALIZATION
                ALGORITHM STANDARD
            &END DIAGONALIZATION
            &MIXING
                METHOD BROYDEN_MIXING
                ALPHA 0.3
                BETA 1.5
                NBROYDEN 14
            &END MIXING
        &END SCF
        # END CONFIGURATION FOR METAL"""

CP2K_MOTION_TEMPLATE = """\
&MOTION
  &MD
    ENSEMBLE NVT
    STEPS       $steps
    TIMESTEP    $timestep
    TEMPERATURE $temp
    &THERMOSTAT
       TYPE CSVR
       REGION MASSIVE
       &CSVR
          TIMECON [fs] 100.0
       &END
    &END
  &END MD
  &PRINT
   &TRAJECTORY
     &EACH
       MD 1
     &END EACH
   &END TRAJECTORY
   &VELOCITIES
     &EACH
       MD 1
     &END EACH
   &END VELOCITIES
   &FORCES
     &EACH
       MD 1
     &END EACH
   &END FORCES
   &RESTART_HISTORY
     &EACH
       MD 1000
     &END EACH
   &END RESTART_HISTORY
   &RESTART
     BACKUP_COPIES 3
     &EACH
       MD 1
     &END EACH
   &END RESTART
  &END PRINT
&END MOTION"""

CP2K_SCF_TABLE = {
    'metal': CP2K_SCF_METAL,
    'semi': CP2K_SCF_SEMICONDUCTOR,
}

CP2K_ACCURACY_TABLE = {
    'high': {'cutoff': 1000, 'rel_cutoff': 90 },
    'medium': {'cutoff': 800, 'rel_cutoff': 70 },
    'low': {'cutoff': 600, 'rel_cutoff': 50 },
}

class ConfigBuilder:

    def __init__(self):
        self._atoms: Optional[Atoms] = None

    def load_system(self, file: str, **kwargs):
        """
        Loading system to memory using ASE
        """
        assert self._atoms is None, 'atoms is already loaded'
        self._atoms = ase.io.read(file, **kwargs)  # type: ignore
        return self

    def gen_mlp_training_input(self,
                               out_dir: str = 'out',
                               template_file: str = MLP_TRAINING_TEMPLATE):
        # Read yaml file as text so that the comments are preserved
        with open(template_file, 'r') as fp:
            text = fp.read()
        # Generate the type_map and mass_map automatically
        assert self._atoms is not None, 'atoms must be loaded first'
        type_map = list(set(self._atoms.get_chemical_symbols()))
        mass_map = [Atom(symbol).mass for symbol in type_map]

        out_data = Template(text).substitute(
            type_map=json.dumps(type_map),
            mass_map=json.dumps(mass_map),
        )
        os.makedirs(out_dir, exist_ok=True)
        mlp_training_input_path = os.path.join(out_dir, 'mlp-training.yml')
        with open(mlp_training_input_path, 'w', encoding='utf-8') as fp:
            fp.write(out_data)


    def gen_deepmd_input(self,
                         out_dir: str = 'out',
                         steps: int = 10000,
                         template_file: str = DEEPMD_DEFAULT_TEMPLATE):
        with open(template_file, 'r') as fp:
            data = json.load(fp)
        os.makedirs(out_dir, exist_ok=True)

        data['training']['numb_steps'] = steps
        deepmd_input_path = os.path.join(out_dir, 'deepmd.json')
        with open(deepmd_input_path, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, indent=4)


    def gen_plumed_input(self, out_dir: str = 'out'):
        """
        Generate Plumed input files
        Args:
            out_dir: output directory, plumed.dat will be generated in this directory
        """
        assert self._atoms is not None, 'atoms must be loaded first'
        plumed_input = [ 'UNITS LENGTH=A' ]
        # define atoms groups by element, e.g:
        # Ag: GROUP ATOMS=1,2,3,4,5,6,7,8,9,10
        # O: GROUP ATOMS=11,12,13,14,15,16,17,18,19,20
        element2ids = {}
        for i, element in enumerate(self._atoms.get_chemical_symbols(), start=1):
            element2ids.setdefault(element, []).append(i)
        for element, ids in element2ids.items():
            plumed_input.append(f'{element}: GROUP ATOMS={",".join(map(str, ids))}')

        # define reaction coordinate
        plumed_input.extend([
            '#define more groups if needed',
            '',
            '# define reaction coordinates, e.g. CV1, CV2, ...',
            '# you may define as many as you want',
            'CV1:',
            '',
            '# define sampling method: metadynamics',
            'metad: METAD ARG=CV1 SIGMA=0.1 HEIGHT=5 PACE=100 TEMP=1000 FILE=HILLS',
            '# define more commands if you need',
            '',
            '# print CVs',
            'PRINT STRIDE=10 ARG=CV1,metad.bias FILE=COLVAR',
        ])
        os.makedirs(out_dir, exist_ok=True)
        plumed_input_path = os.path.join(out_dir, 'plumed.dat')
        with open(plumed_input_path, 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(plumed_input))


    def gen_lammps_input(self):
        ... # TODO


    def gen_report(self):
        ... # TODO


    def gen_cp2k_input(self,
                       basic_set_file: str = 'BASIS_MOLOPT',
                       potential_file: str = 'GTH_POTENTIALS',
                       out_dir: str = 'out',
                       style: Literal['metal', 'semi'] = 'metal',
                       accuracy: Literal['high', 'medium', 'low'] = 'medium',
                       aimd: bool = False,
                       temp: float = 330.,
                       steps: int = 1000,
                       timestep: float = 0.5,
                       parameter_file: str = 'dftd3.dat',
                       template_file: str = CP2K_DEFAULT_TEMPLATE,):
        """
        Generate CP2K input files

        You should call `load_system` first to load system into memory.
        And you may also need to ensure the basic set and potential files
        you want to use are available in CP2K_DATA_DIR,
        or else you have to specify the full path instead of their file name.

        Args:
            basic_set_file: basic set file, can be path or file in CP2K_DATA_DIR
            potential_file: potential file, can be path or file in CP2K_DATA_DIR
            out_dir: output directory, cp2k.inp and coord_n_cell.inc will be generated in this directory
            style: 'metal' or 'semi'
            accuracy: 'high', 'medium' or 'low'
            aimd: whether to run AIMD
            temp: temperature for AIMD
            steps: steps for AIMD
            timestep: timestep for AIMD
            parameter_file: parameter file, can be path or file in CP2K_DATA_DIR
            template_file: template file, no need to change in most cases
        """

        assert self._atoms is not None, 'atoms must be loaded first'
        # get available basic set and potential
        with open(find_cp2k_data_file(basic_set_file), 'r') as fp:
            basic_set_table = parse_cp2k_basic_set_potential(fp)
        with open(find_cp2k_data_file(potential_file), 'r') as fp:
            potential_table = parse_cp2k_basic_set_potential(fp)
        with open(template_file, 'r') as fp:
            template = fp.read()
        # build kinds
        elements = set(self._atoms.get_chemical_symbols())

        kinds = ''
        total_ve = 0  # Valence Electron

        for element in elements:
            basic_set_all = basic_set_table.get(element)
            basic_set = basic_set_all[-1]
            logger.info(f'BASIC_SET {basic_set} is selected for {element}')

            potential_all = potential_table.get(element)
            potential = next(p for p in potential_all if 'PBE' in p)
            logger.info(f'POTENTIAL {potential} is selected for {element}')
            kinds += '\n'.join([
                f'    &KIND {element}',
                f'        BASIS_SET {basic_set}',
                f'        # All available BASIS_SET:',
                f'        # {" ".join(basic_set_all)}',
                f'        POTENTIAL {potential}',
                f'        # All available POTENTIAL:',
                f'        # {" ".join(potential_all)}',
                f'    &END KIND',
                '', # This empty line is required
            ])
            # get ve from the last number of potential
            # for example, if the potential is GTH-OLYP-q6
            # then the ve is 6
            ve = int(re.match(r'.*?(\d+)$', potential).group(1))
            total_ve += ve * self._atoms.get_chemical_symbols().count(element)
        logger.info("total valence electron of the system: %d" % total_ve)

        motion = ''
        if aimd:
            motion = Template(CP2K_MOTION_TEMPLATE).substitute(
                steps=steps,
                timestep=timestep,
                temp=temp,
            )

        template_vars = {
            'run_type': 'MD' if aimd else 'ENERGY_FORCE',
            'basic_set_file': os.path.abspath(basic_set_file),
            'potential_file': os.path.abspath(potential_file),
            'parameter_file': parameter_file,
            'uks': 'T' if total_ve % 2 else 'F',
            'kinds': kinds,
            'motion': motion,
            'scf': CP2K_SCF_TABLE[style],
            **CP2K_ACCURACY_TABLE[accuracy],
        }

        cp2k_input = Template(template).substitute(**template_vars)
        os.makedirs(out_dir, exist_ok=True)
        cp2k_input_path = os.path.join(out_dir, 'cp2k.inp')
        with open(cp2k_input_path, 'w') as fp:
            fp.write(cp2k_input)
        logger.info(f'CP2K input file is generated at {cp2k_input_path}')

        coord_n_cell_path = os.path.join(out_dir, 'coord_n_cell.inc')
        with open(coord_n_cell_path, 'w') as fp:
            dump_coord_n_cell(fp, self._atoms)
        logger.info(f'coord_n_cell.inc is generated at {coord_n_cell_path}')


def find_cp2k_data_file(file: str):
    """
    find CP2K data file in CP2K_DATA_DIR
    """
    if not os.path.exists(file):
        cp2k_data_dir = os.environ.get('CP2K_DATA_DIR')
        if cp2k_data_dir is None:
            raise FileNotFoundError(f'Cannot find {file}')
        file = os.path.join(cp2k_data_dir, file)
        if not os.path.exists(file):
            raise FileNotFoundError(f'Cannot find {file}')
    return file


def parse_cp2k_basic_set_potential(fp):
    """
    get all available basic set and potential from CP2K data file
    """
    parsed = {}
    for line in fp:
        line: str = line.strip()
        if line.startswith('#'):
            continue # skip comment
        tokens = line.split()
        if 0 == len(tokens):
            continue  # skip empty line
        if re.match(r'^[A-Z][a-z]*$', tokens[0]):
            parsed.setdefault(tokens[0], []).append(tokens[1])
    return parsed


def inspect_explore_result(lammps_dir: str, save_to: Optional[str]=None):
    model_devi_file = os.path.join(lammps_dir, 'model_devi.out')
    colvar_file = os.path.join(lammps_dir, 'COLVAR')
    lammps_input_file = os.path.join(lammps_dir, 'lammps.input')

    # read TEMP, N_STEPS lammps input file
    with open(lammps_input_file, 'r') as fp:
        for line in fp:
            line = line.strip()
            if line.startswith('variable'):
                tokens = line.split()
                if tokens[1] == 'TEMP':
                    temp = float(tokens[3])

    # draw colvar
    colvar = np.loadtxt(colvar_file, skiprows=1)
    # read column names from the first line of colvar file
    with open(colvar_file, 'r') as fp:
        col_names = fp.readline().strip().split()[2:]
    # x axis is the first column, and y axis is the rest columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].set_title(f'COLVAR @ TEMP {temp}K')
    axs[0].set_xlabel(r'$time / s$')
    for i, _col in enumerate(col_names[1:-1], start=1):
        axs[0].plot(colvar[:, 0], colvar[:, i])
    axs[0].legend(col_names[1:])
    axs[0].grid()

    # draw model_devi
    model_devi = np.loadtxt(model_devi_file, skiprows=1)  # the 4 colum is max_devi_f
    axs[1].set_title(f'Model Devi at @ TEMP {temp}K')
    axs[1].set_xlabel('step')
    axs[1].set_ylabel('max_devi_f')
    axs[1].plot(model_devi[:, 0], model_devi[:, 4])
    axs[1].grid()

    if save_to is None:
        plt.show()
    else:
        fig.savefig(save_to, dpi=300, bbox_inches='tight')


class CmdEntries:
    def build_config(self):
        """
        Build config file for catalyst
        """
        return ConfigBuilder


class UiHelper:

    def __init__(self) -> None:
        self.aimd_schema_path = os.path.join(res.DIR_PATH, 'catalysis/gen-cp2k-aimd.formily.json')
        self.aimd_form = None
        self.aimd_value = None

        self.training_schema_path = os.path.join(res.DIR_PATH, 'catalysis/gen-training.formily.json')
        self.training_form = None
        self.training_value = None

    def gen_aimd_config(self, cp2k_search_path: str = './', out_dir: str = './'):
        from jupyter_formily import Formily
        from IPython.display import display
        if self.aimd_form is None:
            with open(self.aimd_schema_path, 'r') as fp:
                schema = json.load(fp)
            # patch for FilePicker
            schema = merge_dict(schema, {'schema': {'properties': {
                'system_file':    {'x-component': 'FilePicker', 'x-component-props': {'init_path': cp2k_search_path}},
                'basic_set_file': {'x-component': 'FilePicker', 'x-component-props': {'init_path': cp2k_search_path}},
                'potential_file': {'x-component': 'FilePicker', 'x-component-props': {'init_path': cp2k_search_path}},
                'parameter_file': {'x-component': 'FilePicker', 'x-component-props': {'init_path': cp2k_search_path}},
                'out_dir':        {'x-component': 'FilePicker', 'default': out_dir },
            }}}, quiet=True)
            self.aimd_form = Formily(schema, options={
                "modal_props": {"title": "Config CP2K AIMD", "width": "60vw","style": {"max-width": "800px"}, "styles": {"body": {"max-height": "70vh", "overflow-y": "auto"}}}
            })

        display(self.aimd_form)
        async def _task():
            self.aimd_value = await wait_for_change(self.aimd_form, 'value')
            try:
                config_builder = ConfigBuilder()
                print('Start to generate AMID input files...')
                system_file = self.aimd_value.pop('system_file')
                config_builder.load_system(system_file)
                config_builder.gen_cp2k_input(**self.aimd_value)
                print('Success!')  # TODO: Send a toast message
            except Exception as e:
                print('Failed!', e)  # TODO: Send a alert message
        asyncio.ensure_future(_task())


    def gen_training_config(self):
        ...


_UI_HELPER = None
def get_the_ui_helper():
    """
    Singleton for UiHelper
    """
    global _UI_HELPER  # pylint: disable=global-statement
    if _UI_HELPER is None:
        _UI_HELPER = UiHelper()
    return _UI_HELPER


def cli_main():
    fire.Fire(CmdEntries)


if __name__ == '__main__':
    fire.Fire(ConfigBuilder)
