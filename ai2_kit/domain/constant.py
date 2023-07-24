DP_CHECKPOINT_FILE = 'model.ckpt'
DP_DISP_FILE = 'lcurve.out'
DP_PROFILING_FILE = 'timeline.json'
DP_INPUT_FILE = 'input.json'
DP_FROZEN_MODEL = 'frozen_model.pb'
DP_ORIGINAL_MODEL = 'original_model.pb'

MODEL_DEVI_OUT = 'model_devi.out'
MODEL_DEVI_NEU_OUT = 'model_devi_neu.out'
MODEL_DEVI_RED_OUT = 'model_devi_red.out'

LAMMPS_TRAJ_DIR = 'traj'
LAMMPS_TRAJ_SUFFIX = '.lammpstrj'

SELECTOR_OUTPUT = 'selector_output'


DEFAULT_LASP_IN = {
    "Run_type": 15,
    "SSW.SSWsteps": 20,
    "SSW.Temp": 300,
    "SSW.NG": 10,
    "SSW.NG_cell": 8,
    "SSW.ds_atom": 0.6,
    "SSW.ds_cell": 0.5,
    "SSW.ftol": 0.05,
    "SSW.strtol": 0.05,
    "SSW.MaxOptstep": 300,
    "SSW.printevery": "T",
    "SSW.printselect": 0,
    "SSW.printdelay": -1,
    "SSW.output": "F"
}


DEFAULT_LAMMPS_TEMPLATE_FOR_DP_SSW = """\
units           metal
boundary        p p p
atom_style      atomic
atom_modify map yes

$$read_data_section

$$force_field_section

compute peratom all pressure NULL virial
"""


DEFAULT_ASAP_ASCF_DESC = {
    'preset': None,
    # those params following the convention of dscribe
    # https://singroup.github.io/dscribe/latest/tutorials/descriptors/acsf.html
    'r_cut': 3.5,

    'reducer_type': 'average',
    'element_wise': False,
    'zeta': 1,
}


DEFAULT_ASAP_SOAP_DESC = {
    'preset': None,

    # those params following the convention of dscribe
    # https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html
    'r_cut': 3.5,
    'n_max': 6,
    'l_max': 6,
    'sigma': 0.5,

    'crossover': False,
    'rbf': 'gto',

    'reducer_type': 'average',
    'element_wise': False,
    'zeta': 1,
}

DEFAULT_ASAP_PCA_REDUCER = {
    'type': 'PCA',
            'parameter': {
                'n_components': 3,
                'scalecenter': True,
            }
}
