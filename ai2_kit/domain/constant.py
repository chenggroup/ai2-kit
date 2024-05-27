DP_CHECKPOINT_FILE = 'model.ckpt'
DP_DISP_FILE = 'lcurve.out'
DP_PROFILING_FILE = 'timeline.json'
DP_INPUT_FILE = 'input.json'
DP_FROZEN_MODEL = 'frozen_model.pb'
DP_FINAL_MODEL = 'final_model.pb'
DP_ORIGINAL_MODEL = 'original_model.pb'

MODEL_DEVI_OUT = 'model_devi.out'
MODEL_DEVI_NEU_OUT = 'model_devi_neu.out'
MODEL_DEVI_RED_OUT = 'model_devi_red.out'

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

# LAMMPS

_DEFAULT_LAMMPS_TOP = '''\
$$VARIABLES
# >>>>> EXTRA_VARS
$$EXTRA_VARS
# <<<<< EXTRA_VARS

# >>>>> INITIALIZE
$$INITIALIZE
# <<<<< INITIALIZE

# >>>>> POST_INIT
$$POST_INIT
# <<<<< POST_INIT

# >>>>> READ_DATA
$$READ_DATA
# <<<<< READ_DATA

# >>>>> MASS_MAP
$$MASS_MAP
# <<<<< MASS_MAP


# >>>>> POST_READ_DATA
$$POST_READ_DATA
# <<<<< POST_READ_DATA

# >>>>> SET_ATOM_TYPE
$$SET_ATOM_TYPE
# <<<<< SET_ATOM_TYPE

# >>>>> GROUPS
$$DPFF_GROUPS
# <<<<< GROUPS
'''

_DEFAULT_LAMMPS_BOTTOM = '''\

# >>>>> POST_FORCE_FIELD
$$POST_FORCE_FIELD
# <<<<< POST_FORCE_FIELD

$$SIMULATION

# >>>>> POST_SIMULATION
$$POST_SIMULATION
# <<<<< POST_SIMULATION

$$RUN

# >>>>> POST_RUN
$$POST_RUN
# <<<<< POST_RUN
'''

_DP_FORCE_FIELD = '''\
pair_style deepmd $$DP_MODELS out_freq ${THERMO_FREQ} out_file model_devi.out $$DP_OPT
pair_coeff * *
'''

_DP_FEP_REDOX_FORCE_FIELD = '''\
variable LAMBDA_i equal 1-v_LAMBDA_f

pair_style  hybrid/overlay &
            deepmd $$DP_MODELS out_freq ${THERMO_FREQ} out_file model_devi_ini.out $$FEP_INI_DP_OPT &
            deepmd $$DP_MODELS out_freq ${THERMO_FREQ} out_file model_devi_fin.out $$FEP_FIN_DP_OPT &
            $$PAIR_STYLE_EXT

# >>>>> PAIR_COEFF_EXT
$$PAIR_COEFF_EXT
# <<<<< PAIR_COEFF_EXT

pair_coeff  * * deepmd 1
pair_coeff  * * deepmd 2

fix PES_Sampling ${DEFAULT_GROUP} adapt 0 &
    pair deepmd:1 scale * * v_LAMBDA_i &
    pair deepmd:2 scale * * v_LAMBDA_f
'''

_DP_FEP_PKA_FORCE_FIELD = '''\
variable LAMBDA_i equal 1-v_LAMBDA_f

# >>>>> FEP_GROUPS
$$FEP_GROUPS
# <<<<< FEP_GROUPS

pair_style  hybrid/overlay &
            deepmd $$DP_MODELS out_freq ${THERMO_FREQ} out_file model_devi_ini.out $$FEP_INI_DP_OPT &
            deepmd $$DP_MODELS out_freq ${THERMO_FREQ} out_file model_devi_fin.out $$FEP_FIN_DP_OPT &
            $$PAIR_STYLE_EXT

# >>>>> PAIR_COEFF_EXT
$$PAIR_COEFF_EXT
# <<<<< PAIR_COEFF_EXT

pair_coeff  * * deepmd 1 $$FEP_INI_SPECORDER
pair_coeff  * * deepmd 2 $$FEP_FIN_SPECORDER

fix PES_Sampling ${DEFAULT_GROUP} adapt 0 &
    pair deepmd:1 scale * * v_LAMBDA_i &
    pair deepmd:2 scale * * v_LAMBDA_f
'''

_DPFF_CONFIG = '''\
bond_style    zero
bond_coeff    *
special_bonds lj/coul 1 1 1 angle no

# EWARD_BETA is `ewald_beta` in the training setup
# KMESH should be set by the user
kspace_style    pppm/dplr 1e-5
kspace_modify   gewald ${EWALD_BETA} diff ik mesh ${KMESH} ${KMESH} ${KMESH}

fix dplr all dplr $$DP_MODELS_0 type_associate $$DPLR_TYPE_ASSOCIATION bond_type 1 max_iter 10 efield ${EFIELD}
'''

PRESET_LAMMPS_INPUT_TEMPLATE = {
    'default'  : '\n'.join([_DEFAULT_LAMMPS_TOP, _DP_FORCE_FIELD, _DEFAULT_LAMMPS_BOTTOM]),
    'custom-ff': '\n'.join([_DEFAULT_LAMMPS_TOP, '$$CUSTOM_FF', _DEFAULT_LAMMPS_BOTTOM]),
    'fep-pka'  : '\n'.join([_DEFAULT_LAMMPS_TOP, _DP_FEP_PKA_FORCE_FIELD, _DEFAULT_LAMMPS_BOTTOM]),
    'fep-redox': '\n'.join([_DEFAULT_LAMMPS_TOP, _DP_FEP_REDOX_FORCE_FIELD, _DEFAULT_LAMMPS_BOTTOM ]),
    'dpff'     : '\n'.join([_DEFAULT_LAMMPS_TOP, _DP_FORCE_FIELD, _DPFF_CONFIG, _DEFAULT_LAMMPS_BOTTOM]),
}
