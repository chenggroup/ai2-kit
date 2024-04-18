from ai2_kit import res
import os

AI2CAT_RES_DIR = os.path.join(res.DIR_PATH, 'catalysis')

DEEPMD_DEFAULT_TEMPLATE = os.path.join(AI2CAT_RES_DIR, 'deepmd.json')
MLP_TRAINING_TEMPLATE   = os.path.join(AI2CAT_RES_DIR, 'mlp-training.yml')
CP2K_DEFAULT_TEMPLATE   = os.path.join(AI2CAT_RES_DIR, 'cp2k.inp')

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