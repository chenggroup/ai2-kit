import random


def get_ensemble(ensemble: str, group='all'):
    lines = []
    if ensemble in ('npt', 'npt-i', 'npt-iso',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('npt-a', 'npt-aniso',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('npt-t', 'npt-tri',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('npt-x',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} x ${PRES} ${PRES} ${TAU_P} y 0 0 ${TAU_P} z 0 0 ${TAU_P}')
    elif ensemble in ('npt-y',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} y ${PRES} ${PRES} ${TAU_P} x 0 0 ${TAU_P} z 0 0 ${TAU_P}')
    elif ensemble in ('npt-z',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} z ${PRES} ${PRES} ${TAU_P} x 0 0 ${TAU_P} y 0 0 ${TAU_P}')
    elif ensemble in ('nvt',):
        lines.append('fix 1 %(group)s nvt temp ${TEMP} ${TEMP} ${TAU_T}')
    elif ensemble in ('nve',):
        lines.append('fix 1 %(group)s nve')
    elif ensemble in ('csvr',):
        lines.append('fix 1 %(group)s nve')
        lines.append('fix 2 %(group)s temp/csvr ${TEMP} ${TEMP} ${TIME_CONST} %(seed)d')
    else:
        raise ValueError('unknown ensemble: ' + ensemble)
    return '\n'.join(lines) % {'group': group, 'seed': random.randrange(10^6 - 1) + 1}
