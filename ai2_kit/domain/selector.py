from ai2_kit.core.executor import Executor
from ai2_kit.core.artifact import Artifact
from ai2_kit.core.log import get_logger
from ai2_kit.core.job import DummyFuture

from typing import List
from io import StringIO
from pydantic import BaseModel
from dataclasses import dataclass
import pandas as pd

import os

logger = get_logger(__name__)

class GeneralThresholdInputConfig(BaseModel):
    f_trust_lo: float
    f_trust_hi: float


@dataclass
class GeneralThresholdContext:
    executor: Executor

@dataclass
class GeneralThresholdOutput:
    candidates: List[Artifact]

@dataclass
class LammpsGeneralThresholdInput:
    config: GeneralThresholdInputConfig
    candidates: List[Artifact]
    model_devi_out_file: str

def lammps_general_threshold(input: LammpsGeneralThresholdInput, ctx: GeneralThresholdContext):

    f_trust_lo = input.config.f_trust_lo
    f_trust_hi = input.config.f_trust_hi
    col_force = 'avg_devi_f'
    logger.info('criteria: %f <= %s < %f ', f_trust_lo, col_force, f_trust_hi)

    for candidate in input.candidates:

        model_devi_out_file = os.path.join(candidate.url, input.model_devi_out_file)
        logger.info('start to analysis file: %s', model_devi_out_file)
        text = ctx.executor.load_text(model_devi_out_file)

        df = pd.read_csv(StringIO(text.lstrip('#')), delim_whitespace=True)
        # layout:
        #        step  max_devi_v  min_devi_v  avg_devi_v  max_devi_f  min_devi_f  avg_devi_f
        # 0        0    0.006793    0.000672    0.003490    0.143317    0.005612    0.026106
        # 1      100    0.006987    0.000550    0.003952    0.128178    0.006042    0.022608

        passed   = df[df[col_force] < f_trust_lo]
        selected = df[(df[col_force] >= f_trust_lo) & (df[col_force] < f_trust_hi)]
        rejected = df[df[col_force] >= f_trust_hi]

        logger.info('result: total: %d, passed: %d, selected: %d, rejected: %d', len(df), len(passed), len(selected), len(rejected))

        candidate.attrs['passed']   = passed.step.tolist()
        candidate.attrs['selected'] = selected.step.tolist()
        candidate.attrs['rejected'] = rejected.step.tolist()

    return DummyFuture(GeneralThresholdOutput(
        candidates=input.candidates,
    ))
