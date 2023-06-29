from ai2_kit.core.artifact import Artifact
from ai2_kit.core.log import get_logger

from typing import List
from io import StringIO
from pydantic import BaseModel
from dataclasses import dataclass
import pandas as pd

from .data import LammpsOutputHelper
from .iface import ICllSelectorOutput, BaseCllContext
from .constant import LAMMPS_DUMPS_CLASSIFIED

logger = get_logger(__name__)

class ThresholdSelectorInputConfig(BaseModel):
    f_trust_lo: float
    f_trust_hi: float


@dataclass
class ThresholdSelectorContext(BaseCllContext):
    ...

@dataclass
class ThresholdSelectorOutput(ICllSelectorOutput):
    model_devi_data: List[Artifact]
    passing_rate: float

    def get_model_devi_dataset(self):
        return self.model_devi_data

    def get_passing_rate(self) -> float:
        return self.passing_rate

@dataclass
class ThresholdSelectorInput:
    config: ThresholdSelectorInputConfig
    model_devi_data: List[Artifact]
    model_devi_out_filename: str

    def set_model_devi_dataset(self, data: List[Artifact]):
        self.model_devi_data = data

async def threshold_selector(input: ThresholdSelectorInput, ctx: ThresholdSelectorContext):
    executor = ctx.resource_manager.default_executor

    f_trust_lo = input.config.f_trust_lo
    f_trust_hi = input.config.f_trust_hi
    col_force = 'avg_devi_f'
    logger.info('criteria: %f <= %s < %f ', f_trust_lo, col_force, f_trust_hi)

    total_count = 0
    passed_count = 0

    # TODO: support output of different software
    for candidate in input.model_devi_data:
        if LammpsOutputHelper.is_match(candidate):
            model_devi_out_file = LammpsOutputHelper(candidate).get_model_devi_file(input.model_devi_out_filename).url
        else:
            raise ValueError('unknown model_devi_data types')

        logger.info('start to analysis file: %s', model_devi_out_file)
        text = executor.load_text(model_devi_out_file)

        df = pd.read_csv(StringIO(text.lstrip('#')), delim_whitespace=True)
        # layout:
        #        step  max_devi_v  min_devi_v  avg_devi_v  max_devi_f  min_devi_f  avg_devi_f
        # 0        0    0.006793    0.000672    0.003490    0.143317    0.005612    0.026106
        # 1      100    0.006987    0.000550    0.003952    0.128178    0.006042    0.022608

        passed_df   = df[df[col_force] < f_trust_lo]
        selected_df = df[(df[col_force] >= f_trust_lo) & (df[col_force] < f_trust_hi)]
        rejected_df = df[df[col_force] >= f_trust_hi]

        logger.info('result: total: %d, passed: %d, selected: %d, rejected: %d', len(df), len(passed_df), len(selected_df), len(rejected_df))

        classified_result = {
            'all': df.step.tolist(),
            'passed': passed_df.step.tolist(),
            'selected': selected_df.step.tolist(),
            'rejected': rejected_df.step.tolist(),
        }

        candidate.attrs[LAMMPS_DUMPS_CLASSIFIED] = classified_result

        total_count += len(df)
        passed_count += len(passed_df)

    return ThresholdSelectorOutput(
        model_devi_data=input.model_devi_data,
        passing_rate=passed_count / total_count,
    )
