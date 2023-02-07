from ai2_kit.core.artifact import Artifact
from ai2_kit.core.script import BashTemplate
from ai2_kit.core.executor import Executor
from ai2_kit.core.script import BashScript, BashStep
from ai2_kit.core.job import JobFuture, GatherJobsFuture, retry_fn, map_future
from ai2_kit.core.log import get_logger

from pydantic import BaseModel
from typing import List
from dataclasses import dataclass
import os
import copy
import random
import sys
import json

from .constant import (
    DP_CHECKPOINT_FILE,
    DP_DISP_FILE,
    DP_PROFILING_FILE,
    DP_INPUT_FILE,
    DP_FROZEN_MODEL,
)

logger = get_logger(__name__)


class GeneralDeepmdInputConfig(BaseModel):
    model_num: int = 4
    init_data: List[str]
    input_template: dict


class GeneralDeepmdContextConfig(BaseModel):
    script_template: BashTemplate
    dp_cmd: str = 'dp'


@dataclass
class GeneralDeepmdInput:
    config: GeneralDeepmdInputConfig
    type_map: List[str]
    old_data: List[Artifact]  # training data used by previous iteration
    new_data: List[Artifact]  # training data used by current iteration

@dataclass
class GeneralDeepmdContext:
    path_prefix: str
    executor: Executor
    config: GeneralDeepmdContextConfig

@dataclass
class GeneralDeepmdOutput:
    @dataclass
    class Result:
        output_dir: Artifact

    results: List[Result]
    set_prefix: str



def general_deepmd(input: GeneralDeepmdInput, ctx: GeneralDeepmdContext):

    jobs: List[JobFuture] = []
    results: List[GeneralDeepmdOutput.Result] = []

    # train multiple models at the same time
    for i in range(input.config.model_num):
        # each model should be trained in its own work_dir
        path_prefix = os.path.join(ctx.path_prefix, str(i).zfill(4))
        work_dir = ctx.executor.get_full_path(path_prefix)
        ctx.executor.mkdir(work_dir)
        logger.info('the work_dir is %s', work_dir)

        # create dp train input file
        # NOTE: dp v1 format is supported currently
        # TODO: migrate to dp v2 format
        # TODO: support more params if it is necessary
        # TODO: support train from previous model

        # ref: https://github.com/deepmodeling/dpgen2/blob/master/examples/ch4/param_CH4_deepmd-kit-2.1.1.json
        # ref: https://github.com/deepmodeling/dpgen2/blob/master/dpgen2/op/prep_dp_train.py
        # ref: https://github.com/deepmodeling/dpgen2/blob/master/dpgen2/op/run_dp_train.py

        dp_input = copy.deepcopy(input.config.input_template)
        training: dict = dp_input['training']

        # set output files
        training['disp_file'] = DP_DISP_FILE
        training['save_ckpt'] = DP_CHECKPOINT_FILE
        training['profiling_file'] = DP_PROFILING_FILE

        # set random seed
        discriptor = dp_input['model']['descriptor']
        if discriptor['type'] == 'hybrid':
            for d in discriptor['list']:
                d['seed'] = _random_seed()
        else:
            discriptor['seed'] = _random_seed()
        dp_input['model']['fitting_net']['seed'] = _random_seed()
        dp_input['training']['seed'] = _random_seed()

        # set training data
        systems = [a.url for a in input.old_data + input.new_data]
        training['systems'] = systems

        # set other params
        dp_input['model']['type_map'] = input.type_map
        set_prefix: str = training.setdefault(
            'set_prefix', 'set')  # respect user input
        auto_prob_str = "prob_sys_size"
        training.setdefault('batch_size', 'auto')
        training['auto_prob_style'] = auto_prob_str

        # write config to executor
        dp_input_text = json.dumps(dp_input, indent=2)
        dp_input_path = os.path.join(work_dir, DP_INPUT_FILE)
        ctx.executor.dump_text(dp_input_text, dp_input_path)

        # build script
        dp_cmd = ctx.config.dp_cmd
        dp_train_cmd = [dp_cmd, 'train', DP_INPUT_FILE]
        dp_freeze_cmd = [dp_cmd, 'freeze', '-o', DP_FROZEN_MODEL]

        dp_train_script = BashScript(
            template=ctx.config.script_template,
            steps=[
                BashStep(cmd=dp_train_cmd, checkpoint='dp-train'),  # type: ignore
                BashStep(cmd=dp_freeze_cmd),  # type: ignore
            ] # type: ignore
        )

        results.append(GeneralDeepmdOutput.Result(
            output_dir=Artifact(
                executor=ctx.executor.name,
                url=work_dir,
                attrs=dict(
                    frozen_model=DP_FROZEN_MODEL,
                )
            ) # type: ignore
        ))

        # submit job
        job = ctx.executor.submit(dp_train_script.render(), cwd=work_dir)

        # FIXME: investigate this type issue
        jobs.append(job)  # type: ignore

    future = GatherJobsFuture(jobs, done_fn=retry_fn(max_tries=2), raise_exception=True)

    return map_future(future, GeneralDeepmdOutput(
        set_prefix=set_prefix,  # type: ignore
        results=results,
    ))

def _random_seed():
    return random.randrange(sys.maxsize) % (1 << 32)
