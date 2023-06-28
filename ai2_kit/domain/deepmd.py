from ai2_kit.core.artifact import Artifact
from ai2_kit.core.script import BashTemplate
from ai2_kit.core.script import BashScript, BashStep
from ai2_kit.core.job import JobFuture, gather_jobs
from ai2_kit.core.log import get_logger

from pydantic import BaseModel
from typing import List
from dataclasses import dataclass
import os
import copy
import random
import sys
import json

from .cll import ICllTrainOutput, BaseCllContext
from .data_helper import Cp2kOutputHelper, DeepmdNpyHelper, convert_to_deepmd_npy
from .constant import (
    DP_CHECKPOINT_FILE,
    DP_DISP_FILE,
    DP_PROFILING_FILE,
    DP_INPUT_FILE,
    DP_FROZEN_MODEL,
    DP_ORIGINAL_MODEL
)

logger = get_logger(__name__)


class GenericDeepmdInputConfig(BaseModel):
    model_num: int = 4
    init_dataset: List[str]
    input_template: dict
    compress_model: bool = False

class GenericDeepmdContextConfig(BaseModel):
    script_template: BashTemplate
    dp_cmd: str = 'dp'


@dataclass
class GenericDeepmdInput:
    config: GenericDeepmdInputConfig
    type_map: List[str]
    old_dataset: List[Artifact]  # training data used by previous iteration
    new_dataset: List[Artifact]  # training data used by current iteration
    initiated: bool = False

    def update_training_dataset(self, dataset: List[Artifact]):
        self.old_dataset += self.new_dataset
        self.new_dataset = dataset


@dataclass
class GenericDeepmdContext(BaseCllContext):
    config: GenericDeepmdContextConfig


@dataclass
class GenericDeepmdOutput(ICllTrainOutput):
    outputs: List[Artifact]
    input: GenericDeepmdInput

    def get_mlp_models(self) -> List[Artifact]:
        return [a.join(DP_FROZEN_MODEL) for a in self.outputs]

    def get_training_dataset(self) -> List[Artifact]:
        return self.input.new_dataset + self.input.old_dataset


async def generic_deepmd(input: GenericDeepmdInput, ctx: GenericDeepmdContext):
    executor = ctx.resource_manager.default_executor

    # setup workspace
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    [converted_data_dir, tasks_dir] = executor.setup_workspace(
        work_dir, ['converted_input_data', 'tasks'])

    # initialization
    if not input.initiated:
        input.new_dataset += ctx.resource_manager.resolve_artifacts(
            input.config.init_dataset)

    # convert data type if necessary, only needed for new data as old data is already converted
    new_deepmd_npy_data: List[Artifact] = []
    cp2k_output_data: List[Artifact] = []

    # TODO: refactor data type conversion
    for artifact in input.new_dataset:
        if not artifact.format or DeepmdNpyHelper.is_match(artifact):
            # treated as deepmd npy data if format is not specified
            new_deepmd_npy_data.append(artifact)
        elif Cp2kOutputHelper.is_match(artifact):
            cp2k_output_data.append(artifact)
        else:
            raise ValueError(f'unsupported data type: {artifact.format}')

    # convert data to deepmd/npy format
    # TODO: support more data type
    converted_data_dirs = executor.run_python_fn(convert_to_deepmd_npy)(
        cp2k_outputs=[a.to_dict() for a in cp2k_output_data],
        base_dir=converted_data_dir,
        type_map=input.type_map,
    )
    new_deepmd_npy_data += [Artifact.of(
        url=url, format=DeepmdNpyHelper.format,
    ) for url in converted_data_dirs]

    input.new_dataset = new_deepmd_npy_data

    jobs: List[JobFuture] = []
    output_dirs = []

    # train multiple models at the same time
    for i in range(input.config.model_num):
        # each model should be trained in its own task_dir
        task_dir = os.path.join(tasks_dir, str(i).zfill(3))
        executor.mkdir(task_dir)

        # create dp train input file
        # NOTE: dp v1 format is supported currently
        # TODO: support more params if it is necessary

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
        systems = [a.url for a in input.old_dataset + input.new_dataset]
        training['systems'] = systems
        set_prefix: str = training.setdefault(
            'set_prefix', 'set')  # respect user input
        auto_prob_str = "prob_sys_size"
        training.setdefault('batch_size', 'auto')
        training['auto_prob_style'] = auto_prob_str

        # v2 training data
        training_data = {
            'systems': training['systems'],
            'set_prefix': training['set_prefix'],
            'auto_prob_style': training['auto_prob_style'],
            'batch_size': training['batch_size'],
        }
        training['training_data'] = training_data

        # other params
        dp_input['model']['type_map'] = input.type_map

        # write config to executor
        dp_input_text = json.dumps(dp_input, indent=2)
        dp_input_path = os.path.join(task_dir, DP_INPUT_FILE)
        executor.dump_text(dp_input_text, dp_input_path)

        # build script
        dp_cmd = ctx.config.dp_cmd
        dp_train_cmd = [dp_cmd, 'train', DP_INPUT_FILE]

        steps = [
            BashStep(cmd=dp_train_cmd, checkpoint='dp-train') # type: ignore
        ]

        if input.config.compress_model:
            dp_freeze_cmd = [dp_cmd, 'freeze', '-o', DP_ORIGINAL_MODEL]
            dp_compress_cmd = \
                [dp_cmd, 'compress', '-i', DP_ORIGINAL_MODEL, '-o', DP_FROZEN_MODEL]
            steps.append(BashStep(cmd=dp_freeze_cmd)) # type: ignore
            steps.append(BashStep(cmd=dp_compress_cmd)) # type: ignore
        else:
            dp_freeze_cmd = [dp_cmd, 'freeze', '-o', DP_FROZEN_MODEL]
            steps.append(BashStep(cmd=dp_freeze_cmd)) # type: ignore

        dp_train_script = BashScript(
            template=ctx.config.script_template,
            steps=steps # type: ignore
        )
        output_dirs.append(task_dir)

        # submit job
        job = executor.submit(dp_train_script.render(), cwd=task_dir, checkpoint_key=f'submit-job/dp-train/{i}:{task_dir}')
        jobs.append(job)

    await gather_jobs(jobs, max_tries=2)

    return GenericDeepmdOutput(
        input=input,
        outputs=[Artifact.of(
            url=url,
        ) for url in output_dirs]
    )


def _random_seed():
    return random.randrange(sys.maxsize) % (1 << 32)
