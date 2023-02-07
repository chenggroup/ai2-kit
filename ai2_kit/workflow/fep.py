from ai2_kit.core.executor import BaseExecutorConfig, ExecutorManager
from ai2_kit.core.artifact import ArtifactMap
from ai2_kit.core.log import get_logger
from ai2_kit.core.util import load_yaml_files
from ai2_kit.domain import deepmd
from ai2_kit.domain import lammps
from ai2_kit.domain import selector
from ai2_kit.domain import cp2k
from ai2_kit.domain import constant as const

from pydantic import BaseModel
from typing import Dict, List, Optional
from fire import Fire

import os

logger = get_logger(__name__)


class FepExecutorConfig(BaseExecutorConfig):
    class Context(BaseModel):

        deepmd: deepmd.GeneralDeepmdContextConfig
        lammps: lammps.GeneralLammpsContextConfig
        cp2k: cp2k.GeneralCp2kContextConfig

    context: Context


class FepConfig(BaseModel):
    class Workflow(BaseModel):
        class General(BaseModel):
            type_map: List[str]
            mass_map: List[float]
            max_iters: int = 10

        class Branch(BaseModel):
            deepmd: deepmd.GeneralDeepmdInputConfig
            cp2k: cp2k.GeneralCp2kInputConfig
            threshold: selector.GeneralThresholdInputConfig

        general: General
        neu: Branch
        red: Branch
        lammps: lammps.GeneralLammpsInputConfig

    executors: Dict[str, FepExecutorConfig]
    artifacts: ArtifactMap
    workflow: Workflow

def fep_train_mlp(*config_files, executor: Optional[str] = None, path_prefix: Optional[str] = None):
    """
    Training ML potential for FEP
    """

    config_data = load_yaml_files(*config_files)
    config = FepConfig.parse_obj(config_data)

    if executor not in config.executors:
        raise ValueError(f'executor {executor} is not found')
    if path_prefix is None:
        raise ValueError('path_prefix should not be empty')

    executor_manager = ExecutorManager(config.executors)  # type: ignore
    default_executor = executor_manager.get_executor(executor)

    context_cfg = config.executors[executor].context

    type_map = config.workflow.general.type_map
    mass_map = config.workflow.general.mass_map

    # Init Setting
    neu_deepmd_input = deepmd.GeneralDeepmdInput(
        config=config.workflow.neu.deepmd,
        type_map=type_map,
        old_data=[],
        new_data=[config.artifacts[key] for key in config.workflow.neu.deepmd.init_data],
    )
    neu_deepmd_context = deepmd.GeneralDeepmdContext(
        path_prefix='',
        executor=default_executor,
        config=context_cfg.deepmd,
    )

    red_deepmd_input = deepmd.GeneralDeepmdInput(
        config=config.workflow.red.deepmd,
        type_map=type_map,
        old_data=[],
        new_data=[config.artifacts[key] for key in config.workflow.red.deepmd.init_data],
    )
    red_deepmd_context = deepmd.GeneralDeepmdContext(
        path_prefix='',
        executor=default_executor,
        config=context_cfg.deepmd,
    )

    # Start iteration
    for i in range(config.workflow.general.max_iters):
        # update path prefix for each iteration
        iter_path_prefix = os.path.join(path_prefix, f'iters-{str(i).zfill(3)}')

        # train
        neu_deepmd_context.path_prefix = os.path.join(iter_path_prefix, 'train-neu-deepmd')
        neu_deepmd_output_future = deepmd.general_deepmd(neu_deepmd_input, neu_deepmd_context)

        red_deepmd_context.path_prefix = os.path.join(iter_path_prefix, 'train-red-deepmd')
        red_deepmd_output_future = deepmd.general_deepmd(red_deepmd_input, red_deepmd_context)

        neu_deepmd_output, red_deepmd_output = neu_deepmd_output_future.result(), red_deepmd_output_future.result()

        # explore
        neu_models = [a.output_dir.join(a.output_dir.attrs['frozen_model']) for a in neu_deepmd_output.results]
        red_models = [a.output_dir.join(a.output_dir.attrs['frozen_model']) for a in red_deepmd_output.results]

        lammps_iters = config.workflow.lammps.iters
        md_vars = lammps_iters[i % len(lammps_iters)]

        lammps_input = lammps.GeneralLammpsInput(
            config=config.workflow.lammps,
            type_map=type_map,
            mass_map=mass_map,
            md_vars=md_vars,
            system_vars=[config.artifacts[key] for key in md_vars.system_vars],
            fep_options=lammps.GeneralLammpsInput.FepOptions(red_models=red_models, neu_models=neu_models)
        )
        lammps_context = lammps.GeneralLammpsContext(
            config=context_cfg.lammps,
            path_prefix=os.path.join(iter_path_prefix, 'explore-lammps'),
            executor=default_executor,
        )
        lammps_output = lammps.general_lammps(lammps_input, lammps_context).result()

        # select
        neu_selector_input = selector.LammpsGeneralThresholdInput(
            config=config.workflow.neu.threshold,
            candidates=lammps_output.candidates,
            model_devi_out_file=const.MODEL_DEVI_NEU_OUT,
        )
        neu_selector_context = selector.GeneralThresholdContext(
            executor=default_executor,
        )

        red_selector_input = selector.LammpsGeneralThresholdInput(
            config=config.workflow.red.threshold,
            candidates=lammps_output.candidates,
            model_devi_out_file=const.MODEL_DEVI_RED_OUT,
        )
        red_selector_context = selector.GeneralThresholdContext(
            executor=default_executor,
        )

        neu_selector_output_future = selector.lammps_general_threshold(neu_selector_input, neu_selector_context)
        red_selector_output_future = selector.lammps_general_threshold(red_selector_input, red_selector_context)

        neu_selector_output, red_selector_output = neu_selector_output_future.result(), red_selector_output_future.result()

        # label
        neu_cp2k_input = cp2k.GeneralCp2kInput(
            config=config.workflow.neu.cp2k,
            type_map=config.workflow.general.type_map,
            candidates=neu_selector_output.candidates,
            basic_set_file=config.artifacts.get(config.workflow.neu.cp2k.basic_set_file or ''),
            potential_file=config.artifacts.get(config.workflow.neu.cp2k.potential_file or ''),
        )
        neu_cp2k_context = cp2k.GeneralCp2kContext(
            config=context_cfg.cp2k,
            path_prefix=os.path.join(iter_path_prefix, 'label-neu-cp2k'),
            executor=default_executor,
        )

        red_cp2k_input = cp2k.GeneralCp2kInput(
            config=config.workflow.red.cp2k,
            type_map=config.workflow.general.type_map,
            candidates=red_selector_output.candidates,
            basic_set_file=config.artifacts.get(config.workflow.red.cp2k.basic_set_file or ''),
            potential_file=config.artifacts.get(config.workflow.red.cp2k.potential_file or ''),
        )
        red_cp2k_context = cp2k.GeneralCp2kContext(
            config=context_cfg.cp2k,
            path_prefix=os.path.join(iter_path_prefix, 'label-red-cp2k'),
            executor=default_executor,
        )

        neu_cp2k_output_future = cp2k.general_cp2k(neu_cp2k_input, neu_cp2k_context)
        red_cp2k_output_future = cp2k.general_cp2k(red_cp2k_input, red_cp2k_context)

        neu_cp2k_output, red_cp2k_output = neu_cp2k_output_future.result(), red_cp2k_output_future.result()

        # update input for the next round
        neu_deepmd_input.old_data.extend(neu_deepmd_input.new_data)
        neu_deepmd_input.new_data = neu_cp2k_output.dp_data_sets

        red_deepmd_input.old_data.extend(red_deepmd_input.new_data)
        red_deepmd_input.new_data = red_cp2k_output.dp_data_sets


if __name__ == '__main__':
    # for test, e.g:
    # python -m ai2_kit.workflow.fep fep_config.yaml --executor=hpc01 --path-prefix=fep-test
    Fire(fep_train_mlp)