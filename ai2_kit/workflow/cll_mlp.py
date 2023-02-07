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


class CllWorkflowExecutorConfig(BaseExecutorConfig):
    class Context(BaseModel):
        class Train(BaseModel):
            deepmd: deepmd.GeneralDeepmdContextConfig

        class Explore(BaseModel):
            lammps: lammps.GeneralLammpsContextConfig

        class Label(BaseModel):
            cp2k: cp2k.GeneralCp2kContextConfig

        train: Train
        explore: Explore
        label: Label

    context: Context


class CllWorkflowConfig(BaseModel):
    class Workflow(BaseModel):
        class General(BaseModel):
            type_map: List[str]
            mass_map: List[float]
            max_iters: int = 10

        class Train(BaseModel):
            deepmd: deepmd.GeneralDeepmdInputConfig

        class Explore(BaseModel):
            lammps: lammps.GeneralLammpsInputConfig

        class Select(BaseModel):
            threshold: selector.GeneralThresholdInputConfig

        class Label(BaseModel):
            cp2k: cp2k.GeneralCp2kInputConfig

        general: General
        train: Train
        explore: Explore
        select: Select
        label: Label

    executors: Dict[str, CllWorkflowExecutorConfig]
    artifacts: ArtifactMap
    workflow: Workflow


def cll_train_mlp(*config_files, executor: Optional[str] = None, path_prefix: Optional[str] = None):
    """
    Run Closed-Loop Learning (CLL) workflow to train Machine Learning Potential (MLP) models.
    """

    config_data = load_yaml_files(*config_files)
    config = CllWorkflowConfig.parse_obj(config_data)

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
    deepmd_input = deepmd.GeneralDeepmdInput(
        config=config.workflow.train.deepmd,
        type_map=type_map,
        old_data=[],
        new_data=[config.artifacts[key] for key in config.workflow.train.deepmd.init_data],
    )
    deepmd_context = deepmd.GeneralDeepmdContext(
        path_prefix='',
        executor=default_executor,
        config=context_cfg.train.deepmd,
    )

    # Start iteration
    for i in range(config.workflow.general.max_iters):
        # update path prefix for each iteration
        iter_path_prefix = os.path.join(path_prefix, f'iters-{str(i).zfill(3)}')

        # TODO: change order to: label -> train -> explore -> select
        # TODO: support more tools
        # TODO: support more options

        # train
        deepmd_context.path_prefix = os.path.join(iter_path_prefix, 'train-deepmd')
        deepmd_output = deepmd.general_deepmd(deepmd_input, deepmd_context).result()

        # explore
        models = [a.output_dir.join(a.output_dir.attrs['frozen_model']) for a in deepmd_output.results]
        lammps_iters = config.workflow.explore.lammps.iters
        md_vars = lammps_iters[ i if i < len(lammps_iters) else (len(lammps_iters) - 1) ]

        lammps_input = lammps.GeneralLammpsInput(
            config=config.workflow.explore.lammps,
            type_map=type_map,
            mass_map=mass_map,
            md_vars=md_vars,
            system_vars=[config.artifacts[key] for key in md_vars.system_vars],
            md_options=lammps.GeneralLammpsInput.MdOptions(models=models)
        )
        lammps_context = lammps.GeneralLammpsContext(
            config=context_cfg.explore.lammps,
            path_prefix=os.path.join(iter_path_prefix, 'explore-lammps'),
            executor=default_executor,
        )
        lammps_output = lammps.general_lammps(lammps_input, lammps_context).result()

        # select
        selector_input = selector.LammpsGeneralThresholdInput(
            config=config.workflow.select.threshold,
            candidates=lammps_output.candidates,
            model_devi_out_file=const.MODEL_DEVI_OUT,
        )
        selector_context = selector.GeneralThresholdContext(
            executor=default_executor,
        )
        selector_output = selector.lammps_general_threshold(selector_input, selector_context).result()

        # label
        cp2k_input = cp2k.GeneralCp2kInput(
            config=config.workflow.label.cp2k,
            type_map=config.workflow.general.type_map,
            candidates=selector_output.candidates,
            basic_set_file=config.artifacts.get(config.workflow.label.cp2k.basic_set_file or ''),
            potential_file=config.artifacts.get(config.workflow.label.cp2k.potential_file or ''),
        )
        cp2k_context = cp2k.GeneralCp2kContext(
            config=context_cfg.label.cp2k,
            path_prefix=os.path.join(iter_path_prefix, 'label-cp2k'),
            executor=default_executor,
        )
        cp2k_output = cp2k.general_cp2k(cp2k_input, cp2k_context).result()

        # update input for the next round
        deepmd_input.old_data.extend(deepmd_input.new_data)
        deepmd_input.new_data = cp2k_output.dp_data_sets

        # deepmd_input, lammps_input, selector_input, cp2k_input = smart_update(
        #     i,
        #     deepmd_output, lammps_output, selector_output, cp2k_output,
        #     deepmd_input, lammps_input, selector_input, cp2k_input,
        # )



if __name__ == '__main__':
    # for test, e.g:
    Fire(cll_train_mlp)
