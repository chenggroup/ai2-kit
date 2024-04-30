from ai2_kit.core.executor import BaseExecutorConfig
from ai2_kit.core.artifact import ArtifactMap
from ai2_kit.core.log import get_logger
from ai2_kit.core.util import load_yaml_files
from ai2_kit.core.resource_manager import ResourceManager
from ai2_kit.domain import (
    deepmd,
    iface,
    lammps,
    selector,
    cp2k,
    constant as const,
    updater,
)
from ai2_kit.core.checkpoint import set_checkpoint_dir, apply_checkpoint

from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from fire import Fire

import asyncio
import copy
import itertools
import os

logger = get_logger(__name__)


class FepExecutorConfig(BaseExecutorConfig):
    class Context(BaseModel):

        deepmd: deepmd.CllDeepmdContextConfig
        lammps: lammps.CllLammpsContextConfig
        cp2k: cp2k.CllCp2kContextConfig

    context: Context


class WorkflowConfig(BaseModel):
    class General(BaseModel):
        type_map: List[str]
        mass_map: List[float]
        max_iters: int = 10

    class Branch(BaseModel):
        deepmd: deepmd.CllDeepmdInputConfig
        cp2k: cp2k.CllCp2kInputConfig
        threshold: selector.CllModelDeviSelectorInputConfig

    class Update(BaseModel):
        walkthrough: updater.CllWalkthroughUpdaterInputConfig

    general: General
    neu: Branch
    red: Branch
    lammps: lammps.CllLammpsInputConfig
    update: Update


class FepWorkflowConfig(BaseModel):
    executors: Dict[str, FepExecutorConfig]
    artifacts: ArtifactMap
    workflow: Any


def run_workflow(*config_files,
                 executor: Optional[str] = None,
                 path_prefix: Optional[str] = None,
                 checkpoint: Optional[str] = None):
    """
    Training ML potential for FEP

    Args:
        config_files: path of config files, should be yaml files, can be multiple, support glob pattern
        executor: name of executor, should be defined in config `executors` section
        path_prefix: path prefix for output
        checkpoint: checkpoint file
    """
    if checkpoint is not None:
        set_checkpoint_dir(checkpoint)

    config_data = load_yaml_files(*config_files)
    config = FepWorkflowConfig.parse_obj(config_data)

    if executor not in config.executors:
        raise ValueError(f'executor {executor} is not found')
    if path_prefix is None:
        raise ValueError('path_prefix should not be empty')

    iface.init_artifacts(config.artifacts)
    resource_manager = ResourceManager(
        executor_configs=config.executors,
        artifacts=config.artifacts,
        default_executor=executor,
    )
    return asyncio.run(cll_mlp_training_workflow(config, resource_manager, executor, path_prefix))


async def cll_mlp_training_workflow(config: FepWorkflowConfig, resource_manager: ResourceManager, executor: str, path_prefix: str):
    context_config = config.executors[executor].context
    raw_workflow_config = copy.deepcopy(config.workflow)

    # output of each step
    neu_label_output: Optional[iface.ICllLabelOutput] = None
    red_label_output: Optional[iface.ICllLabelOutput] = None

    neu_selector_output: Optional[iface.ICllSelectorOutput] = None
    red_selector_output: Optional[iface.ICllSelectorOutput] = None

    neu_train_output: Optional[iface.ICllTrainOutput] = None
    red_train_output: Optional[iface.ICllTrainOutput] = None

    explore_output: Optional[iface.ICllExploreOutput] = None

    # cursor of update table
    update_cursor = 0
    # Start iteration
    for i in itertools.count(0):

        # parse workflow config
        workflow_config = WorkflowConfig.parse_obj(raw_workflow_config)
        if i >= workflow_config.general.max_iters:
            logger.info(f'Iteration {i} exceeds max_iters, stop iteration.')
            break

        # shortcut for type_map and mass_map
        type_map = workflow_config.general.type_map
        mass_map = workflow_config.general.mass_map

        # decide path prefix for each iteration
        iter_path_prefix = os.path.join(path_prefix, f'iters-{i:03d}')
        # prefix of checkpoint
        cp_prefix = f'iters-{i:03d}'

        # label: cp2k
        red_cp2k_input = cp2k.CllCp2kInput(
            config=workflow_config.red.cp2k,
            type_map=type_map,
            system_files=[] if red_selector_output is None else red_selector_output.get_model_devi_dataset(),
            initiated=i > 0,
        )
        red_cpk2_context = cp2k.CllCp2kContext(
            config=context_config.cp2k,
            path_prefix=os.path.join(iter_path_prefix, 'red-label-cp2k'),
            resource_manager=resource_manager,
        )

        neu_cp2k_input = cp2k.CllCp2kInput(
            config=workflow_config.neu.cp2k,
            type_map=type_map,
            system_files=[] if neu_selector_output is None else neu_selector_output.get_model_devi_dataset(),
            initiated=i > 0,
        )
        neu_cp2k_context = cp2k.CllCp2kContext(
            config=context_config.cp2k,
            path_prefix=os.path.join(iter_path_prefix, 'neu-label-cp2k'),
            resource_manager=resource_manager,
        )

        red_label_output, neu_label_output = await asyncio.gather(
            apply_checkpoint(f'{cp_prefix}/cp2k/red')(cp2k.cll_cp2k)(red_cp2k_input, red_cpk2_context),
            apply_checkpoint(f'{cp_prefix}/cp2k/neu')(cp2k.cll_cp2k)(neu_cp2k_input, neu_cp2k_context),
        )

        # Train
        red_deepmd_input = deepmd.CllDeepmdInput(
            config=workflow_config.red.deepmd,
            type_map=type_map,
            old_dataset=[] if red_train_output is None else red_train_output.get_training_dataset(),
            new_dataset=red_label_output.get_labeled_system_dataset(),
        )
        red_deepmd_context = deepmd.CllDeepmdContext(
            path_prefix=os.path.join(iter_path_prefix, 'red-train-deepmd'),
            config=context_config.deepmd,
            resource_manager=resource_manager,
        )
        neu_deepmd_input = deepmd.CllDeepmdInput(
            config=workflow_config.neu.deepmd,
            type_map=type_map,
            old_dataset=[] if neu_train_output is None else neu_train_output.get_training_dataset(),
            new_dataset=neu_label_output.get_labeled_system_dataset(),
        )
        neu_deepmd_context = deepmd.CllDeepmdContext(
            path_prefix=os.path.join(iter_path_prefix, 'neu-train-deepmd'),
            config=context_config.deepmd,
            resource_manager=resource_manager,
        )

        red_train_output, neu_train_output = await asyncio.gather(
            apply_checkpoint(f'{cp_prefix}/deepmd/red')(deepmd.cll_deepmd)(red_deepmd_input, red_deepmd_context),
            apply_checkpoint(f'{cp_prefix}/deepmd/neu')(deepmd.cll_deepmd)(neu_deepmd_input, neu_deepmd_context),
        )

        # explore
        lammps_input = lammps.CllLammpsInput(
            config=workflow_config.lammps,
            new_system_files=[],
            type_map=type_map,
            mass_map=mass_map,
            dp_models={
                'NEU': neu_train_output.get_mlp_models(),
                'RED': red_train_output.get_mlp_models(),
            },
            preset_template='fep-2m'
        )
        lammps_context = lammps.CllLammpsContext(
            path_prefix=os.path.join(iter_path_prefix, 'explore-lammps'),
            config=context_config.lammps,
            resource_manager=resource_manager,
        )
        explore_output = await apply_checkpoint(f'{cp_prefix}/lammps')(lammps.cll_lammps)(lammps_input, lammps_context)

        # select
        red_selector_input = selector.CllModelDeviSelectorInput(
            config=workflow_config.red.threshold,
            model_devi_data=explore_output.get_model_devi_dataset(),
            model_devi_file=const.MODEL_DEVI_RED_OUT,
            type_map=type_map,
        )
        red_selector_context = selector.CllModelDevSelectorContext(
            path_prefix=os.path.join(
                iter_path_prefix, 'red-selector-threshold'),
            resource_manager=resource_manager,
        )

        neu_selector_input = selector.CllModelDeviSelectorInput(
            config=workflow_config.neu.threshold,
            model_devi_data=explore_output.get_model_devi_dataset(),
            model_devi_file=const.MODEL_DEVI_NEU_OUT,
            type_map=type_map,
        )
        neu_selector_context = selector.CllModelDevSelectorContext(
            path_prefix=os.path.join(iter_path_prefix, 'neu-selector-threshold'),
            resource_manager=resource_manager,
        )

        red_selector_output, neu_selector_output = await asyncio.gather(
            apply_checkpoint(f'{cp_prefix}/selector/red')(selector.cll_model_devi_selector)(red_selector_input, red_selector_context),
            apply_checkpoint(f'{cp_prefix}/selector/neu')(selector.cll_model_devi_selector)(neu_selector_input, neu_selector_context),
        )

        # Update
        update_config = workflow_config.update.walkthrough

        # nothing to update because the table is empty
        if not update_config.table:
            continue
        # keep using the latest config when it reach the end of table
        if update_cursor >= len(update_config.table):
            continue
        # update config
        update_cursor += 1


if __name__ == '__main__':
    # use python-fire to parse command line arguments
    Fire(run_workflow)
