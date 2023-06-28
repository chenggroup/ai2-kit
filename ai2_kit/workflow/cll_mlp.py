from ai2_kit.core.executor import BaseExecutorConfig
from ai2_kit.core.artifact import ArtifactMap
from ai2_kit.core.log import get_logger
from ai2_kit.core.util import load_yaml_files, merge_dict
from ai2_kit.core.resource_manager import ResourceManager
from ai2_kit.core.checkpoint import set_checkpoint_file, apply_checkpoint
from ai2_kit.domain import (
    deepmd,
    lammps,
    selector,
    cp2k,
    vasp,
    constant as const,
    updater,
    cll,
)

from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from fire import Fire

import asyncio
import itertools
import copy
import os

logger = get_logger(__name__)


class CllWorkflowExecutorConfig(BaseExecutorConfig):
    class Context(BaseModel):
        class Train(BaseModel):
            deepmd: deepmd.GenericDeepmdContextConfig

        class Explore(BaseModel):
            lammps: lammps.GenericLammpsContextConfig

        class Label(BaseModel):
            cp2k: Optional[cp2k.GenericCp2kContextConfig]
            vasp: Optional[vasp.GenericVaspContextConfig]

        train: Train
        explore: Explore
        label: Label

    context: Context


class WorkflowConfig(BaseModel):
    class General(BaseModel):
        type_map: List[str]
        mass_map: List[float]
        max_iters: int = 10

    class Label(BaseModel):
        cp2k: Optional[cp2k.GenericCp2kInputConfig]
        vasp: Optional[vasp.GenericVaspInputConfig]

    class Train(BaseModel):
        deepmd: deepmd.GenericDeepmdInputConfig

    class Explore(BaseModel):
        lammps: lammps.GenericLammpsInputConfig

    class Select(BaseModel):
        by_threshold: selector.ThresholdSelectorInputConfig

    class Update(BaseModel):
        walkthrough: updater.WalkthroughUpdaterInputConfig

    general: General
    train: Train
    explore: Explore
    select: Select
    label: Label
    update: Update


class CllWorkflowConfig(BaseModel):

    executors: Dict[str, CllWorkflowExecutorConfig]
    artifacts: ArtifactMap
    workflow: Any  # Keep it raw here, it should be parsed later in iteration


def run_workflow(*config_files, executor: Optional[str] = None,
                 path_prefix: Optional[str] = None, checkpoint_file: Optional[str] = None):
    """
    Run Closed-Loop Learning (CLL) workflow to train Machine Learning Potential (MLP) models.
    """
    if checkpoint_file is not None:
        set_checkpoint_file(checkpoint_file)

    config_data = load_yaml_files(*config_files)
    config = CllWorkflowConfig.parse_obj(config_data)

    if executor not in config.executors:
        raise ValueError(f'executor {executor} is not found')
    if path_prefix is None:
        raise ValueError('path_prefix should not be empty')

    cll.init_artifacts(config.artifacts)
    resource_manager = ResourceManager(
        executor_configs=config.executors,
        artifacts=config.artifacts,
        default_executor=executor,
    )
    return asyncio.run(cll_mlp_training_workflow(config, resource_manager, executor, path_prefix))


async def cll_mlp_training_workflow(config: CllWorkflowConfig, resource_manager: ResourceManager, executor: str, path_prefix: str):
    context_config = config.executors[executor].context
    raw_workflow_config = copy.deepcopy(config.workflow)

    # output of each step
    label_output: Optional[cll.ICllLabelOutput] = None
    selector_output: Optional[cll.ICllSelectorOutput] = None
    train_output: Optional[cll.ICllTrainOutput] = None
    explore_output: Optional[cll.ICllExploreOutput] = None

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

        # label
        if workflow_config.label.cp2k:
            cp2k_input = cp2k.GenericCp2kInput(
                config=workflow_config.label.cp2k,
                type_map=type_map,
                system_files=[] if selector_output is None else selector_output.get_model_devi_dataset(),
                initiated=i > 0,
            )
            if context_config.label.cp2k is None:
                raise ValueError('label > cp2k should not be empty')
            cp2k_context = cp2k.GenericCp2kContext(
                config=context_config.label.cp2k,
                path_prefix=os.path.join(iter_path_prefix, 'label-cp2k'),
                resource_manager=resource_manager,
            )
            label_output = await apply_checkpoint(f'{cp_prefix}/label-cp2k')(cp2k.generic_cp2k)(cp2k_input, cp2k_context)
        elif workflow_config.label.vasp:
            vasp_input = vasp.GenericVaspInput(
                config=workflow_config.label.vasp,
                type_map=type_map,
                system_files=[] if selector_output is None else selector_output.get_model_devi_dataset(),
                initiated=i > 0,
            )
            if context_config.label.vasp is None:
                raise ValueError('label > vasp should not be empty')
            vasp_context = vasp.GenericVaspContext(
                config=context_config.label.vasp,
                path_prefix=os.path.join(iter_path_prefix, 'label-vasp'),
                resource_manager=resource_manager,
            )
            label_output = await apply_checkpoint(f'{cp_prefix}/label-vasp')(vasp.generic_vasp)(vasp_input, vasp_context)
        else:
            raise ValueError('No label method is specified')

        # train
        if workflow_config.train.deepmd:
            deepmd_input = deepmd.GenericDeepmdInput(
                config=workflow_config.train.deepmd,
                type_map=type_map,
                old_dataset=[] if train_output is None else train_output.get_training_dataset(),
                new_dataset=label_output.get_labeled_system_dataset(),
                initiated=i > 0,
            )
            deepmd_context = deepmd.GenericDeepmdContext(
                path_prefix=os.path.join(iter_path_prefix, 'train-deepmd'),
                config=context_config.train.deepmd,
                resource_manager=resource_manager,
            )
            train_output = await apply_checkpoint(f'{cp_prefix}/train-deepmd')(deepmd.generic_deepmd)(deepmd_input, deepmd_context)
        else:
            raise ValueError('No train method is specified')

        # explore
        if workflow_config.explore.lammps:
            md_options = lammps.GenericLammpsInput.MdOptions(
                models=train_output.get_mlp_models(),
            )
            lammps_input = lammps.GenericLammpsInput(
                config=workflow_config.explore.lammps,
                type_map=type_map,
                mass_map=mass_map,
                md_options=md_options,
            )
            lammps_context = lammps.GenericLammpsContext(
                path_prefix=os.path.join(iter_path_prefix, 'explore-lammps'),
                config=context_config.explore.lammps,
                resource_manager=resource_manager,
            )
            explore_output = await apply_checkpoint(f'{cp_prefix}/explore-lammps')(lammps.generic_lammps)(lammps_input, lammps_context)
        else:
            raise ValueError('No explore method is specified')

        # select
        if workflow_config.select.by_threshold:
            selector_input = selector.ThresholdSelectorInput(
                config=workflow_config.select.by_threshold,
                model_devi_data=explore_output.get_model_devi_dataset(),
                model_devi_out_filename=const.MODEL_DEVI_OUT,
            )
            selector_context = selector.ThresholdSelectorContext(
                path_prefix=os.path.join(iter_path_prefix, 'selector-threshold'),
                resource_manager=resource_manager,
            )
            selector_output = await apply_checkpoint(f'{cp_prefix}/selector-threshold')(selector.threshold_selector)(selector_input, selector_context)
        else:
            raise ValueError('No select method is specified')

        # Update
        update_config = workflow_config.update.walkthrough

        # nothing to update because the table is empty
        if not update_config.table:
            continue
        # keep using the latest config when it reach the end of table
        if update_cursor >= len(update_config.table):
            continue

        # move cursor to next row if passing rate pass threshold
        if selector_output.get_passing_rate() > update_config.passing_rate_threshold:
            raw_workflow_config = merge_dict(copy.deepcopy(
                config.workflow), update_config.table[update_cursor])
            update_cursor += 1


if __name__ == '__main__':
    # use python-fire to parse command line arguments
    Fire(run_workflow)
