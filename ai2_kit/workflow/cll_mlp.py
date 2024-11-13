from ai2_kit.core.executor import BaseExecutorConfig
from ai2_kit.core.artifact import ArtifactMap
from ai2_kit.core.log import get_logger
from ai2_kit.core.util import load_yaml_files, merge_dict
from ai2_kit.core.resource_manager import ResourceManager
from ai2_kit.core.checkpoint import set_checkpoint_dir, apply_checkpoint
from ai2_kit.core.pydantic import BaseModel
from ai2_kit.domain import (
    deepmd,
    iface,
    lammps,
    lasp,
    selector,
    cp2k,
    vasp,
    constant as const,
    updater,
    anyware,

    lammps as _lammps,
    lasp as _lasp,
    cp2k as _cp2k,
    vasp as _vasp,
    anyware as _anyware,
)




from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from fire import Fire

import asyncio
import itertools
import copy
import os

logger = get_logger(__name__)


class CllWorkflowExecutorConfig(BaseExecutorConfig):
    class Context(BaseModel):
        class Train(BaseModel):
            deepmd: deepmd.CllDeepmdContextConfig

        class Explore(BaseModel):
            lammps: Optional[_lammps.CllLammpsContextConfig] = None
            lasp: Optional[_lasp.CllLaspContextConfig] = None
            anyware: Optional[_anyware.AnywareContextConfig] = None

        class Label(BaseModel):
            cp2k: Optional[_cp2k.CllCp2kContextConfig] = None
            vasp: Optional[_vasp.CllVaspContextConfig] = None

        train: Train
        explore: Explore
        label: Label

    context: Context


class WorkflowConfig(BaseModel):
    class General(BaseModel):
        type_map: List[str]
        mass_map: List[float]
        sel_type: Optional[List[str]] = None

        max_iters: int = 1
        mode: iface.TRAINING_MODE = 'default'
        update_explore_systems: bool = False

    class Label(BaseModel):
        cp2k: Optional[_cp2k.CllCp2kInputConfig] = None
        vasp: Optional[_vasp.CllVaspInputConfig] = None

    class Train(BaseModel):
        deepmd: deepmd.CllDeepmdInputConfig

    class Explore(BaseModel):
        lammps: Optional[_lammps.CllLammpsInputConfig] = None
        lasp: Optional[_lasp.CllLaspInputConfig] = None
        anyware: Optional[_anyware.AnywareConfig] = None

    class Select(BaseModel):
        model_devi: selector.CllModelDeviSelectorInputConfig

    class Update(BaseModel):
        walkthrough: updater.CllWalkthroughUpdaterInputConfig

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


def run_workflow(*config_files,
                 executor: Optional[str] = None,
                 path_prefix: Optional[str] = None,
                 checkpoint: Optional[str] = None):
    """
    Run Closed-Loop Learning (CLL) workflow to train Machine Learning Potential (MLP) models.

    Args:
        config_files: path of config files, should be yaml files, can be multiple, support glob pattern
        executor: name of executor, should be defined in config `executors` section
        path_prefix: path prefix for output
        checkpoint: checkpoint file
    """
    if checkpoint is not None:
        set_checkpoint_dir(checkpoint)

    config_data = load_yaml_files(*config_files)
    config = CllWorkflowConfig.parse_obj(config_data)
    for key in config.artifacts:
        if '/' in key:
            raise ValueError(f'Artifact key {key} should not contain /')

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


async def cll_mlp_training_workflow(config: CllWorkflowConfig,
                                    resource_manager: ResourceManager,
                                    executor: str,
                                    path_prefix: str):
    context_config = config.executors[executor].context
    raw_workflow_config = copy.deepcopy(config.workflow)

    # output of each step
    label_output: Optional[iface.ICllLabelOutput] = None
    selector_output: Optional[iface.ICllSelectorOutput] = None
    train_output: Optional[iface.ICllTrainOutput] = None
    explore_output: Optional[iface.ICllExploreOutput] = None

    # cursor of update table
    update_cursor = 0

    # Start iteration
    for i in itertools.count(0):

        # parse workflow config
        workflow_config = WorkflowConfig.parse_obj(raw_workflow_config)
        shared_vars = precondition(workflow_config)

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
        if workflow_config.label.cp2k and context_config.label.cp2k:
            cp2k_input = cp2k.CllCp2kInput(
                config=workflow_config.label.cp2k,
                mode=workflow_config.general.mode,
                type_map=type_map,
                system_files=[] if selector_output is None else selector_output.get_model_devi_dataset(),
                initiated=i > 0,
            )
            cp2k_context = cp2k.CllCp2kContext(
                config=context_config.label.cp2k,
                path_prefix=os.path.join(iter_path_prefix, 'label-cp2k'),
                resource_manager=resource_manager,
            )
            label_output = await apply_checkpoint(f'{cp_prefix}/label-cp2k')(cp2k.cll_cp2k)(cp2k_input, cp2k_context)

        elif workflow_config.label.vasp and context_config.label.vasp:
            vasp_input = vasp.CllVaspInput(
                config=workflow_config.label.vasp,
                type_map=type_map,
                system_files=[] if selector_output is None else selector_output.get_model_devi_dataset(),
                initiated=i > 0,
            )
            vasp_context = vasp.CllVaspContext(
                config=context_config.label.vasp,
                path_prefix=os.path.join(iter_path_prefix, 'label-vasp'),
                resource_manager=resource_manager,
            )
            label_output = await apply_checkpoint(f'{cp_prefix}/label-vasp')(vasp.cll_vasp)(vasp_input, vasp_context)

        else:
            raise ValueError('No label method is specified')

        # return if no new data is generated
        if i > 0 and len(label_output.get_labeled_system_dataset()) == 0:
            logger.info("No new data is generated, stop iteration.")
            break

        # train
        if workflow_config.train.deepmd:
            deepmd_input = deepmd.CllDeepmdInput(
                config=workflow_config.train.deepmd,
                mode=workflow_config.general.mode,
                type_map=type_map,
                old_dataset=[] if train_output is None else train_output.get_training_dataset(),
                new_dataset=label_output.get_labeled_system_dataset(),
                sel_type=shared_vars.dp_sel_type,
                previous=[] if train_output is None else train_output.get_mlp_models(),
            )
            deepmd_context = deepmd.CllDeepmdContext(
                path_prefix=os.path.join(iter_path_prefix, 'train-deepmd'),
                config=context_config.train.deepmd,
                resource_manager=resource_manager,
            )
            train_output = await apply_checkpoint(f'{cp_prefix}/train-deepmd')(deepmd.cll_deepmd)(deepmd_input, deepmd_context)

        else:
            raise ValueError('No train method is specified')

        # explore
        new_explore_system_files = []
        if workflow_config.general.update_explore_systems and selector_output is not None:
            new_explore_system_files = selector_output.get_new_explore_systems()

        if workflow_config.explore.lammps and context_config.explore.lammps:
            lammps_input = lammps.CllLammpsInput(
                config=workflow_config.explore.lammps,
                mode=workflow_config.general.mode,
                type_map=type_map,
                mass_map=mass_map,
                dp_models={'': train_output.get_mlp_models()},
                preset_template='default',
                new_system_files=new_explore_system_files,
                dp_modifier=shared_vars.dp_modifier,
                dp_sel_type=shared_vars.dp_sel_type,
            )
            lammps_context = lammps.CllLammpsContext(
                path_prefix=os.path.join(iter_path_prefix, 'explore-lammps'),
                config=context_config.explore.lammps,
                resource_manager=resource_manager,
            )
            explore_output = await apply_checkpoint(f'{cp_prefix}/explore-lammps')(lammps.cll_lammps)(lammps_input, lammps_context)

        elif workflow_config.explore.lasp and context_config.explore.lasp:
            lasp_input = lasp.CllLaspInput(
                config=workflow_config.explore.lasp,
                type_map=type_map,
                mass_map=mass_map,
                models=train_output.get_mlp_models(),
                new_system_files=new_explore_system_files,
            )
            lasp_context = lasp.CllLaspContext(
                config=context_config.explore.lasp,
                path_prefix=os.path.join(iter_path_prefix, 'explore-lasp'),
                resource_manager=resource_manager,
            )
            explore_output = await apply_checkpoint(f'{cp_prefix}/explore-lasp')(lasp.cll_lasp)(lasp_input, lasp_context)

        elif workflow_config.explore.anyware and context_config.explore.anyware:
            anyware_input = anyware.AnywareInput(
                config=workflow_config.explore.anyware,
                type_map=type_map,
                mass_map=mass_map,
                new_system_files=new_explore_system_files,
                dp_models={'': train_output.get_mlp_models()},
            )
            anyware_context = anyware.AnywareContext(
                config=context_config.explore.anyware,
                path_prefix=os.path.join(iter_path_prefix, 'explore-anyware'),
                resource_manager=resource_manager,
            )
            explore_output = await apply_checkpoint(f'{cp_prefix}/explore-anyware')(anyware.anyware)(anyware_input, anyware_context)

        else:
            raise ValueError('No explore method is specified')

        # select
        if workflow_config.select.model_devi:
            selector_input = selector.CllModelDeviSelectorInput(
                config=workflow_config.select.model_devi,
                model_devi_data=explore_output.get_model_devi_dataset(),
                model_devi_file=const.MODEL_DEVI_OUT,
                type_map=type_map,
            )
            selector_context = selector.CllModelDevSelectorContext(
                path_prefix=os.path.join(iter_path_prefix, 'selector-model-devi'),
                resource_manager=resource_manager,
            )
            selector_output = await apply_checkpoint(f'{cp_prefix}/selector-model-devi')(selector.cll_model_devi_selector)(selector_input, selector_context)
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


@dataclass
class SharedVars():
    dp_modifier: Optional[dict] = None
    dp_sel_type: Optional[List[int]] = None


def precondition(workflow_cfg: WorkflowConfig) -> SharedVars:
    """
    precondition of workflow config to raise error early,
    and extra variables that may shared by multiple steps

    The known shared variables are:
      dp_modifier, which include vars sys_charge_map, model_charge_map, ewald_h, ewald_beta
      sel_type, which is suppose to be used in dplr/dpff mode
    """
    shared_vars = SharedVars()

    mode = workflow_cfg.general.mode
    type_map = workflow_cfg.general.type_map
    sel_type = workflow_cfg.general.sel_type

    if mode == 'dpff':
        assert sel_type is not None, 'sel_type should be specified in general config for dpff mode'
        shared_vars.dp_sel_type = [ type_map.index(t)  for t in sel_type ]

    deepmd_cfg = workflow_cfg.train.deepmd
    if deepmd_cfg is not None:
        if mode == 'dpff':
            modifier = deepmd_cfg.input_template['model'].get('modifier')
            assert modifier is not None, 'modifier should be specified in deepmd input template for dpff mode'
            shared_vars.dp_modifier = modifier
        elif mode == 'fep-redox':
            assert deepmd_cfg.input_template['model']['fitting_net']['numb_fparam'] == 1, 'numb_fparam should be 1 for fep-redox/fep-pka mode'

    lammps_cfg = workflow_cfg.explore.lammps
    if lammps_cfg is not None:
        if mode == 'dpff':
            lammps_cfg.assert_var('EFIELD')
            lammps_cfg.assert_var('KMESH')
            efield = lammps_cfg.explore_vars.get('EFIELD', lammps_cfg.broadcast_vars.get('EFIELD'))
            assert all([isinstance(item, list) for item in efield ]), 'EFIELD should be a list of vector'  # type: ignore
        elif mode in ['fep-redox', 'fep-pka']:
            lammps_cfg.assert_var('LAMBDA_f')


    return shared_vars


if __name__ == '__main__':
    # use python-fire to parse command line arguments
    Fire(run_workflow)
