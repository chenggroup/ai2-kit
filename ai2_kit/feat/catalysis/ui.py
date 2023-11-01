from ai2_kit.core.util import merge_dict, wait_for_change
from ai2_kit.core.log import get_logger
from ai2_kit.tool.deepmd import display_lcurve
from ..catalysis import AI2CAT_RES_DIR, ConfigBuilder, inspect_lammps_output

import matplotlib.pyplot as plt
from jupyter_formily import Formily
from typing import List, Tuple, Callable, Optional
from pathlib import Path
import asyncio
import os
import json
import glob


logger = get_logger(__name__)


class UiHelper:
    """
    Jupyter Widget for AI2CAT
    """

    def __init__(self) -> None:
        self.aimd_schema_path = os.path.join(AI2CAT_RES_DIR, 'gen-cp2k-aimd.formily.json')
        self.aimd_form = None
        self.aimd_value = None

        self.training_schema_path = os.path.join(AI2CAT_RES_DIR, 'gen-training.formily.json')
        self.training_form = None
        self.training_value = None

        self.selector_schema_path = os.path.join(AI2CAT_RES_DIR, 'selector.formily.json')

        self.lammps_schema_path = os.path.join(AI2CAT_RES_DIR, 'gen-lammps.formily.json')
        self.lammps_value = None

    def gen_aimd_config(self, cp2k_search_path: str = './', out_dir: str = './'):
        if self.aimd_form is None:
            with open(self.aimd_schema_path, 'r') as fp:
                schema = json.load(fp)
            # patch for FilePicker
            schema = merge_dict(schema, {'schema': {'properties': {
                'system_file':    file_picker({'init_path': './'}),
                'basic_set_file': file_picker({'init_path': cp2k_search_path}),
                'potential_file': file_picker({'init_path': cp2k_search_path}),
                'parameter_file': file_picker({'init_path': cp2k_search_path}),
                'out_dir':        {**file_picker(), 'default': out_dir },
            }}}, quiet=True)
            self.aimd_form = Formily(schema, options={
                "modal_props": {"title": "Provision AIMD Task", "width": "60vw","style": {"max-width": "800px"}, "styles": {"body": {"max-height": "70vh", "overflow-y": "auto"}}}
            })

        self.aimd_form.display()
        async def _task():
            res = await wait_for_change(self.aimd_form, 'value')
            self.aimd_value = res['data']
            self.aimd_value['aimd'] = True
            logger.info('form value: %s', self.aimd_value)
            try:
                logger.info('Start to generate AMID input files...')
                cp2k_kwargs: dict = self.aimd_value.copy()
                system_file = cp2k_kwargs.pop('system_file')

                config_builder = ConfigBuilder()
                config_builder.load_system(system_file)
                config_builder.gen_cp2k_input(**cp2k_kwargs)  # this is quick but prone to error, fix it later
                logger.info('Success!')  # TODO: Send a toast message
            except Exception as e:
                logger.exception('Failed!')  # TODO: Send a alert message
        asyncio.ensure_future(_task())

    def gen_training_config(self, cp2k_search_path: str = './', out_dir: str = './'):
        if self.training_form is None:
            with open(self.training_schema_path, 'r') as fp:
                schema = json.load(fp)
            # patch for FilePicker
            schema = merge_dict(schema, {'schema': {'properties': { 'collapse': {'properties':{
                'general': {'properties': {
                    'system_file': file_picker({'init_path': './'}),
                    'out_dir':     {**file_picker(), 'default': out_dir},
                }},
                'cp2k': {'properties': {
                    'basic_set_file': file_picker({'init_path': cp2k_search_path}),
                    'potential_file': file_picker({'init_path': cp2k_search_path}),
                    'parameter_file': file_picker({'init_path': cp2k_search_path}),
                }},
            }}}}}, quiet=True)
            self.training_form = Formily(schema, options={
                "modal_props": {"title": "Provision Training Workflow", "width": "60vw","style": {"max-width": "800px"}, "styles": {"body": {"max-height": "70vh", "overflow-y": "auto"}}}
            })
        self.training_form.display()
        async def _task():
            res = await wait_for_change(self.training_form, 'value')
            self.training_value = res['data']
            self.training_value['aimd'] = False
            logger.info('form value: %s', self.training_value)
            try:
                logger.info('Start to generate Training input files...')
                cp2k_kwargs: dict = self.training_value.copy()
                system_file = cp2k_kwargs.pop('system_file')
                dp_steps = cp2k_kwargs.pop('dp_steps')
                out_dir = cp2k_kwargs.get('out_dir', './out')

                config_builder = ConfigBuilder()
                config_builder.load_system(system_file)
                config_builder.gen_plumed_input(out_dir=out_dir)
                config_builder.gen_mlp_training_input(out_dir=out_dir)
                config_builder.gen_cp2k_input(**cp2k_kwargs)  # FIXME: this is quick but prone to error, fix it later
                config_builder.gen_deepmd_input(
                    out_dir=out_dir,
                    steps=dp_steps,
                )
                logger.info('Success!')  # TODO: Send a toast message
            except Exception as e:
                logger.exception('Failed!')  # TODO: Send a alert message
        asyncio.ensure_future(_task())


    def gen_lammps_config(self, work_dir: str = './', out_dir = './'):
        pattern = os.path.join(work_dir, '*/iters-*/train-deepmd/tasks/*/*.pb'  )
        dp_model_files = glob.glob(pattern)
        ensembles = ['nvt', 'nvt-i', 'nvt-a', 'nvt-iso', 'nvt-aniso', 'npt', 'npt-t', 'npt-tri', 'nve', 'csvr']

        with open(self.lammps_schema_path, 'r') as fp:
            schema = json.load(fp)
        # patch for FilePicker
        schema = merge_dict(schema, {'schema': {'properties': {
            'system_file': file_picker({'init_path': './'}),
            'out_dir':     {**file_picker(), 'default': out_dir },
            'dp_models': {
                'enum': [{'children': [], 'label': os.path.relpath(f, work_dir), 'value': f}  for f in sorted(dp_model_files)]
            },
            'ensemble': {
                'enum': [{'children': [], 'label': e.upper(), 'value': e} for e in ensembles]
            },
        }}}, quiet=True)

        form = Formily(schema, options={
            "modal_props": {"title": "Provision AIMD Task", "width": "60vw","style": {"max-width": "800px"}, "styles": {"body": {"max-height": "70vh", "overflow-y": "auto"}}}
        }, default_value=self.lammps_value)
        form.display()
        async def _task():
            res = await wait_for_change(form, 'value')
            self.lammps_value = res['data']
            logger.info('form value: %s', self.lammps_value)
            try:
                kwargs = self.lammps_value.copy()
                system_file = kwargs.pop('system_file')
                out_dir = kwargs.pop('out_dir')
                logger.info('Start to generate LAMMPS input files...')
                config_builder = ConfigBuilder()
                config_builder.load_system(system_file)
                config_builder.gen_lammps_input(out_dir, **kwargs)
                config_builder.gen_plumed_input(out_dir=out_dir)
                logger.info('Success!')  # TODO: Send a toast message
            except Exception as e:
                logger.exception('Failed!')  # TODO: Send a alert message
        asyncio.ensure_future(_task())



    def inspect_deepmd_output(self, work_dir: str):
        pattern = os.path.join(work_dir, '*/iters-*/train-deepmd/tasks/*'  )
        dirs = glob.glob(pattern)
        options = [(os.path.relpath(d, work_dir), d) for d in sorted(dirs)]
        fig_ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        def _task(dir_path):
            display_lcurve(os.path.join(dir_path, 'lcurve.out'), fig_ax=fig_ax)
        self._gen_selector(title='Inspect DeepMD Result', label='Select DeepMD Task', options=options, cb=_task)

    def inspect_lammps_output(self, work_dir: str):
        pattern = os.path.join(work_dir, '*/iters-*/explore-lammps/tasks/*')
        dirs = glob.glob(pattern)
        options = [(os.path.relpath(d, work_dir), d) for d in sorted(dirs)]
        fig_ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        def _task(dir_path):
            inspect_lammps_output(dir_path, fig_ax=fig_ax)
        self._gen_selector(title='Inspect LAMMPS Result', label='Select LAMMPS Task', options=options, cb=_task)

    def _gen_selector(self, title: str, label: str, options: List[Tuple[str, str]], cb: Callable):
        with open(self.selector_schema_path, 'r') as fp:
            schema = json.load(fp)
        # patch options
        schema['schema']['properties']['selected']['enum'] = [{'children': [], 'label': opt[0], 'value': opt[1]} for opt in options]
        schema['schema']['properties']['selected']['title'] = label
        form = Formily(schema, options={
            "modal_props": {"title": title, "width": "60vw","style": {"max-width": "800px"}, "styles": {"body": {"max-height": "70vh", "overflow-y": "auto"}}}
        })
        form.display()
        async def _task():
            res = await wait_for_change(form, 'value')
            logger.info('form data: %s', res)
            value = res['data']['selected']
            try:
                cb(value)
                logger.info('Success!')
            except Exception as e:
                logger.exception('Failed!')  # TODO: Send a alert message
        asyncio.ensure_future(_task())


def file_picker(props: Optional[dict] = None) -> dict:
    if props is None:
        props = {}
    return {'x-component': 'FilePicker', 'x-component-props': props}


_UI_HELPER = None
def get_the_ui_helper():
    """
    Singleton for UiHelper
    """
    global _UI_HELPER  # pylint: disable=global-statement
    if _UI_HELPER is None:
        _UI_HELPER = UiHelper()
    return _UI_HELPER
