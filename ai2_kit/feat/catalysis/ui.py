from ai2_kit.core.util import merge_dict, wait_for_change
from ai2_kit.core.log import get_logger

from jupyter_formily import Formily

import asyncio
import os
import json

from ..catalysis import AI2CAT_RES_DIR, ConfigBuilder


logger = get_logger(__name__)


class UiHelper:
    def __init__(self) -> None:
        self.aimd_schema_path = os.path.join(AI2CAT_RES_DIR, 'gen-cp2k-aimd.formily.json')
        self.aimd_form = None
        self.aimd_value = None

        self.training_schema_path = os.path.join(AI2CAT_RES_DIR, 'gen-training.formily.json')
        self.training_form = None
        self.training_value = None

    def gen_aimd_config(self, cp2k_search_path: str = './', out_dir: str = './'):
        if self.aimd_form is None:
            with open(self.aimd_schema_path, 'r') as fp:
                schema = json.load(fp)
            # patch for FilePicker
            schema = merge_dict(schema, {'schema': {'properties': {
                'system_file':    {'x-component': 'FilePicker', 'x-component-props': {'init_path': './'}},
                'basic_set_file': {'x-component': 'FilePicker', 'x-component-props': {'init_path': cp2k_search_path}},
                'potential_file': {'x-component': 'FilePicker', 'x-component-props': {'init_path': cp2k_search_path}},
                'parameter_file': {'x-component': 'FilePicker', 'x-component-props': {'init_path': cp2k_search_path}},
                'out_dir':        {'x-component': 'FilePicker', 'default': out_dir },
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
                    'system_file':    {'x-component': 'FilePicker', 'x-component-props': {'init_path': './'}},
                    'out_dir':        {'x-component': 'FilePicker', 'default': out_dir},
                }},
                'cp2k' : {'properties': {
                    'basic_set_file': {'x-component': 'FilePicker', 'x-component-props': {'init_path': cp2k_search_path}},
                    'potential_file': {'x-component': 'FilePicker', 'x-component-props': {'init_path': cp2k_search_path}},
                    'parameter_file': {'x-component': 'FilePicker', 'x-component-props': {'init_path': cp2k_search_path}},
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


_UI_HELPER = None
def get_the_ui_helper():
    """
    Singleton for UiHelper
    """
    global _UI_HELPER  # pylint: disable=global-statement
    if _UI_HELPER is None:
        _UI_HELPER = UiHelper()
    return _UI_HELPER

