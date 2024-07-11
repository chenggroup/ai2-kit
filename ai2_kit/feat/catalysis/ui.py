from ai2_kit.core.util import merge_dict, wait_for_change, load_json, dump_json
from ai2_kit.core.log import get_logger
from ai2_kit.tool.deepmd import display_lcurve
from ..catalysis import AI2CAT_RES_DIR, ConfigBuilder, inspect_lammps_output

import matplotlib.pyplot as plt
from jupyter_formily import Formily
from typing import List, Tuple, Callable, Optional
import asyncio
import os
import glob


logger = get_logger(__name__)


class UiHelper:
    """
    Jupyter Widget for AI2CAT
    """

    def __init__(self) -> None:
        self.aimd_args = None
        self.train_args = None
        self.label_explore_args = None
        self.lammps_args = None

        self.system_file = None
        self.cp2k_basis_set_file = None
        self.cp2k_potential_file = None
        self.cp2k_parameter_file = None

        self.lammps_schema_path = os.path.join(AI2CAT_RES_DIR, 'gen-lammps.formily.json')
        self.selector_schema_path = os.path.join(AI2CAT_RES_DIR, 'selector.formily.json')

    def _set_default_system_file(self, args: dict):
        if self.system_file is not None:
            args['system_file'] = self.system_file

    def _update_default_system_file(self, args: dict):
        self.system_file = args.get('system_file', self.system_file)

    def _set_default_cp2k_basic_args(self, args: dict):
        if self.cp2k_basis_set_file is not None:
            args['basis_set_file'] = self.cp2k_basis_set_file
        if self.cp2k_potential_file is not None:
            args['potential_file'] = self.cp2k_potential_file
        if self.cp2k_parameter_file is not None:
            args['parameter_file'] = self.cp2k_parameter_file

    def _update_default_cp2k_basic_args(self, args: dict):
        self.cp2k_basis_set_file = args.get('basis_set_file', self.cp2k_basis_set_file)
        self.cp2k_potential_file = args.get('potential_file', self.cp2k_potential_file)
        self.cp2k_parameter_file = args.get('parameter_file', self.cp2k_parameter_file)


    def gen_aimd_config(self, out_dir: str, **default_value):
        os.makedirs(out_dir, exist_ok=True)
        if self.aimd_args is None:
            self.aimd_args = default_value
        self._set_default_system_file(self.aimd_args)
        self._set_default_cp2k_basic_args(self.aimd_args)

        schema = self._get_aimd_schema()
        options = _get_form_options('Provision AIMD Task')
        form = Formily(schema, options=options , default_value=self.aimd_args)

        form.display()
        async def _task():
            res = await wait_for_change(form, 'value')
            self.aimd_args = res['data']
            self._update_default_system_file(self.aimd_args)
            self._update_default_cp2k_basic_args(self.aimd_args)
            logger.info('form value: %s', self.aimd_args)
            try:
                logger.info('Start to generate AMID input files...')
                cp2k_kwargs: dict = self.aimd_args.copy()
                system_file = cp2k_kwargs.pop('system_file')
                config_builder = ConfigBuilder()
                config_builder.load_system(system_file)
                config_builder.gen_cp2k_input(out_dir=out_dir, **cp2k_kwargs, aimd=True)
                logger.info('Success!')  # TODO: Send a toast message
            except Exception as e:
                logger.exception('Failed!')  # TODO: Send a alert message
        asyncio.ensure_future(_task())


    def gen_train_vendors_config(self, out_dir: str, **default_value):
        os.makedirs(out_dir, exist_ok=True)
        if self.label_explore_args is None:
            self.label_explore_args = default_value
        self._set_default_system_file(self.label_explore_args)
        self._set_default_cp2k_basic_args(self.label_explore_args)

        schema = self._get_label_explore_schema()
        options = _get_form_options('Generate CP2K and Plumed Inputs')
        form = Formily(schema, options=options, default_value=self.label_explore_args)
        form.display()
        async def _task():
            res = await wait_for_change(form, 'value')
            self.label_explore_args = res['data']
            self._update_default_system_file(self.label_explore_args)
            self._update_default_cp2k_basic_args(self.label_explore_args)

            logger.info('form value: %s', self.label_explore_args)
            try:
                logger.info('Start to generate CP2K and plumed input files...')
                cp2k_kwargs: dict = self.label_explore_args.copy()
                system_file = cp2k_kwargs.pop('system_file')
                config_builder = ConfigBuilder()
                config_builder.load_system(system_file)
                config_builder.gen_cp2k_input(out_dir=out_dir, **cp2k_kwargs, aimd=False)
                config_builder.gen_plumed_input(out_dir=out_dir)
                logger.info('Success!')  # TODO: Send a toast message
            except Exception as e:
                logger.exception('Failed!')
        asyncio.ensure_future(_task())

    def gen_train_config(self, out_dir: str, **default_value):
        os.makedirs(out_dir, exist_ok=True)
        if self.train_args is None:
            self.train_args = default_value
        self._set_default_system_file(self.train_args)

        schema = self._get_train_schema()
        options = _get_form_options('Provision Training Workflow')
        form = Formily(schema, options=options, default_value=self.train_args)
        form.display()
        async def _task():
            res = await wait_for_change(form, 'value')
            self.train_args = res['data']
            self._update_default_system_file(self.train_args)
            logger.info('form value: %s', self.train_args)
            try:
                logger.info('Start to generate Training input files...')
                config_builder = ConfigBuilder()
                config_builder.load_system(self.train_args['system_file'])
                config_builder.gen_mlp_training_input(
                    out_dir=out_dir,
                    train_data=self.train_args.get('train_data', []),
                    explore_data=self.train_args.get('explore_data', []),
                    artifacts=self.train_args.get('artifacts', []),
                )
                config_builder.gen_deepmd_input(
                    out_dir=out_dir,
                    steps=self.train_args.get('steps'),  # TODO: provide default value
                )
                logger.info('Success!')  # TODO: Send a toast message
            except Exception as e:
                logger.exception('Failed!')  # TODO: Send a alert message
        asyncio.ensure_future(_task())

    def gen_lammps_config(self, out_dir: str, work_dir: str, **default_value):
        os.makedirs(out_dir, exist_ok=True)
        if self.lammps_args is None:
            self.lammps_args = default_value
        self._set_default_system_file(self.lammps_args)

        pattern = os.path.join(work_dir, '*/iters-*/train-deepmd/tasks/*/*.pb'  )
        dp_model_files = glob.glob(pattern)
        ensembles = ['nvt', 'nvt-i', 'nvt-a', 'nvt-iso', 'nvt-aniso', 'npt', 'npt-t', 'npt-tri', 'nve', 'csvr']

        schema = load_json(self.lammps_schema_path)
        # patch for FilePicker
        schema = merge_dict(schema, {'schema': {'properties': {
            'system_file': _get_file_picker(),
            'ensemble': {
                'enum': [{'children': [], 'label': e.upper(), 'value': e} for e in ensembles]
            },
            'dp_models': {
                'enum': [{'children': [], 'label': os.path.relpath(f, work_dir), 'value': os.path.abspath(f)}
                         for f in sorted(dp_model_files)]
            },
        }}}, quiet=True)
        options = _get_form_options('Provision LAMMPS Task')
        form = Formily(schema, options=options, default_value=self.lammps_args)
        form.display()

        async def _task():
            res = await wait_for_change(form, 'value')
            self.lammps_args = res['data']
            logger.info('form value: %s', self.lammps_args)
            try:
                kwargs = self.lammps_args.copy()
                system_file = kwargs.pop('system_file')
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
        options = [(os.path.relpath(d, work_dir), os.path.abspath(d)) for d in sorted(dirs)]
        fig_ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        def _task(dir_path):
            display_lcurve(os.path.join(dir_path, 'lcurve.out'), fig_ax=fig_ax)
        self._gen_selector(title='Inspect DeepMD Result', label='Select DeepMD Task', options=options, cb=_task)

    def inspect_lammps_output(self, work_dir: str):
        pattern = os.path.join(work_dir, '*/iters-*/explore-lammps/tasks/*')
        dirs = glob.glob(pattern)
        options = [(os.path.relpath(d, work_dir), os.path.abspath(d)) for d in sorted(dirs)]
        fig_ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        def _task(dir_path):
            inspect_lammps_output(dir_path, fig_ax=fig_ax)
        self._gen_selector(title='Inspect LAMMPS Result', label='Select LAMMPS Task', options=options, cb=_task)

    def _gen_selector(self, title: str, label: str, options: List[Tuple[str, str]], cb: Callable):
        schema = load_json(self.selector_schema_path)
        # patch options
        schema['schema']['properties']['selected']['enum'] = [{'children': [], 'label': opt[0], 'value': opt[1]} for opt in options]
        schema['schema']['properties']['selected']['title'] = label
        form = Formily(schema, options=_get_form_options(title))
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

    def _get_aimd_schema(self):
        schema_path = os.path.join(AI2CAT_RES_DIR, 'gen-cp2k-aimd.formily.json')
        schema = load_json(schema_path)
        # patch for FilePicker
        return merge_dict(schema, {'schema': {'properties': {
            'system_file':    _get_file_picker(),
            'basis_set_file': _get_file_picker(),
            'potential_file': _get_file_picker(),
            'parameter_file': _get_file_picker(),
        }}}, quiet=True)


    def _get_label_explore_schema(self):
        schema_path = os.path.join(AI2CAT_RES_DIR, 'gen-label-explore.formily.json')
        schema = load_json(schema_path)
        return merge_dict(schema, {'schema': {'properties': {
            'system_file':    _get_file_picker(),
            'basis_set_file': _get_file_picker(),
            'potential_file': _get_file_picker(),
            'parameter_file': _get_file_picker(),
        }}}, quiet=True)


    def _get_train_schema(self):
        schema_path = os.path.join(AI2CAT_RES_DIR, 'gen-training.formily.json')
        schema = load_json(schema_path)
        select_artifact_expr = "($deps[0] || []).map(item => ({label:item.key, value: item.key}))"
        schema = merge_dict(schema, {'schema': {'properties': {
            'system_file': _get_file_picker(),
            'train_data': _get_select_reactor(['artifacts'], select_artifact_expr),
            'explore_data': _get_select_reactor(['artifacts'], select_artifact_expr),
            'artifacts': {'items': { 'properties': {'artifact': {'properties': {
                'url': _get_file_picker(),
                'plumed_file': _get_file_picker(),
                'cp2k_file': _get_file_picker(),
            }}}}},
        }}}, quiet=True)
        return schema


def _get_select_reactor(deps: List[str], expr: str):
    return {"x-reactions": {
        "dependencies": deps,
        "fulfill": {
            "schema": {
                "enum": "{{%s}}" % expr,
            }
        }
    } }


def _get_file_picker(props: Optional[dict] = None) -> dict:
    if props is None:
        props = {}
    return {'x-component': 'FilePicker', 'x-component-props': props}


def _get_form_options(title: str):
    return {
        "modal_props": {
            "title": title,
            "width": "60vw",
            "style": {"max-width": "1200px"},
            "styles": {
                "body": {"max-height": "75vh", "overflow-y": "auto"}
            }
        }
    }


_UI_HELPER = None
def get_the_ui_helper():
    """
    Singleton for UiHelper
    """
    global _UI_HELPER  # pylint: disable=global-statement
    if _UI_HELPER is None:
        _UI_HELPER = UiHelper()
    return _UI_HELPER
