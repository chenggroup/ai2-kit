from fire import Fire

import warnings
import logging
import os

from ai2_kit.core.cmd import CmdGroup

logger = logging.getLogger(__name__)


class AlgorithmGroup:
    """
    Algorithms for specific domains.
    """

    def proton_transfer(self):
        """
        Proton transfer analysis toolkit.
        """
        from ai2_kit.algorithm import proton_transfer
        return proton_transfer.cli_entry

    def aosa(self):
        """
        Amorphous oxides structure analysis toolkit.
        """
        from ai2_kit.algorithm import aos_analysis
        return aos_analysis.cli_entry

    def reweighting(self):
        """
        Reweightning toolkit
        """
        from ai2_kit.algorithm.reweighting import ReweightingTool
        return ReweightingTool()


class WorkflowGroup:
    """
    Workflows for specific domains.
    """

    @property
    def cll_mlp_training(self):
        from ai2_kit.workflow.cll_mlp import run_workflow
        return run_workflow

    @property
    def fep_mlp_training(self):
        from ai2_kit.workflow.fep_mlp import run_workflow
        return run_workflow

class ToolGroup:
    """
    Tools for specific domains.
    """

    @property
    def ase(self):
        from ai2_kit.tool.ase import AseTool
        return AseTool

    @property
    def dpdata(self):
        from ai2_kit.tool.dpdata import DpdataTool
        return DpdataTool

    @property
    def model_devi(self):
        from ai2_kit.tool.model_devi import ModelDevi
        return ModelDevi

    @property
    def yaml(self):
        from ai2_kit.tool.yaml import YamlTool
        return YamlTool

    @property
    def batch(self):
        from ai2_kit.tool.batch import BatchTool
        return BatchTool

    @property
    def frame(self):
        from ai2_kit.tool.frame import FrameTool
        return FrameTool

    @property
    def hpc(self):
        # deprecated, use `oh-my-batch` instead
        warnings.warn('The hpc module is deprecated, please use oh-my-batch instead.',
                        DeprecationWarning, stacklevel=2)

        from ai2_kit.tool.hpc import cmd_entry
        return cmd_entry

    @property
    def lammps(self):
        from ai2_kit.tool.lammps import LammpsTool
        return LammpsTool()

    @property
    def misc(self):
        from ai2_kit.tool.misc import cmd_entry
        return cmd_entry

class FeatureGroup:
    """
    Featuring tools for specific domains.
    """

    @property
    def catalysis(self):
        """
        Catalyst specific tools.
        """
        from ai2_kit.feat.catalysis import CmdEntries
        return CmdEntries

    @property
    def cat(self):
        """
        Shortcut for catalyst.
        """
        return self.catalysis

    @property
    def spectr(self):
        """
        Spectrum specific tools.
        """
        from ai2_kit.feat.spectrum import CmdEntries
        return CmdEntries

    @property
    def nmrnet(self):
        """
        NMRNet specific tools.
        """
        try:
            from ai2_kit.algorithm.uninmr import CmdEntries
        except ImportError:
            logging.info('In order to use nmrnet, you need to ensure the following packages are installed: ')
            logging.info('"rdkit", "scipy" and "unicore"')
            raise
        return CmdEntries


ai2_kit = CmdGroup({
    'workflow': WorkflowGroup(),
    'algorithm': AlgorithmGroup(),
    'tool': ToolGroup(),
    'feat': FeatureGroup(),

}, doc="Welcome to use ai2-kit!")


def _setup_logging():
    level_name = os.environ.get('LOG_LEVEL', 'INFO')
    level = logging._nameToLevel.get(level_name, logging.INFO)
    logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', level=level)
    logging.getLogger('transitions.core').setLevel(logging.WARNING)


def main():
    _setup_logging()
    Fire(ai2_kit)


def rpc_main():
    from fire_rpc import make_fire_cmd
    from ai2_kit.core.util import json_dumps
    _setup_logging()
    Fire(make_fire_cmd(ai2_kit, json_dumps=json_dumps))


if __name__ == '__main__':
    main()
