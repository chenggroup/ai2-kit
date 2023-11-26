from fire import Fire


class Group:
    def __init__(self, items: dict, doc: str = '') -> None:
        self.__doc__ = doc
        self.__dict__.update(items)


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
        from ai2_kit.tool.ase import AseHelper
        return AseHelper

    @property
    def dpdata(self):
        from ai2_kit.tool.dpdata import DpdataHelper
        return DpdataHelper

    @property
    def yaml(self):
        from ai2_kit.tool.yaml import Yaml
        return Yaml

    @property
    def checkpoint(self):
        from ai2_kit.core.checkpoint import CheckpointCmd
        return CheckpointCmd

    @property
    def batch(self):
        from ai2_kit.tool.batch import BatchHelper
        return BatchHelper

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


ai2_kit = Group({
    'workflow': WorkflowGroup(),
    'algorithm': AlgorithmGroup(),
    'tool': ToolGroup(),
    'feat': FeatureGroup(),

}, doc="Welcome to use ai2-kit!")


def main():
    Fire(ai2_kit)


if __name__ == '__main__':
    main()
