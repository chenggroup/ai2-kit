from fire import Fire


class Group:
    def __init__(self, items: dict, doc: str = '') -> None:
        self.__doc__ = doc
        self.__dict__.update(items)


class ProtonTransferGroup:
    @property
    def analyze(self):
        from ai2_kit.algorithm import proton_transfer
        return proton_transfer.proton_transfer_detection

    @property
    def visualize(self):
        from ai2_kit.algorithm import proton_transfer
        return proton_transfer.visualize_transfer

    @property
    def show_transfer_paths(self):
        from ai2_kit.algorithm import proton_transfer
        return proton_transfer.analysis_transfer_paths

    @property
    def show_type_change(self):
        from ai2_kit.algorithm import proton_transfer
        return proton_transfer.detect_type_change

    @property
    def calculate_distances(self):
        from ai2_kit.algorithm import proton_transfer
        return proton_transfer.calculate_distances

    @property
    def show_distance_change(self):
        from ai2_kit.algorithm import proton_transfer
        return proton_transfer.show_distance_change


class WorkflowGroup:
    @property
    def cll_mlp_training(self):
        from ai2_kit.workflow.cll_mlp import run_workflow
        return run_workflow

    @property
    def fep_mlp_training(self):
        from ai2_kit.workflow.fep_mlp import run_workflow
        return run_workflow

class ToolGroup:

    @property
    def ase(self):
        from ai2_kit.tool.ase import AseHelper
        return AseHelper

    @property
    def dpdata(self):
        from ai2_kit.tool.dpdata import DpdataHelper
        return DpdataHelper

    @property
    def checkpoint(self):
        from ai2_kit.core.checkpoint import CheckpointCmd
        return CheckpointCmd

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
    'algorithm': Group({
        'proton-transfer': ProtonTransferGroup(),
    }),
    'tool': ToolGroup(),
    'feat': FeatureGroup(),

}, doc="Welcome to use ai2-kit!")


def main():
    Fire(ai2_kit)

if __name__ == '__main__':
    main()
