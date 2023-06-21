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
    def checkpoint(self):
        from ai2_kit.core.checkpoint import CheckpointCmd
        return CheckpointCmd

kit = Group({
    'workflow': WorkflowGroup(),
    'algorithm': Group({
        'proton-transfer': ProtonTransferGroup(),
    }),
    'tool': ToolGroup(),
}, doc="Welcome to use ai2-kit!")


def main():
    Fire(kit)

if __name__ == '__main__':
    main()
