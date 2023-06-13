from fire import Fire
from ai2_kit.workflow.cll_mlp import cll_train_mlp
from ai2_kit.workflow.ec_fep import fep_train_mlp
from ai2_kit.algorithm import proton_transfer


class Group:

    def __init__(self, items: dict, doc: str = '') -> None:
        self.__doc__ = doc
        self.__dict__.update(items)


# TODO: use lazy loading to speed up the startup time
kit = Group({
    'workflow': Group({
        'cll-mlp-training': cll_train_mlp,
        'fep-mlp-training': fep_train_mlp,

    }),
    'algorithm': Group({
        'proton-transfer': Group({
            'analyze': proton_transfer.proton_transfer_detection,
            'visualize': proton_transfer.visualize_transfer,
            'show-transfer-paths': proton_transfer.analysis_transfer_paths,
            'show-type-change': proton_transfer.detect_type_change,
        })
    }),
}, doc="Welcome to use ai2-kit!")



def main():
    Fire(kit)

if __name__ == '__main__':
    main()
