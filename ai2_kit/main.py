from fire import Fire
from ai2_kit.workflow.cll_mlp import cll_train_mlp
from ai2_kit.workflow.fep import fep_train_mlp
from collections import UserDict



# a placeholder to cheating Fire
nil_fn = lambda: None

class Group:

    def __init__(self, items: dict, doc: str = '') -> None:
        self.__doc__ = doc
        self.__dict__.update(items)


toolkit = Group({
    'ec': Group({
        'fep': Group({
            'train-mlp': fep_train_mlp,
        }),
    }),
    'cll-workflow': Group({
        'train-mlp': cll_train_mlp,
    }),
}, doc="Welcome to use ai2-kit!")




def main():
    Fire(toolkit)

if __name__ == '__main__':
    main()
