import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def display_lcurve(in_file: str = 'lcurve.out', save_to: Optional[str] = None):
    lcurve = np.loadtxt(in_file)
    step  = lcurve[:,0]  # step
    e_trn = lcurve[:,2]  # rmse_e_trn
    f_trn = lcurve[:,3]  # rmse_f_trn

    # plot e_trn and f_trn in log scale,
    # and make them fit the size of jupyter notebook
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ax[0].plot(step, e_trn)
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$STEP$')
    ax[0].set_ylabel(r'$RMSE_{E}$')
    ax[0].grid()
    ax[0].set_title('Learning Curve: RMSE of Energy')

    ax[1].plot(step, f_trn)
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$STEP$')
    ax[1].set_ylabel(r'$RMSE_{F}$')
    ax[1].grid()
    ax[1].set_title('Learning Curve: RMSE of Force')


    if save_to is None:
        plt.show()
    else:
        fig.savefig(save_to, dpi=300, bbox_inches='tight')
