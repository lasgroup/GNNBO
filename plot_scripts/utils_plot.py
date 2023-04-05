import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def plt_regret(t: int, net: str, print_every: int, regrets: list, regrets_bp: list,
               plot_vars: bool = True, avg_vars: Optional[list] = None, pick_vars_all: Optional[list] = None):
    if plot_vars:
        fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
        # plot regret
        ax[0].plot(regrets, color='#9dc0bc', label=net+'-UCB')
        ax[0].plot(regrets_bp, color='#c6ecae', label=net + 'UCB, Best Point R_T')
        ax[0].set_title(f'Cumulative Regret, t = {t}')
        ax[0].legend()
        ax[0].set(xlabel='t', ylabel=r'$R_t$')

        time_stamps = np.arange(0, t + 1, print_every)
        ax[1].plot(pick_vars_all, label=net + r', $G_t$', color='#9dc0bc')
        ax[1].plot(time_stamps, avg_vars, '-.', label= net + r', $\bar{G}$', color='#c6ecae')
        ax[1].set_title('uncertainty')
        ax[1].legend()
        ax[1].set(xlabel='t', ylabel=r'$\sigma_t$')
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        # plot regret
        plt.plot(regrets, color='#9dc0bc', label=net+'UCB, R_T')
        plt.plot(regrets_bp, color='#c6ecae', label=net+'UCB, Best Point R_T')
        plt.title(f'Cumulative Regret (BP), t = {t}')
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$R_t$')

