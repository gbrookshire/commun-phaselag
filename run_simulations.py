"""
Run the simulations of various patterns of data, and check whether the analysis
can reconstruct these patterns correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import simulate
import pdc
from tqdm import tqdm


plot_dir = '/media/geoff/Seagate2TB1/geoff/commun-phaselag/data/plots/'
plot_dir += 'simulated/2021-07-12/'

plt.ion()

# Low-freq 'modulator' frequencies
lf_centers = np.arange(5, 15)
lf_bandwidth = lf_centers / 2.5  # ~4 cycles

# High-freq 'carrier' frequencies
hf_centers = np.arange(50, 150, 10)
hf_bandwidth = hf_centers / 3  # ~5 cycles

# Parameters for the simulated signals
sim_params = dict(dur=500,
                  fs=400,
                  noise_amp=1.5)
lag = 6
lag_seconds = lag / sim_params['fs']
n_subjects = 1

sim_funcs = {
        'phase_dep_comm': lambda: simulate.sim(**sim_params,
                                               gamma_lag_a_to_b=lag_seconds),
        # 'lf_coh_plus_pac': lambda: simulate.sim_lf_coh_with_pac(
        #     **sim_params, lag=lag),
        # 'lf_coh_plus_hf_comm': lambda: simulate.sim_lf_coh_with_hf_comm(
        #     **sim_params, lag=lag)
        }

# Parameters for the MI phase-lag analysis
mi_params = dict(fs=sim_params['fs'],
                 lf_centers=lf_centers,
                 lf_bandwidth=lf_bandwidth,
                 hf_centers=hf_centers,
                 hf_bandwidth=hf_bandwidth,
                 lag=lag,
                 n_bins=2**3)


def plot_results_group(results, sim_type, save=True):
    figsize = (14, 4)
    plt.figure(f'avg: {sim_type}', figsize=figsize)
    i_plot = 0
    for direc in ('a', 'b', 'PD(AB-BA)', 'PD(AB)-PD(BA)'):

        if direc == 'a':
            title = 'AB'
        elif direc == 'b':
            title = 'BA'
        else:
            title = f'{direc}'

        te_all = np.stack([e[direc] for e in results])
        i_plot += 1

        x = np.mean(te_all, axis=0)

        plt.subplot(1, 4, i_plot)
        plt.title(title)

        maxabs = np.max(np.abs(x))
        if direc == 'PD(AB)-PD(BA)':
            cb_ticks = [-maxabs, 0, maxabs]
            vmin = -maxabs
            cmap = plt.cm.RdBu_r
        else:
            cb_ticks = [0, maxabs]
            vmin = 0
            cmap = plt.cm.plasma

        plt.imshow(x.T,
                   origin="lower",
                   interpolation="none",
                   aspect='auto',
                   vmin=vmin, vmax=maxabs,
                   cmap=cmap)
        ytick_spacing = 4
        xtick_spacing = 4
        plt.yticks(range(len(mi_params['hf_centers']))[::ytick_spacing],
                   mi_params['hf_centers'][::ytick_spacing])
        plt.xticks(range(len(mi_params['lf_centers']))[::xtick_spacing],
                   mi_params['lf_centers'][::xtick_spacing])
        plt.xlabel('LF freq (Hz)')
        plt.ylabel('HF freq (Hz)')
        cb = plt.colorbar(orientation='horizontal',
                          format='%.3f')
        cb.set_ticks(cb_ticks)
        if direc != 'PD(AB)-PD(BA)':
            cb.set_ticklabels(['0', f'{maxabs:.4f}'])
        cb.ax.set_xlabel('bits $^2$ / Hz')
        plt.ylabel('HF freq (Hz)')
        plt.xlabel('Phase freq (Hz)')
        plt.tight_layout()

    plt.savefig(f'{plot_dir}{sim_type}.pdf')


def run():
    for sim_type, sim_func in sim_funcs.items():
        print('\n', sim_type)
        results = []
        for i_subject in tqdm(range(n_subjects), desc='Subjects'):
            t, s_a, s_b = sim_func()  # Simulate the data
            res = pdc.pdc(s_a, s_b, **mi_params)  # Run the analysis
            results.append(res)

        plot_results_group(results, sim_type)


if __name__ == '__main__':
    run()
