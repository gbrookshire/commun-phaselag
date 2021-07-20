"""
Run the simulations of various patterns of data, and check whether the analysis
can reconstruct these patterns correctly.
"""

import numpy as np
from scipy import stats
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
sim_params = dict(dur=50,
                  fs=400,
                  noise_amp=1.5)
lag = 6
n_subjects = 25

sim_funcs = {
        'phase_dep_comm': lambda: simulate.sim(**sim_params),
        # 'lf_coh': lambda: simulate.sim_lf_coh_plus_noise(
        #     **sim_params, lag=lag),
        'lf_coh_plus_pac': lambda: simulate.sim_lf_coh_with_pac(
            **sim_params, lag=lag),
        'lf_coh_plus_hf_comm': lambda: simulate.sim_lf_coh_with_hf_comm(
            **sim_params, lag=lag)
        }

# Parameters for the MI phase-lag analysis
mi_params = dict(fs=sim_params['fs'],
                 lf_centers=lf_centers,
                 lf_bandwidth=lf_bandwidth,
                 hf_centers=hf_centers,
                 hf_bandwidth=hf_bandwidth,
                 lag=lag,
                 n_bins=2**3)


def plot_results_single(res, stat_info):
    # contour_color = {'PD(AB)-PD(BA)': 'black',
    #                  'PD(AB-BA)': 'white'}

    figsize = (8, 5)
    plt.figure(figsize=figsize)
    i_plot = 1
    for diff_type in ('PD(AB)-PD(BA)', 'PD(AB-BA)'):
        for direc in ('a', 'b', 'diff'):
            # Individual directions are the same across diff types
            # Only plot one of them
            if diff_type == 'PD(AB-BA)' and direc != 'diff':
                continue

            plt.subplot(2, 2, i_plot)
            i_plot += 1
            if direc == 'a':
                title = 'AB'
            elif direc == 'b':
                title = 'BA'
            else:
                # title = f'{diff_type}, p={stat_info[diff_type]["pval"]:.2f}'
                title = f'{diff_type}'
            plt.title(title)
            te_full = res
            # clust_info = stat_info

            te = te_full[diff_type]

            # z-transform
            te[direc] = stats.zscore(te[direc], axis=None)

            x = te[direc][0, :, :, 0]  # Plot the empirical data

            maxabs = np.max(np.abs(x))
            if diff_type == 'PD(AB)-PD(BA)' and direc == 'diff':
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
            plt.yticks(range(len(mi_params['f_car']))[::ytick_spacing],
                       mi_params['f_car'][::ytick_spacing])
            plt.xticks(range(len(mi_params['f_mod']))[::xtick_spacing],
                       mi_params['f_mod'][::xtick_spacing])
            plt.xlabel('LF freq (Hz)')
            plt.ylabel('HF freq (Hz)')
            cb = plt.colorbar(format='%.2f')
            # # Plot significant clusters
            # if direc == 'diff':
            #     cl_info = clust_info[diff_type]
            #     clust_labels = cl_info['labels'][:, :, 0].astype(int)
            #     thresh = np.percentile(cl_info['max_per_perm'],
            #                            [(1 - 0.05) * 100])
            #     for i_clust in np.unique(clust_labels):
            #         if i_clust == 0:
            #             continue
            #         elif cl_info['stats'][i_clust-1] > thresh:
            #             plt.contour(np.arange(len(mi_params['f_mod'])),
            #                         np.arange(len(mi_params['f_car'])),
            #                         (clust_labels == i_clust).T,
            #                         levels=[0.5],
            #                         colors=contour_color[diff_type])
            cb.set_ticks(cb_ticks)
            cb.ax.set_ylabel('bits $^2$ / Hz')
            plt.ylabel('HF freq (Hz)')
            plt.xlabel('Phase freq (Hz)')
        plt.tight_layout()


def plot_results_group(results, sim_type, save=True):
    figsize = (8, 5)
    plt.figure(f'avg: {sim_type}', figsize=figsize)
    plt.figure(f't: {sim_type}', figsize=figsize)
    i_plot = 0
    for direc in ('a', 'b', 'PD(AB)-PD(BA)', 'PD(AB-BA)'):

        if direc == 'a':
            title = 'AB'
        elif direc == 'b':
            title = 'BA'
        else:
            title = f'{direc}'

        te_all = np.stack([e[direc] for e in results])
        i_plot += 1

        for dv in ('avg', 't'):
            if dv == 'avg':  # The averaged values
                x = np.mean(te_all, axis=0)
            elif dv == 't':  # The t-statistics
                if direc in ('a', 'b', 'PD(AB-BA)'):
                    continue
                elif direc == 'PD(AB)-PD(BA)':
                    t, p = stats.ttest_1samp(te_all, popmean=0, axis=0)
                    x = t
            else:
                raise(Exception('Code should not get here'))

            plt.figure(f'{dv}: {sim_type}')

            plt.subplot(2, 2, i_plot)
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
            cb = plt.colorbar(format='%.1e')
            cb.set_ticks(cb_ticks)
            cb.ax.set_ylabel('bits $^2$ / Hz')
            plt.ylabel('HF freq (Hz)')
            plt.xlabel('Phase freq (Hz)')
            plt.tight_layout()

    for dv in ('avg', 't'):
        plt.figure(f'{dv}: {sim_type}', figsize=figsize)
        plt.savefig(f'{plot_dir}{sim_type}_{dv}.png')


for sim_type, sim_func in sim_funcs.items():
    print('\n', sim_type)
    results = []
    for i_subject in tqdm(range(n_subjects), desc='Subjects'):
        # Run the simulation
        t, s_a, s_b = sim_func()
        s = (s_a, s_b)

        # Split the data into epochs
        epoch_dur = 5.0
        epoch_len = int(epoch_dur * sim_params['fs'])
        n_splits = len(s[0]) // epoch_len
        sig_len = n_splits * epoch_len
        s = [np.stack(np.split(sig[:sig_len], n_splits), axis=1) for sig in s]
        s_a, s_b = s

        # Run the analysis
        # res, stat_info = comlag.cfc_phaselag_transferentropy(s_a, s_b,
        #                                                      **mi_params)
        res = pdc.pdc(s_a, s_b, **mi_params)
        results.append(res)

    plot_results_group(results, sim_type)
