"""
Run the simulations of various patterns of data, and check whether the analysis
can reconstruct these patterns correctly.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import simulate
import comlag


plt.ion()

# Low-freq 'modulator' frequencies
f_mod = np.arange(5, 15)
f_mod_bw = f_mod / 2.5  # ~4 cycles

# High-freq 'carrier' frequencies
f_car = np.arange(50, 120, 10)
f_car_bw = f_car / 3  # ~5 cycles

# Parameters for the simulated signals
sim_params = dict(dur=200,
                  fs=500,
                  noise_amp=1.0)
lag = 6

sim_funcs = {
        'phase_dep_comm': lambda: simulate.sim(**sim_params),
        # 'lf_coh': lambda: simulate.sim_lf_coh_plus_noise(
        #     **sim_params, lag=lag),
        # 'lf_coh_plus_pac': lambda: simulate.sim_lf_coh_with_pac(
        #     **sim_params, lag=lag),
        # 'lf_coh_plus_hf_comm': lambda: simulate.sim_lf_coh_with_hf_comm(
        #     **sim_params, lag=lag)
        }

# Parameters for the MI phase-lag analysis
mi_params = dict(fs=sim_params['fs'],
                 f_mod=f_mod,
                 f_mod_bw=f_mod_bw,
                 f_car=f_car,
                 f_car_bw=f_car_bw,
                 lag=[lag],
                 n_bins=2**3,
                 method='sine psd',
                 n_perm_signal=0,
                 perm_phasebin_flip=True,
                 cluster_alpha=0.05,
                 calc_type=2,
                 diff_method='both',
                 verbose=True)


def plot_results(res, stat_info):
    contour_color = {'PD(AB)-PD(BA)': 'black',
                     'PD(AB-BA)': 'white'}

    plt.figure(figsize=(8, 5))
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
                title = f'{diff_type}, p = {stat_info[diff_type]["pval"]:.2f}'
            plt.title(title)
            te_full = res
            clust_info = stat_info

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
            cb = plt.colorbar(format='%.2e')
            # Plot significant clusters
            if direc == 'diff':
                cl_info = clust_info[diff_type]
                clust_labels = cl_info['labels'][:, :, 0].astype(int)
                # signif_clusters = np.nonzero(
                #     np.array(cl_info['stats']) > cl_info['cluster_thresh'])
                # clust_highlight = np.isin(clust_labels,
                #                           1 + signif_clusters[0]).astype(int)
                thresh = np.percentile(cl_info['max_per_perm'],
                                       [(1 - 0.05) * 100])
                for i_clust in np.unique(clust_labels):
                    if i_clust == 0:
                        continue
                    elif cl_info['stats'][i_clust-1] > thresh:
                        plt.contour(np.arange(len(mi_params['f_mod'])),
                                    np.arange(len(mi_params['f_car'])),
                                    (clust_labels == i_clust).T,
                                    levels=[0.5],
                                    colors=contour_color[diff_type])
            cb.set_ticks(cb_ticks)
            cb.ax.set_ylabel('bits $^2$')
            plt.ylabel('HF freq (Hz)')
            plt.xlabel('Phase freq (Hz)')
        plt.tight_layout()


for sim_type, sim_func in sim_funcs.items():
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
    res, stat_info = comlag.cfc_phaselag_transferentropy(s_a, s_b, **mi_params)

    plot_results(res, stat_info)
