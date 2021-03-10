#!/usr/bin/env python3
"""
Run simulations, including statistics
- Randomization
    - Randomly shift the HF signal
    - Randomly shuffle the phase bins
- Different types of signals
    - HF communication depends on LF phase difference
    - Coherent LF oscillations plus pink noise
        - With and without a lag between signals
    - Coherent LF oscillations with independent PAC in each signal

Call in bash like this: 

data_options=("phase-dep-comm" "lf-coh-plus-noise" \
              "lf-coh-plus-noise-lag" "lf-coh-plus-pac")
        
for data_type in ${data_options[*]}
do
    sbatch_submit.py \
        -s 'source load_python-simulated_rhythmic_sampling.sh' \
        -i "python sbatch_simulate.py $data_type" \
        -t 12:00:00 -m 10G -d ../slurm_results/ -c 5
done


"""

import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt
import comlag
import simulate

# Get the command-line arguments
sim_type = sys.argv[1] # Type of data simulation

now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

data_dir = '../data/'
plot_dir = f'{data_dir}plots/simulated/stats/'

# Low-freq 'modulator' frequencies
f_mod = np.arange(6, 15)
f_mod_bw = f_mod / 2.5 # ~4 cycles

# High-freq 'carrier' frequencies
f_car = np.arange(30, 150, 10)
f_car_bw = f_car / 3 # ~5 cycles

# Parameters for the simulated signals
sim_params = dict(dur=100,
                  fs=1000,
                  noise_amp=0.1,
                  common_noise_amp=0.1,
                  shared_gamma=True)

# Parameters for the MI phase-lag analysis
mi_params = dict(fs=sim_params['fs'],
                 f_mod=f_mod,
                 f_mod_bw=f_mod_bw,
                 f_car=f_car,
                 f_car_bw=f_car_bw,
                 lag=[15],
                 n_bins=2**3,
                 method='sine psd',
                 n_perm_phasebin=0, # 1000,
                 n_perm_shift= 0, # 100
                 min_shift=None, max_shift=None,
                 cluster_alpha=0.05,
                 calc_type=2)

# Determine which type of simulation to run
sim_opts = ['phase-dep-comm',
            'lf-coh-plus-noise',
            'lf-coh-plus-noise-lag',
            'lf-coh-plus-pac']
assert sim_type in sim_opts, f'Simulation type "{sim_type}" not recognized'


def generate_plots(te, clust_stat_info, fname):
    # Plot the raw signals
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 1, 1)
    plt.plot(s_a)
    plt.plot(s_b)
    plt.xlim(0, 1000)
    plt.title('Raw signals')

    # Which permutation to plot (0 is the empirical data)
    i_perm = 0

    for n_plot, lagged_sig in enumerate('ab'):
        plt.subplot(2, 3, 4 + n_plot)
        x = te[lagged_sig][i_perm, :, :, 0]
        plt.contourf(f_mod, f_car, x.T)
        cb = plt.colorbar(format='%.2f')
        cb.ax.set_ylabel('I (bits)')
        plt.ylabel('HF freq (Hz)')
        plt.xlabel('Phase freq (Hz)')
        plt.title(f'Lagged: {lagged_sig}')

    plt.subplot(2, 3, 6)
    # Plot the comodulogram
    x = te['diff'][i_perm, :, :, 0]
    levels = np.linspace(*np.array([-1, 1]) * np.max(np.abs(x)), 50)
    plt.contourf(f_mod, f_car, x.T,
                levels=levels,
                cmap=plt.cm.RdBu_r)
    cb = plt.colorbar(format='%.2f')
    # Plot significant clusters
    clust_labels = clust_stat_info['labels'][:,:,0]
    signif_clusters = np.nonzero(
            np.array(clust_stat_info['stats']) > clust_stat_info['cluster_thresh'])
    clust_highlight = np.isin(clust_labels, 1 + signif_clusters[0]).astype(int)
    plt.contour(f_mod, f_car,
                clust_highlight.T,
                levels=[0.5],
                colors='black')
    cb.ax.set_ylabel('Diff (bits)')
    plt.ylabel('HF freq (Hz)')
    plt.xlabel('Phase freq (Hz)')
    plt.title(f'p = {clust_stat_info["pval"]:.3f}')
    plt.tight_layout()

    plt.savefig(f'{plot_dir}{fname}', dpi=300)
    #plt.show()


if __name__ == '__main__':

    if sim_type == 'phase-dep-comm':
        t, s_a, s_b = simulate.sim(**sim_params)
    elif sim_type == 'lf-coh-plus-noise':
        t, s_a, s_b = simulate.sim_lf_coh_plus_noise(sim_params['dur'],
                                                     sim_params['fs'])
    elif sim_type == 'lf-coh-plus-noise-lag':
        t, s_a, s_b = simulate.sim_lf_coh_plus_noise(sim_params['dur'],
                                                     sim_params['fs'],
                                                     lag=15)
    elif sim_type == 'lf-coh-plus-pac':
        t, s_a, s_b = simulate.sim_lf_coh_with_pac(sim_params['dur'],
                                                   sim_params['fs'])

    # Use each randomization scheme on the same underlying data

    # First run the analysis by permuting the base-bins
    rand_type = 'phasebin'
    mi_params['n_perm_phasebin'] = 1000
    mi_params['n_perm_shift'] = 0
    te, clust_stat_info = comlag.cfc_phaselag_transferentropy(s_a.copy(),
                                                              s_b.copy(),
                                                              **mi_params)
    fname = f'te_stats_{sim_type}_{rand_type}_{now}.png'
    generate_plots(te, clust_stat_info, fname)

    # Then run the analysis by randomly shifting the HF time-series
    rand_type = 'shift'
    mi_params['n_perm_phasebin'] = 0
    mi_params['n_perm_shift'] = 100
    te, clust_stat_info = comlag.cfc_phaselag_transferentropy(s_a.copy(),
                                                              s_b.copy(),
                                                              **mi_params)
    fname = f'te_stats_{sim_type}_{rand_type}_{now}.png'
    generate_plots(te, clust_stat_info, fname)




