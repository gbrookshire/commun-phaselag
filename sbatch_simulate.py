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

data_options=("phase-dep-comm" "volume-cond-plus-noise" \
              "lf-coh-plus-noise" "lf-coh-plus-pac")
        
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
from tqdm import tqdm
import comlag
import simulate

# Get the command-line arguments
try:
    sim_type = sys.argv[1] # Type of data simulation
except IndexError:
    sim_type = 'phase-dep-comm'

now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

data_dir = '../data/'
plot_dir = f'{data_dir}plots/simulated/stats/'

## # Low-freq 'modulator' frequencies
## f_mod = np.arange(6, 15)
## f_mod_bw = f_mod / 2.5 # ~4 cycles
## 
## # High-freq 'carrier' frequencies
## f_car = np.arange(30, 150, 10)
## f_car_bw = f_car / 3 # ~5 cycles

# Low-freq 'modulator' frequencies
f_mod = np.array([10])
f_mod_bw = np.array([2.5])

# High-freq 'carrier' frequencies
f_car = np.array([80])
f_car_bw = np.array([20])


# Parameters for the simulated signals
lag = 6 # ms
sim_params = dict(dur=100,
                  fs=1000,
                  noise_amp=0.1,
                  common_noise_amp=0.1,
                  gamma_lag_a_to_b=(lag / 1000),
                  shared_gamma=True)

# Parameters for the MI phase-lag analysis
mi_params = dict(fs=sim_params['fs'],
                 f_mod=f_mod,
                 f_mod_bw=f_mod_bw,
                 f_car=f_car,
                 f_car_bw=f_car_bw,
                 lag=[lag],
                 n_bins=2**3,
                 method='sine psd',
                 n_perm_phasebin=0, # 1000,
                 n_perm_shift= 0, # 100
                 min_shift=None, max_shift=None,
                 cluster_alpha=0.05,
                 calc_type=2,
                 verbose=False)

# Determine which type of simulation to run
sim_opts = ['phase-dep-comm',
            'volume-cond-plus-noise',
            'lf-coh-plus-noise',
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

    #plt.savefig(f'{plot_dir}{fname}', dpi=300)
    plt.show()


def compare_shift_vs_phasebin():
    """
    Compare randomization with shifting the HF signal vs shuffling the
    communication scores across the phase bins
    """
    if sim_type == 'phase-dep-comm':
        t, s_a, s_b = simulate.sim(**sim_params)
    elif sim_type == 'volume-cond-plus-noise':
        t, s_a, s_b = simulate.sim_lf_coh_plus_noise(sim_params['dur'],
                                                     sim_params['fs'])
    elif sim_type == 'lf-coh-plus-noise':
        t, s_a, s_b = simulate.sim_lf_coh_plus_noise(sim_params['dur'],
                                                     sim_params['fs'],
                                                     lag=lag)
    elif sim_type == 'lf-coh-plus-pac':
        t, s_a, s_b = simulate.sim_lf_coh_with_pac(sim_params['dur'],
                                                   sim_params['fs'],
                                                   lag=lag)

    # Use each randomization scheme on the same underlying data

    ### In this simulation, using only 1 frequency band, this takes about 1.4 s
    # First run the analysis by permuting the base-bins
    rand_type = 'phasebin'
    mi_params['n_perm_phasebin'] = 1000
    mi_params['n_perm_shift'] = 0
    te, clust_stat_info = comlag.cfc_phaselag_transferentropy(s_a.copy(),
                                                              s_b.copy(),
                                                              **mi_params)
    fname = f'te_stats_{sim_type}_{rand_type}_{now}.png'
    generate_plots(te, clust_stat_info, fname)

    ### In this simulation, using only 1 frequency band, this takes about 2 min
    # Then run the analysis by randomly shifting the HF time-series
    rand_type = 'shift'
    mi_params['n_perm_phasebin'] = 0
    mi_params['n_perm_shift'] = 100
    te, clust_stat_info = comlag.cfc_phaselag_transferentropy(s_a.copy(),
                                                              s_b.copy(),
                                                              **mi_params)
    fname = f'te_stats_{sim_type}_{rand_type}_{now}.png'
    generate_plots(te, clust_stat_info, fname)



if __name__ == '__main__':

    #plt.ion()

    n_sim = 100

    rand_opts = ['phasebin', 'shift']

    colors = ['tab:green', 'tab:purple']

    for i_plot,sim_type in enumerate(sim_opts):
        plt.subplot(2, 2, i_plot + 1)
        
        for i_rand,rand_type in enumerate(rand_opts):
            if rand_type == 'phasebin':
                mi_params['n_perm_phasebin'] = 1000
                mi_params['n_perm_shift'] = 0
            elif rand_type == 'shift':
                mi_params['n_perm_phasebin'] = 0
                mi_params['n_perm_shift'] = 100

            pvals = []
            for k in tqdm(range(n_sim)):

                if sim_type == 'phase-dep-comm':
                    t, s_a, s_b = simulate.sim(**sim_params)
                elif sim_type == 'volume-cond-plus-noise':
                    t, s_a, s_b = simulate.sim_lf_coh_plus_noise(
                                            sim_params['dur'],
                                            sim_params['fs'],
                                            noise_amp=2)
                elif sim_type == 'lf-coh-plus-noise':
                    t, s_a, s_b = simulate.sim_lf_coh_plus_noise(
                                            sim_params['dur'],
                                            sim_params['fs'],
                                            lag=mi_params['lag'],
                                            noise_amp=2)
                elif sim_type == 'lf-coh-plus-pac':
                    t, s_a, s_b = simulate.sim_lf_coh_with_pac(
                                            sim_params['dur'],
                                            sim_params['fs'],
                                            lag=mi_params['lag'])
                else:
                    msg = f"simulation type '{sim_type}' not recognized"
                    raise(Exception(msg))

                # Run the analysis and save the p-value
                try:
                    te, clust_stat_info = comlag.cfc_phaselag_transferentropy(
                                                    s_a,
                                                    s_b,
                                                    **mi_params)
                    # Cluster-based statistic is not sensible when there's only 1
                    # frequency bin. If there's only 1 frequency bin, use the value of
                    # the TE-difference.
                    if len(f_mod) > 1:
                        p = clust_stat_info['pval']
                    else:
                        x = te['diff'].flatten()
                        p = np.mean(x[1:] > x[0])

                except np.linalg.LinAlgError:
                    print('Linear algebra error')
                    p = np.nan

                pvals.append(p)

            pvals = np.array(pvals)

            # Plot a histogram of the results
            plt.hist(pvals, np.arange(0, 1, 0.05),
                     color=colors[i_rand], alpha=0.5)
            plt.xlabel('P value')
            plt.ylabel('Count')
            plt.xlim(0, 1)
            plt.ylim(0, n_sim)
            plt.axvline(x=0.05, color='red', linestyle='--')
            plt.title(sim_type)
            plt.text(0.3, n_sim * (1 - ((1 + i_rand) / 6)),
                    f"P(Signif) = {np.mean(pvals <= 0.05)}",
                    color=colors[i_rand])

    fname = 'randomization_showdown.png'
    plt.savefig(f'{plot_dir}{fname}', dpi=300)



