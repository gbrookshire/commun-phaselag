#!/usr/bin/env python3
"""
Call with:
# ratioLevels=$(seq 1.5 0.5 6)  # Full tiling analysis
ratioLevels=$(seq 2 1 6)  # Reduced levels for SNR computation
for iRat in {0..7}
do
    for lfRatio in $ratioLevels
    do
        for hfRatio in $ratioLevels
        do
            echo $iRat $lfRatio $hfRatio
            sbatch_submit.py \
                -s 'source load_python-simulated_rhythmic_sampling.sh' \
                -i "python sbatch_rat.py $iRat $lfRatio $hfRatio" \
                -t 48:00:00 -m 20G -c 5 -d ../slurm_results/
        done
    done
done
"""

import sys
import datetime
import copy
import numpy as np
from scipy.io import loadmat
from scipy import signal
import comlag

now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

base_data_dir = '../'  # Bluebear
# base_data_dir = '/media/geoff/Seagate2TB1/geoff/commun-phaselag/'  # Desktop
data_dir = base_data_dir + 'data/RatData/'

fnames = ['EEG_speed_allData_Rat17_20120616_begin1.mat',
          'EEG_speed_allData_Rat6_20111021_begin1.mat',
          'EEG_speed_allData_Rat13_20120131_begin1_CA3_CSC9.mat',
          'EEG_speed_allData_Rat45_20140522_begin1.mat',
          'EEG_speed_allData_Rat44_20140506_begin1_CA3_CSC4_CA1_TT6.mat',
          'EEG_speed_allData_Rat47_20140923_begin1_CA3_CSC11_CA1_TT3.mat',
          'EEG_speed_allData_Rat31_20140110_begin1_CA3_CSC7_CA1_TT2.mat']

f_bw_ratios = np.arange(1.5, 6.1, 0.5)  # From ~2 to ~10 cycles
f_mod = np.arange(4, 16)  # Centers of the LF filters (Hz)
f_car = np.arange(30, 150, 5)  # Centers of the HF filters (Hz)
downsamp_factor = 5  # Factor by which to downsample the data
epoch_dur = 5.0  # in seconds

# Parameters for the MI phase-lag analysis
k_perm = 0
lag_sec = 0.006
mi_params = dict(f_mod=f_mod,
                 f_mod_bw=None,
                 f_car=f_car,
                 f_car_bw=None,
                 n_bins=2**3,
                 decimate=None,
                 n_perm_phasebin=0,
                 n_perm_phasebin_indiv=0,
                 n_perm_signal=100,
                 n_perm_shift=0,
                 min_shift=None, max_shift=None,
                 cluster_alpha=0.05,
                 diff_method='both',
                 calc_type=2,
                 method='sine psd',
                 return_phase_bins=True,
                 verbose=True)


def te_fnc(i_rat, lf_ratio, hf_ratio):
    """ Helper function for parallel computation
    """
    fn = fnames[i_rat]
    print(fn, lf_ratio, hf_ratio)

    # Load the data
    d = loadmat(data_dir + fn)
    fs = d['Fs'][0][0]
    s = [d['Data_EEG'][:, inx] for inx in [1, 2]]

    # Downsample the data
    if downsamp_factor is not None:
        s = [signal.decimate(sig, downsamp_factor) for sig in s]
        fs /= downsamp_factor

    # Split the data into epochs
    epoch_len = int(epoch_dur * fs)
    n_splits = len(s[0]) // epoch_len
    sig_len = n_splits * epoch_len
    s = [np.stack(np.split(sig[:sig_len], n_splits), axis=1) for sig in s]

    # Get the filters ready
    mip = copy.deepcopy(mi_params)
    mip['f_mod_bw'] = mip['f_mod'] / lf_ratio
    mip['f_car_bw'] = mip['f_car'] / hf_ratio

    lag = int(lag_sec * fs)

    te_out = comlag.cfc_phaselag_transferentropy(s[0], s[1],
                                                 fs=fs,
                                                 lag=[lag],
                                                 **mip)

    # Save the data
    save_fname = \
        f"te_{now}_rat{i_rat}_lfratio-{lf_ratio}_hfratio-{hf_ratio}.npz"
    save_fname = f"{data_dir}te/{save_fname}"
    np.savez(save_fname, te=te_out, mi_params=mi_params, lag_sec=lag_sec)
    print(save_fname)


def run_single_rat_sbatch():
    i_rat = int(sys.argv[1])
    lf_ratio = float(sys.argv[2])
    hf_ratio = float(sys.argv[3])
    te_fnc(i_rat, lf_ratio, hf_ratio)
    msg = f"i_rat: {i_rat}, lf ratio: {lf_ratio:.2f}, hf ratio: {hf_ratio:.2f}"
    print(msg)


def run_all_rats_desktop():
    for i_rat in range(len(fnames)):
        print(f'Analyzing rat {i_rat}')
        te_fnc(i_rat, None)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print('Running single analysis on sbatch')
        run_single_rat_sbatch()
    else:
        print('Running all rat analyses without sbatch')
        run_all_rats_desktop()
