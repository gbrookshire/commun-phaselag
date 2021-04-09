#!/usr/bin/env python3
"""
Call with:
for iRat in {0..7}
do
    for permType in signal shift
    do
        sbatch_submit.py \
            -s 'source load_python-simulated_rhythmic_sampling.sh' \
            -i "python sbatch_rat.py $iRat $permType" \
            -t 72:00:00 -m 10G -c 10 -d ../slurm_results/
    done
done
"""

import sys
import datetime
import numpy as np
from scipy.io import loadmat
from scipy import signal
import comlag

now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

base_data_dir = '../' # Bluebear
base_data_dir = '/media/geoff/Seagate2TB1/geoff/commun-phaselag/' # Desktop
data_dir = base_data_dir + 'data/RatData/' 

fnames = ['EEG_speed_allData_Rat17_20120616_begin1.mat',
          'EEG_speed_allData_Rat6_20111021_begin1.mat',
          'EEG_speed_allData_Rat13_20120131_begin1_CA3_CSC9.mat',
          'EEG_speed_allData_Rat45_20140522_begin1.mat',
          'EEG_speed_allData_Rat44_20140506_begin1_CA3_CSC4_CA1_TT6.mat',
          'EEG_speed_allData_Rat47_20140923_begin1_CA3_CSC11_CA1_TT3.mat',
          'EEG_speed_allData_Rat31_20140110_begin1_CA3_CSC7_CA1_TT2.mat']


## ## Parameters from Jiang
## ## On 4/9, we decided to use the other parameter set
## # Low-freq 'modulator' frequencies
## # Jiang et al (2015): "a choice of 3-5 cycles in relation to the slower
## # oscillation is sensible"
## f_mod = np.arange(4, 16)
## f_mod_bw = f_mod / 2.5 # ~4 cycles
## 
## # High-freq 'carrier' frequencies
## # Jiang et al (2015): "a range of 4 to 6 cycles is appropriate when analyzing
## # how gamma band power is related to the phase of slower oscillations."
## f_car = np.arange(30, 150, 10)
## f_car_bw = f_car / 3 # ~5 cycles
##
## downsamp_factor = 5 # 2000 Hz / 5 = 400 Hz

## ## Parameters from Feb 17
f_mod = np.logspace(np.log10(4), np.log10(20), 15)
f_mod_bw = f_mod / 2
f_car = np.arange(20, 150, 10)
f_car_bw = f_car / 80 * 20 # Keep 20 Hz bandwidth at 80 Hz (~ 7 cycles)
downsamp_factor = 5

# Parameters for the MI phase-lag analysis
k_perm = 500
lag_sec = 0.006
mi_params = dict(f_mod=f_mod,
                 f_mod_bw=f_mod_bw,
                 f_car=f_car,
                 f_car_bw=f_car_bw,
                 n_bins=2**3,
                 decimate=None,
                 n_perm_phasebin=0,
                 n_perm_phasebin_indiv=0,
                 n_perm_signal=0,
                 n_perm_shift=0,
                 min_shift=None, max_shift=None,
                 cluster_alpha=0.05,
                 diff_method='both',
                 calc_type=2,
                 method='sine psd',
                 verbose=True)


def te_fnc(i_rat, perm_type):
    """ Helper function for parallel computation
    """
    fn = fnames[i_rat]
    print(fn)

    # Load the data
    d = loadmat(data_dir + fn)
    fs = d['Fs'][0][0]
    s = [d['Data_EEG'][:,inx] for inx in [1, 2]]

    # Downsample the data
    if downsamp_factor is not None:
        s = [signal.decimate(sig, downsamp_factor) for sig in s]
        fs /= downsamp_factor

    lag = int(lag_sec * fs)

    if perm_type == 'shift':
        mi_params['n_perm_shift'] = k_perm
        mi_params['n_perm_signal'] = 0

    elif perm_type == 'signal':
        mi_params['n_perm_shift'] = 0
        mi_params['n_perm_signal'] = k_perm

        # Split the data into epochs for the signal-permuting analysis
        epoch_dur = 1 # seconds
        epoch_length = int(epoch_dur * fs) # Samples
        n_samps_to_keep = len(s[0]) // epoch_length * epoch_length
        n_splits = n_samps_to_keep / epoch_length
        s = [np.stack(np.split(sig[:n_samps_to_keep], n_splits), axis=1)
                for sig in s]
    elif perm_type == None:
        mi_params['n_perm_shift'] = 0
        mi_params['n_perm_signal'] = 0
    else:
        raise(NotImplementedError(f"perm_type {perm_type} is not supported"))

    te_out = comlag.cfc_phaselag_transferentropy(s[0], s[1],
                                                fs=fs,
                                                lag=[lag],
                                                **mi_params)

    # Save the data
    save_fname = f"{data_dir}te/te_{now}_rat{i_rat}_{perm_type}.npz"
    np.savez(save_fname, te=te_out, mi_params=mi_params, lag_sec=lag_sec)
    print(save_fname)


def run_single_rat_sbatch():
    i_rat = sys.argv[1]
    perm_type = sys.argv[2]
    assert i_rat.isnumeric(), \
            f'arg must be the index of the animal, got "{i_rat}"'
    i_rat = int(i_rat)
    te_fnc(i_rat, perm_type)


def run_all_rats_desktop():
    for i_rat in range(len(fnames)):
        print(f'Analyzing rat {i_rat}')
        te_fnc(i_rat, None)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print('Running single analysis on sbatch')
        run_single_rat_on_sbatch()
    else:
        print('Running all rat analyses without sbatch')
        run_all_rats_desktop()
