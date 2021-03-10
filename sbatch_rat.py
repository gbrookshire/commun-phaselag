#!/usr/bin/env python3
"""
Call with:
for iRat in {0..7}
do
    sbatch_submit.py \
        -s 'source load_python-simulated_rhythmic_sampling.sh' \
        -i "python sbatch_rat.py $iRat" \
        -t 24:00:00 -m 10G -d ../slurm_results/
done
"""



import sys
import datetime
import numpy as np
from scipy.io import loadmat
from scipy import signal
import comlag

now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

data_dir = '../data/RatData/'
fnames = ['EEG_speed_allData_Rat17_20120616_begin1.mat',
          'EEG_speed_allData_Rat6_20111021_begin1.mat',
          'EEG_speed_allData_Rat13_20120131_begin1_CA3_CSC9.mat',
          'EEG_speed_allData_Rat45_20140522_begin1.mat',
          'EEG_speed_allData_Rat44_20140506_begin1_CA3_CSC4_CA1_TT6.mat',
          'EEG_speed_allData_Rat47_20140923_begin1_CA3_CSC11_CA1_TT3.mat',
          'EEG_speed_allData_Rat31_20140110_begin1_CA3_CSC7_CA1_TT2.mat']

# Low-freq 'modulator' frequencies
# Jiang et al (2015): "a choice of 3-5 cycles in relation to the slower
# oscillation is sensible"
f_mod = np.arange(4, 21)
f_mod_bw = f_mod / 2.5 # ~4 cycles

# High-freq 'carrier' frequencies
# Jiang et al (2015): "a range of 4 to 6 cycles is appropriate when analyzing
# how gamma band power is related to the phase of slower oscillations."
f_car = np.arange(30, 150, 10)
f_car_bw = f_car / 3 # ~5 cycles


# Parameters for the MI phase-lag analysis
n_jobs = len(fnames)
downsamp_factor = 4 # 2000 Hz / 4 = 500 Hz
lag_sec = 0.006
mi_params = dict(f_mod=f_mod,
                 f_mod_bw=f_mod_bw,
                 f_car=f_car,
                 f_car_bw=f_car_bw,
                 n_bins=2**3,
                 method='sine psd',
                 decimate=None,
                 n_perm=100,
                 min_shift=None, max_shift=None, cluster_alpha=0.05,
                 calc_type=2)

def te_fnc(i_rat):
    """ Helper function for parallel computation
    """
    fn = fnames[i_rat]
    print(fn)

    # Load the data
    d = loadmat(data_dir + fn)
    fs = d['Fs'][0][0]
    s = [d['Data_EEG'][:,inx] for inx in [1, 2]]

    # Downsample the data
    s = [signal.decimate(sig, downsamp_factor) for sig in s] # Downsample
    fs /= downsamp_factor

    # Run the analysis
    lag = int(lag_sec * fs)
    te_out = comlag.cfc_phaselag_transferentropy(s[0], s[1],
                                                 fs=fs,
                                                 lag=[lag],
                                                 **mi_params)
    # Save the data
    save_fname = f"{data_dir}te/te_{now}_rat{i_rat}.npz"
    np.savez(save_fname, te=te_out, mi_params=mi_params, lag_sec=lag_sec)


if __name__ == '__main__':
    i_rat = sys.argv[1]
    assert i_rat.isnumeric(), \
            f'arg must be the index of the animal, got "{i_rat}"'
    i_rat = int(i_rat)
    te_fnc(i_rat)

