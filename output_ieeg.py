"""
Test the analysis on iEEG data with simultaneous recordings of hippocampus and
cerebral cortex.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import mne
import pactools

import comlag

import sys
sys.path.append('../../wm-seeg/analysis/')
import wm

####################
# Read in the data #
####################

#TODO
epochs_cort = xxxxx
epochs_hipp = xxxxx


###################################
# Coherence between the two areas #
###################################

plt.figure()
nfft = 2 ** 8
est = pactools.utils.Coherence(block_length=nfft, fs=epochs.info['sfreq'])
# Swap axes to get the right data format: (n_signal, n_epoch, n_points)
coh = est.fit(np.swapaxes(epochs_cort.get_data(), 0, 1),
              np.swapaxes(epochs_hipp.get_data(), 0, 1))
freq = np.arange(1 + nfft / 2) * epochs.info['sfreq'] / nfft
plt.plot(freq, np.abs(np.squeeze(coh)).T)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence (norm)')
plt.xlim(2, 100)


#################
# Calculate CFC #
#################

# TODO Make this work with multiple channels

# Frequencies of the LF phase signal
f_mod_centers = np.logspace(np.log10(2), np.log10(30), 20)
f_mod_width = f_mod_centers / 6
f_mod = np.tile(f_mod_width, [2, 1]).T \
            * np.tile([-1, 1], [len(f_mod_centers), 1]) \
            + np.tile(f_mod_centers, [2, 1]).T

# Frequencies of the HF power signal
f_car = np.arange(30, 200, 5)

# Extract the data
x_cort = np.squeeze(epochs_cort.get_data()).T
x_hipp = np.squeeze(epochs_hipp.get_data()).T

cfc_args = [epochs.info['sfreq'], f_mod, f_car]
mi_cort = comlag.cfc_tort(x_cort, x_cort, *cfc_args)
mi_hipp = comlag.cfc_tort(x_hipp, x_hipp, *cfc_args)
mi_hipp_cort = comlag.cfc_tort(x_hipp, x_cort, *cfc_args)
mi_cort_hipp = comlag.cfc_tort(x_cort, x_hipp, *cfc_args)
mi_phase = comlag.cfc_phasediff_tort(x_hipp, x_cort, *cfc_args)

def plot_cfc(mi):
    plt.contourf(np.mean(f_mod, axis=1),
                    f_car, mi,
                    levels=np.linspace(0, mi.max(), 100))
    plt.ylabel('Amp freq (Hz)')
    plt.colorbar(ticks=[0, mi.max()],
                    format='%.3f')
    

