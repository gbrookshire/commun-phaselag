"""
Test the analysis on real data looking at beta/gamma coupling in MEG & EMG
"""

import re
import mne

import numpy as np
import matplotlib.pyplot as plt

import comlag

raw = mne.io.read_raw_ctf('../data/SubjectCMC/SubjectCMC.ds')
# This has numbers after the chan names -- remove them
new_names = [re.sub("-\d+", "", e) for e in raw.ch_names]
raw.rename_channels({old:new for old,new in zip(raw.ch_names, new_names)})
# Read in the epochs from Fieldtrip
epochs = mne.io.fieldtrip.read_epochs_fieldtrip(
            fname='../data/SubjectCMC/data.mat',
            info=raw.info)

# Read in the data with no info section
epochs_meg = epochs.copy()
epochs_meg.pick_channels(['MRC21'])
epochs_emg = epochs.copy()
epochs_emg.pick_channels(['EMGlft'])

# Look at coherence between EMG and MEG
# It doesn't look as nice as in FT. Why not? Some params must be different
import pactools
plt.figure()
nfft = 2**8
est = pactools.utils.Coherence(block_length=nfft, fs=epochs.info['sfreq'])
# Swap axes to get the right data format: (n_signal, n_epoch, n_points)
coh = est.fit(np.swapaxes(epochs_meg.get_data(), 0, 1), 
                np.swapaxes(epochs_emg.get_data(), 0, 1))
freq = np.arange(1 + nfft / 2) * epochs.info['sfreq'] / nfft
plt.plot(freq, np.abs(np.squeeze(coh)).T)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence (norm)')
plt.xlim(2, 100)

# Look at CFC within EMG
# TODO figure out how to make it work with multiple channels

# Which frequencies to calculate phase for
f_mod_centers = np.logspace(np.log10(2), np.log10(30), 20)
f_mod_width = f_mod_centers / 6
f_mod = np.tile(f_mod_width, [2, 1]).T \
            * np.tile([-1, 1], [len(f_mod_centers), 1]) \
            + np.tile(f_mod_centers, [2, 1]).T

# Which frequencies to calculate power for
f_car = np.arange(30, 200, 5)

x_meg = np.squeeze(epochs_meg.get_data()).T
x_emg = np.squeeze(epochs_emg.get_data()).T

mi_emg = comlag.cfc_tort(x_emg, x_emg, epochs.info['sfreq'], f_mod, f_car)
mi_meg = comlag.cfc_tort(x_meg, x_meg, epochs.info['sfreq'], f_mod, f_car)
mi_meg_emg = comlag.cfc_tort(x_meg, x_emg, epochs.info['sfreq'], f_mod, f_car)
mi_emg_meg = comlag.cfc_tort(x_emg, x_meg, epochs.info['sfreq'], f_mod, f_car)
mi_phase = comlag.cfc_phasediff_tort(x_meg, x_emg, epochs.info['sfreq'],
                                        f_mod, f_car)

def plot_cfc(mi):
    plt.contourf(np.mean(f_mod, axis=1),
                    f_car, mi,
                    levels=np.linspace(0, mi.max(), 100))
    plt.ylabel('Amp freq (Hz)')
    plt.colorbar(ticks=[0, mi.max()],
                    format='%.3f')
    
plt.figure()

plt.subplot(3,2,1)
plot_cfc(mi_meg)
plt.xlabel('Phase freq (Hz)')
plt.title('CFC within MEG')

plt.subplot(3,2,2)
plot_cfc(mi_emg)
plt.xlabel('Phase freq (Hz)')
plt.title('CFC within EMG')

plt.subplot(3,2,3)
plot_cfc(mi_meg_emg)
plt.xlabel('Phase freq (Hz)')
plt.title('Phase(MEG) to Amp(EMG)')

plt.subplot(3,2,4)
plot_cfc(mi_emg_meg)
plt.xlabel('Phase freq (Hz)')
plt.title('Phase(EMG) to Amp(MEG)')

# CFC from LF phase-diff between MEG and MEG to HG amp in EGM (or MEG)
plt.subplot(3,2,5)
plot_cfc(mi_phase['a'])
plt.xlabel('Phase-diff freq (Hz)')
plt.title('Phase-diff to MEG')

plt.subplot(3,2,6)
plot_cfc(mi_phase['b'])
plt.xlabel('Phase-diff freq (Hz)')
plt.title('Phase-diff to EMG')

plt.tight_layout()


