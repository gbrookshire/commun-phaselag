"""
Run the analyses and generate plots.
"""

import numpy as np
import matplotlib.pyplot as plt

import simulate
import comlag

plt.ion()
plt.clf()

#######################################
# Test the analysis on simulated data #
#######################################

dur = 100
fs = 1000
volume_conduction = 0.0
t, s_a, s_b = simulate.sim(dur=dur, fs=fs,
                           noise_amp=0.1,
                           signal_leakage=volume_conduction,
                           gamma_lag_a_to_b=0.050,
                           common_noise_amp=0.0,
                           common_alpha_amp=0.0)

plt.subplot(2, 1, 1)
plt.plot(t, s_a, label='$s_a$')
plt.plot(t, s_b, label='$s_b$')
plt.legend(loc='upper right')
plt.xlim(1, 2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')


# Which frequencies to calculate phase for
f_mod_centers = np.logspace(np.log10(2), np.log10(30), 20)
f_mod_width = f_mod_centers / 6
f_mod = np.tile(f_mod_width, [2, 1]).T \
            * np.tile([-1, 1], [len(f_mod_centers), 1]) \
            + np.tile(f_mod_centers, [2, 1]).T

# Which frequencies to calculate power for
f_car = np.arange(30, 200, 5)


#####################################
# Compute CFC using the Tort method #
#####################################

# CFC within signal A
mi_a = comlag.cfc_tort(s_a, s_a, fs, f_mod, f_car)
plt.subplot(2, 3, 4)
plt.contourf(np.mean(f_mod, axis=1), f_car, mi_a)
plt.xlabel('Phase freq (Hz)')
plt.ylabel('Amp freq (Hz)')
plt.title('CFC within $s_a$')
plt.colorbar()

# CFC from LF phase-diff between A and B to HG amp in B
mi = comlag.cfc_phasediff_tort(s_a, s_b, fs, f_mod, f_car)
plt.subplot(2, 3, 5)
plt.contourf(np.mean(f_mod, axis=1), f_car, mi['a'])
plt.xlabel('Phase-diff freq (Hz)')
#plt.ylabel('Amp freq (Hz)')
plt.title('CFC: phase-diff to $s_a$')
plt.colorbar()

plt.subplot(2, 3, 6)
plt.contourf(np.mean(f_mod, axis=1), f_car, mi['b'])
plt.xlabel('Phase-diff freq (Hz)')
#plt.ylabel('Amp freq (Hz)')
plt.title('CFC: phase-diff to $s_b$')
plt.colorbar()

plt.tight_layout()

plt.savefig('../data/plots/sim.pdf')
################################################
# Compute CFC using modified-Tort w/ sine-fits #
################################################

mi_amp, mi_r = comlag.cfc_sine(s_a, s_a, fs, f_mod, f_car)
plt.subplot(2, 3, 4)
plt.contourf(np.mean(f_mod, axis=1), f_car, mi_r)
plt.xlabel('Phase freq (Hz)')
plt.ylabel('Amp freq (Hz)')
plt.title('CFC within $s_a$')
plt.colorbar()

# CFC from LF phase-diff between A and B to HG amp in B
mi = comlag.cfc_phasediff_sine(s_a, s_b, fs, f_mod, f_car)
plt.subplot(2, 3, 5)
plt.contourf(np.mean(f_mod, axis=1), f_car, mi['a']['r'])
plt.xlabel('Phase-diff freq (Hz)')
plt.title('CFC: phase-diff to $s_a$')
plt.colorbar()

plt.subplot(2, 3, 6)
plt.contourf(np.mean(f_mod, axis=1), f_car, mi['b']['r'])
plt.xlabel('Phase-diff freq (Hz)')
plt.title('CFC: phase-diff to $s_b$')
plt.colorbar()

plt.tight_layout()


###########################################
# Compute CFC using cross-spectrum method #
###########################################

nfft = 2 ** 10
xlim = [0, 40]
mi, f_mod = comlag.cfc_xspect(s_a, s_a, fs, nfft, nfft / 2, f_car)
plt.subplot(2, 3, 4)
plt.contourf(f_mod, f_car, mi[:,:,0].T)
plt.xlabel('Phase freq (Hz)')
plt.ylabel('Amp freq (Hz)')
plt.xlim(xlim)
plt.title('CFC within $s_a$')
plt.colorbar()

# CFC from LF phase-diff between A and B to HG amp in B
mi, f_mod = comlag.cfc_phasediff_xspect(s_a, s_b, fs, nfft, nfft / 2, f_car)
plt.subplot(2, 3, 5)
plt.contourf(f_mod, f_car, mi['a'][:,:,0].T)
plt.xlabel('Phase-diff freq (Hz)')
#plt.ylabel('Amp freq (Hz)')
plt.xlim(xlim)
plt.title('CFC: phase-diff to $s_a$')
plt.colorbar()

plt.subplot(2, 3, 6)
plt.contourf(f_mod, f_car, mi['b'][:,:,0].T)
plt.xlabel('Phase-diff freq (Hz)')
#plt.ylabel('Amp freq (Hz)')
plt.xlim(xlim)
plt.title('CFC: phase-diff to $s_b$')
plt.colorbar()

plt.tight_layout()



k = 130 # Antiphase segment
k = 10 # In phase segment

i_f_car = 15
i_f_mod = 10
i_chan = 0
plt.clf()
plt.plot(x_split['a'][:,k])
plt.plot(x_split['b'][:,k])
plt.plot(amp_split['b'][:, i_f_car, i_chan, k])
np.angle(x_phasediff_fft[i_f_mod, k])

x_phasediff_fft[i_f_mod, k] * np.conj(amp_fft['b'][i_f_mod, i_f_car, i_chan, k])
x_phasediff_fft[i_f_mod, :, :, k] * np.conj(amp_fft['b'][i_f_mod, i_f_car, i_chan, k])
xspec['b'][i_f_mod, i_f_car, i_chan, k]

