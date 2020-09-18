"""
Run the analyses and generate plots.
"""

import numpy as np
import matplotlib.pyplot as plt

import simulate
import comlag

plt.ion()

#######################################
# Test the analysis on simulated data #
#######################################

dur = 100
fs = 1000
volume_conduction = 0.0
t, s_a, s_b = simulate.sim(dur=dur, fs=fs,
                           noise_amp=0.5,
                           signal_leakage=volume_conduction,
                           gamma_lag_a=0.010,
                           gamma_lag_a_to_b=0.015,
                           common_noise_amp=0.0,
                           common_alpha_amp=0.0)

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

plt.clf()
plt.subplot(2, 1, 1)
plt.plot(t, s_a, label='$s_a$')
plt.plot(t, s_b, label='$s_b$')
plt.legend(loc='upper right')
plt.xlim(1, 2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

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
plt.savefig(f'../data/plots/sim_tort.png')


################################################
# Compute CFC using modified-Tort w/ sine-fits #
################################################

plt.clf()
plt.subplot(2, 1, 1)
plt.plot(t, s_a, label='$s_a$')
plt.plot(t, s_b, label='$s_b$')
plt.legend(loc='upper right')
plt.xlim(1, 2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

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
plt.savefig(f'../data/plots/sim_fit_sine.png')


###########################################
# Compute CFC using cross-spectrum method #
###########################################

plt.clf()
plt.subplot(2, 1, 1)
plt.plot(t, s_a, label='$s_a$')
plt.plot(t, s_b, label='$s_b$')
plt.legend(loc='upper right')
plt.xlim(1, 2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

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
plt.savefig(f'../data/plots/sim_xspect.png')


#########################################
# Compute CFC using 2D Von Mises method #
#########################################

fits, rsq = comlag.cfc_vonmises_2d(s_a, s_b, fs, f_mod, f_car)

def plot_contour(x, **kwargs):
    plt.contourf(f_mod_centers, f_car, x.T, **kwargs)
    plt.colorbar(format='%.2f', ticks=[x.min(), x.max()])
    plt.ylabel('Amp freq (Hz)')
    plt.xlabel('Phase freq (Hz)')

plt.clf()

plt.subplot(2, 2, 1)
plot_contour(fits['a'][:, :, 0], levels=50)
plt.title('$\\kappa$: phase in A')

plt.subplot(2, 2, 2)
plot_contour(fits['b'][:, :, 0], levels=50)
plt.title('$\\kappa$: phase in B')

plt.subplot(2, 2, 3)
plot_contour(fits['2d'][:, :, 0], levels=50)
plt.title('$\\kappa$: combined phase')

plt.subplot(2, 2, 4)
plot_contour(fits['2d_cont'][:, :, 0], levels=50)
plt.title('$\\kappa$: combined (controlled)')

plt.tight_layout()


##########################
###### Sketchpad #########
##########################

#######################################################
# 1) HF power in region 2 as a function of phase diff #
#######################################################

from comlag import *
from comlag import _wavelet_tfr, _buffer, _match_dims

f_mod = [[7, 14]]
f_car = [90]
n_cycles = 5
n_bins = 18
phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

fm = f_mod[0]
i_fc = 0

x = {'a': s_a, 'b': s_b}
amp = {}
phase = {}
for sig in 'ab':

    # Get high frequency amplitude using a wavelet transform
    amp[sig] = _wavelet_tfr(x[sig], f_car, n_cycles, fs)

    # Compute LF phase
    s_filt = bp_filter(x[sig].T, fm[0], fm[1], fs, 2).T
    s_phase = np.angle(hilbert(s_filt, axis=0))
    phase[sig] = s_phase

plt.clf()

# Plot HF power as a function phase-difference.
# On first glance, it looks pretty good.
# Compute the phase difference
phase_diff = phase['a'] - phase['b']
phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi # wrap to +/-pi
phase_diff = np.digitize(phase_diff, phase_bins) - 1 # Binned
# Average HF amplitude per LF phase bin
amplitude_dist = np.ones(n_bins) # default is 1 to avoid log(0)
for phase_bin in np.unique(phase_diff):
    amplitude_dist[phase_bin] = np.mean(amp['b'][phase_diff == phase_bin, i_fc])
# Plot the result
plt.subplot(2, 1, 1)
plt.plot(phase_bins[:-1], amplitude_dist)
plt.xticks([-np.pi, 0, np.pi], ['$-\\pi$', '0', '$\\pi$'])
plt.xlabel('LF phase difference (rad)')
plt.ylabel('HF power')

# Average HF amplitude per LF phase bin in BOTH signals.
# When we look at HF power as a function of phase in each signal, we can see
# that it's not phase-difference per se that determines HF power -- if it were,
# HF power would follow a downward diagonal line. Instead, HF power depends on
# a specific combination of LF phase in each signal.
phase_dig = {} # Digitize the phase into bins
for sig in 'ab':
    phase_dig[sig] = np.digitize(phase[sig], phase_bins) - 1
amplitude_dist = np.ones([n_bins, n_bins]) # default is 1 to avoid log(0)
for bin_a in np.unique(phase_diff):
    for bin_b in np.unique(phase_diff):
        phase_sel = (phase_dig['a'] == bin_a) & (phase_dig['b'] == bin_b)
        amplitude_dist[bin_a, bin_b] = np.mean(amp['b'][phase_sel, i_fc])
plt.subplot(2, 2, 3)
plt.imshow(amplitude_dist)
plt.xlabel('Binned LF phase in B')
plt.ylabel('Binned LF phase in A')
plt.colorbar(label='HF power in B')

# Plot the phase difference as a function of phase in each signal. If phase
# difference directly determines HF power, we should see a line along one of
# the equal color values in this plot.
plt.subplot(2, 2, 4)
phase_bin_mat = np.tile(phase_bins, [len(phase_bins), 1])
phase_diff_mat = phase_bin_mat - phase_bin_mat.T
phase_diff_mat = (phase_diff_mat + np.pi) % (2 * np.pi) - np.pi # wrap to +/-pi
plt.imshow(phase_diff_mat, cmap=plt.cm.twilight)
plt.colorbar(label='Phase difference')
plt.xlabel('Binned LF phase in B')
plt.ylabel('Binned LF phase in A')

plt.tight_layout()


######################################################################
# 2) HF coherence between region 1 and 2 as a function of phase diff #
######################################################################

# I'm not actually sure how to do this



####################################################################
# 3) HF mutual info b/w region 1 and 2 as a function of phase diff #
####################################################################

# Use the setup from (1)
import gcmi

filt = {}
x_2d = {}
for sig in 'ab':
    # Filter into the HG band
    filt = bp_filter(x[sig], 70, 150, fs)
    # Make a 2d version of the signal with it's Hilbert transform
    h = hilbert(filt)
    sig_2d = np.stack([np.real(h), np.imag(h)])
    x_2d[sig] = sig_2d

plt.clf()

# MI between HF signals per LF phase bin
mi = [] 
for phase_bin in np.unique(phase_diff):
    phase_sel = phase_diff == phase_bin
    i = gcmi.gcmi_cc(x_2d['a'][:, phase_sel],
                     x_2d['b'][:, phase_sel])
    mi.append(i)
plt.subplot(2, 1, 1)
plt.plot(phase_bins[:-1], mi)
plt.xticks([-np.pi, 0, np.pi], ['$-\\pi$', '0', '$\\pi$'])
plt.xlabel('LF phase difference (rad)')
plt.ylabel('I(A; B)')

# MI between HF signals per LF phase in each signal
mi = np.full([n_bins, n_bins], np.nan) 
for bin_a in np.unique(phase_diff):
    for bin_b in np.unique(phase_diff):
        phase_sel = (phase_dig['a'] == bin_a) & (phase_dig['b'] == bin_b)
        i = gcmi.gcmi_cc(x_2d['a'][:, phase_sel],
                         x_2d['b'][:, phase_sel])
        mi[bin_a, bin_b] = i
plt.subplot(2, 2, 3)
plt.imshow(mi)
plt.xlabel('Binned LF phase in B')
plt.ylabel('Binned LF phase in A')
plt.colorbar(label='I(A; B)')

# Plot the phase difference as a function of phase in each signal
plt.subplot(2, 2, 4)
phase_bin_mat = np.tile(phase_bins, [len(phase_bins), 1])
phase_diff_mat = phase_bin_mat - phase_bin_mat.T
phase_diff_mat = (phase_diff_mat + np.pi) % (2 * np.pi) - np.pi # wrap to +/-pi
plt.imshow(phase_diff_mat, cmap=plt.cm.twilight)
plt.colorbar(label='Phase difference')
plt.xlabel('Binned LF phase in B')
plt.ylabel('Binned LF phase in A')

plt.tight_layout()


#########################################
# Phase difference vs phase combination #
#########################################

# Explicitly compare the two models. Is it phase-difference that's important,
# or phase combination?

# Get the average HF power for each time-point
# Compare 2 regressions
#   sine-cosine transform of the phase difference
#   sine-cosine transform of phase of each signal

import statsmodels.api as sm
from matplotlib import gridspec

y = amp['b']
y = np.squeeze(y)
y = 10 * np.log10(y)

# Fit a model based on phase-difference
x_phasediff = phase['a'] - phase['b']
x_phasediff = (x_phasediff + np.pi) % (2 * np.pi) - np.pi # wrap to +/-pi
x_phasediff = np.stack([np.sin(x_phasediff), # Sine-cosine transform
                        np.cos(x_phasediff)]).T
x_phasediff = sm.add_constant(x_phasediff)
# Fit the model
model_phasediff = sm.OLS(y, x_phasediff)
results_phasediff = model_phasediff.fit()
print(results_phasediff.summary())

# Fit a model based on separate phase in each signal
x_separate = np.stack([np.sin(phase['a']),
                       np.cos(phase['a']),
                       np.sin(phase['b']),
                       np.cos(phase['b'])]).T
x_separate = sm.add_constant(x_separate)
# Fit the model
model_separate = sm.OLS(y, x_separate)
results_separate = model_separate.fit()
print(results_separate.summary())

# Fit a model based on combined phase in each signal
x_combined = np.stack([np.sin(phase['a']),
                       np.cos(phase['a']),
                       np.sin(phase['b']),
                       np.cos(phase['b']),
                       np.sin(phase['a']) * np.sin(phase['b']),
                       np.sin(phase['a']) * np.cos(phase['b']),
                       np.cos(phase['a']) * np.sin(phase['b']),
                       np.cos(phase['a']) * np.cos(phase['b']),
                       ]).T
x_combined = sm.add_constant(x_combined)
# Fit the model
model_combined = sm.OLS(y, x_combined)
results_combined = model_combined.fit()
print(results_combined.summary())

# Plot the model fits
fig = plt.figure(figsize=(6, 3))
spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[5, 1])

ax0 = fig.add_subplot(spec[0])
ax0.plot(t, y, color='black', label='Empirical')
ax0.plot(t, results_phasediff.fittedvalues,
         color='darkorange',
         label='Fits: Phase diff')
ax0.plot(t, results_separate.fittedvalues,
         color='seagreen',
         label='Fits: Sep phase')
ax0.plot(t, results_combined.fittedvalues, 
         color='blueviolet',
         label='Fits: Comb phase')
plt.xlim(3, 4)
#plt.ylim(-0.5, 5)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('HF amplitude')

def likelihood_ratio_test(results_1, results_2):
    if results_1.df_resid < results_2.df_resid:
        results_1, results_2 = results_2, results_1
    llf_1 = results_1.llf
    llf_2 = results_2.llf
    df_1 = results_1.df_resid 
    df_2 = results_2.df_resid 
    lrdf = (df_1 - df_2)
    lrstat = -2 * (llf_1 - llf_2)
    lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)
    return lr_pvalue

print(f"p = {likelihood_ratio_test(results_phasediff, results_separate)}")
print(f"p = {likelihood_ratio_test(results_separate, results_combined)}")

# Which model has the lowest/best AIC and BIC?
msg = f"""
AIC
Phase diff: {results_phasediff.aic}
Separate  : {results_separate.aic}
Combined  : {results_combined.aic}

BIC
Phase diff: {results_phasediff.bic}
Separate  : {results_separate.bic}
Combined  : {results_combined.bic}
"""
print(msg)

models = [results_phasediff, results_separate, results_combined]
labels = ['Diff', 'Sep', 'Comb']
bic = np.array([r.bic for r in models])
bic /= 1e5
xpos = range(len(models))
ax1 = fig.add_subplot(spec[1])
ax1.plot(xpos, bic, 'o')
plt.xticks(xpos, labels, rotation=45)
plt.ylabel('BIC ($\\times 10 ^ 5$)')

plt.tight_layout()

