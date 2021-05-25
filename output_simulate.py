"""
Run the analyses and generate plots.

How does the TE method deal with:
    - independent LF signals and independent HF signals with PAC
        - It shows no relationship (good)
    - Coherent alpha oscillations with independent noise in each signal
        - It shows alpha phase-diff communication between the signals (bad)
        - This means the measure does not uniquely measure communication

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
from tqdm import tqdm

import simulate
import comlag

plt.ion()

data_dir = '/media/geoff/Seagate2TB1/geoff/commun-phaselag/data/'
plot_dir = data_dir + 'plots/simulated/'

plot_dir = plot_dir + 'stats/perm_phasebin/'

#######################################
# Test the analysis on simulated data #
#######################################

np.random.seed(1)

dur = 1000
fs = 1000
volume_conduction = 0.0
t, s_a, s_b = simulate.sim(dur=dur, fs=fs,
                           noise_amp=1.0,
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


####################################
# Plot a snippet of simulated data #
####################################

plt.figure(figsize=(6, 3))
plt.plot(t, s_a, label='$s_a$: Sender')
plt.plot(t, s_b, label='$s_b$: Receiver')
plt.xlim(0, 1)
plt.xlabel('Time (s)')
plt.ylabel('Signal amplitude (mV or fT)')
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}timeseries_example.png')


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
plt.savefig(f'{plot_dir}tort.png')


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
plt.savefig(f'{plot_dir}fit_sine.png')


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
plt.savefig(f'{plot_dir}xspect.png')


#########################################
# Compute CFC using 2D Von Mises method #
#########################################

fits, rsq = comlag.cfc_vonmises_2d(s_a, s_b, fs, f_mod, f_car)

def plot_contour(x, colorbar_label='', **kwargs):
    plt.contourf(f_mod_centers, f_car, x.T, **kwargs)
    cb = plt.colorbar(format='%.2f', ticks=[x.min(), x.max()])
    cb.ax.set_ylabel(colorbar_label)
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


###################################################################
# Compute MI between HF time-series as a function of LF phase lag #
###################################################################

mi, mi_comod = comlag.cfc_phaselag_mutualinfo(s_a, s_b, fs, f_mod, f_car)
plot_contour(mi_comod, colorbar_label='')
plt.savefig(f'{plot_dir}phase-diff_mutual-info.png')


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

plt.figure(figsize=(5, 2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

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
ax0 = plt.subplot(gs[0])
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
ax1 = plt.subplot(gs[1])
# Mirror the results for -pi at +pi to make the plot prettier
amp_dist = np.c_[amplitude_dist, amplitude_dist[:,:1]]
amp_dist = np.r_[amp_dist, amp_dist[:1,:]]
plt.contourf(phase_bins, phase_bins, amp_dist,
             levels=np.linspace(0, amp_dist.max(), 100))
plt.xticks([-np.pi, 0, np.pi], ['$-\\pi$', '0', '$\\pi$'])
plt.yticks([-np.pi, 0, np.pi], ['$-\\pi$', '0', '$\\pi$'])
plt.xlabel('Receiver LF phase')
plt.ylabel('Sender LF phase')
plt.colorbar(label='Receiver HF power', ticks=[0, 4])

plt.tight_layout()

plt.savefig(f'{plot_dir}phase-diff_HFPower.png', dpi=300)


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

plt.figure(figsize=(5, 2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# MI between HF signals per LF phase bin
mi = [] 
for phase_bin in np.unique(phase_diff):
    phase_sel = phase_diff == phase_bin
    i = gcmi.gcmi_cc(x_2d['a'][:, phase_sel],
                     x_2d['b'][:, phase_sel])
    mi.append(i)
ax0 = plt.subplot(gs[0])
plt.plot(phase_bins[:-1], mi)
plt.xticks([-np.pi, 0, np.pi], ['$-\\pi$', '0', '$\\pi$'])
plt.xlabel('LF phase difference (rad)')
plt.ylabel('I(Sender; Receiver)')

# MI between HF signals per LF phase in each signal
mi = np.full([n_bins, n_bins], np.nan) 
for bin_a in np.unique(phase_diff):
    for bin_b in np.unique(phase_diff):
        phase_sel = (phase_dig['a'] == bin_a) & (phase_dig['b'] == bin_b)
        i = gcmi.gcmi_cc(x_2d['a'][:, phase_sel],
                         x_2d['b'][:, phase_sel])
        mi[bin_a, bin_b] = i
ax1 = plt.subplot(gs[1])

# Mirror the results for -pi at +pi to make the plot prettier
mi_to_plot = np.c_[mi, mi[:,:1]]
mi_to_plot = np.r_[mi_to_plot, mi_to_plot[:1,:]]
plt.contourf(phase_bins, phase_bins, mi_to_plot, 100)
             #levels=np.linspace(0, mi_to_plot.max(), 100))
plt.xticks([-np.pi, 0, np.pi], ['$-\\pi$', '0', '$\\pi$'])
plt.yticks([-np.pi, 0, np.pi], ['$-\\pi$', '0', '$\\pi$'])
plt.xlabel('Receiver LF phase')
plt.ylabel('Sender LF phase')
plt.colorbar(label='I(Sender; Receiver)', ticks=[0, 1.0])

plt.tight_layout()

plt.savefig(f'{plot_dir}phase-diff_HF_MI.png', dpi=300)


#########################################
# Phase difference vs phase combination #
#########################################

# Explicitly compare the two models. Is it phase-difference that's important,
# or phase combination?

phase_offset = -np.pi / 2

# Plot the hypotheses
phases = np.linspace(-np.pi, np.pi, 100)
phase_mat_a = np.tile(phases, [len(phases), 1])
phase_mat_b = phase_mat_a.copy().T


def phase_mat_plot(z):
    plt.contourf(phases, phases, z, 100)
    plt.xticks([-np.pi, 0, np.pi], ['$-\\pi$', '0', '$\\pi$'])
    plt.yticks([-np.pi, 0, np.pi], ['$-\\pi$', '0', '$\\pi$'])
    plt.xlabel('Phase in sender')
    plt.ylabel('Phase in receiver')

plt.figure(figsize=(6.5, 2))

# Plot what we'd find if excitability depends on individual phase in each
plt.subplot(1, 3, 1)
z_indiv = np.cos(phase_mat_b + phase_offset) \
          + np.cos(phase_mat_a + phase_offset)
phase_mat_plot(z_indiv)

# Plot what we'd expect to find if the phase-difference per se is important
plt.subplot(1, 3, 2)
phase_mat_diff = phase_mat_b - phase_mat_a
phase_mat_diff = (phase_mat_diff + np.pi) % (2 * np.pi) - np.pi # wrap to +/-pi
z_diff = np.cos(phase_mat_diff + phase_offset)
phase_mat_plot(z_diff)

# Plot what we'd find if the important thing is aligning excitable periods
plt.subplot(1, 3, 3)
cos_positive = lambda x: (1/2 * np.cos(x)) + 1/2 # only positive values
z_comb = cos_positive(phase_mat_a + phase_offset) \
         * cos_positive(phase_mat_b + phase_offset)
phase_mat_plot(z_comb)

plt.tight_layout()

plt.savefig(f'{plot_dir}hypotheses.png', dpi=300)


# Get the average HF power for each time-point
# Compare 2 regressions
#   sine-cosine transform of the phase difference
#   sine-cosine transform of phase of each signal

import statsmodels.api as sm
from matplotlib import gridspec

colors = ['seagreen', 'darkorange', 'blueviolet',
          'deepskyblue', 'deeppink', 'slategrey'] 

y = amp['b']
y = np.squeeze(y)
#y = 10 * np.log10(y)

# Fit a model based on phase in each signal individually
x_indiv = np.stack([np.sin(phase['a']),
                    np.cos(phase['a']),
                    np.sin(phase['b']),
                    np.cos(phase['b'])]).T
x_indiv = sm.add_constant(x_indiv)
# Fit the model
model_indiv = sm.OLS(y, x_indiv)
results_indiv = model_indiv.fit()
print(results_indiv.summary())

# Phase difference alone
phasediff = phase['a'] - phase['b']
phasediff = (phasediff + np.pi) % (2 * np.pi) - np.pi # wrap to +/-pi
x_phasediff = np.stack([np.sin(phasediff),
                        np.cos(phasediff)]).T
x_phasediff = sm.add_constant(x_phasediff)
# Fit the model
model_phasediff = sm.OLS(y, x_phasediff)
results_phasediff = model_phasediff.fit()
print(results_phasediff.summary())

# Phase-difference plus indiv phase
x_phasediff_indiv = np.stack([np.sin(phase['a']), # Include terms for phase of each
                              np.cos(phase['a']),
                              np.sin(phase['b']),
                              np.cos(phase['b']),
                              np.sin(phasediff),
                              np.cos(phasediff)]).T
x_phasediff_indiv = sm.add_constant(x_phasediff_indiv)
# Fit the model
model_phasediff_indiv = sm.OLS(y, x_phasediff_indiv)
results_phasediff_indiv = model_phasediff_indiv.fit()
print(results_phasediff_indiv.summary())

# Combined phase alone
x_combined = np.stack([np.sin(phase['a']) * np.sin(phase['b']),
                       np.sin(phase['a']) * np.cos(phase['b']),
                       np.cos(phase['a']) * np.sin(phase['b']),
                       np.cos(phase['a']) * np.cos(phase['b']),
                       ]).T
x_combined = sm.add_constant(x_combined)
# Fit the model
model_combined = sm.OLS(y, x_combined)
results_combined = model_combined.fit()
print(results_combined.summary())

# Combined phase plus individual
x_combined_indiv = np.stack([np.sin(phase['a']),
                             np.cos(phase['a']),
                             np.sin(phase['b']),
                             np.cos(phase['b']),
                             np.sin(phase['a']) * np.sin(phase['b']),
                             np.sin(phase['a']) * np.cos(phase['b']),
                             np.cos(phase['a']) * np.sin(phase['b']),
                             np.cos(phase['a']) * np.cos(phase['b']),
                             ]).T
x_combined_indiv = sm.add_constant(x_combined_indiv)
# Fit the model
model_combined_indiv = sm.OLS(y, x_combined_indiv)
results_combined_indiv = model_combined_indiv.fit()
print(results_combined_indiv.summary())

# Fit a model based on combined phase + phase-difference + individual
x_total = np.stack([np.sin(phase['a']),
                    np.cos(phase['a']),
                    np.sin(phase['b']),
                    np.cos(phase['b']),
                    np.sin(phase['a']) * np.sin(phase['b']),
                    np.sin(phase['a']) * np.cos(phase['b']),
                    np.cos(phase['a']) * np.sin(phase['b']),
                    np.cos(phase['a']) * np.cos(phase['b']),
                    np.sin(phasediff),
                    np.cos(phasediff)
                    ]).T
x_total = sm.add_constant(x_total)
# Fit the model
model_total = sm.OLS(y, x_total)
results_total = model_total.fit()
print(results_total.summary())

# Plot each model's fit against the HG activity
results = [results_indiv, results_phasediff, results_combined]
labels = ['Individual', 'Phase diff', 'Combined']
t_adj = t - 2
for n,res,lab,col in zip(range(len(results)), results, labels, colors):
    #plt.subplot(len(results), 1, n+1)
    fig = plt.figure(figsize=(3, 1.5))
    plt.plot(t_adj, y, color='black', label='Empirical')
    plt.plot(t_adj, res.fittedvalues, color=col, label=f'Fits: {lab}')
    plt.xlim(0, 3)
    plt.ylim(-0.5, 5)
    #plt.legend(loc='upper center')
    plt.xlabel('Time (s)')
    plt.ylabel('HF amplitude')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}model_comp_{lab}.png', dpi=300)
    plt.close()

# Plot the model fits
# fig = plt.figure(figsize=(5, 2.5))
# spec = gridspec.GridSpec(ncols=2, nrows=1,
#                          width_ratios=[5, 1])
# 
# # Plot predicted HF time-courses all together
# ax0 = fig.add_subplot(spec[0])
# t_adj = t - 3.5
# ax0.plot(t_adj, y, color='black', label='Empirical')
# ax0.plot(t_adj, results_indiv.fittedvalues,
#          color=colors[0],
#          label='Individual phases')
# ax0.plot(t_adj, results_phasediff.fittedvalues,
#          color=colors[1],
#          label='Phase diff model')
# ax0.plot(t_adj, results_combined.fittedvalues, 
#          color=colors[2],
#          label='Phase combo model')
# plt.xlim(0, 1.5)
# plt.xticks([0, 0.5, 1, 1.5])
# #plt.ylim(-50, 15)
# plt.legend(loc='upper right')
# plt.xlabel('Time (s)')
# plt.ylabel('HF amplitude')
# 
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

models = [results_indiv,
          results_phasediff, results_combined,
          results_phasediff_indiv, results_combined_indiv, 
          results_total]
labels = ['Individual',
          'Difference', 'Combined',
          'Indiv + Diff', 'Indiv + Comb',
          'Indiv + Diff + Comb']
bic = np.array([r.bic for r in models])
bic /= 1e5
xpos = range(len(models))
#ax1 = fig.add_subplot(spec[1])
plt.figure(figsize=(3.5, 3))
import matplotlib
matplotlib.rcParams['hatch.linewidth'] = 4.0
plt.clf()
plt.bar(xpos[:3], bic[:3], color=colors[:3])
# Draw faces
plt.bar(xpos[3:], bic[3:], color=colors[0], zorder=0)
# Draw hatches
plt.bar(xpos[3:5], bic[3:5], color='none', edgecolor=colors[1:3],
        hatch="//", zorder = 1)
plt.bar(xpos[-1], bic[-1], color='none', edgecolor=colors[1],
        hatch="//", zorder = 1)
plt.bar(xpos[-1], bic[-1], color='none', edgecolor=colors[2],
        hatch="/", zorder = 1)

plt.xticks(xpos, labels, rotation=65)
plt.xlim(-0.5, 5.5)
plt.ylim(15, 23)
plt.ylabel('BIC ($\\times 10 ^ 5$)')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{plot_dir}model_comp.png', dpi=300)

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

print(f"Diff vs Sep: p = {likelihood_ratio_test(results_phasediff, results_indiv)}")
print(f"Sep vs Comb: p = {likelihood_ratio_test(results_indiv, results_combined)}")
print(f"Diff vs Comb: p = {likelihood_ratio_test(results_phasediff, results_combined)}")

# Which model has the lowest/best AIC and BIC?
msg = f"""
AIC
Phase diff: {results_phasediff.aic}
Separate  : {results_indiv.aic}
Combined  : {results_combined.aic}

BIC
Phase diff: {results_phasediff.bic}
Separate  : {results_indiv.bic}
Combined  : {results_combined.bic}
"""
print(msg)


results = comlag.cfc_modelcomp(s_a, s_b, fs, f_mod, f_car)
# Plot the results
stat = {}
for summary_stat in ('bic', 'rsquared_adj'):
    stat_summ = {}
    for model_type in results[0][0].keys():
        stat_mt = [] # Stats for this model type
        for res_fm in results: # For each phase freq
            stat_fm = [] # Stats for this model type and phase freq
            for res_fc in res_fm: # For each amplitude freq
                stat_fm.append(res_fc[model_type][summary_stat])
            stat_mt.append(stat_fm)
        stat_summ[model_type] = np.array(stat_mt)
    stat[summary_stat] = stat_summ

plt.clf()

plt.subplot(1, 3, 1)
plot_contour(stat['rsquared_adj']['diff'], levels=50)
plt.title('$R^2$: Indiv + Diff')

plt.subplot(1, 3, 2)
plot_contour(stat['rsquared_adj']['combined'], levels=50)
plt.title('$R^2$: Indiv + Comb')

plt.subplot(1, 3, 3)
comparison = stat['bic']['combined'] - stat['bic']['diff']
plot_contour(comparison,
                levels=np.linspace(-np.max(np.abs(comparison)),
                                np.max(np.abs(comparison)),
                                100),
                cmap=plt.cm.RdBu_r)
plt.title('BIC difference')

plt.tight_layout()

plt.savefig(f'{plot_dir}model_comp_comodulogram.png', dpi=300)


#################################################
# Test the role of cross-talk in the MI measure #
#################################################
# assume theta-gamma PAC in sender and receiver - gamma independent in sender
# and receiver. Now mix e.g. by 20%. Does supprious info-transfer arise? 

def mi_fnc(s_a, s_b, **mi_params):
    """ Helper function
    """
    mi, mi_comod, counts = comlag.cfc_phaselag_mutualinfo(
                                            s_a, s_b,
                                            **mi_params)
    d = dict(mi=mi, mi_comod=mi_comod, counts=counts)
    return d

# Which frequencies to calculate phase for
f_mod_centers = np.logspace(np.log10(4), np.log10(20), 15)
f_mod_width = f_mod_centers / 8
f_mod = np.tile(f_mod_width, [2, 1]).T \
            * np.tile([-1, 1], [len(f_mod_centers), 1]) \
            + np.tile(f_mod_centers, [2, 1]).T

# Which frequencies to calculate power for
f_car = np.arange(20, 100, 10)

def plot_contour(x, caxis_label='', **kwargs):
    plt.contourf(f_mod_centers, f_car, x.T,
                 #levels=np.linspace(0, 1, 50),
                 **kwargs)
    cb = plt.colorbar(format='%.2f', ticks=[x.min(), x.max()])
    cb.ax.set_ylabel(caxis_label)
    plt.ylabel('HF freq (Hz)')
    plt.xlabel('Phase freq (Hz)')

# Parameters for the simulated signals
sim_params = dict(dur=100, fs=1000,
                  shared_gamma=True,
                  noise_amp=0.01, common_noise_amp=0.0)

# Parameters for the MI phase-lag analysis
mi_params = dict(fs=sim_params['fs'],
                 f_mod=f_mod, f_car=f_car,
                 f_car_bw=10, n_bins=2**4,
                 method='sine psd')

methods = {'tort': 'Mod. Index',
           'sine psd': 'bits$^2$ / Hz',
           'sine amp': 'bits / Hz',
           'sine fit adj': 'Mod. Index',
           'rsquared': '$R^2$',
           'vector': 'Vector length',
           'vector-imag': 'Imag(Vector length)',
           'itlc': 'ITLC'}

#plt.figure(figsize=(4, 6))
for shared_gamma in [True]: #, False]:
    sim_params['shared_gamma'] = shared_gamma

    # Simulate signals with no cross-talk
    t, s_a, s_b = simulate.sim(signal_leakage=0, **sim_params)
    sig_no_crosstalk = {'t': t, 's_a': s_a, 's_b': s_b}

    ### Simulate signals with a lot of cross-talk
    ##t, s_a, s_b = simulate.sim(signal_leakage=0.5, **sim_params)
    ##sig_crosstalk = {'t': t, 's_a': s_a, 's_b': s_b}

    for method in methods.keys():
        plt.figure()
        mi_params['method'] = method
        res_no_crosstalk = mi_fnc(sig_no_crosstalk['s_a'],
                                  sig_no_crosstalk['s_b'],
                                  **mi_params)
        ##res_crosstalk = mi_fnc(sig_crosstalk['s_a'],
        ##                       sig_crosstalk['s_b'],
        ##                       **mi_params)

        # Plot it
        plt.clf()

        ##plt.subplot(2, 1, 1)
        plot_contour(res_no_crosstalk['mi_comod'], methods[method])
        ##plt.title('No cross-talk')
        plt.title(method)

        ##plt.subplot(2, 1, 2)
        ##plot_contour(res_crosstalk['mi_comod'], methods[method])
        ##plt.title('Includes cross-talk')

        plt.tight_layout()

        ##gamma_cond = 'shared' if sim_params['shared_gamma'] else 'separate'
        ##fname_stem = 'mi_comod_cross-talk'
        ##fname = f'{fname_stem}_{mi_params["method"]}_{gamma_cond}-gamma.png'
        ##plt.savefig(f'{plot_dir}{fname}', dpi=300)


# Vary the lag between the signals
lag = 15
res_no_lag = mi_fnc(s_a, s_b, **mi_params)
res_pos_lag = mi_fnc(np.roll(s_a, lag), s_b, **mi_params)
res_neg_lag = mi_fnc(np.roll(s_a, -lag), s_b, **mi_params)

plt.figure(figsize=(12, 3))
levels = np.linspace(0, 2.6, 50)
plt.subplot(1, 4, 1)
plot_contour(res_neg_lag['mi_comod'], methods[method], levels=levels)
plt.title('I(L$^{-}$A;B)')

plt.subplot(1, 4, 2)
plot_contour(res_no_lag['mi_comod'], methods[method], levels=levels)
plt.title('I(A;B)')

plt.subplot(1, 4, 3)
plot_contour(res_pos_lag['mi_comod'], methods[method], levels=levels)
plt.title('I(L$^{+}$A;B)')

plt.subplot(1, 4, 4)
plot_contour(res_pos_lag['mi_comod'] - res_neg_lag['mi_comod'],
             methods[method],
             levels=np.linspace(-2, 2, 50), cmap=plt.cm.RdBu_r)
plt.title('I(L$^{+}$A;B) - I(L$^{-}$A;B)')

plt.tight_layout()

# TODO
# Lagged MI as a function of cross-talk: I(LA;B) - I(A;LB)
# Lagged CMI as a function of cross-talk: I(A;B|LA) - I(A;B|LB)


################################################
# Test effects of varying levels of cross-talk #
################################################

def mi_fnc(s_a, s_b, **mi_params):
    """ Helper function
    """
    mi, mi_comod, counts = comlag.cfc_phaselag_mutualinfo(
                                            s_a, s_b,
                                            **mi_params)
    d = dict(mi=mi, mi_comod=mi_comod, counts=counts)
    return d

# Which frequencies to calculate phase for
f_mod_centers = np.logspace(np.log10(4), np.log10(20), 15)
f_mod_width = f_mod_centers / 8
f_mod = np.tile(f_mod_width, [2, 1]).T \
            * np.tile([-1, 1], [len(f_mod_centers), 1]) \
            + np.tile(f_mod_centers, [2, 1]).T

# Which frequencies to calculate power for
f_car = np.arange(20, 100, 10)

def plot_contour(x, caxis_label='', **kwargs):
    plt.contourf(f_mod_centers, f_car, x.T,
                 #levels=np.linspace(0, 1, 50),
                 **kwargs)
    cb = plt.colorbar(format='%.2f', ticks=[x.min(), x.max()])
    cb.ax.set_ylabel(caxis_label)
    plt.ylabel('HF freq (Hz)')
    plt.xlabel('Phase freq (Hz)')

# Parameters for the simulated signals
sim_params = dict(dur=100, fs=1000,
                  shared_gamma=True,
                  noise_amp=0.01, common_noise_amp=0.0)

# Parameters for the MI phase-lag analysis
mi_params = dict(fs=sim_params['fs'],
                 f_mod=f_mod, f_car=f_car,
                 f_car_bw=10, n_bins=2**4)

methods = {'sine psd': 'bits$^2$ / Hz',
           'vector': 'Vector length',
           'vector-imag': 'Imag(Vector length)'}
cross_talk_levels = np.linspace(0, 0.6, num=7)

methods = {#'tort': 'Mod. Index',
           #'sine psd': 'bits$^2$ / Hz',
           #'sine amp': 'bits / Hz',
           #'sine fit adj': 'Mod. Index',
           'rsquared': '$R^2$',
           'vector': 'Vector length',
           'vector-imag': 'Imag(Vector length)',
           'itlc': 'ITLC'}
cross_talk_levels = [0, 0.5]

plt.figure(figsize=(4, 3))
for leakage in cross_talk_levels:
    # Simulate signals
    t, s_a, s_b = simulate.sim(signal_leakage=leakage, **sim_params)

    # Run the analyses and save the plots
    for method in methods.keys():
        res = mi_fnc(s_a, s_b, 
                     method=method,
                     **mi_params)
        plt.clf()
        plot_contour(res['mi_comod'], methods[method])
        plt.title(f'Leakage: {leakage:.1f}')
        plt.tight_layout()
        fname_stem = 'XX_mi_comod_cross-talk'
        fname = f'{fname_stem}_{method}_leak_{leakage:.1f}.png'
        plt.savefig(f'{plot_dir}{fname}', dpi=300)

!notify-send 'Analyses finished'


#############################################
# Test the effects of cross-talk on the PSI #
#############################################

# Parameters for the simulated signals
sim_params = dict(dur=100, fs=1000,
                  shared_gamma=True,
                  noise_amp=0.01, common_noise_amp=0.0)

# Parameters for the PSI phase-lag analysis
mi_params = dict(fs=sim_params['fs'],
                 nfft=2**9, step_size=2**4,
                 n_bins=2**4,
                 psi_bw=10)
#for k,v in mi_params.items(): globals()[k] = v # FOR TESTING

phase_diff_lims = [2, 20] # Freqs at which phase diff is calculated
psi_lims = [40, 120] # Freqs at which PSI is calculated

# Simulate signals
t, s_a_orig, s_b_orig = simulate.sim(signal_leakage=leakage, **sim_params)

cross_talk_levels = np.linspace(0, 0.6, num=7)
plt.figure(figsize=(4, 3))
for leakage in cross_talk_levels:

    # Mix the signals together
    s_a = s_a_orig + (s_b_orig * signal_leakage)
    s_b = (s_a_orig * signal_leakage) + s_b_orig

    # Run the analyses and save the plots
    res = comlag.psi_phaselag(s_a, s_b, 
                              **mi_params)

    plt.clf()
    z = res['mi_comod']
    # Set color scale to max within range of interest
    f = res['freqs']
    fx = (phase_diff_lims[0] < f) & (f < phase_diff_lims[1])
    fy = (psi_lims[0] < f) & (f < psi_lims[1])
    z_max = np.nanmax(z[fy,:][:,fx])
    plt.contourf(f, f, z,
                 levels=np.linspace(0, z_max, 10))
    cb = plt.colorbar(ticks=[0, z_max], format='%.2f')
    cb.ax.set_ylabel('Sine amp')
    plt.xlabel('Phase freq (Hz)')
    plt.ylabel('HF freq (Hz)')
    plt.xlim(phase_diff_lims)
    plt.ylim(psi_lims) #20, 90) #
    plt.title(f'Leakage: {leakage:.1f}')
    plt.tight_layout()
    fname_stem = 'mi_comod_cross-talk'
    fname = f'{fname_stem}_psi_leak_{leakage:.1f}.png'
    plt.savefig(f'{plot_dir}{fname}', dpi=300)

!notify-send 'Analyses finished'

# # Plots for testing
# plt.clf()

# plt.subplot(1, 2, 1)
# i_freq = 10
# x = Psi_ij[:,:,i_freq]
# levels = np.linspace(-np.max(np.abs(x)),
#                      np.max(np.abs(x)),
#                      100)
# plt.contourf(phase_bins, fft_freqs[keep_freqs], 
#              x,
#              levels=levels,
#              cmap=plt.cm.RdBu_r)
# plt.xlabel('LF phase diff')
# plt.ylabel('HF PSI freq')
# plt.ylim(psi_lims)
# cb = plt.colorbar(ticks=np.array([-1, 0, 1]) * np.max(np.abs(x)))
# cb.ax.set_ylabel('PSI')

# plt.subplot(1, 2, 2)
# max_val = 0.04
# plt.contourf(fft_freqs[keep_freqs], fft_freqs[keep_freqs],
#              mi_comod,
#              levels=np.linspace(0, max_val, 100))
# plt.xlabel('LF phase freq')
# plt.ylabel('HF PSI freq')
# plt.xlim(phase_diff_lims)
# plt.ylim(psi_lims)
# cb = plt.colorbar(ticks=[0, 0.04])
# cb.ax.set_ylabel('Mod. Ind.')

# plt.tight_layout()


###########################################
# Communication based on Transfer-entropy #
###########################################

# Low-freq 'modulator' frequencies
f_mod = np.arange(6, 15)
f_mod_bw = f_mod / 2.5 # ~4 cycles

# High-freq 'carrier' frequencies
f_car = np.arange(30, 150, 10)
f_car_bw = f_car / 3 # ~5 cycles


# Parameters for the simulated signals
lag = 6
sim_params = dict(dur=100,
                  fs=1000,
                  noise_amp=1.5,
                  common_noise_amp=0.1,
                  shared_gamma=True)

# Simulate signals with phase-lagged communication
t, s_a, s_b = simulate.sim(**sim_params)
fname = f'te_stats_phase-dep-comm.png'

## # Sig A: Alpha oscillation plus pink noise
## # Sig B: Same alpha oscillation as Sig A plus independent pink noise
## t, s_a, s_b = simulate.sim_lf_coh_plus_noise(sim_params['dur'],
##                                              sim_params['fs'],
##                                              noise_amp=2)
## fname = f'te_stats_lf-coh-plus-noise.png'

## # Same as above, but lag Sig B (and therefore offset the alpha oscillations).
## # This simulates alpha coherence that is not reducible to cross-talk.
## # Result: Alpha-limited phase-dependent communication b/w A & B, with
## # directionality mostly in one direction but with some switches by HF frequency
## t, s_a, s_b = simulate.sim_lf_coh_plus_noise(sim_params['dur'],
##                                              sim_params['fs'],
##                                              noise_amp=2,
##                                              lag=lag)
## fname = f'te_stats_lf-coh-plus-noise-lag.png'

## # Two signals with LF coherence and independent PAC
## t, s_a, s_b = simulate.sim_lf_coh_with_pac(sim_params['dur'],
##                                            sim_params['fs'])
## fname = f'te_stats_lf-coh-plus-pac.png'

## # Two signals with LF coherence and HF communication, but no phase-dependence
## # of the communication
## t, s_a, s_b = simulate.sim_lf_coh_with_hf_comm(sim_params['dur'],
##                                                sim_params['fs'],
##                                                lag=lag,
##                                                noise_amp=2)

# Split the data into epochs
epoch_length = 1000 # Samples
n_splits = len(s_a) / epoch_length
s_a = np.stack(np.split(s_a, n_splits), axis=1)
s_b = np.stack(np.split(s_b, n_splits), axis=1)

# Parameters for the MI phase-lag analysis
mi_params = dict(fs=sim_params['fs'],
                 f_mod=f_mod,
                 f_mod_bw=f_mod_bw,
                 f_car=f_car,
                 f_car_bw=f_car_bw,
                 lag=[lag],
                 n_bins=2**3,
                 method='sine psd',
                 n_perm_phasebin=0, # 1000, ## Works better at identifying real frequencies
                 n_perm_phasebin_indiv=0, # 1000,
                 n_perm_signal=0,
                 n_perm_shift=3, #100,
                 min_shift=None, max_shift=None,
                 cluster_alpha=0.05,
                 calc_type=2,
                 diff_method='both', #'PD(AB-BA)',
                 verbose=True)

## for k,v in mi_params.items(): globals()[k] = v # FOR DEBUGGING
## i_fm = 0
## i_fc = 0
## fm = f_mod[i_fm]
## fm_bw = f_mod_bw[i_fm]
## fc = f_car[i_fc]
## fc_bw = f_car_bw[i_fc]


# Compute transfer entropy
res, stat_info = comlag.cfc_phaselag_transferentropy(s_a, s_b,
                                                          **mi_params)
diff_meth = 'PD(AB)-PD(BA)'
te = res[diff_meth]
clust_stat_info = stat_info[diff_meth]

# Plots

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
    cb.ax.set_ylabel('bits$^2$')
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


###########################################################################
# Does lagging one LF signal plus noise result in spurious communication? #
###########################################################################
"""
Test for significant communication across "subjects". Does a lagged signal lead
to significant spurious communication in the same direction across subjects?
"""


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
                  gamma_lag_a_to_b=(lag / 1000))

# Parameters for the MI phase-lag analysis
mi_params = dict(fs=sim_params['fs'],
                 f_mod=f_mod,
                 f_mod_bw=f_mod_bw,
                 f_car=f_car,
                 f_car_bw=f_car_bw,
                 lag=[lag],
                 n_bins=2**3,
                 method='sine psd',
                 n_perm_phasebin=0,
                 n_perm_shift= 0,
                 min_shift=None, max_shift=None,
                 cluster_alpha=0.05,
                 calc_type=2,
                 verbose=False)

# Determine which type of simulation to run
sim_opts = ['phase-dep-comm',
            'volume-cond-plus-noise',
            'lf-coh-plus-noise',
            'lf-coh-plus-pac']
n_sim = 100
plt.clf()
for n_plot, sim_type in enumerate(sim_opts):
    te_a = []
    te_b = []
    te_d = []
    for k in tqdm(range(n_sim)):

        if sim_type == 'phase-dep-comm':
            t, s_a, s_b = simulate.sim(**sim_params)
        elif sim_type == 'volume-cond-plus-noise':
            t, s_a, s_b = simulate.sim_lf_coh_plus_noise(sim_params['dur'],
                                                         sim_params['fs'],
                                                         noise_amp=2)
        elif sim_type == 'lf-coh-plus-noise':
            t, s_a, s_b = simulate.sim_lf_coh_plus_noise(sim_params['dur'],
                                                         sim_params['fs'],
                                                         noise_amp=2,
                                                         lag=mi_params['lag'])
        elif sim_type == 'lf-coh-plus-pac':
            t, s_a, s_b = simulate.sim_lf_coh_with_pac(sim_params['dur'],
                                                       sim_params['fs'],
                                                       lag=mi_params['lag'])
    
        te = comlag.cfc_phaselag_transferentropy(s_a, s_b, **mi_params)
        te_a.append(np.squeeze(te['a']))
        te_b.append(np.squeeze(te['b']))
        te_d.append(np.squeeze(te['diff']))

    plt.subplot(2, 2, n_plot + 1)
    min_te = np.min(np.array([te_a, te_b]))
    max_te = np.max(np.array([te_a, te_b]))
    plt.loglog(te_a, te_b, 'o', alpha=0.5)
    plt.plot([min_te, max_te], [min_te, max_te], '--k')
    plt.title(f"{sim_type}; p = {stats.wilcoxon(te_a, te_b).pvalue:.2f}")
    plt.xlabel('TE(A$\\rightarrow$B)')
    plt.ylabel('TE(B$\\rightarrow$A)')
plt.tight_layout()
plt.savefig(f"{plot_dir}stats/subject-wise_by_simulation_type.png", dpi=300)


n = 7 # Subjects
k = 1000
mean_diffs = np.linspace(0, 3, 20) # Difference between means (in units of SD)
p = []
for d in mean_diffs:
    pvals = []
    for _ in range(k):
        x = np.random.normal(size=n)
        y = np.random.normal(size=n) + d
        pvals.append(stats.wilcoxon(x, y).pvalue)
    pvals = np.array(pvals)
    p.append(np.mean(pvals < .05))

plt.clf()
plt.plot(mean_diffs, p)
plt.xlabel('$\mu_x - \mu_y$ (in $\sigma$)')
plt.ylabel('Prop(Signif)')
plt.ylim(0, 1)
