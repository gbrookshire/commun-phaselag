"""
Things to rule out

- Statistical test
    - Split the data into trials or epochs
    - Compute the comodulogram of the empirical phase-lagged TE
    - For k permutations
        - Randomly shuffle the HF filtered data between epochs
        - Compute the comodulograms for each permutation
        - Compute a cluster stat
            - The summed z-value across comodulograms and permutations
            - Check for normal distribution?
            - For pos/neg, this is a two-tailed test (so z thresh for .025)
- Do changes to theta phase difference actually reflect interruptions in the
  theta cycle?
- Do we see the same pattern of results from cross-talk of a theta rhythm, plus
  unequal noise in each signal?
"""


import numpy as np
from scipy.io import loadmat
from scipy import signal, stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re
import os
from joblib import Parallel, delayed
import datetime
import comlag
import gcmi
from tqdm import tqdm
import copy
import simulate

plt.ion()

data_dir = '/media/geoff/Seagate2TB1/geoff/commun-phaselag/data/RatData/'
plot_dir = data_dir + 'plots/'
fnames = ['EEG_speed_allData_Rat17_20120616_begin1.mat',
          'EEG_speed_allData_Rat6_20111021_begin1.mat',
          'EEG_speed_allData_Rat13_20120131_begin1_CA3_CSC9.mat',
          'EEG_speed_allData_Rat45_20140522_begin1.mat',
          'EEG_speed_allData_Rat44_20140506_begin1_CA3_CSC4_CA1_TT6.mat',
          'EEG_speed_allData_Rat47_20140923_begin1_CA3_CSC11_CA1_TT3.mat',
          'EEG_speed_allData_Rat31_20140110_begin1_CA3_CSC7_CA1_TT2.mat']


##################################################
# Check the length of the recording for each rat #
##################################################

rec_len = [loadmat(data_dir + fn)['Data_EEG'].shape[0] for fn in fnames]
rec_len = [x / 2000 for x in rec_len]  # Length in s
plt.clf()
plt.bar(range(len(fnames)), rec_len)
plt.xticks(range(len(fnames)),
           [re.search('Rat[0-9]+', fn).group() for fn in fnames],
           rotation=45)
plt.ylabel('Recording length (s)')
plt.tight_layout()
plt.savefig(f'{plot_dir}recording_length.png')


##########################################
# Plot a sample of the data for each rat #
##########################################

labels = ['CA3', 'CA1']
plt.figure(figsize=(5, 10))
for n, fn in enumerate(fnames):
    plt.subplot(7, 1, n + 1)
    d = loadmat(data_dir + fn)
    s = [d['Data_EEG'][:, inx] for inx in [1, 2]]
    t = np.arange(s[0].size) / d['Fs'][0][0]
    for sig, lab in zip(s, labels):
        plt.plot(t, sig, label=lab)
    plt.xlim(10, 12.5)
    ylims = 0.0015
    plt.ylim(-ylims, ylims)
    plt.title(re.search('Rat[0-9]+', fn).group())
    if n < len(fnames) - 1:
        plt.xticks([])

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}example_traces.png')


################################################
# Plot the PSD for each rat and recording site  #
################################################

labels = ['CA3', 'CA1']
nfft = 2 ** 10
plt.clf()
for n, fn in enumerate(fnames):
    plt.subplot(3, 3, n + 1)
    d = loadmat(data_dir + fn)
    s = [d['Data_EEG'][:, inx] for inx in [1, 2]]
    for sig, lab in zip(s, labels):
        f, y = signal.welch(sig, nperseg=nfft, noverlap=nfft / 2,
                            fs=d['Fs'][0][0])
        f_sel = f <= 100
        f = f[f_sel]
        y = y[f_sel]
        plt.loglog(f, y, label=lab)
        # plt.plot(f, y, label=lab)
    plt.xticks([1, 10, 100])
    plt.xlim(1, 100)
    plt.title(re.search('Rat[0-9]+', fn).group())

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}spectra.png')


###############################################
# Plot the coherence spectrum for each animal #
###############################################

fs = 2000
nfft = 2 ** 12

plt.clf()
for n, fn in enumerate(fnames):
    plt.subplot(3, 3, n + 1)
    d = loadmat(data_dir + fn)
    s = [d['Data_EEG'][:, inx] for inx in [1, 2]]
    coh, freq = comlag.coherence(*s, fs=fs, nfft=nfft)
    plt.plot(freq, coh)
    plt.xlim(0, 100)
    plt.title(re.search('Rat[0-9]+', fn).group())

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig(f'{plot_dir}spectra.png')


#################################################
# Check the distribution of LF phase difference  #
#################################################

n_bins = 8
phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
for n, fn in enumerate(fnames):
    plt.subplot(3, 3, n + 1)
    d = loadmat(data_dir + fn)
    s = [d['Data_EEG'][:, inx] for inx in [1, 2]]
    filt = [comlag.bp_filter(sig, 4, 12, d['Fs'], 2) for sig in s]
    phase = [np.angle(signal.hilbert(sig)) for sig in filt]
    phase_diff = comlag.wrap_to_pi(phase[0] - phase[1])
    phase_diff = np.digitize(phase_diff, phase_bins) - 1  # Binned
    plt.hist(phase_diff, n_bins)
    plt.xlabel('Phase diff (rad)')
    plt.ylabel('Density')
    plt.title(re.search('Rat[0-9]+', fn).group())
    # plt.xlim([-np.pi, np.pi])
    # plt.xticks([-np.pi, 0, np.pi], ['$-\pi$', 0, '$\pi$'] )
plt.tight_layout()
plt.savefig(f'{plot_dir}phase-diff_hist.png')

"""
We see very strong coherence between the two signals, so it's not clear that
we'll be able to look at connectivity as a function of phase difference --
because we essentially only have data from one phase difference.
"""

# Do phase differences actually reflect interruptions in the theta cycles?
# Look at whether power correlates with phase
n_bins = 8
phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
for n, fn in enumerate(fnames):
    plt.subplot(3, 3, n + 1)
    d = loadmat(data_dir + fn)
    s = [d['Data_EEG'][:, inx] for inx in [1, 2]]
    filt = [comlag.bp_filter(sig, 4, 12, d['Fs'], 2) for sig in s]
    hilb = [signal.hilbert(sig) for sig in filt]
    phase = [np.angle(sig) for sig in hilb]
    phase_diff = comlag.wrap_to_pi(phase[0] - phase[1])
    phase_diff = np.digitize(phase_diff, phase_bins) - 1  # Binned
    pwr = [np.abs(sig) for sig in hilb]
    pwr_mean = [[np.mean(p[phase_diff == b]) for b in np.unique(phase_diff)]
                for p in pwr]
    _, counts = np.unique(phase_diff, return_counts=True)
    plt.plot(phase_bins[:-1], counts / counts.max())
    plt.plot(phase_bins[:-1], pwr_mean[0] / np.max(pwr_mean[0]))
    plt.plot(phase_bins[:-1], pwr_mean[1] / np.max(pwr_mean[1]))
    plt.xlabel('Phase diff (rad)')
    plt.ylabel('Density')
    plt.title(re.search('Rat[0-9]+', fn).group())
    # plt.xlim([-np.pi, np.pi])
    # plt.xticks([-np.pi, 0, np.pi], ['$-\pi$', 0, '$\pi$'] )
plt.tight_layout()


###################################################################
# Compute MI between HF time-series as a function of LF phase lag  #
###################################################################

# Which frequencies to calculate phase for
f_mod_centers = np.logspace(np.log10(4), np.log10(20), 15)
f_mod_width = f_mod_centers / 8
f_mod = np.tile(f_mod_width, [2, 1]).T \
            * np.tile([-1, 1], [len(f_mod_centers), 1]) \
            + np.tile(f_mod_centers, [2, 1]).T

# Which frequencies to calculate power for
f_car = np.arange(20, 100, 10)

f_car_bw = 10  # Bandwidth of the HF bandpass filter
n_phase_bins = 8  # Number of bins for the phase-difference
n_jobs = 3  # How many parallel jobs to run

save_fname = f"{data_dir}mi_comod/bw{int(f_car_bw)}_nbins{n_phase_bins}.npz"


def mi_fnc(fn):
    """ Helper function for parallel computation
    """
    print(f'f_car_bw: {f_car_bw}')
    print(f'n_phase_bins: {n_phase_bins}')
    d = loadmat(data_dir + fn)
    s = [d['Data_EEG'][:, inx] for inx in [1, 2]]
    mi_i, mi_comod_i, counts_i = comlag.cfc_phaselag_mutualinfo(
                                s[0], s[1],
                                d['Fs'], f_mod, f_car,
                                f_car_bw=f_car_bw,
                                n_bins=n_phase_bins)
    return (mi_i, mi_comod_i, counts_i)


mi_out = Parallel(n_jobs=n_jobs)(delayed(mi_fnc)(fn) for fn in fnames)
mi_full, mi_comod, counts = zip(*mi_out)
mi_full = np.array(mi_full)
mi_comod = np.array(mi_comod)
counts = np.array(counts)

# Save the data
np.savez(save_fname, mi_full=mi_full, mi_comod=mi_comod, counts=counts)

# Load the saved data
saved_data = np.load(save_fname)
mi_full = saved_data.get('mi_full')
mi_comod = saved_data.get('mi_comod')
counts = saved_data.get('counts')

# Mask out the lower right corner where there's spurious communication
mask = np.full(mi_comod[0].shape, False)
for i_fm in range(len(f_mod)):
    for i_fc in range(len(f_car)):
        if (f_car[i_fc] - f_car_bw / 2) < f_mod[i_fm, 1]:
            mask[i_fm, i_fc] = True


# Plot the MI comodulogram
def plot_contour(x, **kwargs):
    x[mask] = np.nan
    plt.contourf(f_mod_centers, f_car, x.T,
                 # levels=np.linspace(0, np.nanmax(x), 50),
                 **kwargs)
    cb = plt.colorbar(format='%.2f',
                      ticks=[np.nanmin(x), np.nanmax(x)])
    cb.ax.set_ylabel('Sine-fit amp.')
    plt.ylabel('HF freq (Hz)')
    plt.xlabel('Phase freq (Hz)')


plt.clf()
for n, fn in enumerate(fnames):
    plt.subplot(3, 3, n + 1)
    plot_contour(mi_comod[n].copy())
    plt.title(re.search('Rat[0-9]+', fn).group())
# Plot the average
plt.subplot(3, 3, len(fnames) + 1)
mi_comod_avg = np.mean(np.array(mi_comod), axis=0)
plot_contour(mi_comod_avg)
plt.title('Average')

plt.tight_layout()

fn_details = f'bw{int(f_car_bw)}_nbins{n_phase_bins}'
plt.savefig(f'{plot_dir}phase-diff_mi_by_animal_{fn_details}.png')

# Plot the number of data points per phase bin
plt.clf()
colors = plt.cm.plasma(np.linspace(0, 1, len(f_mod_centers)))
phase_bins = np.linspace(-np.pi, np.pi, n_phase_bins)
for n, fn in enumerate(fnames):
    plt.subplot(3, 3, n + 1)
    for i_fm in range(len(f_mod_centers)):
        plt.plot(phase_bins, counts[n, i_fm, :],
                 color=colors[i_fm],
                 label=f_mod_centers[i_fm])
    plt.title(re.search('Rat[0-9]+', fn).group())
    plt.xlabel('Phase (rad)')
    plt.ylabel('Count')
    plt.ylim(0, plt.ylim()[1])
# Make a legend
plt.subplot(3, 3, n + 2)
for i_fm, fm in enumerate(f_mod_centers):
    plt.plot(f_mod[i_fm], [i_fm, i_fm],
             '-', color=colors[i_fm], linewidth=5)
plt.yticks([])
plt.xticks([0, 10, 20])
plt.xlabel('Frequency (Hz)')
plt.title('Legend')

plt.tight_layout()

plt.savefig(f'{plot_dir}phase-diff_hist_{fn_details}.png')


# Plot MI as a function of phase-diff

# Which phase freq and amp freq to choose
lf_range = [7, 10]
hf_range = [60, 90]
lf_freq_sel = (lf_range[0] < f_mod_centers) & (f_mod_centers < lf_range[1])
hf_freq_sel = (hf_range[0] < f_car) & (f_car < hf_range[1])

phase_bins = np.linspace(-np.pi, np.pi, n_phase_bins + 1)[:-1]

# Extract mutual information as a function of phase difference
fig, axs = plt.subplots(3, 3)
for n in range(len(fnames)):
    i_ax = [n // 3, n % 3]
    ax1 = axs[i_ax[0], i_ax[1]]

    # Plot average MI
    x = mi_full[n, ...]  # Select this animal
    x = np.mean(x[lf_freq_sel, ...], 0)  # Select the LF freqs
    x = np.mean(x[hf_freq_sel, ...], 0)  # Select the HF freqs

    # Plot the number of observations per bin
    c = np.mean(counts[n, lf_freq_sel, :], 0)
    c /= c.max()  # Normalize to amplitude 1

    color = 'tab:blue'
    ax1.set_xlabel('Phase diff (rad)')
    ax1.set_ylabel('MI', color=color)
    ax1.plot(phase_bins, x, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks([-np.pi, 0, np.pi])
    ax1.set_xticklabels(['$-\\pi$', 0, '$\\pi$'])
    ax1.set_xlim([-np.pi, np.pi])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Hist (norm)', color=color)
    ax2.plot(phase_bins, c, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(re.search('Rat[0-9]+', fnames[n]).group())
plt.tight_layout()

# Delete the empty plots
for k in [1, 2]:
    axs[-1, -k].axis('off')

fn_details = f'lf{lf_range}_hf{hf_range}_nbins{n_phase_bins}'
fn_details = fn_details.replace('[', '_').replace(']', '').replace(', ', '-')
plt.savefig(f'{plot_dir}phase-diff_mi_{fn_details}.png')


##################################################################
# Check whether MI depends on the number of observations in GCMI  #
##################################################################
# It seems that the bias correction fully accounts for the differences in
# sample size: MI is not correlated with sample size. So we should be alright
# to go on and analyze as a function of phase.

def rand():
    np.random.normal(size=int(n))


ns = [10 ** n for n in range(2, 6)]
n_sims = int(1e3)
bias = {n: [gcmi.gcmi_cc(rand(), rand()) for _ in range(n_sims)] for n in ns}

plt.clf()
plt.subplot(1, 2, 1)
for n, x in bias.items():
    plt.semilogx(np.ones(n_sims) * n, x, 'o', alpha=0.1)
    plt.plot(n, np.mean(x), 'ok', fillstyle='none')
    plt.plot(n, np.median(x), '+k')
    plt.xlim(1e1, 1e6)
plt.xlabel('N obs')
plt.ylabel('MI (bits)')

plt.subplot(1, 2, 2)
for n, x in bias.items():
    plt.semilogx(n, np.mean(x), 'ok', fillstyle='none')
    plt.plot(n, np.median(x), '+k')

plt.tight_layout()

plt.savefig(f'{plot_dir}simulate_MI_sample_size_bias.png')


###################################################################
# Look at mutual information at high frequencies between channels  #
###################################################################
"""
The first signal is CA3, and the second is CA1.
Positive values in the cross-MI indicate that CA3 precedes CA1.
"""


def cross_mi(x, y, max_lag, lag_step=1):
    """
    Compute MI between 2 signals as a function of lag

    Positive numbers indicate that x precedes y, and negative numbers indicate
    that y precedes x.

    Parameters
    ----------
    x, y : list or np.ndarray
        Signals to analyze. If x or y are multivariate, columns (dim 2)
        correspond to samples, and rows (dim 1) to variables.
    max_lag : int
        Maximum lag at which MI will be computed
    lag_step : int
        How many samples to step in between lags
    """
    lags = np.arange(-max_lag, max_lag + 1, lag_step)
    mi = []
    for lag in tqdm(lags):
        m = gcmi.gcmi_cc(x, np.roll(y, lag))
        mi.append(m)
    return lags, mi


freq_bands = {'theta-alpha': [4, 12],
              'beta': [15, 30],
              'low gamma': [30, 60],
              'high gamma': [60, 150]}
max_lags = 200  # In samples
lag_step = 5  # In samples
plt.figure()
xmi = {}
for band_name, band_lims in freq_bands.items():
    print(band_name)
    xmi[band_name] = []

    plt.clf()
    plt.axvline(x=0, linestyle='--', color='k')
    for i_rat in range(len(fnames)):
        print(i_rat)
        d = loadmat(data_dir + fnames[i_rat])
        s = [d['Data_EEG'][:, inx] for inx in [1, 2]]
        filt = [comlag.bp_filter(sig.T,
                                 band_lims[0], band_lims[1],
                                 d['Fs'], 2).T
                for sig in s]
        # Make a 2D version of the signal with it's Hilbert transform
        # This makes mutual information more informative
        h = [signal.hilbert(sig) for sig in filt]
        sig_2d = [np.stack([np.real(sig), np.imag(sig)]) for sig in h]

        # n = 1000
        # plt.plot(s[0][:n] / s[0].std())
        # plt.plot(filt[0][:n] / s[0].std())
        # plt.plot(sig_2d[0][:, :n].T / s[0].std())

        lags, xmi_i = cross_mi(sig_2d[0], sig_2d[1], 100, 5)
        xmi[band_name].append(xmi_i)

        lags_s = np.squeeze(lags / d['Fs'])
        plt.plot(lags_s, xmi_i, label=i_rat)
        max_inx = np.argmax(xmi_i)
        plt.plot(lags_s[max_inx], xmi_i[max_inx], 'ok')

    plt.legend()
    plt.title(f'Cross-MI: {band_name}')
    plt.savefig(f"{plot_dir}highfreq_crossMI_{band_name}.png")


####################################
# Look at phase-amplitude coupling  #
####################################

# Which frequencies to calculate phase for
f_mod_centers = np.logspace(np.log10(4), np.log10(20), 10)
f_mod_width = f_mod_centers / 6
# f_mod_centers = np.logspace(np.log10(4), np.log10(20), 5)
# f_mod_width = f_mod_centers / 4
f_mod = np.tile(f_mod_width, [2, 1]).T \
            * np.tile([-1, 1], [len(f_mod_centers), 1]) \
            + np.tile(f_mod_centers, [2, 1]).T

# Which frequencies to calculate power for
f_car = np.arange(20, 100, 5)
# f_car = np.arange(20, 100, 10)

f_car_cycles = 4  # Number of cycles in the HF wavelet transform
n_jobs = 3  # How many parallel jobs to run
nfft = 2 ** 11  # FFT length for CFC


def cfc_fnc(fn):
    """ Helper function for parallel computation
    """
    d = loadmat(data_dir + fn)
    d['Fs'] = np.squeeze(d['Fs'])
    s = [d['Data_EEG'][:, inx] for inx in [1, 2]]
    pac_out = [comlag.cfc_xspect(sig, sig, fs=d['Fs'],
                                 nfft=nfft, n_overlap=nfft/2,
                                 f_car=f_car, n_cycles=f_car_cycles)
               for sig in s]
    return pac_out


pac_out = Parallel(n_jobs=n_jobs)(delayed(cfc_fnc)(fn) for fn in fnames)
freqs = np.squeeze(pac_out[0][0][1])  # Get the vector of freqs from one run
pac_0 = [p[0][0] for p in pac_out]  # Get PAC for each signal
pac_1 = [p[1][0] for p in pac_out]

# Save the data
fname = f"{data_dir}pac/nfft{nfft}_ncyc{f_car_cycles}.npz"
np.savez(fname, freqs=freqs, pac_0=pac_0, pac_1=pac_1)


def plot_contour(x, colorbar_label='', **kwargs):
    plt.contourf(freqs, f_car, x.T,
                 **kwargs)
    cb = plt.colorbar(format='%.2f', ticks=[x.min(), x.max()])
    cb.ax.set_ylabel(colorbar_label)
    plt.ylabel('Amp freq (Hz)')
    plt.xlabel('Phase freq (Hz)')
    plt.xlim(0, 20)


# Which signal to plot
n_sig = 0
if n_sig == 0:
    pac = pac_0
elif n_sig == 1:
    pac = pac_1

for n, fn in enumerate(fnames):
    plt.subplot(3, 3, n + 1)
    plot_contour(np.squeeze(pac[n]), colorbar_label='Amplitude')
    plt.title(re.search('Rat[0-9]+', fn).group())
# Plot the average
plt.subplot(3, 3, len(fnames) + 1)
avg = np.squeeze(np.mean(np.array(pac), axis=0))
plot_contour(avg, colorbar_label='Amplitude')
plt.title('Average')

plt.tight_layout()

fname = f"pac_sig{n_sig}_nfft{nfft}_ncyc{f_car_cycles}"
plt.savefig(f'{plot_dir}{fname}.png')

# !notify-send "Analysis finished"


###########################################
# Communication based on transfer entropy  #
###########################################
'''
Signal 1: CA3
Signal 2: CA1
A positive TE difference indicates CA3 --> CA1.
A negative TE difference indicates CA1 --> CA3.
'''

# Parameters from Feb 17
f_mod = np.logspace(np.log10(4), np.log10(20), 15)
f_mod_bw = f_mod / 2
f_car = np.arange(20, 150, 10)
f_car_bw = f_car / 80 * 20  # Keep 20 Hz bandwidth at 80 Hz (~ 7 cycles)


#  ## Parameters from Jiang
#  # Low-freq 'modulator' frequencies
#  # Jiang et al (2015): "a choice of 3-5 cycles in relation to the slower
#  # oscillation is sensible"
# f_mod = np.arange(4, 16)
# f_mod_bw = f_mod / 2.5  # ~4 cycles
#
#  # High-freq 'carrier' frequencies
#  # Jiang (2015): "a range of 4 to 6 cycles is appropriate when analyzing
#  # how gamma band power is related to the phase of slower oscillations."
# f_car = np.arange(30, 150, 10)
# f_car_bw = f_car / 3  # ~5 cycles

# Plot the lowest and highest frequencies for the LF and HF filters
plt.clf()
i_plot = 1
for f_center, f_bw in ([f_mod, f_mod_bw], [f_car, f_car_bw]):
    for i_freq in [0, -1]:  # Plot the first and last ones
        plt.subplot(4, 1, i_plot)
        f_low = f_center[i_freq] - (f_bw[i_freq] / 2)
        f_high = f_center[i_freq] + (f_bw[i_freq] / 2)
        comlag.plot_filter_kernel(f_low, f_high, 2000, 1.5)
        i_plot += 1
plt.tight_layout()
now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
plt.savefig(f'{plot_dir}filter_kernels_{now}.png')

# Parameters for the MI phase-lag analysis
k_perm = 0
downsamp_factor = None  # 5 # 2000 Hz / 5 = 400 Hz
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


def te_fnc(fn):
    """ Helper function for parallel computation
    """
    d = loadmat(data_dir + fn)
    s = [d['Data_EEG'][:, inx] for inx in [1, 2]]
    if downsamp_factor is not None:
        s = [signal.decimate(sig, downsamp_factor) for sig in s]  # Downsample
    fs = d['Fs'][0][0]
    lag = int(lag_sec * fs)
    te_full = comlag.cfc_phaselag_transferentropy(s[0], s[1],
                                                  fs=fs,
                                                  lag=[lag],
                                                  **mi_params)
    # te = {'a': te_full[:, :, 0, 0],
    #       'b': te_full[:, :, 0, 1]}
    # te['diff'] = te['a'] - te['b']
    return te_full


te_out = Parallel(n_jobs=n_jobs)(delayed(te_fnc)(fn) for fn in fnames)

# Save the data
now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
save_fname = f"{data_dir}te/te_{now}.npz"
np.savez(save_fname, te=te_out, mi_params=mi_params, lag_sec=lag_sec)

# Load the saved data
# save_fname = f"{data_dir}te/te_2021-02-16-1559.npz"
saved_data = np.load(save_fname, allow_pickle=True)
te_out = saved_data.get('te')
mi_params = saved_data.get('mi_params').item()
lag_sec = saved_data.get('lag_sec')
f_mod_centers = np.mean(mi_params['f_mod'], axis=1)
f_car = mi_params['f_car']

te_a = np.array([x[:, :, 0, 0] for x in te_out])
te_b = np.array([x[:, :, 0, 1] for x in te_out])
te_diff = te_a - te_b
te = {'a': te_a,
      'b': te_b,
      'diff': te_diff}


# Plot it
def plot_contour(x):
    max_abs = np.max(np.abs(x))
    max_abs = np.max(np.abs(x[:, 2:]))
    if x.min() < 0:
        levels = np.linspace(-max_abs, max_abs, 50)
        cm = plt.cm.RdBu_r
        ticks = [-max_abs, 0, max_abs]
    else:
        levels = np.linspace(0, max_abs, 50)
        cm = plt.cm.viridis
        ticks = [0, max_abs]
    x[x > max_abs] = max_abs
    x[x < -max_abs] = -max_abs
    plt.contourf(f_mod_centers, f_car, x.T,
                 levels=levels,
                 cmap=cm)
    cb = plt.colorbar(format='%.0e', ticks=ticks)
    cb.ax.set_ylabel('Sine power (bits$^2$)')
    plt.ylabel('Amp freq (Hz)')
    plt.xlabel('Phase freq (Hz)')


def subset_freqs(x):
    return x[:, f_car >= 40]


# Plot it for each rat
for direction in te.keys():
    x = te[direction]
    plt.figure(figsize=(8, 5))
    plt.clf()
    for n, fn in enumerate(fnames):
        plt.subplot(3, 3, n + 1)
        plot_contour(np.squeeze(x[n, :, :]))
        plt.title(re.search('Rat[0-9]+', fn).group())
    # Plot the average
    plt.subplot(3, 3, len(fnames) + 1)
    avg = np.squeeze(np.mean(x, axis=0))
    plot_contour(avg)
    plt.title('Average')

    plt.tight_layout()

    data_timestamp = save_fname[save_fname.find('2'):-4]
    fname = f"te_{data_timestamp}_{direction}"
    plt.savefig(f'{plot_dir}{fname}.png')


# Plot analyses that were run in sbatch on the Bluebear cluster

# Load data from the cluster
timestamp = '2021-04-08-1033'
p_thresh = 0.05
cmaps = {'PD(AB)-PD(BA)': plt.cm.RdBu_r,
         'PD(AB-BA)': plt.cm.plasma}
contour_color = {'PD(AB)-PD(BA)': 'black',
                 'PD(AB-BA)': 'white'}
perm_type = None  # None | shift | signal
te_all = []
mi_params_all = []
lag_sec_all = []
for diff_type in ('PD(AB)-PD(BA)', 'PD(AB-BA)'):
    plt.figure(figsize=(8, 5))
    for i_rat in range(len(fnames)):
        fn = f"te_{timestamp}_rat{i_rat}_{perm_type}.npz"
        try:
            saved_data = np.load(f"{data_dir}te/{fn}",
                                 allow_pickle=True)
        except FileNotFoundError:
            print(f'No file found: {fn}')
            plt.subplot(3, 3, i_rat + 1)
            continue
        te_full, clust_info = saved_data.get('te')
        mi_params = saved_data.get('mi_params').item()
        lag_sec = saved_data.get('lag_sec')
        if i_rat == 0:
            print(mi_params)

        te = te_full[diff_type]

        x = te['diff'][0, :, :, 0]  # Plot the empirical data for the first lag

        plt.subplot(3, 3, i_rat + 1)

        maxabs = np.max(np.abs(x))
        if diff_type == 'PD(AB)-PD(BA)':
            levels = np.linspace(-maxabs, maxabs, 50)
            cb_ticks = [-maxabs, 0, maxabs]
        elif diff_type == 'PD(AB-BA)':
            levels = np.linspace(0, maxabs, 50)
            cb_ticks = [0, maxabs]
        plt.contourf(mi_params['f_mod'],
                     mi_params['f_car'],
                     x.T,
                     levels=levels,
                     cmap=cmaps[diff_type])
        cb = plt.colorbar(format='%.2e')
        # Plot significant clusters
        cl_info = clust_info[diff_type]
        clust_labels = cl_info['labels'][:, :, 0].astype(int)
        signif_clusters = np.nonzero(
            np.array(cl_info['stats']) > cl_info['cluster_thresh'])
        clust_highlight = np.isin(clust_labels,
                                  1 + signif_clusters[0]).astype(int)
        thresh = np.percentile(cl_info['max_per_perm'],
                               [(1 - p_thresh) * 100])
        for i_clust in np.unique(clust_labels):
            if i_clust == 0:
                continue
            elif cl_info['stats'][i_clust-1] > thresh:
                plt.contour(mi_params['f_mod'],
                            mi_params['f_car'],
                            (clust_labels == i_clust).T,
                            levels=[0.5],
                            colors=contour_color[diff_type],
                            alpha=0.5)
        cb.set_ticks(cb_ticks)
        cb.ax.set_ylabel('bits $^2$')
        plt.ylabel('HF freq (Hz)')
        plt.xlabel('Phase freq (Hz)')
        plt.title(f'p = {cl_info["pval"]:.3f}')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}te/{timestamp}_{diff_type}_{perm_type}.png')

timestamp = '2021-04-08-1033'
timestamp = '2021-04-09-1503'
timestamp = '2021-04-09-1537'
timestamp = '2021-04-09-1608'
timestamp = '2021-04-09-1648'  # 2/17 w/ downsampling
timestamp = '2021-04-09-1656'  # 2/17 w/out downsampling
diff_type = 'PD(AB)-PD(BA)'
perm_type = 'None'
for direction in 'ab':
    plt.figure(figsize=(8, 5))
    for i_rat in range(len(fnames)):
        fn = f"te_{timestamp}_rat{i_rat}_{perm_type}.npz"
        try:
            saved_data = np.load(f"{data_dir}te/{fn}",
                                 allow_pickle=True)
        except FileNotFoundError:
            print(f'No file found: {fn}')
            continue
        te_full, clust_info = saved_data.get('te')
        mi_params = saved_data.get('mi_params').item()
        lag_sec = saved_data.get('lag_sec')
        if i_rat == 0 and direction == 'a':
            for k, v in mi_params.items():
                print(k, ":", v)

        te = te_full[diff_type]

        x = te[direction][0, :, :, 0]  # Plot the emp data for the first lag

        plt.subplot(3, 3, i_rat + 1)
        fm = mi_params['f_mod']
        fc = mi_params['f_car']
        max_level = np.max(x[:, fc >= 25])

        levels = np.linspace(0, max_level, 50)
        plt.contourf(fm, fc, x.T, levels=levels)
        cb = plt.colorbar(format='%.0e')
        cb.set_ticks([0, max_level])
    plt.tight_layout()
    plt.savefig(f'{plot_dir}te/{timestamp}_{direction}.png')


####################################################
# Plot the impulse responses for the HF BP-filters  #
####################################################

fs = 1000  # Sampling rate in Hz
impulse_dur = 2  # seconds
impulse_len = int(impulse_dur * fs)  # samples
t = np.arange(impulse_len) / fs  # Time vector in seconds
t -= t.mean()

impulse = np.zeros(impulse_len)
impulse[impulse_len // 2] = 1

f_bw_ratio = np.arange(1.5, 6.1, 0.5)
f_center = 10 * np.ones(f_bw_ratio.shape)
f_bw = f_center / f_bw_ratio
assert len(f_center) == len(f_bw)

plt.figure()
for i_freq in range(len(f_center)):
    plt.subplot(3, 4, i_freq + 1)
    f = f_center[i_freq]
    bw = f_bw[i_freq]
    msg = f"Ratio: {f_bw_ratio[i_freq]:.1f}\n{f:.1f} $\\pm$ {bw / 2:.1f} Hz"
    plt.title(msg)
    f_low = f - (bw / 2)
    f_high = f + (bw / 2)
    ir = comlag.bp_filter(impulse, f_low, f_high, fs)
    plt.plot(t, ir, label=f)
    plt.yticks([])
    plt.xticks([])
plt.tight_layout()
plt.savefig(f'{plot_dir}te/tile_by_param_filter_kernels.png')


##################################################
# Tile the space of HF and LF filter bandwidths  #
# Analyses run on Bluebear                       #
##################################################

# Full tiled analyses
ratio_levels = np.arange(1.5, 6.1, 0.5)
# ratio_levels = np.arange(2.0, 6.01, 1.0)
files = os.listdir(data_dir + 'te/')
date = '2021-04-15'
pattern = 'rat{i_rat}_lfratio-{lf_ratio:.1f}_hfratio-{hf_ratio:.1f}.npz'
pattern = f'te_{date}-[0-9]+_{pattern}'
diff_type = 'PD(AB)-PD(BA)'
use_zscore = False
direc = 'a'
plot_fname_stem = plot_dir + "te/tile_by_param_{anim}_{direc}.png"

# "SNR" analyses
ratio_levels = np.arange(2.0, 6.01, 1.0)
files = os.listdir(data_dir + 'te/')
date = '2021-04-26'
pattern = 'rat{i_rat}_lfratio-{lf_ratio:.1f}_hfratio-{hf_ratio:.1f}.npz'
pattern = f'te_{date}-[0-9]+_{pattern}'
diff_type = 'PD(AB)-PD(BA)'
use_zscore = True
direc = 'b'
plot_fname_stem = plot_dir + "te/tile_by_param_{anim}_{direc}_zscore.png"

ytick_spacing = 4
xtick_spacing = 4
common_color_scale = True
directions = ['a', 'b', 'diff']

te_all = np.full([len(ratio_levels),
                  len(ratio_levels),
                  12,
                  24,
                  3,
                  len(fnames)],
                 np.nan)  # LF ratio, HF ratio, LF freq, HF freq, direc, animal

max_abs_vals = np.full([len(ratio_levels), len(ratio_levels), len(fnames) + 1],
                       np.nan)
for i_lf, lf_ratio in enumerate(ratio_levels):
    for i_hf, hf_ratio in enumerate(ratio_levels[::-1]):
        for i_rat in range(len(fnames)):
            pat = pattern.format(i_rat=i_rat,
                                 lf_ratio=lf_ratio,
                                 hf_ratio=hf_ratio)
            match_inx = [i for i, f in enumerate(files) if re.match(pat, f)]
            if len(match_inx) > 1:
                raise(Exception('More than one matching file found'))
            else:
                match_inx = match_inx[0]
            fn = files[match_inx]
            saved_data = np.load(f"{data_dir}te/{fn}",
                                 allow_pickle=True)
            mi_params = saved_data.get('mi_params').item()
            for i_direc, direc in enumerate(['a', 'b', 'diff']):
                te_indiv = saved_data.get('te')[0][diff_type][direc]
                if use_zscore:
                    te_indiv = stats.zscore(te_indiv, axis=None)
                te_indiv = np.squeeze(te_indiv[0, ...])  # Get real data
                te_all[i_lf, i_hf, :, :, i_direc, i_rat] = te_indiv
                max_level = np.max(np.abs(te_indiv))
                max_abs_vals[i_hf, i_lf, i_rat] = max_level

for i_direc, direc in enumerate(directions):
    plt.close('all')
    figsize = (10, 8)
    for i_rat in range(len(fnames)):
        plt.figure(i_rat, figsize=figsize)
    plt.figure('average', figsize=figsize)
    for i_lf, lf_ratio in enumerate(ratio_levels):
        for i_hf, hf_ratio in enumerate(ratio_levels[::-1]):
            i_plot = (i_hf * len(ratio_levels)) + i_lf + 1
            for i_rat in range(len(fnames)):
                # Plot the results for this individual rat
                plt.figure(i_rat)
                plt.subplot(len(ratio_levels), len(ratio_levels), i_plot)
                te_x = te_all[i_lf, i_hf, :, :, i_direc, i_rat]
                if common_color_scale:
                    if direc == 'diff':
                        dirslice = -1
                    else:
                        dirslice = slice(0, 2)
                    max_level = np.max(np.abs(te_all[:, :, :, :,
                                                     dirslice, i_rat]))
                else:
                    max_level = np.max(np.abs(te_x))
                plt.imshow(te_x.T, origin="lower",
                           interpolation="none",
                           aspect='auto',
                           vmin=-max_level, vmax=max_level,
                           cmap=plt.cm.RdBu_r)
                if i_plot == 1:
                    if use_zscore:
                        label = f'{max_level:.1f}'
                    else:
                        label = f'{max_level:.1e}'
                    plt.text(1, 20, label)
                if i_lf == 0:
                    plt.ylabel(f'HF: {hf_ratio:.1f}')
                    plt.yticks(range(len(mi_params['f_car']))[::ytick_spacing],
                               mi_params['f_car'][::ytick_spacing])
                else:
                    plt.yticks([])
                if i_hf == len(ratio_levels) - 1:
                    plt.xlabel(f'LF: {lf_ratio:.1f}')
                    plt.xticks(range(len(mi_params['f_mod']))[::xtick_spacing],
                               mi_params['f_mod'][::xtick_spacing])
                else:
                    plt.xticks([])
                if lf_ratio == max(ratio_levels) \
                        and hf_ratio == max(ratio_levels):
                    plt.tight_layout()

            # Plot the average over all the rats
            te_avg = np.mean(te_all[i_lf, i_hf, :, :, i_direc, :], axis=-1)
            if common_color_scale:
                if direc == 'diff':
                    dirslice = -1
                else:
                    dirslice = slice(0, 2)
                max_level = np.max(np.abs(np.mean(te_all[:, :, :, :,
                                                         dirslice, :],
                                                  axis=-1)))
            else:
                max_level = np.max(np.abs(np.mean(te_all[i_lf, i_hf, ...],
                                                  axis=-1)))
            max_abs_vals[i_hf, i_lf, -1] = max_level
            mi_params = saved_data.get('mi_params').tolist()
            plt.figure('average')
            plt.subplot(len(ratio_levels), len(ratio_levels), i_plot)
            plt.imshow(te_avg.T, origin="lower",
                       interpolation="none",
                       aspect='auto',
                       vmin=-max_level, vmax=max_level,
                       cmap=plt.cm.RdBu_r)
            if i_plot == 1:
                if use_zscore:
                    label = f'{max_level:.1f}'
                else:
                    label = f'{max_level:.1e}'
                plt.text(1, 20, label)
            if i_lf == 0:
                plt.ylabel(f'HF: {hf_ratio:.1f}')
                plt.yticks(range(len(mi_params['f_car']))[::ytick_spacing],
                           mi_params['f_car'][::ytick_spacing])
            else:
                plt.yticks([])
            if i_hf == len(ratio_levels) - 1:
                plt.xlabel(f'LF: {lf_ratio:.1f}')
                plt.xticks(range(len(mi_params['f_mod']))[::xtick_spacing],
                           mi_params['f_mod'][::xtick_spacing])
            else:
                plt.xticks([])
            if lf_ratio == max(ratio_levels) and hf_ratio == max(ratio_levels):
                plt.tight_layout()

    plt.figure('average')
    plt.tight_layout()
    plt.savefig(plot_fname_stem.format(anim='average', direc=direc))

    for i_rat in range(len(fnames)):
        plt.figure(i_rat)
        rat_num = re.search('Rat[0-9]+', fnames[i_rat]).group()
        plt.savefig(plot_fname_stem.format(anim=rat_num, direc=direc))

plt.figure('color-scale', figsize=figsize)
for i_rat in range(len(fnames)):
    plt.subplot(3, 3, i_rat + 1)
    plt.title(re.search('Rat[0-9]+', fnames[i_rat]).group())
    plt.imshow(max_abs_vals[:, :, i_rat],
               vmin=0)
    cb = plt.colorbar(format='%.4f')
    cb.set_ticks([0,
                  max_abs_vals[:, :, i_rat].max()])
    plt.xticks([])
    plt.yticks([])
tick_slice = slice(1, None, 2)
for tick_fnc in (plt.xticks, plt.yticks):
    tick_fnc(range(len(ratio_levels))[tick_slice],
             [int(e) for e in ratio_levels[tick_slice]])
plt.xlabel('LF ratio')
plt.xlabel('HF ratio')
# Plot the scale of the average
plt.subplot(3, 3, len(fnames) + 1)
plt.title('Average')
plt.imshow(max_abs_vals[:, :, -1],
           vmin=0)
cb = plt.colorbar(format='%.4f')
cb.set_ticks([0, max_abs_vals[:, :, -1].max()])
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'{plot_dir}te/tile_by_param_colorbar_{direc}.png')


##################################
# Dig into a specific comparison #
##################################

"""
Rat 44, bottom left with neighboring points that go in opposite directions
"""

lf_ratio = 2.0
hf_ratio = 3.0
i_rat = 4
fn = fnames[i_rat]
lf_freqs = [8, 9, 10]  # Which LF frequencies to zoom in on
hf_freqs = [50]  # Which HF frequencies to zoom in on
i_perm = 0  # 0: Empirical data; >0: Permutations

ratio_levels = np.arange(2.0, 6.01, 1.0)
files = os.listdir(data_dir + 'te/')
# date = '2021-04-26'  # Randomly shifted
date = '2021-05-13'  # Randomly shuffled A/B
pattern = f'rat{i_rat}_lfratio-{lf_ratio:.1f}_hfratio-{hf_ratio:.1f}.npz'
pattern = f'te_{date}-[0-9]+_{pattern}'
diff_type = 'PD(AB)-PD(BA)'

pat = pattern.format(i_rat=i_rat,
                     lf_ratio=lf_ratio,
                     hf_ratio=hf_ratio)
match_inx = [i for i, f in enumerate(files) if re.match(pat, f)]
if len(match_inx) > 1:
    raise(Exception('More than one matching file found'))
else:
    match_inx = match_inx[0]
fn = files[match_inx]
saved_data = np.load(f"{data_dir}te/{fn}", allow_pickle=True)
mi_params = saved_data.get('mi_params').item()
te = saved_data.get('te')[0][diff_type]

plt.figure(figsize=(10, 6))
for i_cond, (cond, te_cond) in enumerate(te.items()):
    plt.subplot(2, 3, i_cond + 1)
    plt.title(cond)
    te_cond = np.squeeze(te_cond[i_perm, :, :])

    # Plot the average over all the rats
    max_level = np.max(np.abs(te_cond))
    mi_params = saved_data.get('mi_params').tolist()
    plt.imshow(te_cond.T,
               origin="lower",
               interpolation="none",
               aspect='auto',
               vmin=-max_level, vmax=max_level,
               cmap=plt.cm.RdBu_r)
    ytick_spacing = 4
    xtick_spacing = 2
    plt.yticks(range(len(mi_params['f_car']))[::ytick_spacing],
               mi_params['f_car'][::ytick_spacing])
    plt.xticks(range(len(mi_params['f_mod']))[::xtick_spacing],
               mi_params['f_mod'][::xtick_spacing])
    plt.xlabel('LF freq (Hz)')
    plt.ylabel('HF freq (Hz)')
    plt.gca().add_patch(  # Draw a box around the frequencies we're focusing on
            Rectangle([mi_params['f_mod'].tolist().index(lf_freqs[0]) - 0.5,
                       mi_params['f_car'].tolist().index(hf_freqs[0]) - 0.5],
                      width=len(lf_freqs),
                      height=len(hf_freqs),
                      fill=False,
                      edgecolor='k',
                      linestyle='--',
                      linewidth=2))
    plt.colorbar(ticks=[0, np.max(np.abs(te_cond))],
                 format='%2.0e',
                 orientation='vertical')
plt.tight_layout()

# Choose the cells of the analysis to look at
i_fm = [4, 5, 6]
i_fc = [4]

mi_params_orig = copy.deepcopy(mi_params)
mi_params['f_mod'] = mi_params['f_mod'][i_fm]
mi_params['f_car'] = mi_params['f_car'][i_fc]
mi_params['f_mod_bw'] = mi_params['f_mod'] / lf_ratio
mi_params['f_car_bw'] = mi_params['f_car'] / hf_ratio
mi_params['n_perm_shift'] = 0
mi_params['n_perm_signal'] = 1
mi_params['return_phase_bins'] = True

d = loadmat(data_dir + fnames[i_rat])
fs = d['Fs'][0][0]
s = [d['Data_EEG'][:, inx] for inx in [1, 2]]

# Downsample the data
downsamp_factor = 5
s = [signal.decimate(sig, downsamp_factor) for sig in s]
fs /= downsamp_factor

# Split the data into epochs
epoch_dur = 5.0
epoch_len = int(epoch_dur * fs)
n_splits = len(s[0]) // epoch_len
sig_len = n_splits * epoch_len
s = [np.stack(np.split(sig[:sig_len], n_splits), axis=1) for sig in s]

lag_sec = 0.006
lag = int(lag_sec * fs)
mi_c, mi, _ = comlag.cfc_phaselag_transferentropy(s[0], s[1],
                                                  fs=fs,
                                                  lag=[lag],
                                                  **mi_params)

# mi dims: Permutation, LF freq, HF freq, CMI lag, direction, LF phase bin
labels = ('a', 'b', 'diff')
colors = ['firebrick', 'grey', 'royalblue']
plt.figure()
for i_direc in range(2):  # a, b, diff
    plt.subplot(2, 3, i_direc + 4)
    for i_freq in range(3):
        x = np.squeeze(mi[i_perm, i_freq, 0, 0, i_direc, :])
        plt.plot(x,
                 label=mi_params['f_mod'][i_freq],
                 color=colors[i_freq])
    plt.xticks([])
    plt.xlabel('Phase bins')
    plt.ylabel('TE (bits)')
    if i_direc == 0:
        plt.legend()
plt.tight_layout()

plt.savefig(f'{plot_dir}te/zooming_in_on_phase_bins_perm{i_perm}.png')

# Plot the TE by phase-bin for a few permutations
n_perms = 5
for rand_type in ('shift', 'shuffleAB'):
    if rand_type == 'shift':
        mi_params['n_perm_shift'] = n_perms
        mi_params['n_perm_signal'] = 0
    elif rand_type == 'shuffleAB':
        mi_params['n_perm_shift'] = 0
        mi_params['n_perm_signal'] = n_perms
    mi_c, mi, _ = comlag.cfc_phaselag_transferentropy(s[0], s[1],
                                                      fs=fs,
                                                      lag=[lag],
                                                      **mi_params)
    plt.figure()
    for i_perm in range(n_perms):
        for i_direc in range(2):
            plt.subplot(n_perms, 2, i_direc + 1 + (i_perm * 2))
            for i_freq in range(3):
                x = np.squeeze(mi[i_perm, i_freq, 0, 0, i_direc, :])
                plt.plot(x,
                         label=mi_params['f_mod'][i_freq],
                         color=colors[i_freq])
            plt.xticks([])
            plt.xlabel('Phase bins')
            plt.ylabel('TE (bits)')
            # if i_direc == 0 and i_perm == 0:
            #     plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}te/zooming_in_on_phase_bins_perms_{rand_type}.png')


# Show the randomization distributions for the adjacent flipped pixels
lf_ratio = 2.0
hf_ratio = 3.0
i_rat = 4
fn = fnames[i_rat]
lf_freqs = [8, 10]  # Which LF frequencies to zoom in on
hf_freqs = [50]  # Which HF frequencies to zoom in on

ratio_levels = np.arange(2.0, 6.01, 1.0)
files = os.listdir(data_dir + 'te/')

# date = '2021-04-26'
# perm_type = 'shifted'  # Randomly shifted

date = '2021-05-13'
perm_type = 'shuffledAB'  # Randomly shuffled A/B

pattern = f'rat{i_rat}_lfratio-{lf_ratio:.1f}_hfratio-{hf_ratio:.1f}.npz'
pattern = f'te_{date}-[0-9]+_{pattern}'
diff_type = 'PD(AB)-PD(BA)'

pat = pattern.format(i_rat=i_rat,
                     lf_ratio=lf_ratio,
                     hf_ratio=hf_ratio)
match_inx = [i for i, f in enumerate(files) if re.match(pat, f)]
if len(match_inx) > 1:
    raise(Exception('More than one matching file found'))
else:
    match_inx = match_inx[0]
fn = files[match_inx]
saved_data = np.load(f"{data_dir}te/{fn}", allow_pickle=True)
mi_params = saved_data.get('mi_params').item()
te = saved_data.get('te')[0][diff_type]

# Plot raw TE distributions
colors = ['tab:blue', 'tab:green', 'tab:red']
plt.figure()
for i_freq in range(len(lf_freqs)):
    plt.subplot(2, 1, i_freq + 1)
    plt.title(f"{lf_freqs[i_freq]} Hz")
    for i_direc, direc in enumerate(directions):
        lf_inx = mi_params['f_mod'].tolist().index(lf_freqs[i_freq])
        hf_inx = mi_params['f_car'].tolist().index(hf_freqs[0])
        x = te[direc][:, lf_inx, hf_inx, 0]
        plt.hist(x[1:], 20,
                 color=colors[i_direc],
                 histtype='step', density=True)
        plt.axvline(x=x[0], color=colors[i_direc],
                    linestyle='--', label=direc)
        plt.xlabel('PhaseDep TE (bits$^2$)')
        plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}te/zooming_in_on_phase_bins_hist-{perm_type}.png')

# Plot z-scores
for direc in directions:
    te[direc] = stats.zscore(te[direc], axis=None)

plt.figure()
directions = ['a', 'b', 'diff']
i_plot = 1
for direc in directions:
    for i_freq in range(len(lf_freqs)):
        lf_inx = mi_params['f_mod'].tolist().index(lf_freqs[i_freq])
        hf_inx = mi_params['f_car'].tolist().index(hf_freqs[0])
        x = te[direc][:, lf_inx, hf_inx, 0]
        plt.subplot(3, 2, i_plot)
        plt.title(f"{direc}, {lf_freqs[i_freq]} Hz")
        plt.hist(x[1:], 20)
        plt.axvline(x=x[0], color='r')
        i_plot += 1
plt.tight_layout()
plt.savefig(f'{plot_dir}te/zooming_in_on_phase_bins_z-hist-{perm_type}.png')


#######
# Does swapping A/B cause TE to be drastically reduced?
# - Plot raw TE histogram instead of PhaseDep(AB-BA)
# - Simulated data along with raw data
data_type = 'rat'  # rat, sim, sim-pac
lf_ratio = 2.0
hf_ratio = 3.0
i_rat = 4
# LF frequency is meaningless if we're collapsing over phase bins
f_mod = np.array([1])
f_car = np.array([50])  # Which HF frequencies to zoom in on
f_mod_bw = f_mod / lf_ratio
f_car_bw = f_car / hf_ratio

mi_params = dict(f_mod=f_mod,
                 f_mod_bw=f_mod_bw,
                 f_car=f_car,
                 f_car_bw=f_car_bw,
                 n_bins=8,
                 decimate=None,
                 n_perm_phasebin=0,
                 n_perm_phasebin_indiv=0,
                 n_perm_signal=0,
                 n_perm_shift=0,
                 min_shift=None, max_shift=None,
                 perm_phasebin_flip=False,
                 cluster_alpha=0.05,
                 diff_method='both',
                 calc_type=2,
                 method=None,
                 return_phase_bins=True,
                 verbose=True)
if mi_params['n_perm_signal'] > 0:
    perm_type = 'shuffledAB'
elif mi_params['perm_phasebin_flip']:
    perm_type = 'phasebin_flip'
else:
    perm_type = 'NOT_SPECIFIED'
# for k,v in mi_params.items(): globals()[k] = v

d = loadmat(data_dir + fnames[i_rat])
fs = d['Fs'][0][0]
s = [d['Data_EEG'][:, inx] for inx in [1, 2]]

# Downsample the data
downsamp_factor = 5
s = [signal.decimate(sig, downsamp_factor) for sig in s]
fs /= downsamp_factor

if data_type == 'sim':  # Simulate data
    t, s_a, s_b = simulate.sim(dur=len(s[0]) / fs,
                               fs=fs,
                               gamma_freq=(40, 60),
                               noise_amp=1.0,
                               signal_leakage=0,
                               gamma_lag_a=0.010,
                               gamma_lag_a_to_b=0.006,
                               common_noise_amp=0.5,
                               common_alpha_amp=0.0)
    s = [s_a, s_b]
elif data_type == 'sim-pac':
    t, s_a, s_b = simulate.sim_lf_coh_with_pac(
            dur=len(s[0]) / fs,
            fs=fs,
            lag=int(0.006 * fs),
            gamma_freq=(40, 60))
    s = [s_a, s_b]

# Split the data into epochs
epoch_dur = 5.0
epoch_len = int(epoch_dur * fs)
n_splits = len(s[0]) // epoch_len
sig_len = n_splits * epoch_len
s = [np.stack(np.split(sig[:sig_len], n_splits), axis=1) for sig in s]

lag_sec = 0.006
lag = int(lag_sec * fs)
mi_c, mi, _ = comlag.cfc_phaselag_transferentropy(s[0], s[1],
                                                  fs=fs,
                                                  lag=[lag],
                                                  **mi_params)
# mi dims: Permutation, LF freq, HF freq, CMI lag, direction, LF phase bin

plt.figure()
colors = ['tab:blue', 'tab:green']
for i_direc, direc in enumerate(['a', 'b']):
    x = np.squeeze(mi[:, 0, 0, 0, i_direc, 0])
    print(x[:4])
    plt.hist(x[1:], 20,
             color=colors[i_direc],
             histtype='step', density=True)
    plt.axvline(x=x[0], color=colors[i_direc],
                linestyle='--', label=direc)
    plt.xlabel('TE (bits)')
    plt.ylabel('Density')
plt.legend()
plt.title(f'Raw TE at ${f_car[0]} \\pm {f_car_bw[0] / 2:.1f}$ Hz')
plt.savefig(f'{plot_dir}te/te_real-vs-shuffled-{perm_type}-{data_type}.png')

# Do the same analysis by hand
s_filt = [comlag.bp_filter(sig.T,
                           f_car[0] - (f_car_bw[0] / 2),
                           f_car[0] + (f_car_bw[0] / 2),
                           fs, 2).T
          for sig in s]
s_hilb = [signal.hilbert(sig, axis=0) for sig in s_filt]
s_2d = [np.stack([np.real(sig), np.imag(sig)]) for sig in s_hilb]
s_2d_orig = s_2d.copy()

plt.figure
plt.subplot(2, 1, 1)
i_epoch = 0
plt.plot(s[0][:, i_epoch])
plt.plot(s_filt[0][:, i_epoch])
plt.plot(np.real(s_hilb[0][:, i_epoch]))
plt.plot(np.imag(s_hilb[0][:, i_epoch]))
plt.plot(s_filt[0][:, i_epoch])
plt.xlim(0, 100)


def L(x):
    """ lag function """
    return np.roll(x, lag, axis=1)


# Shuffle the two signals A/B
k_perm = 100
mi = []
for i_perm in tqdm(range(k_perm)):
    del s_2d
    s_2d = s_2d_orig.copy()
    if i_perm > 0:  # Shuffle signals
        for i_epoch in range(s_2d[0].shape[2]):
            if np.random.choice([True, False]):
                s0_tmp = s_2d[0][:, :, i_epoch].copy()
                s1_tmp = s_2d[1][:, :, i_epoch].copy()
                s_2d[0][:, :, i_epoch] = s1_tmp.copy()
                s_2d[1][:, :, i_epoch] = s0_tmp.copy()
    s_2d_append = [np.reshape(sig, (2, -1), order='F') for sig in s_2d]
    try:
        i = gcmi.gccmi_ccc(L(s_2d_append[0]),
                           s_2d_append[1],
                           L(s_2d_append[1]))
    except np.linalg.LinAlgError:
        i = np.nan
    mi.append(i)

plt.figure()
plt.hist(mi[1:], 20,
         color='tab:blue',
         label='Shuffle A/B',
         histtype='step', density=True)
plt.axvline(x=mi[0], color='red',
            linestyle='--',
            label='True value')

# Shuffle the epochs. This should have even lower MI than shuffling signals
s_2d = s_2d_orig.copy()
mi = []
for i_perm in tqdm(range(k_perm)):
    shuffle_inx = np.arange(s_2d[0].shape[-1])  # Shuffle the epochs
    np.random.shuffle(shuffle_inx)
    s_2d[0] = s_2d[0][:, :, shuffle_inx]
    s_2d_append = [np.reshape(sig, (2, -1), order='F') for sig in s_2d]
    try:
        i = gcmi.gccmi_ccc(L(s_2d_append[0]),
                           s_2d_append[1],
                           L(s_2d_append[1]))
    except np.linalg.LinAlgError:
        i = np.nan
    mi.append(i)
plt.hist(mi, 20,
         color='tab:green',
         label='Shuffle epochs',
         histtype='step', density=True)
plt.xlabel('TE (bits)')
plt.ylabel('Density')
plt.legend()
plt.savefig(f'{plot_dir}te/te_real-vs-shuffled-simulated.png')

# Cut the segment into 2 pieces
# and see if flipping one half drastically reduces TE
lf_ratio = 2.0
hf_ratio = 3.0
i_rat = 4
# LF frequency is meaningless if we're collapsing over phase bins
f_mod = np.array([1])
f_car = np.array([50])  # Which HF frequencies to zoom in on
f_mod_bw = f_mod / lf_ratio
f_car_bw = f_car / hf_ratio

d = loadmat(data_dir + fnames[i_rat])
fs = d['Fs'][0][0]
s = [d['Data_EEG'][:, inx] for inx in [1, 2]]

# Downsample the data
downsamp_factor = 5
s = [signal.decimate(sig, downsamp_factor) for sig in s]
fs /= downsamp_factor

# Do the same analysis by hand
s_filt = [comlag.bp_filter(sig.T,
                           f_car[0] - (f_car_bw[0] / 2),
                           f_car[0] + (f_car_bw[0] / 2),
                           fs, 2).T
          for sig in s]
s_hilb = [signal.hilbert(sig, axis=0) for sig in s_filt]
s_2d = [np.stack([np.real(sig), np.imag(sig)]) for sig in s_hilb]

# Compute real TE between the two measures
te_real = {}
te_real['a'] = gcmi.gccmi_ccc(L(s_2d[0]),
                              s_2d[1],
                              L(s_2d[1]))
te_real['b'] = gcmi.gccmi_ccc(L(s_2d[1]),
                              s_2d[0],
                              L(s_2d[0]))

# Flip the second half of the signal and recompute it
s_2d_flip = [sig.copy() for sig in s_2d]
n_samps = s[0].size
s_2d_flip[0][:, n_samps//2:] = s_2d[1][:, n_samps//2:]
s_2d_flip[1][:, n_samps//2:] = s_2d[0][:, n_samps//2:]
te_flip = {}
te_flip['a'] = gcmi.gccmi_ccc(L(s_2d_flip[0]),
                              s_2d_flip[1],
                              L(s_2d_flip[1]))
te_flip['b'] = gcmi.gccmi_ccc(L(s_2d_flip[1]),
                              s_2d_flip[0],
                              L(s_2d_flip[0]))

te = list(te_real.values()) + list(te_flip.values())
te = np.array(te)
xpos = [1, 2, 4, 5]
plt.clf()
plt.plot(xpos, te, 'o')
plt.axhline(y=0, linestyle='--', color='k')
plt.xticks(xpos, ['Emp AB', 'Emp BA', 'Flip AB', 'Flip BA'])
plt.ylabel("TE (bits)")
plt.savefig(f'{plot_dir}te/te_flip_half.png')

###########################################################################
# Try this with the new analysis that flips the signal for each phase-bin #
###########################################################################
lf_ratio = 2.0
hf_ratio = 3.0
i_rat = 4
# LF frequency is meaningless if we're collapsing over phase bins
f_mod = np.array([1])
f_car = np.array([50])  # Which HF frequencies to zoom in on
f_mod_bw = f_mod / lf_ratio
f_car_bw = f_car / hf_ratio

mi_params = dict(f_mod=f_mod,
                 f_mod_bw=f_mod_bw,
                 f_car=f_car,
                 f_car_bw=f_car_bw,
                 n_bins=8,
                 decimate=None,
                 n_perm_phasebin=0,
                 n_perm_phasebin_indiv=0,
                 n_perm_signal=0,
                 n_perm_shift=0,
                 min_shift=None, max_shift=None,
                 perm_phasebin_flip=True,
                 cluster_alpha=0.05,
                 diff_method='both',
                 calc_type=2,
                 method='sine psd',
                 return_phase_bins=True,
                 verbose=True)
if mi_params['n_perm_signal'] > 0:
    perm_type = 'shuffledAB'
elif mi_params['perm_phasebin_flip']:
    perm_type = 'phasebin_flip'
else:
    perm_type = 'NOT_SPECIFIED'
# for k,v in mi_params.items(): globals()[k] = v

d = loadmat(data_dir + fnames[i_rat])
fs = d['Fs'][0][0]
s = [d['Data_EEG'][:, inx] for inx in [1, 2]]

# Downsample the data
downsamp_factor = 5
s = [signal.decimate(sig, downsamp_factor) for sig in s]
fs /= downsamp_factor

# Split the data into epochs
epoch_dur = 5.0
epoch_len = int(epoch_dur * fs)
n_splits = len(s[0]) // epoch_len
sig_len = n_splits * epoch_len
s = [np.stack(np.split(sig[:sig_len], n_splits), axis=1) for sig in s]

lag_sec = 0.006
lag = int(lag_sec * fs)
mi_c, mi, _ = comlag.cfc_phaselag_transferentropy(s[0], s[1],
                                                  fs=fs,
                                                  lag=[lag],
                                                  **mi_params)

