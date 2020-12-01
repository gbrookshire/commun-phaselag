import numpy as np
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import re
from joblib import Parallel, delayed
import comlag

plt.ion()

data_dir = '../data/RatData/'
plot_dir = data_dir + 'plots/'
fnames = ['EEG_speed_allData_Rat17_20120616_begin1.mat',
          'EEG_speed_allData_Rat6_20111021_begin1.mat',
          'EEG_speed_allData_Rat13_20120131_begin1_CA3_CSC9.mat',
          'EEG_speed_allData_Rat45_20140522_begin1.mat',
          'EEG_speed_allData_Rat44_20140506_begin1_CA3_CSC4_CA1_TT6.mat',
          'EEG_speed_allData_Rat47_20140923_begin1_CA3_CSC11_CA1_TT3.mat',
          'EEG_speed_allData_Rat31_20140110_begin1_CA3_CSC7_CA1_TT2.mat']


################################################
# Plot the PSD for each rat and recording site #
################################################
labels = ['CA3', 'CA1']
nfft = 2 ** 10
plt.clf()
for n,fn in enumerate(fnames):
    plt.subplot(3, 3, n + 1)
    d = loadmat(data_dir + fn)
    s = [d['Data_EEG'][:,inx] for inx in [1, 2]]
    for sig, lab in zip(s, labels):
        f, y = signal.welch(sig, nperseg=nfft, noverlap=nfft / 2,
                            fs=d['Fs'][0][0])
        plt.loglog(f, y, label=lab)
    plt.xlim(1, 200)
    plt.xticks([1, 10, 100])
    plt.title(re.search('Rat[0-9]+', fn).group())

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}spectra.png')


#################################################
# Check the distribution of LF phase difference #
#################################################

n_bins = 8
phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
for n,fn in enumerate(fnames):
    plt.subplot(3, 3, n + 1)
    d = loadmat(data_dir + fn)
    s = [d['Data_EEG'][:,inx] for inx in [1, 2]]
    filt = [comlag.bp_filter(sig, 4, 12, d['Fs'], 2) for sig in s]
    phase = [np.angle(signal.hilbert(sig)) for sig in filt]
    phase_diff = comlag.wrap_to_pi(phase[0] - phase[1])
    phase_diff = np.digitize(phase_diff, phase_bins) - 1 # Binned    
    plt.hist(phase_diff, n_bins)
    plt.xlabel('Phase diff (rad)')
    plt.ylabel('Density')
    plt.title(re.search('Rat[0-9]+', fn).group())
    #plt.xlim([-np.pi, np.pi])
    #plt.xticks([-np.pi, 0, np.pi], ['$-\pi$', 0, '$\pi$'] )
plt.tight_layout()
plt.savefig(f'{plot_dir}phase-diff_hist.png')

"""
We see very strong coherence between the two signals, so it's not clear that
we'll be able to look at connectivity as a function of phase difference --
because we essentially only have data from one phase difference.
"""


###################################################################
# Compute MI between HF time-series as a function of LF phase lag #
###################################################################

# Which frequencies to calculate phase for
f_mod_centers = np.logspace(np.log10(4), np.log10(20), 10)
f_mod_width = f_mod_centers / 6
##f_mod_centers = np.logspace(np.log10(4), np.log10(20), 5)
##f_mod_width = f_mod_centers / 4
f_mod = np.tile(f_mod_width, [2, 1]).T \
            * np.tile([-1, 1], [len(f_mod_centers), 1]) \
            + np.tile(f_mod_centers, [2, 1]).T

# Which frequencies to calculate power for
f_car = np.arange(20, 100, 5)
##f_car = np.arange(20, 100, 10)

save_fname = f"{data_dir}mi_comod/bw{int(f_car_bw)}_nbins{n_phase_bins}.npz"

f_car_bw = 10 # Bandwidth of the HF bandpass filter
n_phase_bins = 8 # Number of bins for the phase-difference
n_jobs = 3 # How many parallel jobs to run

def mi_fnc(fn):
    """ Helper function for parallel computation
    """
    print(f'f_car_bw: {f_car_bw}')
    print(f'n_phase_bins: {n_phase_bins}')
    d = loadmat(data_dir + fn)
    s = [d['Data_EEG'][:,inx] for inx in [1, 2]]
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
!notify-send "Analysis finished"

# Load the saved data
saved_data = np.load(save_fname)
mi_full = saved_data.get('mi_full')
mi_comod = saved_data.get('mi_comod')
counts = saved_data.get('counts')

# Plot the MI comodulogram
def plot_contour(x, colorbar_label='', **kwargs):
    plt.contourf(f_mod_centers, f_car, x.T,
                 levels=np.linspace(0, 1, 50),
                 **kwargs)
    cb = plt.colorbar(format='%.2f', ticks=[x.min(), x.max()])
    cb.ax.set_ylabel(colorbar_label)
    plt.ylabel('HF freq (Hz)')
    plt.xlabel('Phase freq (Hz)')

for n,fn in enumerate(fnames):
    plt.subplot(3, 3, n + 1)
    plot_contour(mi_comod[n], colorbar_label='$R^2$')
    plt.title(re.search('Rat[0-9]+', fn).group())
# Plot the average
plt.subplot(3, 3, len(fnames) + 1)
mi_comod_avg = np.mean(np.array(mi_comod), axis=0)
plot_contour(mi_comod_avg, colorbar_label='$R^2$')
plt.title('Average')

plt.tight_layout()

fn_details = f'bw{int(f_car_bw)}_nbins{n_phase_bins}'
plt.savefig(f'{plot_dir}phase-diff_mi_by_animal_{fn_details}.png')

# Plot the number of data points per phase bin
plt.clf()
colors = plt.cm.plasma(np.linspace(0, 1, len(f_mod_centers)))
phase_bins = np.linspace(-np.pi, np.pi, n_phase_bins)
for n,fn in enumerate(fnames):
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


# Which phase freq and amp freq to choose
i_f_mod = 5
i_f_car = 11
i_rat = 5

# Extract mutual information as a function of phase difference
x = mi_full[i_rat][i_f_mod, i_f_car]
plt.plot(x)


##################################################################
# Check whether MI depends on the number of observations in GCMI #
##################################################################
# It seems that the bias correction fully accounts for the differences in
# sample size: MI is not correlated with sample size. So we should be alright
# to go on and analyze as a function of phase.

ns = [10 ** n for n in range(2, 6)]
n_sims = int(1e3)
rand = lambda: np.random.normal(size=int(n))
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
# Look at mutual information at high frequencies between channels #
###################################################################
"""
The first signal is CA3, and the second is CA1.
Positive values in the cross-MI indicate that CA3 precedes CA1.
"""

import gcmi
def cross_mi(x, y, max_lag, lag_step=1):
    """
    Compute MI between 2 signals as a function of lag

    Positive numbers indicate that x precedes y, and negative numbers indicate
    that y precedes x.

    Parameters
    ----------
    x, y : list or np.ndarray
        Signals to analyze. If x or y are multivariate, columns (dim 2) correspond to
        samples, and rows (dim 1) to variables.
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
max_lags = 200 # In samples
lag_step = 5 # In samples
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
        s = [d['Data_EEG'][:,inx] for inx in [1, 2]]
        filt = [comlag.bp_filter(sig.T,
                                 band_lims[0], band_lims[1],
                                 d['Fs'], 2).T
                    for sig in s]
        # Make a 2D version of the signal with it's Hilbert transform
        # This makes mutual information more informative
        h = [signal.hilbert(sig) for sig in filt]
        sig_2d = [np.stack([np.real(sig), np.imag(sig)]) for sig in h]
        
        #n = 1000
        #plt.plot(s[0][:n] / s[0].std())
        #plt.plot(filt[0][:n] / s[0].std())
        #plt.plot(sig_2d[0][:, :n].T / s[0].std())
        
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
# Look at phase-amplitude coupling #
####################################

# Which frequencies to calculate phase for
f_mod_centers = np.logspace(np.log10(4), np.log10(20), 10)
f_mod_width = f_mod_centers / 6
##f_mod_centers = np.logspace(np.log10(4), np.log10(20), 5)
##f_mod_width = f_mod_centers / 4
f_mod = np.tile(f_mod_width, [2, 1]).T \
            * np.tile([-1, 1], [len(f_mod_centers), 1]) \
            + np.tile(f_mod_centers, [2, 1]).T

# Which frequencies to calculate power for
f_car = np.arange(20, 100, 5)
##f_car = np.arange(20, 100, 10)

f_car_cycles = 4 # Number of cycles in the HF wavelet transform
n_jobs = 3 # How many parallel jobs to run
nfft = 2 ** 11 # FFT length for CFC

def cfc_fnc(fn):
    """ Helper function for parallel computation
    """
    d = loadmat(data_dir + fn)
    d['Fs'] = np.squeeze(d['Fs'])
    s = [d['Data_EEG'][:,inx] for inx in [1, 2]]
    pac_out = [comlag.cfc_xspect(sig, sig, fs=d['Fs'],
                                 nfft=nfft, n_overlap=nfft/2,
                                 f_car=f_car, n_cycles=f_car_cycles)
                    for sig in s]
    return pac_out

pac_out = Parallel(n_jobs=n_jobs)(delayed(cfc_fnc)(fn) for fn in fnames)
freqs = np.squeeze(pac_out[0][0][1]) # Get the vector of freqs from one run
pac_0 = [p[0][0] for p in pac_out] # Get PAC for each signal
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

for n,fn in enumerate(fnames):
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

!notify-send "Analysis finished"
