import numpy as np
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import re
import os
from joblib import Parallel, delayed
import datetime
import comlag

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
f_mod_centers = np.logspace(np.log10(4), np.log10(20), 15)
f_mod_width = f_mod_centers / 8
f_mod = np.tile(f_mod_width, [2, 1]).T \
            * np.tile([-1, 1], [len(f_mod_centers), 1]) \
            + np.tile(f_mod_centers, [2, 1]).T

# Which frequencies to calculate power for
f_car = np.arange(20, 100, 10)

f_car_bw = 10 # Bandwidth of the HF bandpass filter
n_phase_bins = 8 # Number of bins for the phase-difference
n_jobs = 3 # How many parallel jobs to run

save_fname = f"{data_dir}mi_comod/bw{int(f_car_bw)}_nbins{n_phase_bins}.npz"

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
                 #levels=np.linspace(0, np.nanmax(x), 50),
                 **kwargs)
    cb = plt.colorbar(format='%.2f',
                      ticks=[np.nanmin(x), np.nanmax(x)])
    cb.ax.set_ylabel('Sine-fit amp.')
    plt.ylabel('HF freq (Hz)')
    plt.xlabel('Phase freq (Hz)')

plt.clf()
for n,fn in enumerate(fnames):
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


#### Plot MI as a function of phase-diff

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
    x = mi_full[n, ...] # Select this animal
    x = np.mean(x[lf_freq_sel, ...], 0) # Select the LF freqs
    x = np.mean(x[hf_freq_sel, ...], 0) # Select the HF freqs

    # Plot the number of observations per bin
    c = np.mean(counts[n, lf_freq_sel, :], 0)
    c /= c.max() # Normalize to amplitude 1

    color = 'tab:blue'
    ax1.set_xlabel('Phase diff (rad)')
    ax1.set_ylabel('MI', color=color)
    ax1.plot(phase_bins, x, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks([-np.pi, 0, np.pi])
    ax1.set_xticklabels(['$-\pi$', 0, '$\pi$'])
    ax1.set_xlim([-np.pi, np.pi])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Hist (norm)', color=color)
    ax2.plot(phase_bins, c, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(re.search('Rat[0-9]+', fnames[n]).group())
plt.tight_layout()

# Delete the empty plots
for k in [1,2]:
    axs[-1, -k].axis('off')

fn_details = f'lf{lf_range}_hf{hf_range}_nbins{n_phase_bins}'
fn_details = fn_details.replace('[', '_').replace(']', '').replace(', ', '-')
plt.savefig(f'{plot_dir}phase-diff_mi_{fn_details}.png')


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


###########################################
# Communication based on transfer entropy #
###########################################
'''
Signal 1: CA3
Signal 2: CA1
A positive TE difference indicates CA3 --> CA1.
A negative TE difference indicates CA1 --> CA3.
'''

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

# Plot the lowest and highest frequencies for the LF and HF filters
plt.clf()
i_plot = 1
for f_center, f_bw in ([f_mod, f_mod_bw], [f_car, f_car_bw]):
    for i_freq in [0, -1]: # Plot the first and last ones
        plt.subplot(4, 1, i_plot)
        f_low = f_center[i_freq] - (f_bw[i_freq] / 2)
        f_high = f_center[i_freq] + (f_bw[i_freq] / 2)
        comlag.plot_filter_kernel(f_low, f_high, 2000, 1.5)
        i_plot += 1
plt.tight_layout()

# Parameters for the MI phase-lag analysis
n_jobs = 3
lag_sec = 0.005 # By eyeballing the plot of high-gamma cross-MI
mi_params = dict(f_mod=f_mod,
                 f_mod_bw=f_mod_bw,
                 f_car=f_car,
                 f_car_bw=f_car_bw,
                 n_bins=2**3,
                 method='sine psd',
                 n_perm=100,
                 min_shift=None, max_shift=None, cluster_alpha=0.05,
                 calc_type=2)


def te_fnc(fn):
    """ Helper function for parallel computation
    """
    downsamp_factor = 4
    d = loadmat(data_dir + fn)
    s = [d['Data_EEG'][:,inx] for inx in [1, 2]]
    s = [signal.decimate(sig, downsamp_factor) for sig in s] # Downsample
    fs = d['Fs'][0][0]
    lag = int(lag_sec * fs)
    te_full = comlag.cfc_phaselag_transferentropy(s[0], s[1],
                                                  fs=fs,
                                                  lag=[lag],
                                                  **mi_params)
    ##te = {'a': te_full[:, :, 0, 0],
    ##      'b': te_full[:, :, 0, 1]}
    ##te['diff'] = te['a'] - te['b']
    return te_full

te_out = Parallel(n_jobs=n_jobs)(delayed(te_fnc)(fn) for fn in fnames)

# Save the data
now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
save_fname = f"{data_dir}te/te_{now}.npz"
np.savez(save_fname, te=te_out, mi_params=mi_params, lag_sec=lag_sec)

# Load the saved data
#save_fname = f"{data_dir}te/te_2021-02-16-1559.npz"
saved_data = np.load(save_fname, allow_pickle=True)
te_out = saved_data.get('te')
mi_params = saved_data.get('mi_params').item()
lag_sec = saved_data.get('lag_sec')
f_mod_centers = np.mean(mi_params['f_mod'], axis=1)
f_car = mi_params['f_car']

te_a = np.array([x[:,:,0,0] for x in te_out])
te_b = np.array([x[:,:,0,1] for x in te_out])
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
timestamp = '2021-03-31-1417'
p_thresh = 0.05
cmaps = {'PD(AB)-PD(BA)': plt.cm.RdBu_r,
         'PD(AB-BA)': plt.cm.plasma}
contour_color = {'PD(AB)-PD(BA)': 'black',
                 'PD(AB-BA)': 'white'}
for perm_type in ('shift', 'signal'):
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

            te = te_full[diff_type]

            x = te['diff'][0, :, :, 0] # Plot the empirical data for the first lag

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
            clust_labels = cl_info['labels'][:,:,0].astype(int)
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

    
for direction in 'ab':
    plt.figure(figsize=(8, 5))
    for i_rat in range(len(fnames)):
        fn = f"te_{timestamp}_rat{i_rat}_{perm_type}.npz"
        try:
            saved_data = np.load(f"{data_dir}te/{fn}",
                                allow_pickle=True)
        except FileNotFoundError:
            print(f'No file found: {fn}')
        te_full, clust_info = saved_data.get('te')
        mi_params = saved_data.get('mi_params').item()
        lag_sec = saved_data.get('lag_sec')

        te = te_full[diff_type]

        x = te[direction ][0, :, :, 0] # Plot the empirical data for the first lag

        plt.subplot(3, 3, i_rat + 1)

        levels = np.linspace(*np.array([0, 1]) * np.max(x), 50)
        plt.contourf(mi_params['f_mod'],
                    mi_params['f_car'],
                    x.T,
                    levels=levels)
        cb = plt.colorbar(format='%.2e')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}te/{timestamp}_{direction}.png')


####################################################
# Plot the impulse responses for the HF BP-filters #
####################################################

fs = 1000 # Sampling rate in Hz
impulse_dur = 2 # seconds
impulse_len = int(impulse_dur * fs) # samples
t = np.arange(impulse_len) / fs # Time vector in seconds
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
    msg = f"{f:.1f} $\pm$ {bw / 2:.1f} Hz\nRatio: {f_bw_ratio[i_freq]:.1f}"
    plt.title(msg)
    f_low = f - (bw / 2)
    f_high = f + (bw / 2)
    ir = comlag.bp_filter(impulse, f_low, f_high, fs)
    plt.plot(t, ir, label=f)
    plt.yticks([])
    plt.xticks([])
plt.tight_layout()


#################################################
# Tile the space of HF and LF filter bandwidths #
# Analyses run on Bluebear                      #
#################################################

ratio_levels = np.arange(1.5, 6.1, 0.5)
#ratio_levels = np.arange(2.0, 6.01, 1.0)
files = os.listdir(data_dir + 'te/')
pattern = 'rat{i_rat}_lfratio-{lf_ratio:.1f}_hfratio-{hf_ratio:.1f}.npz'
diff_type = 'PD(AB)-PD(BA)'

figsize = (10, 8)
for i_rat in range(len(fnames)):
    plt.figure(i_rat, figsize=figsize)
plt.figure('average', figsize=figsize)

for i_lf, lf_ratio in enumerate(ratio_levels):
    for i_hf, hf_ratio in enumerate(ratio_levels[::-1]):
        i_plot = (i_hf * len(ratio_levels)) + i_lf + 1
        te = [] # hold the results for all the rats
        for i_rat in range(len(fnames)):
            pat = pattern.format(i_rat=i_rat,
                                 lf_ratio=lf_ratio,
                                 hf_ratio=hf_ratio)
            match_inx = [i for i, f in enumerate(files) if f.endswith(pat)]
            if len(match_inx) > 1:
                raise(Exception('More than one matching file found'))
            else:
                match_inx = match_inx[0]
            fn = files[match_inx]
            saved_data = np.load(f"{data_dir}te/{fn}",
                                 allow_pickle=True)
            te_indiv = np.squeeze(saved_data.get('te')[0][diff_type]['diff'])
            te.append(te_indiv)

            # Plot the results for this individual rat
            plt.figure(i_rat)
            plt.subplot(len(ratio_levels), len(ratio_levels), i_plot)
            max_level = np.max(np.abs(te_indiv))
            levels = np.linspace(-max_level, max_level, 50)
            plt.contourf(mi_params['f_mod'], mi_params['f_car'], te_indiv.T,
                         cmap=plt.cm.RdBu_r, levels=levels)
            #cb = plt.colorbar(format='%.0e')
            #cb.set_ticks([0, max_level])
            if i_lf == 0:
                plt.ylabel(f'HF: {hf_ratio:.1f}')
            else:
                plt.yticks([])
            if i_hf == len(ratio_levels) - 1:
                plt.xlabel(f'LF: {lf_ratio:.1f}')
            else:
                plt.xticks([])
            if lf_ratio == max(ratio_levels) and hf_ratio == max(ratio_levels):
                plt.tight_layout()

        # Plot the average over all the rats
        te = np.stack(te)
        te_avg = np.mean(te, 0)
        mi_params = saved_data.get('mi_params').tolist()
        plt.figure('average')
        plt.subplot(len(ratio_levels), len(ratio_levels), i_plot)
        #plt.title(f'LF:{lf_ratio}, HF:{hf_ratio}')
        max_level = np.max(np.abs(te_avg))
        levels = np.linspace(-max_level, max_level, 50)
        plt.contourf(mi_params['f_mod'], mi_params['f_car'], te_avg.T,
                     cmap=plt.cm.RdBu_r, levels=levels)
        #cb = plt.colorbar(format='%.0e')
        #cb.set_ticks([0, max_level])
        if i_lf == 0:
            plt.ylabel(f'HF: {hf_ratio:.1f}')
        else:
            plt.yticks([])
        if i_hf == len(ratio_levels) - 1:
            plt.xlabel(f'LF: {lf_ratio:.1f}')
        else:
            plt.xticks([])

plt.figure('average')
plt.tight_layout()
plt.savefig(f'{plot_dir}te/tile_by_param_average.png')

for i_rat in range(len(fnames)):
    plt.figure(i_rat)
    rat_num = re.search('Rat[0-9]+', fnames[i_rat]).group()
    plt.savefig(f'{plot_dir}te/tile_by_param_{rat_num}.png')

