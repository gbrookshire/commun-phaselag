"""
Test the analysis on sEEG data with simultaneous recordings of hipp and cortex

Sample: 0.15 s
Delay: 1.2 s

"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mne.externals.h5io import write_hdf5, read_hdf5

import pactools
import comlag

import sys
sys.path.append('../../wm-seeg/analysis/') # Desktop
sys.path.append('../../2020_wm-seeg/wm-seeg/analysis/') # Laptop
sys.path.append('/rds/projects/p/panyz-wm-seeg/wm-seeg/analysis') # Bluebear
import wm

plt.ion()

plot_dir = '../data/plots/seeg/'
save_dir = '../data/seeg/'

####################
# Read in the data #
####################

n = 14

epochs, trial_info = wm.load_clean_epochs(
                        n,
                        trig=wm.exp_info['event_dict']['d'],
                        tmin=0.0,
                        tmax=1.2,
                        baseline=None)
x = epochs.get_data()

# Get channel information
chan_info = wm.read_chan_info(n)
keep_chans = np.isin(chan_info['number'], epochs.ch_names)
chan_info = chan_info[keep_chans]

picks_hipp = (chan_info['region'] == 'Hippocampus').to_numpy()
picks_parietal = np.isin(chan_info['region'], wm.regions['parietal'])


######################################
# Plot the spectrum for each channel #
######################################

plt.clf()
ax = plt.subplot(1, 1, 1)
colors = ['blue', 'red']
for color,picks in zip(colors, [picks_parietal, picks_hipp]):
    #nfft = 2 ** 11
    #psds, freqs = mne.time_frequency.psd_welch(
    #                epochs,
    #                fmin=0, fmax=200,
    #                n_fft=nfft,
    #                n_overlap=nfft / 2,
    #                picks=np.array(epochs.ch_names)[picks])
    psds, freqs = mne.time_frequency.psd_multitaper(
                    epochs,
                    fmin=0, fmax=180,
                    bandwidth=2,
                    picks=np.array(epochs.ch_names)[picks])
    plt.loglog(freqs, np.mean(psds, axis=0).T,
               color=color)
plt.ylabel('Power (mV$^2$/Hz)')
plt.xlabel('Frequency (Hz)')
plt.xlim(0, 180)
plt.text(50, 1e-5, 'Parietal', color=colors[0])
plt.text(50, 5e-6, 'Hipp', color=colors[1])
plt.tight_layout()
plt.savefig(plot_dir + 'spectra.png')


###############################
# Plot the correlation matrix #
###############################

x_conc = np.concatenate(list(x), 1)
cor = np.corrcoef(x_conc)
clim = np.max(np.abs(cor))
plt.clf()
plt.imshow(cor, vmin=-clim, vmax=clim, cmap='RdBu_r')
ax = plt.gca()
ax.set_yticks(range(epochs.info['nchan']))
ax.set_yticklabels(chan_info['region'])
ax.set_xticks(range(epochs.info['nchan']))
ax.set_xticklabels(epochs.info['ch_names'], rotation=90)
plt.savefig(plot_dir + 'correl_mat.png')


#####################
# Look at coherence #
#####################

x_hipp = x[:, picks_hipp, :]
x_hipp_parietal = x[:, picks_hipp | picks_parietal, :]
nfft = 2**10
print(f'Window size: {nfft / epochs.info["sfreq"]} s')
est = pactools.utils.Coherence(block_length=nfft, fs=epochs.info['sfreq'])
# Swap axes to get the right data format: (n_signal, n_epoch, n_points)
x_coh = np.swapaxes(x_hipp_parietal, 0, 1) 
coh = est.fit(x_coh, x_coh)
freq = np.arange(1 + nfft / 2) * epochs.info['sfreq'] / nfft

# Plot coherence between hippocampal chans by frequency
plt.clf()
for i_chan in range(picks_hipp.sum()):
    plt.subplot(2, 2, i_chan + 1)
    c = np.squeeze(coh[i_chan, :, :])
    plt.plot(freq, np.abs(c).T)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence (norm)')
    plt.xlim(2, 100)
plt.savefig(plot_dir + 'coh_hipp.png')

# Plot theta-band coherence between each pair of channels
freq_of_interest = (4 < freq) & (freq < 8)
theta_coh = np.mean(np.abs(coh[:, :, freq_of_interest]), axis=-1)
plt.clf()
plt.imshow(theta_coh, aspect='equal', origin='upper')
chan_num = np.array(epochs.ch_names)[picks_hipp | picks_parietal]
chan_region = chan_info['region'][picks_hipp | picks_parietal]
ax = plt.gca()
ax.set_xticks(range(theta_coh.shape[0]))
ax.set_yticks(range(theta_coh.shape[1]))
ax.set_yticklabels(chan_region)
ax.set_xticklabels(chan_num, rotation=90)
plt.colorbar()
plt.savefig(plot_dir + 'coh_theta.png')


###################################
# Look at CFC within each channel # 
###################################

# Which frequencies to calculate phase for
# f_mod_centers = np.logspace(np.log10(3), np.log10(30), 20)
# f_mod_width = f_mod_centers / 6
# f_mod = np.tile(f_mod_width, [2, 1]).T \
#             * np.tile([-1, 1], [len(f_mod_centers), 1]) \
#             + np.tile(f_mod_centers, [2, 1]).T
f_mod_centers = np.arange(3, 31)
f_mod = f_mod_centers + [[-1], [1]]
f_mod = f_mod.T

# Which frequencies to calculate power for
f_car = np.arange(30, 120, 5)

# Helper function to allow for switching between CFC functions
def cfc_func(a, b, method='tort'):
    cfc_kwargs = {'fs': epochs.info['sfreq'],
                  'f_mod': f_mod,
                  'f_car': f_car}
    if method== 'tort':
        mi = comlag.cfc_tort(a, b, **cfc_kwargs)
    elif method== 'sine-amp':
        mi = comlag.cfc_sine(a, b, **cfc_kwargs)
        mi = mi[0]
    elif method== 'sine-r':
        mi = comlag.cfc_sine(a, b, **cfc_kwargs)
        mi = mi[1]
    else:
        raise(Exception(f"type '{type}' not recognized"))

    return mi

def plot_cfc(mi):
    # Plots on a log-axis if the frequencies are log-spaced
    plt.imshow(mi, aspect='auto', origin='lower')
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, len(f_mod_centers)))
    ax.set_yticks(np.arange(0, len(f_car)))
    # Labels for major ticks
    ax.set_xticklabels(np.round(f_mod_centers))
    ax.set_yticklabels(f_car)
    plt.ylabel('Amp freq (Hz)')
    plt.xlabel('Phase freq (Hz)')
    #plt.colorbar(ticks=[0, mi.max()],
    #                format='%.3f')

# Look at CFC within single electrodes
# Choose a parietal channel with high CFC to look at connectivity
cfc_method = 'sine-amp'
keep_chans = np.where(picks_hipp | picks_parietal)[0]
mi_within = []
for i_chan in tqdm(keep_chans):
    x_chan = np.squeeze(x[:, i_chan, :]).T
    mi_chan = cfc_func(x_chan, x_chan, cfc_method)
    mi_within.append(mi_chan)
# Plot it
plt.clf()
for i, i_chan in enumerate(keep_chans):
    plt.subplot(5, 6, i + 1)
    #plot_cfc(mi_within[i])
    m = mi_within[i]
    extent = [np.min(f_mod), np.max(f_mod),
              np.min(f_car), np.max(f_car)]
    plt.imshow(m,
               aspect='auto',
               extent=extent,
               origin='lower')
    chan_num = chan_info['number'].to_list()[i_chan]
    chan_reg = chan_info['region'].to_list()[i_chan]
    plt.title(f"{chan_num} {chan_reg}")
plt.tight_layout()
plt.savefig(f"{plot_dir}cfc_within_{cfc_method}.png")


####################################################################
# Look at cross-region CFC between hippocampal channels and cortex #
####################################################################

# Plot cross-region CFC b/w hippocampus and cortex
chan_combos = list(itertools.product(np.nonzero(picks_hipp)[0],
                                     np.nonzero(picks_parietal)[0]))
mi = []
for chan_1, chan_2 in tqdm(chan_combos):
    mi_x = cfc_func(x[:, chan_1, :].T,
                    x[:, chan_2, :].T,
                    cfc_method)
    mi.append(mi_x)
# Plot it
plt.clf()
for i_combo in range(len(chan_combos)):
    plt.subplot(4, 25, i_combo + 1)
    plot_cfc(mi[i_combo])
    plt.axis('off')
    if i_combo < 25:
        plt.title(f'{combo[0]} -> {combo[1]}')
        plt.title(np.array(epochs.ch_names)[chan_combos[i_combo][1]])
    #combo = chan_combos[i_combo]
    #plt.title(f'{combo[0]} -> {combo[1]}')
plt.tight_layout()
plt.savefig(f"{plot_dir}cfc_cross-region_{cfc_method}.png")

# Plot the mean of CFC across hippocampal channels
mi_split = np.array(np.split(np.array(mi),
                             picks_hipp.sum(),
                             axis=0))
mi_avg = np.mean(mi_split, axis=0)
plt.clf()
for i_chan in range(mi_avg.shape[0]):
    plt.subplot(5, 5, i_chan + 1)
    plot_cfc(mi[i_chan])
    plt.axis('off')
    plt.title(np.array(epochs.ch_names)[chan_combos[i_chan][1]])
plt.tight_layout()


##################################################################
# Look at phase-diff CFC between hippocampal channels and cortex #
##################################################################

def cfc_phasediff_func(a, b, method='tort'):
    cfc_kwargs = {'fs': epochs.info['sfreq'],
                  'f_mod': f_mod,
                  'f_car': f_car}
    if method== 'tort':
        mi = comlag.cfc_phasediff_tort(a, b, **cfc_kwargs)
    elif method== 'sine':
        mi = comlag.cfc_phasediff_sine(a, b, **cfc_kwargs)
    else:
        raise(Exception(f"type '{type}' not recognized"))

    return mi

chan_combos = list(itertools.product(np.nonzero(picks_hipp)[0],
                                     np.nonzero(picks_parietal)[0]))
mi = []
for chan_1, chan_2 in tqdm(chan_combos):
    mi_x = cfc_phasediff_func(x[:, chan_1, :].T,
                              x[:, chan_2, :].T,
                              'sine')
    mi.append(mi_x)
# Plot it
plt.clf()
for i_combo in range(len(chan_combos)):
    chan_name = np.array(epochs.ch_names)[chan_combos[i_combo][1]]

    # Top rows: Phase-diff --> Hippocampus
    plt.subplot(9, 25, i_combo + 1)
    plot_cfc(mi[i_combo]['a'])
    plt.axis('off')
    if i_combo < 25:
        plt.title(f'{chan_name}')
    
    # Bottom rows: Phase-diff --> Cortex
    plt.subplot(9, 25, i_combo + 1 + 125)
    plot_cfc(mi[i_combo]['b'])
    plt.axis('off')
    if i_combo < 25:
        chan_name = np.array(epochs.ch_names)[chan_combos[i_combo][1]]
        plt.title(f'{chan_name}')

plt.tight_layout()
plt.savefig(f"{plot_dir}cfc_phase-diff_cross-region{cfc_method}.png")

# For each channel, make a separate figure
#   CFC within cortex
#   for each hipp channel
#      CFC hipp phase to cortex amp
#      CFC cortex phase to hipp amp
#      CFC phase diff to cortex amp
#      CFC phase diff to hipp amp
def plot_cfc_no_label(mi):
    plt.imshow(mi, aspect='auto', origin='lower',
               vmin=0, vmax=mi.max())
    ax = plt.gca()

    x_ticks_locs = f_mod_centers % 5 == 0
    x_ticks_locs = np.nonzero(x_ticks_locs)[0]
    ax.set_xticks(x_ticks_locs)
    ax.set_xticklabels(f_mod_centers[x_ticks_locs])

    y_ticks_locs = f_car % 20 == 0
    y_ticks_locs = np.nonzero(y_ticks_locs)[0]
    ax.set_yticks(y_ticks_locs)
    ax.set_yticklabels(f_car[y_ticks_locs])

    plt.colorbar(ticks=[0, mi.max()],
                 format='%.3f')

cfc_kwargs = {'fs': epochs.info['sfreq'],
              'f_mod': f_mod,
              'f_car': f_car}
chans = [705, 801, 802, 806, 807, 808, 901, 902, 903, 1003, 1004]
chans = [str(e) for e in chans]
n_hipp_chans = picks_hipp.sum()
for cort_chan in tqdm(chans):
    cort_chan_inx = epochs.ch_names.index(cort_chan)
    x_cort = x[:, cort_chan_inx, :].T
    plt.clf()
    for i_hipp_chan in range(n_hipp_chans):
        hipp_chan_inx = np.nonzero(picks_hipp)[0][i_hipp_chan]
        hipp_chan = epochs.ch_names[hipp_chan_inx]
        x_hipp = x[:, hipp_chan_inx, :].T
        # CFC hipp phase to cortex amp
        mi_hipp_to_cort = comlag.cfc_tort(x_hipp, x_cort, **cfc_kwargs)
        # CFC cortex phase to hipp amp
        mi_cort_to_hipp = comlag.cfc_tort(x_cort, x_hipp, **cfc_kwargs)
        # CFC phase diff
        mi_phasediff = comlag.cfc_phasediff_tort(x_hipp, x_cort, **cfc_kwargs)
        # Phase diff to hipp amp
        mi_phasediff_to_hipp = mi_phasediff['a']
        # Phase diff to cortex amp
        mi_phasediff_to_cort = mi_phasediff['b']

        base_plot_n = i_hipp_chan * n_hipp_chans 

        title = '$\\Phi(\\mathrm{{{}}}) \\rightarrow '\
                '\\mathrm{{A}}(\\mathrm{{{}}})$'

        plt.subplot(n_hipp_chans, 4, base_plot_n + 1)
        plot_cfc_no_label(mi_hipp_to_cort)
        plt.text(0.95, 0.05,
                 f'Hipp: {hipp_chan}', #\nCort: {cort_chan}',
                 color='white',
                 horizontalalignment='right',
                 verticalalignment='bottom',
                 transform=plt.gca().transAxes)
        if i_hipp_chan == 0:
            plt.title(title.format('Hipp', 'Cort'))

        plt.subplot(n_hipp_chans, 4, base_plot_n + 2)
        plot_cfc_no_label(mi_cort_to_hipp)
        if i_hipp_chan == 0:
            plt.title(title.format('Cort', 'Hipp'))

        title = '$\\Delta\\Phi \\rightarrow '\
                '\\mathrm{{A}}(\\mathrm{{{}}})$'

        plt.subplot(n_hipp_chans, 4, base_plot_n + 3)
        plot_cfc_no_label(mi_phasediff_to_hipp)
        if i_hipp_chan == 0:
            plt.title(title.format('Hipp'))

        plt.subplot(n_hipp_chans, 4, base_plot_n + 4)
        plot_cfc_no_label(mi_phasediff_to_cort)
        if i_hipp_chan == 0:
            plt.title(title.format('Cort'))

    plt.tight_layout()
    save_fname = f'cfc_{cort_chan}.png'
    plt.savefig(plot_dir + save_fname)


###################################################################
# Look at mutual information at high frequencies between channels #
###################################################################
epochs_hg = epochs.copy()
epochs_hg.filter(70, 200)
epochs_hg.apply_hilbert()

def cross_mi(x, y, max_lag):
    """ Compute MI between 2 signals as a function of lag

    Parameters
    ----------
    x, y : list or np.ndarray
        Signals to analyze. If x or y are multivariate, columns (dim 2) correspond to
        samples, and rows (dim 1) to variables.
    max_lag : int
        Maximum lag at which MI will be computed
    """
    lags = np.arange(-max_lag, max_lag + 1)
    mi = []
    for lag in (lags):
        m = gcmi.gcmi_cc(x, np.roll(y, lag))
        mi.append(m)
    return lags, mi

max_lag = 50

# Pad the end of each trial with zeros
x_hg = epochs_hg.get_data()
x_hg = np.pad(x_hg, ((0,0), (0,0), (max_lag, max_lag)))
# Concatenate all the trials in time
x_hg = np.concatenate(np.split(x_hg, x_hg.shape[0], axis=0), axis=-1)
x_hg = np.squeeze(x_hg)
# Separate analytic signal into two variables
split_analytic = lambda x: np.vstack([np.real(x), np.imag(x)])
# Get MI between hippocampal and cortical channels
max_lag_sec = 0.2 # Maximum lag in seconds
max_lag = int(max_lag_sec * epochs.info['sfreq'])
combos = itertools.product(np.nonzero(picks_hipp)[0],
                           np.nonzero(picks_hipp | picks_parietal)[0])
combos = [c for c in combos if c[0] < c[1]] # Avoid duplicates & self-connect
#mi_combos = []
#for inx_1, inx_2 in tqdm(combos):
#    lags, mi = cross_mi(split_analytic(x_hg[inx_1,:]),
#                        split_analytic(x_hg[inx_2,:]),
#                        max_lag)
#    mi_combos.append(mi)

def cross_mi_parhelper(chan_indices):
    inx_1, inx_2 = chan_indices
    lags, mi = cross_mi(split_analytic(x_hg[inx_1,:]),
                        split_analytic(x_hg[inx_2,:]),
                        max_lag)
    return mi
mi_combos = Parallel(n_jobs=3)(delayed(cross_mi_parhelper)(c) for c in combos)
mi_combos = np.array(mi_combos)
# Save the results
write_hdf5(save_dir + 'xmi_combos.h5',
        {'xmi': mi_combos,
         'lags': lags,
         'combos': combos})

# Load the saved results
d = read_hdf5(save_dir + 'xmi_combos.h5')
mi_combos = d['xmi']
lags = d['lags']
combos = d['combos']

# Plot MI 
combos = np.array(combos)
hipp_to_hipp = np.isin(combos[:,1], np.nonzero(picks_hipp)[0])
plt.clf()
t = lags / epochs.info['sfreq']
#plt.plot(t, mi_combos[hipp_to_hipp,:].T, '-k', alpha=0.5)
plt.semilogy(t, mi_combos[~hipp_to_hipp,:].T)
plt.axvline(x=0, linestyle='--', color='k')
plt.xlabel('Lag (s)')
plt.ylabel('MI (bits)')
# Get the lag at which there's maximal mi
max_inx = np.argmax(mi_combos, axis=1)
for k in range(len(combos)):
    if hipp_to_hipp[k]: continue
    inx = max_inx[k]
    plt.plot(t[inx], mi_combos[k, inx], 'o')

# Make separate plots for each channel pair
plt.clf()
for k in range(len(mi_combos)): 
     plt.subplot(10, 11, k + 1) 
     plt.plot(t, mi_combos[k,:].T) 
     plt.plot(t[max_inx[k]], mi_combos[k, max_inx[k]], 'o') 
     plt.axvline(x=0, linestyle='--', color='k')  
     plt.axhline(y=0, linestyle='--', color='k') 
     plt.axis(False) 
     c1, c2 = combos[k]
     plt.title(f'{epochs.ch_names[c1]} $\\rightarrow$ {epochs.ch_names[c2]}')
plt.tight_layout()
save_fname = f'xmi.png'
plt.savefig(plot_dir + save_fname)


#########################
# 2D Von Mises analysis #
#########################

chan_combos = list(itertools.product(np.nonzero(picks_hipp)[0],
                                     np.nonzero(picks_parietal)[0]))
fits = []
rsq = []
for chan_1, chan_2 in tqdm(chan_combos):
    fits_i, rsq_i = comlag.cfc_vonmises_2d(x[:, chan_1, :].T,
                                           x[:, chan_2, :].T,
                                           epochs.info['sfreq'],
                                           f_mod,
                                           f_car)
    fits.append(fits_i)
    rsq.append(rsq_i)

# Save the results
write_hdf5(save_dir + 'vonmises_2d.h5',
           {'fits': fits,
            'rsq': rsq,
            'f_mod': f_mod,
            'f_car': f_car,
            'f_mod_centers': f_mod_centers},
           overwrite=True)

# Read the results and plot it 
res = read_hdf5(save_dir + 'vonmises_2d.h5')

def plot_contour(x, **kwargs):
    plt.contourf(res['f_mod_centers'], res['f_car'], x.T, **kwargs)
    plt.colorbar(format='%.2f', ticks=[x.min(), x.max()])
    plt.ylabel('Amp freq (Hz)')
    plt.xlabel('Phase freq (Hz)')


# # Get the average fits across all channel pairs
# avg_fit = {key: np.mean([e[key] for e in res['fits']], axis=0)
#             for key in res['fits'][0].keys()}
# f = avg_fit

for i_combo in range(len(chan_combos)):
    f = res['fits'][i_combo]

    plt.close()
    plt.clf()

    plt.subplot(2, 2, 1)
    plot_contour(f['a'][:, :, 0], levels=50)
    plt.title('$\\kappa$: phase in Hipp')

    plt.subplot(2, 2, 2)
    plot_contour(f['b'][:, :, 0], levels=50)
    plt.title('$\\kappa$: phase in Cortex')

    plt.subplot(2, 2, 3)
    plot_contour(f['2d'][:, :, 0], levels=50)
    plt.title('$\\kappa$: combined phase')

    plt.subplot(2, 2, 4)
    plot_contour(f['2d_cont'][:, :, 0], levels=50)
    plt.title('$\\kappa$: combined (controlled)')

    plt.tight_layout()

    input('Press ENTER to go on')


###################
###################
####           ####
#### SKETCHPAD ####
####           ####
###################
###################



from scipy import signal


x = epochs.get_data()
x = x[0,0,:]
x_hg = epochs_hg.get_data()
x_hg = x_hg[0,0,:]
f, Pxx_raw = signal.welch(x, epochs.info['sfreq'], nperseg=2 ** 10)
f, Pxx_hg = signal.welch(x_hg, epochs.info['sfreq'], nperseg=2 ** 10)

plt.clf()
plt.subplot(2, 1, 1)
plt.plot(x)
plt.plot(x_hg)

plt.subplot(2, 1, 2)
plt.plot(f, Pxx_raw)
plt.plot(f, Pxx_hg)
plt.xlim(0, 150)
plt.ylim(0, 5e-11)


import multiprocessing as mp
n_cpus = mp.cpu_count() - 1
pool = mp.Pool(n_cpus)

a = [1, 2, 3, 4, 5]
b = [1, 10, 100, 1000, 10000]
def test_func(x):
    return x[0] * x[1]
r = pool.map(test_func, zip(a, b))
pool.close()


from joblib import Parallel, delayed, parallel_backend
with parallel_backend('threading', n_jobs=3):
    Parallel()(delayed(lambda x: x ** (1/2))(i ** 2) for i in range(10))
