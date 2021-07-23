import numpy as np
from scipy.signal import butter, filtfilt, hilbert, fftconvolve


def pac_tort(x, fs, f_mod, f_car, n_bins=18, n_cycles=5):
    """
    Compute CFC as in Tort et al (2010, J Neurophysiol).

    x_{raw}(t) is filtered at the two freq ranges of interest: f_p (phase)
    and f_A (amplitude).

    Get phase of x_{f_p}(t) using the Hilbert transform: Phi_{f_p}(t).

    Get amplitude of x_{f_A}(t) using the Hilbert transform: A_{f_A}(t).

    Bin the phases of Phi_{f_p}(t), and get the mean of A_{f_A}(t) for each bin.

    Normalize the mean amps by dividing each bin value by the sum over bins.

    Get phase-amplitude coupling by computing the Kullback-Leibler distance
    D_{KL} between the mean binned amplitudes and a uniform distribution.

    Modulation Index (MI) := D_{KL}(normed binned amps, uniform dist) / log(n bins)


    Parameters
    ----------
    x : ndarray (time,) or (time, trial)
        Signal array. If 2D, first dim must be time, and 2nd dim is trial.
    fs : int,float
        Sampling rate
    f_mod : ndarray (n, 2)
        Array with a row of cutoff frequencies for each bandpass filter for
        computing the phase of the LF modulation frequencies.
    f_car : ndarray (n,)
        List of center frequencies for the wavelet transform to get the amp of
        the HF carrier frequencies.
    n_bins : int
        Number of phase bins for computing D_{KL}
    n_cycles : int
        Number of cycles for the wavelet analysis to compute high-freq power

    Returns
    -------
    mi : ndarray
        (Modulation frequecy, Carrier frequency)
    """
    #TODO test this with multichannel inputs

    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    # Get high frequency amplitude using a wavelet transform
    x_hf_amp = _wavelet_tfr(x, f_car, n_cycles, fs)
    # Append trials over time if data includes multiple trials
    # x_hf_amp shape: (time, carrier freq)
    if x.ndim == 2:
        x_hf_amp = np.concatenate(
                    [x_hf_amp[:,:,k] for k in range(x_hf_amp.shape[2])],
                    axis=0)

    mi = np.full([f_car.shape[0], f_mod.shape[0]], np.nan)
    for i_fm,fm in enumerate(f_mod):
        # Compute LF phase
        x_lf_filt = bp_filter(x.T, fm[0], fm[1], fs, 2).T
        x_lf_phase = np.angle(hilbert(x_lf_filt, axis=0))
        x_lf_phase = np.digitize(x_lf_phase, phase_bins) - 1 # Binned
        # Append trials over time if data includes multiple trials
        if x_lf_phase.ndim == 2:
            x_lf_phase = np.ravel(x_lf_phase, 'F')
        # Compute CFC for each carrier freq using KL divergence
        for i_fc,fc in enumerate(f_car):
            # Average HF amplitude per LF phase bin
            amplitude_dist = np.ones(n_bins)  # default is 1 to avoid log(0)
            for b in np.unique(x_lf_phase):
                amplitude_dist[b] = np.mean(x_hf_amp[x_lf_phase == b, i_fc])
            # Kullback-Leibler divergence of the amp distribution vs uniform
            amplitude_dist /= np.sum(amplitude_dist)
            d_kl = np.sum(amplitude_dist * np.log(amplitude_dist * n_bins))
            mi_mc = d_kl / np.log(n_bins)
            mi[i_fc, i_fm] = mi_mc

    return mi


def bp_filter(data, lowcut, highcut, fs, order=2):
    def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def _wavelet_tfr(x, freqs, n_cycles, fs):
    """ Compute the time-frequency representation using wavelets.

    Parameters
    ----------
    x : ndarray (time,) or (time, channel) or (time, trial)
        Data. If multidimensional, time must be the first dimension
    freqs : sequence of numbers
        Frequency of the wavelet (in Hz)
    n_cycles : int
        Number of cycles to include in the wavelet
    fs : int|float
        Sampling rate of the signal

    Returns
    -------
    pwr : np.ndarray (time, freq, channel|trial)
        Power time-courses
    """

    # Set up the wavelets
    n_cycles = n_cycles * np.ones(len(freqs))
    wavelets = [_wavelet(f, n, fs) for f,n in zip(freqs, n_cycles)]

    # Compute power timecourses of data with multiple channels
    if x.ndim == 1:
        x = np.reshape(x, [-1, 1])
    pwr = []
    for w in wavelets:
        w = np.reshape(w, [-1, 1])
        cnv = fftconvolve(x, w, mode='same', axes=0)
        p = abs(cnv) ** 2
        pwr.append(p)
    pwr = np.array(pwr)
    pwr = np.swapaxes(pwr, 0, 1)

    return pwr


def _wavelet(freq, n_cycles, fs):
    """ Make a complex wavelet for convolution with a signal.

    Parameters
    ----------
    freq : int|float
        Frequency of the wavelet (in Hz)
    n_cycles : int
        Number of cycles to include in the wavelet
    fs : int|float
        Sampling rate of the signal

    Returns
    -------
    w : np.ndarray
        Complex-valued wavelet
    """
    n = int(np.floor(n_cycles * fs / freq))
    taper = np.hanning(n)
    osc = np.exp(1j * 2 * np.pi * freq * np.arange(n) / fs)
    w = taper * osc
    return w


def test():
    import matplotlib.pyplot as plt
    plt.ion()

    # Which frequencies to calculate phase for
    f_mod_centers = np.logspace(np.log10(4), np.log10(20), 15)
    f_mod_width = f_mod_centers / 8
    f_mod = np.tile(f_mod_width, [2, 1]).T \
                * np.tile([-1, 1], [len(f_mod_centers), 1]) \
                + np.tile(f_mod_centers, [2, 1]).T

    # Which frequencies to calculate power for
    f_car = np.arange(20, 150, 10)

    # Simulate a signal
    fs = 1000
    n = 1e4
    t = np.arange(n) / fs
    freq_lf = 10
    freq_hf = 60
    x_lf = 1 + np.sin(2 * np.pi * freq_lf * t)
    x_hf = 1 + np.sin(2 * np.pi * freq_hf * t)
    x = x_lf + ((x_lf ** 2) * x_hf)
    x = x + np.random.normal(scale=0.5, size=x.shape)
    
    pac_params = dict(fs=fs, f_mod=f_mod, f_car=f_car,
                      n_cycles=4, n_bins=10)

    mi = pac_tort(x, **pac_params)

    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.xlim(0, 1)

    plt.subplot(2, 1, 2)
    plt.contourf(f_mod_centers, f_car, mi)
    cb = plt.colorbar(format='%.2f')
    cb.ax.set_ylabel('MI')
    plt.ylabel('Amp freq (Hz)')
    plt.xlabel('Phase freq (Hz)')


if __name__ == '__main__':
    test()
