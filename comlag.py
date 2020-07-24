"""
Look for communication between two brain areas based on phase lag

# Algorithm

Take two areas A and B
BP-filter the two areas in a low-frequency band (alpha)
Compute the LF phase difference between A and B at each timepoint
Make overlapping bins of LF phase difference
For each LF phase bin
    Compute gamma-band phase-lag index (PLI) between the areas

Use plain old PAC between areas but with LF phase difference instead of LF phase
    Compute gamma power in the sender and receiver

2d plot
- MI b/w brain areas in the gamma band

"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, fftconvolve


def cfc_xspect(s_a, s_b, fs, nfft, n_overlap, f_car, n_cycles=5):
    """
    Cross-frequency coupling between two signals.
    This uses the Fourier-based cross-spectrum method (Jiang et al).

    Parameters
    ----------
    s_a, s_b : ndarray (time,) or (time, trial)
        Signal arrays. If 2D, first dim must be time, and 2nd dim is trial.
        Phase is extracted from s_a. Amplitude is extracted from s_b.
    fs : int,float
        Sampling rate
    nfft : int
        Size of the FFT window 
    n_overlap : int
        Number of samples of overlap in the FFT windows
    f_car : list or ndarray
        Center frequencies for the power-timecourses.
    n_cycles : int
        Number of cycles for the wavelet analysis to compute high-freq power

    Returns
    -------
    cfc_data : Coherence values
    mod_freq : The frequencies of modulation for coherence
    """

    # Compute the power time-course
    amp = _wavelet_tfr(s_b, f_car, n_cycles, fs)

    # Split the data into segments of length nfft
    x_split = _buffer(s_a, nfft, int(nfft / 2))
    amp_split = _buffer(amp, nfft, int(nfft / 2))
    
    # Apply hanning taper to each segment
    taper = np.hanning(nfft)
    x_taper = _match_dims(taper, x_split) * x_split
    amp_taper = _match_dims(taper, amp_split) * amp_split

    # FFT of each segment
    x_fft = np.fft.fft(x_taper, nfft, axis=0)
    amp_fft = np.fft.fft(amp_taper, nfft, axis=0)

    # Reshape so we can take the cross-spectrum of phase diff and amp FFT
    new_shape = list(amp_fft.shape)
    for inx in range(1, len(new_shape) - 1):
        new_shape[inx] = 1
    x_fft = np.reshape(x_fft, new_shape)
 
    # Cross spectra
    xspec = x_fft * np.conj(amp_fft)

    # Cross-frequency coupling
    num = np.abs(np.nansum(xspec, axis=-1)) # Combine over segments
    denom_a = np.nansum(np.abs(x_fft) ** 2, axis=-1)
    denom_b = np.nansum(np.abs(amp_fft) ** 2, axis=-1)
    denom  = np.sqrt(denom_a * denom_b)
    cfc_data = num / denom

    # Only keep the meaningful frequencies
    n_keep_freqs = int(np.floor(nfft / 2))
    cfc_data = cfc_data[:n_keep_freqs, :]

    # Compute the modulation frequencies
    f_mod = np.arange(nfft - 1) * fs / nfft
    f_mod = f_mod[:n_keep_freqs]
     
    return cfc_data, f_mod


def cfc_phasediff_xspect(s_a, s_b, fs, nfft, n_overlap, f_car, n_cycles=5):
    """
    Compute CFC based on the low-frequency phase-difference between two signals.
    Compute CFC as in Tort et al (2010, J Neurophysiol).

    ------- This approach doesn't work --------
    In this method, we use the cross-spectrum to get the phase difference
    between two signals -- in our case, LF phase difference and phase of HF
    amplitude. The problem arises because when we take the phase difference, we
    lose the absolute phase of either signal. So when we try to look at the
    relationship between LF phase difference and phase of HF amplitude, there's
    nothing there -- because the absolute phase of both LF signals has been
    lost. We would probably be able to connect LF phase difference with the
    overall amplitude of HF power at each segment, but not with the phase of
    the LF envelope.

    Parameters
    ----------
    s_a, s_b : ndarray (time,) or (time, trial)
        Signal arrays. If 2D, first dim must be time, and 2nd dim is trial.
    fs : int,float
        Sampling rate
    nfft : int
        Size of the FFT window 
    n_overlap : int
        Number of samples of overlap in the FFT windows
    f_car : list or ndarray
        Center frequencies for the power-timecourses.
    n_cycles : int
        Number of cycles for the wavelet analysis to compute high-freq power

    Returns
    -------
    cfc_data : dict of ndarrays
        (Modulation frequecy, Carrier frequency)
        Phase difference a-b influences HF activity in signal a and signal b
    """

    x = {'a': s_a, 'b': s_b}
    amp = {}
    x_split = {}
    amp_split = {}
    x_taper = {}
    amp_taper = {}
    x_fft = {}
    amp_fft = {}
    xspec = {}
    cfc_data = {}

    for sig in 'ab':

        # Get high frequency amplitude using a wavelet transform
        amp[sig] = _wavelet_tfr(x[sig], f_car, n_cycles, fs)

        # TODO Check whether this works with multiple channels and trials
        # Split the data into segments of length nfft
        x_split[sig] = _buffer(x[sig], nfft, int(nfft / 2))
        amp_split[sig] = _buffer(amp[sig], nfft, int(nfft / 2))

        # Apply hanning taper to each segment
        taper = np.hanning(nfft)
        x_taper[sig] = _match_dims(taper, x_split[sig]) * x_split[sig]
        amp_taper[sig] = _match_dims(taper, amp_split[sig]) * amp_split[sig]

        # FFT of each segment
        x_fft[sig] = np.fft.fft(x_taper[sig], nfft, axis=0)
        amp_fft[sig] = np.fft.fft(amp_taper[sig], nfft, axis=0)

    # Use the cross-spectra to get the phase diff b/w the low-frequency signals
    x_phasediff_fft = x_fft['a'] * np.conj(x_fft['b'])
    # Normalize the phase difference to unit amplitude
    x_phasediff_fft = np.exp(1j * np.angle(x_phasediff_fft))

    # Reshape so we can take the cross-spectrum of phase diff and amp FFT
    new_shape = list(amp_fft['a'].shape)
    for inx in range(1, len(new_shape) - 1):
        new_shape[inx] = 1
    x_phasediff_fft = np.reshape(x_phasediff_fft, new_shape)

    # ##### TESTING
    # # Make amp_fft follow x_phasediff_fft as a test
    # freq_car_to_change = 20
    # freq_mod_to_change = slice(40, 60)
    # amp_fft['a'][freq_mod_to_change, freq_to_change, :, :] = \
    #                                 x_phasediff_fft[freq_mod_to_change, 0, :, :]

    for sig in 'ab':
        # Get the cross-spec b/w the phase-diff signal and the phase of the
        # amplitude signals
        xspec[sig] = x_phasediff_fft * np.conj(amp_fft[sig])

        # Cross-frequency coupling
        num = np.abs(np.nansum(xspec[sig], axis=-1)) # Combine over segments
        denom_a = np.nansum(np.abs(x_phasediff_fft) ** 2, axis=-1)
        denom_b = np.nansum(np.abs(amp_fft[sig]) ** 2, axis=-1)
        denom  = np.sqrt(denom_a * denom_b)
        cfc_data[sig] = num / denom

        # Only keep the meaningful frequencies
        n_keep_freqs = int(np.floor(nfft / 2))
        cfc_data[sig] = cfc_data[sig][:n_keep_freqs, ...] #TODO check with 3D data

    # Compute the modulation frequencies
    f_mod = np.arange(nfft - 1) * fs / nfft
    f_mod = f_mod[:n_keep_freqs]
     
    return cfc_data, f_mod


def _match_dims(arr1, arr2):
    """ Reshape arr1 so it has the same #dims as arr1
    """
    arr1 = np.reshape(arr1, [-1] + ([1] * (arr2.ndim - 1)))
    return arr1

    
def cfc_within(x, fs, f_carrier, nfft, n_cycles):
    """
    Cross-frequency coupling within one signal

    Parameters
    ----------
    x : Vector of data
    f_carrier : Vector of carrier frequencies to look for modulation
    nfft : size of the FFT window (FIXME -- window for what?)
    n_cycles : How many cycles to include in the wavelet analysis (vec or int)
    fs : Sampling rate of the signal

    Returns
    -------
    cfc_data : Coherence values
    mod_freq : The frequencies of modulation for coherence
    """
    #FIXME args are out of order
    cfc_data, f_mod = cfc_two_signals(x, x, fs, f_carrier, nfft, n_cycles) 
    return cfc_data, f_mod


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


def _buffer(x, n, p):
    '''
    Buffers ndarray x into segments of length n with overlap p. Creates a new
    dimension for segments. Excess data at the end of x is discarded.

    Parameters
    ----------
    x : ndarray
        Signal array. If 2D, first dim must be time, and 2nd dim is channel
    n : int
        Number of samples in each data segment
    p : int
        Number of values to overlap

    Returns
    -------
    result : ndarray (<input dims>, n segments)
        Buffer array created from x

    Adapted from https://stackoverflow.com/a/57491913
    '''
    start_points = range(0, x.shape[0] - n + 1, n - p) # Where each seg starts
    assert len(x.shape) <= 2, 'Data must be 1- or 2-dimensional'
    if len(x.shape) == 1:
        result_shape = [n, len(start_points)]
    elif len(x.shape) == 2:
        result_shape = [n, x.shape[1], len(start_points)]
    result = np.full(result_shape, np.nan) # initialize data matrix
    for i_seg, start_inx in enumerate(start_points):
        result[..., i_seg] = x[start_inx:(start_inx + n), ...] #fill in by column
    return result


def cfc_tort(s_a, s_b, fs, f_mod, f_car, n_bins=18, n_cycles=5):
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
    s_a : ndarray (time,) or (time, trial)
        Signal array with the modualting signal. If 2D, first dim must be time,
        and 2nd dim is trial.
    s_b : ndarray (time,) or (time, trial)
        Signal array with the amplitude variations. If 2D, first dim must be
        time, and 2nd dim is trial.
    fs : int,float
        Sampling rate
    f_mod : list of lists (n, 2)
        Array with a row of cutoff frequencies for each bandpass filter for
        computing the phase of the modulation frequencies.
    f_car : list of lists (n, 2)
        Array with a row of cutoff frequencies for each bandpass filter for
        computing the amplitude at the carrier frequencies.
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
    s_b_amp = _wavelet_tfr(s_b, f_car, n_cycles, fs)
    # Append trials over time if data includes multiple trials
    # s_b_amp shape: (time, carrier freq)
    if s_b.ndim == 2:
        s_b_amp = np.concatenate(
                    [s_b_amp[:,:,k] for k in range(s_b_amp.shape[2])],
                    axis=0)

    mi = np.full([f_car.shape[0], f_mod.shape[0]], np.nan)
    for i_fm,fm in enumerate(f_mod):
        # Compute LF phase
        s_a_filt = bp_filter(s_a.T, fm[0], fm[1], fs, 2).T
        s_a_phase = np.angle(hilbert(s_a_filt, axis=0))
        s_a_phase = np.digitize(s_a_phase, phase_bins) - 1 # Binned
        # Append trials over time if data includes multiple trials
        if s_a_phase.ndim == 2:
            s_a_phase = np.ravel(s_a_phase, 'F')
        # Compute CFC for each carrier freq using KL divergence
        for i_fc,fc in enumerate(f_car):
            # Average HF amplitude per LF phase bin
            amplitude_dist = np.ones(n_bins)  # default is 1 to avoid log(0)
            for b in np.unique(s_a_phase):
                amplitude_dist[b] = np.mean(s_b_amp[s_a_phase == b, i_fc])
            # Kullback-Leibler divergence of the amp distribution vs uniform
            amplitude_dist /= np.sum(amplitude_dist)
            d_kl = np.sum(amplitude_dist * np.log(amplitude_dist * n_bins))
            mi_mc = d_kl / np.log(n_bins)
            mi[i_fc, i_fm] = mi_mc

    return mi


def cfc_phasediff_tort(s_a, s_b, fs, f_mod, f_car, n_bins=18, n_cycles=5):
    """
    Compute CFC based on the low-frequency phase-difference between two signals.
    Compute CFC as in Tort et al (2010, J Neurophysiol).

    Parameters
    ----------
    s_a : ndarray (time,) or (time, trial)
        Signal array. If 2D, first dim must be time, and 2nd dim is trial.
    s_b : ndarray (time,) or (time, trial)
        Signal array. If 2D, first dim must be time, and 2nd dim is trial.
    fs : int,float
        Sampling rate
    f_mod : list of lists (n, 2)
        Array with a row of cutoff frequencies for each bandpass filter for
        computing the phase of the modulation frequencies.
    f_car : list of lists (n, 2)
        Array with a row of cutoff frequencies for each bandpass filter for
        computing the amplitude at the carrier frequencies.
    n_bins : int
        Number of phase bins for computing D_{KL}
    n_cycles : int
        Number of cycles for the wavelet analysis to compute high-freq power

    Returns
    -------
    mi_a : ndarray
        (Modulation frequecy, Carrier frequency)
        Phase difference a-b influences HF activity in signal a
    mi_b : ndarray
        (Modulation frequecy, Carrier frequency)
        Phase difference a-b influences HF activity in signal b
    """

    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    # Get high frequency amplitude using a wavelet transform
    amp = {}
    amp['a'] = _wavelet_tfr(s_a, f_car, n_cycles, fs)
    amp['b'] = _wavelet_tfr(s_b, f_car, n_cycles, fs)
    # Append trials over time if data includes multiple trials
    # amp.shape: (time, carrier freq)
    for sig in 'ab':
        if amp[sig].ndim == 3:
            amp[sig] = np.concatenate(
                            [amp[sig][:,:,k] for k in range(amp[sig].shape[2])],
                            axis=0)

    mi = {}
    mi['a'] = np.full([f_car.shape[0], f_mod.shape[0]], np.nan)
    mi['b'] = mi['a'].copy()
    for i_fm,fm in enumerate(f_mod):
        # Compute LF phase difference
        s_a_filt = bp_filter(s_a.T, fm[0], fm[1], fs, 2).T
        s_a_phase = np.angle(hilbert(s_a_filt, axis=0))
        s_b_filt = bp_filter(s_b.T, fm[0], fm[1], fs, 2).T
        s_b_phase = np.angle(hilbert(s_b_filt, axis=0))
        # Append trials over time if data includes multiple trials
        if s_a_phase.ndim == 2:
            s_a_phase = np.ravel(s_a_phase , 'F')
        if s_b_phase.ndim == 2:
            s_b_phase = np.ravel(s_b_phase , 'F')
        phase_diff = s_a_phase - s_b_phase
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi # wrap to +/-pi
        phase_diff = np.digitize(phase_diff, phase_bins) - 1 # Binned
        for i_fc,fc in enumerate(f_car):
            for sig in ('a', 'b'):
                # Average HF amplitude per LF phase bin
                amplitude_dist = np.ones(n_bins) # default is 1 to avoid log(0)
                for b in np.unique(phase_diff):
                    amplitude_dist[b] = np.mean(amp[sig][phase_diff == b, i_fc])
                # Kullback-Leibler divergence of the amp distribution vs uniform
                amplitude_dist /= np.sum(amplitude_dist)
                d_kl = np.sum(amplitude_dist * np.log(amplitude_dist * n_bins))
                mi_mc = d_kl / np.log(n_bins)
                mi[sig][i_fc, i_fm] = mi_mc

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
