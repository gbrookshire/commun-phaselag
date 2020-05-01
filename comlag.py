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
from scipy.signal import butter, filtfilt, hilbert


def cfc_two_signals(phase_sig, amp_sig, fs, f_carrier, nfft, n_cycles):
    """
    Cross-frequency coupling between two signals.

    Parameters
    ----------
    phase_sig : Vector of data to calculate LF phase
    amp_sig : Vector of data to calculate HF amplitude
    fs : Sampling rate of the signal
    f_carrier : Vector of carrier frequencies to look for modulation
    nfft : size of the FFT window (FIXME -- window for what?)
    n_cycles : How many cycles to include in the wavelet analysis (vec or int)

    Returns
    -------
    cfc_data : Coherence values
    mod_freq : The frequencies of modulation for coherence
    """

    # Compute the power time-course
    pwr = _wavelet_tfr(amp_sig, f_carrier, n_cycles, fs)

    # Split the data into segments of length nfft
    x_split = _buffer(phase_sig, nfft, int(nfft / 2))
    pwr_split = _buffer(pwr, nfft, int(nfft / 2))
    
    # Apply hanning taper to each segment
    taper = np.reshape(np.hanning(nfft), [-1, 1])
    x_taper = taper * x_split
    pwr_taper = np.reshape(np.hanning(nfft), [-1, 1, 1]) * pwr_split

    # FFT of each segment
    x_fft = np.fft.fft(x_taper, nfft, axis=0)
    pwr_fft = np.fft.fft(pwr_taper, nfft, axis=0)
    
    # Cross spectra
    x_fft = np.reshape(x_fft, [1024, 1, -1]) # Reshape to combine w/ power
    xspec = x_fft * np.conj(pwr_fft)

    # Cross-frequency coupling
    num = np.abs(np.nansum(xspec, axis=-1)) # Combine over segments
    denom_a = np.nansum(np.abs(x_fft) ** 2, axis=-1)
    denom_b = np.nansum(np.abs(pwr_fft) ** 2, axis=-1)
    denom  = np.sqrt(denom_a * denom_b)
    cfc_data = num / denom

    # Only keep the meaningful frequencies
    n_keep_freqs = int(np.floor(nfft / 2))
    cfc_data = cfc_data[:n_keep_freqs, :]

    # Compute the modulation frequencies
    f_mod = np.arange(nfft - 1) * fs / nfft
    f_mod = f_mod[:n_keep_freqs]
     
    return cfc_data, f_mod


def cfc_phasediff(a, b, fs, f_carrier, nfft, n_cycles):
    """ Compute cross-frequency coupling based on the low-frequency phase
    difference between signals a and b. Return the CFC between low-freq phase
    diff and HF activity in each signal.

    ****** This doesn't seem to work!

    Parameters
    ----------
    a : Vector of data
    b : Vector of data
    f_carrier : Vector of carrier frequencies to look for modulation
    nfft : size of the FFT window (FIXME -- window for what?)
    n_cycles : How many cycles to include in the wavelet analysis (vec or int)
    fs : Sampling rate of the signal

    Returns
    -------
    cfc_data : Coherence values #FIXME Update for two signals
    mod_freq : The frequencies of modulation for coherence
    """

    def buffered_fft(x):
        """ Get the FFT of a signal split into segments and then tapered
        """
        # Compute power time-course
        x_pwr = _wavelet_tfr(x, f_carrier, n_cycles, fs)
        # Split the data into segments of length nfft
        buf = lambda x: _buffer(x, nfft, int(nfft / 2))
        x_split = buf(x)
        x_pwr_split = buf(x_pwr)
        # Apply hanning taper to each segment
        taper = np.reshape(np.hanning(nfft), [-1, 1])
        x_taper = taper * x_split
        x_pwr_taper = np.reshape(np.hanning(nfft), [-1, 1, 1]) * x_pwr_split
        # FFT of each segment
        x_fft = np.fft.fft(x_taper, nfft, axis=0)
        x_fft = np.reshape(x_fft, [1024, 1, -1]) # Reshape to combine w/ power
        x_pwr_fft = np.fft.fft(x_pwr_taper, nfft, axis=0)
        return x_fft, x_pwr_fft

    # Get the FFTs of each segment
    a_fft, a_pwr_fft = buffered_fft(a)
    b_fft, b_pwr_fft = buffered_fft(b)

    # Cross-spectrum of a and b
    xspec_ab = a_fft * np.conj(b_fft)
    cospectrum = np.real(xspec_ab)
    quadspectrum = np.imag(xspec_ab)
    amp_spec = np.sqrt(cospectrum ** 2 + quadspectrum ** 2) # like coherence
    phase_spec = np.arctan(quadspectrum / cospectrum) # Phase diff per freq
    # If LF activity is independent in A and B, the phase spect will be pretty random

    # Only keep the meaningful frequencies
    n_keep_freqs = int(np.floor(nfft / 2))

    # Cross-frequency coupling
    def cfc_helper(spec1, spec2):
        xspec = spec1 * np.conj(spec2)
        num = np.abs(np.nansum(xspec, axis=-1)) # Combine over segments
        denom_a = np.nansum(np.abs(spec1) ** 2, axis=-1)
        denom_b = np.nansum(np.abs(spec2) ** 2, axis=-1)
        denom  = np.sqrt(denom_a * denom_b)
        cfc_data = num / denom
        cfc_data = cfc_data[:n_keep_freqs, :]
        return cfc_data

    cfc_phase_a = cfc_helper(phase_spec, a_pwr_fft)
    cfc_phase_b = cfc_helper(phase_spec, b_pwr_fft)
    cfc_diff = cfc_phase_b - cfc_phase_a
    cfc_ratio = cfc_phase_b / cfc_phase_a

    cfc_ab_a = cfc_helper(xspec_ab, a_pwr_fft)
    cfc_ab_b = cfc_helper(xspec_ab, b_pwr_fft)
    cfc_diff = cfc_ab_b - cfc_ab_a
    cfc_ratio = cfc_ab_b / cfc_ab_a


    # Compute the modulation frequencies
    f_mod = np.arange(nfft - 1) * fs / nfft
    f_mod = f_mod[:n_keep_freqs]
     
    return cfc_data, f_mod

    
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
    x : Vector of data
    freqs : sequence of numbers
        Frequency of the wavelet (in Hz)
    n_cycles : int
        Number of cycles to include in the wavelet
    fs : int|float
        Sampling rate of the signal

    Returns
    -------
    pwr : np.ndarray
        Power time-courses
    """

    # Set up the wavelets
    n_cycles = n_cycles * np.ones(len(freqs))
    wavelets = [_wavelet(f, n, fs) for f,n in zip(freqs, n_cycles)]

    # Compute the power timecourses
    pwr = [abs(np.convolve(x, w, 'same')) ** 2 for w in wavelets]
    pwr = np.array(pwr).T

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
    s_a : ndarray
        Signal array with the modualting signal. If 2D, first dim must be time,
        and 2nd dim is channel.
    s_b : ndarray
        Signal array with the amplitude variations. If 2D, first dim must be
        time, and 2nd dim is channel.
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

    mi = np.full([f_car.shape[0], f_mod.shape[0]], np.nan)
    for i_fm,fm in enumerate(f_mod):
        # Compute LF phase
        s_a_filt = bp_filter(s_a.T, fm[0], fm[1], fs, 2).T
        s_a_phase = np.angle(hilbert(s_a_filt, axis=0))
        s_a_phase = np.digitize(s_a_phase, phase_bins) - 1 # Binned
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
    Compute CFC based on the phase-difference between two signals.
    Compute CFC as in Tort et al (2010, J Neurophysiol).

    Parameters
    ----------
    s_a : ndarray
        Signal array. If 2D, first dim must be time, and 2nd dim is channel.
    s_b : ndarray
        Signal array with the amplitude variations. If 2D, first dim must be
        time, and 2nd dim is channel.
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
    mi = {}
    mi['a'] = np.full([f_car.shape[0], f_mod.shape[0]], np.nan)
    mi['b'] = mi['a'].copy()
    for i_fm,fm in enumerate(f_mod):
        # Compute LF phase difference
        s_a_filt = bp_filter(s_a.T, fm[0], fm[1], fs, 2).T
        s_a_phase = np.angle(hilbert(s_a_filt, axis=0))
        s_b_filt = bp_filter(s_b.T, fm[0], fm[1], fs, 2).T
        s_b_phase = np.angle(hilbert(s_b_filt, axis=0))
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
