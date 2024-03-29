"""
Look for communication between two brain areas based on phase lag
"""

import itertools
import copy
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, fftconvolve
from scipy import stats, optimize
from skimage import measure
import statsmodels.api as sm
import gcmi


def wrap_to_pi(x):
    """ Wrap the input phase to the range [-pi, pi]
    """
    x = (x + np.pi) % (2 * np.pi) - np.pi
    return x


def coherence(s_a, s_b, fs, nfft, n_overlap=None):
    """
    Compute coherence between the two signals
    """
    if n_overlap is None:
        n_overlap = nfft // 2
    s = [s_a, s_b]

    # Split the data into segments of length nfft
    s_split = [_buffer(sig, nfft, int(nfft / 2)) for sig in s]

    # Apply hanning taper to each segment
    taper = np.hanning(nfft)
    s_taper = [_match_dims(taper, sig) * sig for sig in s_split]

    # FFT of each segment
    s_fft = [np.fft.fft(sig, nfft, axis=0) for sig in s_taper]

    # Coherence
    xspec = s_fft[0] * np.conj(s_fft[1])  # Cross-spectrum
    num = np.abs(np.nansum(xspec, axis=-1))  # Combine over segments
    denom_a = np.nansum(np.abs(s_fft[0]) ** 2, axis=-1)
    denom_b = np.nansum(np.abs(s_fft[1]) ** 2, axis=-1)
    denom = np.sqrt(denom_a * denom_b)
    coh_data = num / denom

    # Only keep the meaningful frequencies
    n_keep_freqs = int(np.floor(nfft / 2))
    coh_data = coh_data[:n_keep_freqs, ...]

    # Compute the modulation frequencies
    freq = np.arange(nfft - 1) * fs / nfft
    freq = freq[:n_keep_freqs]

    return coh_data, freq


def cfc_xspect(s_a, s_b, fs, nfft, n_overlap, f_car, n_cycles=5):
    """
    Cross-frequency coupling between two signals.
    This uses the Fourier-based cross-spectrum method (Jiang et al).

    Parameters
    ----------
    s_a, s_b : ndarray (time, ) or (time, trial)
        Signal arrays. If 2D, first dim must be time, and 2nd dim is trial.
        Phase is extracted from s_a. Amplitude is extracted from s_b.
    fs : int, float
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
    num = np.abs(np.nansum(xspec, axis=-1))  # Combine over segments
    denom_a = np.nansum(np.abs(x_fft) ** 2, axis=-1)
    denom_b = np.nansum(np.abs(amp_fft) ** 2, axis=-1)
    denom = np.sqrt(denom_a * denom_b)
    cfc_data = num / denom

    # Only keep the meaningful frequencies
    n_keep_freqs = int(np.floor(nfft / 2))
    cfc_data = cfc_data[:n_keep_freqs, ...]

    # Compute the modulation frequencies
    f_mod = np.arange(nfft - 1) * fs / nfft
    f_mod = f_mod[:n_keep_freqs]

    return cfc_data, f_mod


def cfc_phasediff_xspect(s_a, s_b, fs, nfft, n_overlap, f_car, n_cycles=5):
    """
    Compute CFC based on the low-frequency phase-difference between two sigs.
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
    s_a, s_b : ndarray (time, ) or (time, trial)
        Signal arrays. If 2D, first dim must be time, and 2nd dim is trial.
    fs : int, float
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

    #  ##### TESTING
    #  # Make amp_fft follow x_phasediff_fft as a test
    # freq_car_to_change = 20
    # freq_mod_to_change = slice(40, 60)
    # amp_fft['a'][freq_mod_to_change, freq_to_change, :, :] = \
    #                                 x_phasediff_fft[freq_mod_to_change, 0,
    #                                                 :, :]

    for sig in 'ab':
        # Get the cross-spec b/w the phase-diff signal and the phase of the
        # amplitude signals
        xspec[sig] = x_phasediff_fft * np.conj(amp_fft[sig])

        # Cross-frequency coupling
        num = np.abs(np.nansum(xspec[sig], axis=-1))  # Combine over segments
        denom_a = np.nansum(np.abs(x_phasediff_fft) ** 2, axis=-1)
        denom_b = np.nansum(np.abs(amp_fft[sig]) ** 2, axis=-1)
        denom = np.sqrt(denom_a * denom_b)
        cfc_data[sig] = num / denom

        # Only keep the meaningful frequencies
        n_keep_freqs = int(np.floor(nfft / 2))
        cfc_data[sig] = cfc_data[sig][:n_keep_freqs, ...]  # TODO check with 3D

    # Compute the modulation frequencies
    f_mod = np.arange(nfft - 1) * fs / nfft
    f_mod = f_mod[:n_keep_freqs]

    return cfc_data, f_mod


def _match_dims(arr1, arr2):
    """ Reshape arr1 so it has the same  #dims as arr1
    """
    arr1 = np.reshape(arr1, [-1] + ([1] * (arr2.ndim - 1)))
    return arr1


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
    x : ndarray (time, ) or (time, channel) or (time, trial)
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
    wavelets = [_wavelet(f, n, fs) for f, n in zip(freqs, n_cycles)]

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
        Signal array. If more than 1D, first dim must be time.
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
    start_points = range(0, x.shape[0] - n + 1, n - p)  # Where each seg starts
    # assert len(x.shape) <= 2, 'Data must be 1- or 2-dimensional'
    if len(x.shape) == 1:
        result_shape = [n, len(start_points)]
    elif len(x.shape) == 2:
        result_shape = [n, x.shape[1], len(start_points)]
    elif len(x.shape) == 3:
        result_shape = [n, x.shape[1], x.shape[2], len(start_points)]
    result = np.full(result_shape, np.nan)  # initialize data matrix
    for i_seg, start_inx in enumerate(start_points):  # fill in by column
        result[..., i_seg] = x[start_inx:(start_inx + n), ...]
    return result


def cfc_tort(s_a, s_b, fs, f_mod, f_car, n_bins=18, n_cycles=5):
    """
    Compute CFC as in Tort et al (2010, J Neurophysiol).

    x_{raw}(t) is filtered at the two freq ranges of interest: f_p (phase)
    and f_A (amplitude).

    Get phase of x_{f_p}(t) using the Hilbert transform: Phi_{f_p}(t).

    Get amplitude of x_{f_A}(t) using the Hilbert transform: A_{f_A}(t).

    Bin the phases of Phi_{f_p}(t), and get the mean of A_{f_A}(t) for each bin

    Normalize the mean amps by dividing each bin value by the sum over bins.

    Get phase-amplitude coupling by computing the Kullback-Leibler distance
    D_{KL} between the mean binned amplitudes and a uniform distribution.

    Modulation Index
        MI := D_{KL}(normed binned amps, uniform dist) / log(n bins)


    Parameters
    ----------
    s_a : ndarray (time, ) or (time, trial)
        Signal array with the modualting signal. If 2D, first dim must be time,
        and 2nd dim is trial.
    s_b : ndarray (time, ) or (time, trial)
        Signal array with the amplitude variations. If 2D, first dim must be
        time, and 2nd dim is trial.
    fs : int, float
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
    # TODO test this with multichannel inputs

    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    # Get high frequency amplitude using a wavelet transform
    s_b_amp = _wavelet_tfr(s_b, f_car, n_cycles, fs)
    # Append trials over time if data includes multiple trials
    # s_b_amp shape: (time, carrier freq)
    if s_b.ndim == 2:
        s_b_amp = np.concatenate(
                    [s_b_amp[:, :, k] for k in range(s_b_amp.shape[2])],
                    axis=0)

    mi = np.full([f_car.shape[0], f_mod.shape[0]], np.nan)
    for i_fm, fm in enumerate(f_mod):
        # Compute LF phase
        s_a_filt = bp_filter(s_a.T, fm[0], fm[1], fs, 2).T
        s_a_phase = np.angle(hilbert(s_a_filt, axis=0))
        s_a_phase = np.digitize(s_a_phase, phase_bins) - 1  # Binned
        # Append trials over time if data includes multiple trials
        if s_a_phase.ndim == 2:
            s_a_phase = np.ravel(s_a_phase, 'F')
        # Compute CFC for each carrier freq using KL divergence
        for i_fc, fc in enumerate(f_car):
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
    Compute CFC based on the low-frequency phase-difference between two signals
    Compute CFC as in Tort et al (2010, J Neurophysiol).

    Parameters
    ----------
    s_a, s_b : ndarray (time, ) or (time, trial)
        Signal arrays. If 2D, first dim must be time, and 2nd dim is trial.
    fs : int, float
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
    mi : dict of ndarrays
        (Modulation frequecy, Carrier frequency)
        Phase difference a-b influences HF activity in signal a and signal b
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
                            [amp[sig][:, :, k]
                             for k in range(amp[sig].shape[2])],
                            axis=0)

    mi = {}
    mi['a'] = np.full([f_car.shape[0], f_mod.shape[0]], np.nan)
    mi['b'] = mi['a'].copy()
    for i_fm, fm in enumerate(f_mod):
        # Compute LF phase difference
        s_a_filt = bp_filter(s_a.T, fm[0], fm[1], fs, 2).T
        s_a_phase = np.angle(hilbert(s_a_filt, axis=0))
        s_b_filt = bp_filter(s_b.T, fm[0], fm[1], fs, 2).T
        s_b_phase = np.angle(hilbert(s_b_filt, axis=0))
        # Append trials over time if data includes multiple trials
        if s_a_phase.ndim == 2:
            s_a_phase = np.ravel(s_a_phase, 'F')
        if s_b_phase.ndim == 2:
            s_b_phase = np.ravel(s_b_phase, 'F')
        phase_diff = s_a_phase - s_b_phase
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi  # wrap to pi
        phase_diff = np.digitize(phase_diff, phase_bins) - 1  # Binned
        for i_fc, fc in enumerate(f_car):
            for sig in ('a', 'b'):
                # Average HF amplitude per LF phase bin
                amplitude_dist = np.ones(n_bins)  # default 1 to avoid log(0)
                for b in np.unique(phase_diff):
                    amplitude_dist[b] = np.mean(amp[sig][phase_diff == b,
                                                         i_fc])
                # Kullback-Leibler divergence of amp distribution vs uniform
                amplitude_dist /= np.sum(amplitude_dist)
                d_kl = np.sum(amplitude_dist * np.log(amplitude_dist * n_bins))
                mi_mc = d_kl / np.log(n_bins)
                mi[sig][i_fc, i_fm] = mi_mc

    return mi


def cfc_sine(s_a, s_b, fs, f_mod, f_car, n_cycles=5):
    """
    Compute CFC similar to Tort et al (2010, J Neurophysiol), but fitting the
    gamma amplitudes at each phase value to a sine-wave instead of comparing
    against a uniform distribution using KL distance.

    Parameters
    ----------
    s_a : ndarray (time, ) or (time, trial)
        Signal array with the modualting signal. If 2D, first dim must be time,
        and 2nd dim is trial.
    s_b : ndarray (time, ) or (time, trial)
        Signal array with the amplitude variations. Structure like s_a.
    fs : int, float
        Sampling rate
    f_mod : list of lists (n, 2)
        Array with a row of cutoff frequencies for each bandpass filter for
        computing the phase of the modulation frequencies.
    f_car : list of lists (n, 2)
        Array with a row of cutoff frequencies for each bandpass filter for
        computing the amplitude at the carrier frequencies.
    n_cycles : int
        Number of cycles for the wavelet analysis to compute high-freq power

    Returns
    -------
    sin_amp : ndarray
        Amplitude of sine-wave fits. (Modulation frequecy, Carrier frequency)
    sin_r : ndarray
        Pearson correlation of sine-wave fits.
    """
    # TODO test this with multichannel inputs

    # Get high frequency amplitude using a wavelet transform
    s_b_amp = _wavelet_tfr(s_b, f_car, n_cycles, fs)
    # Append trials over time if data includes multiple trials
    # s_b_amp shape: (time, carrier freq)
    if s_b.ndim == 2:
        s_b_amp = np.concatenate(
                    [s_b_amp[:, :, k] for k in range(s_b_amp.shape[2])],
                    axis=0)

    sin_amp = np.full([f_car.shape[0], f_mod.shape[0]], np.nan)
    sin_r = sin_amp.copy()
    for i_fm, fm in enumerate(f_mod):
        # Compute LF phase
        s_a_filt = bp_filter(s_a.T, fm[0], fm[1], fs, 2).T
        s_a_phase = np.angle(hilbert(s_a_filt, axis=0))
        # Append trials over time if data includes multiple trials
        if s_a_phase.ndim == 2:
            s_a_phase = np.ravel(s_a_phase, 'F')
        # Compute CFC for each carrier freq by checking for sinusoidal
        # modulation of HF power along with LF phase
        for i_fc, fc in enumerate(f_car):
            p = sine_ols(s_a_phase, np.squeeze(s_b_amp[:, i_fc]))
            # Save the amplitude of the sine fit
            sin_amp[i_fc, i_fm] = p[0]
            # Save the correlation of the sine fit to the real data
            rval, _ = stats.pearsonr(np.squeeze(s_b_amp[:, i_fc]),
                                     sine_helper(s_a_phase, *p))
            sin_r[i_fc, i_fm] = rval

    return sin_amp, sin_r


def cfc_phasediff_sine(s_a, s_b, fs, f_mod, f_car, n_cycles=5):
    """
    Compute CFC based on the low-frequency phase-difference between two signals

    Compute CFC similar to Tort et al (2010, J Neurophysiol), but fitting the
    gamma amplitudes at each phase value to a sine-wave instead of comparing
    against a uniform distribution using KL distance.

    Parameters
    ----------
    s_a, s_b : ndarray (time, ) or (time, trial)
        Signal arrays. If 2D, first dim must be time, and 2nd dim is trial.
    fs : int, float
        Sampling rate
    f_mod : list of lists (n, 2)
        Array with a row of cutoff frequencies for each bandpass filter for
        computing the phase of the modulation frequencies.
    f_car : list of lists (n, 2)
        Array with a row of cutoff frequencies for each bandpass filter for
        computing the amplitude at the carrier frequencies.
    n_cycles : int
        Number of cycles for the wavelet analysis to compute high-freq power

    Returns
    -------
    mi : dict of dict of ndarrays
        (Modulation frequecy, Carrier frequency)
        Phase difference a-b influences HF activity in signal a and signal b.
        Dict mi contains two dicts 'amp' and 'r' that contain arrays of the
        amplitude and goodness-of-fit of the sine-wave fits.

    """
    # TODO test this with multichannel inputs

    # Get high frequency amplitude using a wavelet transform
    # Get high frequency amplitude using a wavelet transform
    amp = {}
    amp['a'] = _wavelet_tfr(s_a, f_car, n_cycles, fs)
    amp['b'] = _wavelet_tfr(s_b, f_car, n_cycles, fs)
    # Append trials over time if data includes multiple trials
    # amp.shape: (time, carrier freq)
    for sig in 'ab':
        if amp[sig].ndim == 3:
            amp[sig] = np.concatenate(
                            [amp[sig][:, :, k]
                             for k in range(amp[sig].shape[2])],
                            axis=0)

    # Initialize data structures to hold CFC values
    init_mat = np.full([f_car.shape[0], f_mod.shape[0]], np.nan)
    mi = {}
    for sig in ('a', 'b'):
        mi[sig] = {}
        for out in ('amp', 'r'):
            mi[sig][out] = init_mat.copy()

    # Compute the CFC
    for i_fm, fm in enumerate(f_mod):
        # Compute LF phase difference
        s_a_filt = bp_filter(s_a.T, fm[0], fm[1], fs, 2).T
        s_a_phase = np.angle(hilbert(s_a_filt, axis=0))
        s_b_filt = bp_filter(s_b.T, fm[0], fm[1], fs, 2).T
        s_b_phase = np.angle(hilbert(s_b_filt, axis=0))
        # Append trials over time if data includes multiple trials
        if s_a_phase.ndim == 2:
            s_a_phase = np.ravel(s_a_phase, 'F')
        if s_b_phase.ndim == 2:
            s_b_phase = np.ravel(s_b_phase, 'F')
        phase_diff = s_a_phase - s_b_phase
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi  # wrap to pi
        # Compute CFC for each carrier freq by checking for sinusoidal
        # modulation of HF power along with LF phase
        for i_fc, fc in enumerate(f_car):
            for sig in ('a', 'b'):
                amp_sig = np.squeeze(amp[sig][:, i_fc])
                p = sine_ols(phase_diff, amp_sig)
                # Save the correlation of the sine fit to the real data
                rval, _ = stats.pearsonr(amp_sig, sine_helper(phase_diff, *p))
                mi[sig]['amp'][i_fc, i_fm] = p[0]
                mi[sig]['r'][i_fc, i_fm] = rval

    return mi


def sine_helper(x, a, phi, o):
    """ Output a sine wave at the points in x, given a (amp), phi (phase),
    and o (offset) of the sine wave.
    """
    return o + (a * np.sin(x + phi))


def sine_ols(s_phase, s_amp):
    """ Given vectors of phase and amplitude, fit a sine wave.
    Return a list of Amplitude, Phase, and Offset.
    """
    a = np.stack([np.ones(s_phase.shape),
                  np.sin(s_phase),
                  np.cos(s_phase)])
    b = np.squeeze(s_amp)
    x, _, _, _ = np.linalg.lstsq(a.T, b, rcond=None)
    z = np.complex(*x[1:])
    # Amp, phase, offset
    p = [np.abs(z), np.angle(z), x[0]]
    return p


# plt.clf()
# plt.subplot(1, 2, 1)
# plt.contourf(np.mean(f_mod, axis=1), f_car, sin_amp)
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.contourf(np.mean(f_mod, axis=1), f_car, sin_r)
# plt.colorbar()

# plt.clf()
# plt.plot(s_a_phase, s_b_amp[:, i_fc], 'o', alpha=0.2)
# x_phase = np.linspace(-np.pi, np.pi, 100)
# plt.plot(x_phase, sine_helper(x_phase, popt[0], popt[1]))

#  # Two ways to fit a sine wave
#  # First simulate a signal
# k = 1000
# sim_amp = np.random.uniform(0.5, 2.0)
# sim_offset = np.random.uniform(0.0, 10.0)
# sim_phase = np.random.uniform(-np.pi, np.pi)
# sin_fnc = lambda a, p, x: a * np.sin(x + p)
# x = np.random.uniform(-np.pi, np.pi, k)
# y = sine_helper(x, sim_amp, sim_phase, sim_offset)
# y = y + np.random.normal(size=y.shape)
# plt.clf()
# plt.plot(x, y, 'o', alpha=0.2)
#  # 1. Using optimize.curve_fit to estimate the amp and phase of a sine
# def sine_curve_fit(s_phase, s_amp):
#     popt, _ = curve_fit(sine_helper,
#                         s_phase,
#                         np.squeeze(s_amp))
#     return popt
# p_curve_fit = sine_curve_fit(x, y)
# phase_x = np.arange(-np.pi, np.pi, 1/fs)
# plt.plot(phase_x, sine_helper(phase_x, *p_curve_fit),
#          linestyle='--')
#  # 2. Using OLS
#  # Fit a sine and a cosine, then calculate amp and phase
# def sine_ols(s_phase, s_amp):
#     a = np.stack([np.ones(s_phase.shape),
#                   np.sin(s_phase),
#                   np.cos(s_phase)])
#     b = np.squeeze(s_amp)
#     x, _, _, _ = np.linalg.lstsq(a.T, b, rcond=None)
#     z = np.complex(*x[1:])
#     p = [np.abs(z), np.angle(z), x[0]]  # Amp, phase, offset
#     return p
# p_ols = sine_ols(x, y)
# plt.plot(phase_x, sine_helper(phase_x, *p_ols),
#          linestyle=':')
#  # Compare the timing
#  # curve_fit is about 5 times slower
# %timeit p_curve_fit = sine_curve_fit(x, y)
# %timeit p_ols = sine_ols(x, y)


def mod_index(x, method):
    """
    Compute the modulation index given some dependent measure across phase bins

    Parameters
    ----------
    x : np.ndarray
        The dependent measure across different phase bins. The last dimension
        is phase bins.
    method : str
        The method used to compute the modulation index.

    Returns
    -------
    mi_comod : np.ndarray
        Modulation index, returned as a comodulogram. Same shape as the input
        array x, but without the final dimension.

    """

    n_bins = x.shape[-1]
    phase_bins = np.linspace(-np.pi, np.pi, n_bins)

    if method == 'tort':
        # Kullback-Leibler divergence of the distribution vs uniform
        # Unitless
        # First, set any negative MI values to equal the minimum positive value
        # This is necesary for the logarithms to work
        x[x < 0] = x[x > 0].min()
        # Make sure each LF/HF pair sums to 1 so KL-divergence works
        x_sums = np.sum(x, axis=-1)
        x_sums = np.swapaxes(np.swapaxes(np.tile(x_sums,  # Make dims the same
                                                 [n_bins, 1, 1]),
                                         0, 1),
                             1, 2)
        x /= x_sums
        d_kl = np.sum(x * np.log(x * n_bins), 2)
        mi_comod = d_kl / np.log(n_bins)

    elif method == 'sine psd':
        # PSD of a sine wave fit to the MI as a function of phase-difference
        # Units: bits^2 / Hz
        sine_psd = (np.abs(np.fft.fft(x)) ** 2) / n_bins
        mi_comod = sine_psd[..., 1]  # Take the freq matching the whole signal

    elif method == 'sine amp':
        # Amplitude of a sine wave fit, normalized by sequence length
        # Units: bits / Hz
        sine_amp = np.abs(np.fft.fft(x)) / n_bins
        mi_comod = sine_amp[..., 1]

    elif method == 'sine fit adj':
        """
        Adjusted PSD of a sine-fit.

        Pros
        - Bigger DC offset means smaller mod inx
        - Good behavior when osc is whole signal

        Cons
        - Noisier signals with low DC can have bigger mod inx
        """
        y = np.abs(np.fft.fft(x))  # Amp spect
        y = y[..., :(n_bins // 2)]  # Only take positive frequencies
        y[..., 0] /= 2  # non-DC coefs are doubled in DFTs
        # Proportion of the spectrum accounted for by this freq
        y = y / np.reshape(np.sum(y, axis=-1),
                           list(y.shape[:2]) + [1])
        mod_inx = y[..., 1] / np.sum(np.delete(y, 1, axis=-1), axis=-1)
        mi_comod = mod_inx

    elif method == 'rsquared':
        # Find the R^2 of a sine wave fit to the MI by phase-lag
        x_arr = np.stack([np.sin(phase_bins), np.cos(phase_bins)]).T
        mi_comod = np.full(x.shape[:2], np.nan)
        for i_fm in range(x.shape[0]):
            for i_fc in range(x.shape[1]):
                y = x[i_fm, i_fc, :]
                y = y - np.mean(y)  # Remove mean to focus on sine-wave fits
                model = sm.OLS(y, x_arr)
                results = model.fit()
                rsq = results.rsquared
                mi_comod[i_fm, i_fc] = rsq

    elif method == 'vector':
        # Look at mean vector length. Inspired by the ITPC
        theta = np.exp(1j * phase_bins)
        phase_vectors = x * theta
        mean_vector_length = np.abs(np.mean(phase_vectors, axis=-1))
        mi_comod = mean_vector_length

    elif method == 'vector-imag':
        # Imaginary part of the mean vector length.
        # Inspired by the ITPC and imaginary coherence.
        theta = np.exp(1j * phase_bins)
        phase_vectors = x * theta
        mean_vector_length = np.abs(np.imag(np.mean(phase_vectors, axis=-1)))
        mi_comod = mean_vector_length

    elif method == 'itlc':
        # Something like the inter-trial linear coherence (ITLC)
        # Delorme & Makeig (2004)
        F = x * np.exp(1j * phase_bins)
        n = n_bins
        num = np.sum(F, axis=-1)
        den = np.sqrt(n * np.sum(np.abs(F) ** 2, axis=-1))
        itlc = num / den
        itlc = np.abs(itlc)
        mi_comod = itlc

    else:
        raise(Exception(f'method {method} not recognized'))

    return mi_comod


def cfc_phaselag_transferentropy(s_a, s_b, fs,
                                 f_mod, f_mod_bw,
                                 f_car, f_car_bw,
                                 lag, n_bins,
                                 decimate=None,
                                 n_perm_phasebin=0,
                                 n_perm_phasebin_indiv=0,
                                 n_perm_signal=0,
                                 n_perm_shift=0,
                                 min_shift=None, max_shift=None,
                                 perm_phasebin_flip=False,
                                 cluster_alpha=0.05,
                                 method='sine psd', calc_type=2,
                                 diff_method='both',
                                 min_cluster_size=1,
                                 return_phase_bins=False,
                                 ignore_corner=False,
                                 verbose=True):
    """
    Compute conditional mutual information between two signals, and lagged
    copies of those two signals.

    Using the conditional mutual information calculations (calc_type = 2), when
    A-->B, I(A;B|LA) << I(A;B|LB). This happens because conditioning on LA
    reduces the CMI when LA predicts both A and B independently (i.e. past A
    predicts future A and B); but conditioning on LB does not reduce the CMI
    when LB does not predict A or B.

    Using the transfer entropy calculation (calc_type = 2), A-->B leads the
    first output [I(LA;B|LB)] to be greater than the second [I(A;LB|LA)]. This
    leads to a positive blob in the difference comodulogram.


    - Test: Preserve LF phase diff, randomize HF trials (to test communication)
        - Split the data into trials or epochs
        - Compute the comodulogram of the empirical phase-lagged TE
        - For k permutations
            - Randomly shuffle the HF filtered data between epochs
            - Compute the comodulograms for each permutation
            - Compute a cluster stat
                - The summed z-value across comodulograms and permutations
                - Check for normal distribution?
                - For pos/neg, this is a two-tailed test (so z thresh for .025)


    Parameters
    ----------

    s_a, s_b : np.ndarray (time, ) or (time, trial)
        The two signals
    fs : scalar (int, float)
        The sampling rate of the signals
    f_mod : list, nd.array
        The center frequencies of the low-frequency bandpass filters (in Hz)
    f_mod_bw : scalar (int, float) or sequence (list or np.ndarray)
        The bandwidth of the low-frequency bandpass filters (in Hz)
    f_car : list, nd.array
        The center frequencies of the high-frequency bandpass filters (in Hz)
    f_car_bw : scalar (int, float) or sequence (list or np.ndarray)
        The bandwidth of the low-frequency bandpass filters (in Hz)
    lag : sequence of ints
        The lags (in samples) to test between the two variables. Should be
        positive integers.
    n_bins : int
        The number of phase-difference bins
    decimate : None, int
        The factor by which to decimate the data. For example, a value of 3
        only considers every 3rd sample in the MI calculations.
    n_perm_phasebin : int, 'full'
        The number of random permutations for the cluster test. Randomly
        shuffle the transfer entropy between each phase bin. If the string
        "full" is given, compute every permutation of the data, for k =
        fact(phase_bins) permutations.
    n_perm_phasebin_indiv : int, 'full'
        Like n_perm_phasebin, but separately shuffles the phase-bins of each HF
        signal before computing the diff and phase-dependence. DO NOT USE!
        Results in false positives every time for structured data.
    n_perm_signal : int
        The number of random permutations for the cluster test. Randomly
        shuffle which signal is considered to be signal A or B, and then
        recompute the difference between them to get the directionality of
        communication.
    n_perm_shift : int
        The number of random permutations for the cluster test. Randomly shift
        one of the high-frequency traces to destroy communication between the
        signals.
    min_shift, max_shift : int
        The minimum and maximum number of samples to shift by when performing
        random permutation tests
    perm_phasebin_flip : bool
        Run a permutation test by switching the TE in each phase bin between
        the two directions (A-->B and B--A) before calculating the phase
        dependence. Because we're dealing with a fairly small number of phase
        bins, exhaustively test every permutation.
    cluster_alpha : float
        The alpha threshold for including values in the clusters.
    method : str
        The method to use to test for phase-dependence of communication. Must
        be a value accepted by mod_index().
    calc_type : int
        How to calculate the directionality.
        1: Mutual information conditioned on a lagged copy of each signal
            I(A;B|LA) and I(A;B|LB)
        2: Transfer entropy
            I(LA;B|LB) and I(A;LB|LA)
    min_cluster_size : int
        The minimum number of samples ("pixels") that a cluster can have to be
        included in the analysis.
    return_phase_bins : bool
        If true, return the TE values for each individual phase bin.
    diff_method : str
        How to calculate the difference.
        'PD(AB)-PD(BA)': PhaseDep(TE(A-->B)) - PhaseDep(TE(B-->A))
        'PD(AB-BA)': PhaseDep( TE(A-->B) - TE(B-->A) )
        'both': Calculate the difference using both of the methods above
    ignore_corner : bool
        If True, restrict the values that are included in the cluster
        permutation test. Only includes samples for which HF > 2 * LF.
    """

    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    assert s_a.shape == s_b.shape, 'Data s_a and s_b must be the same shape'
    s = {'a': s_a, 'b': s_b}

    assert calc_type in (1, 2)

    assert isinstance(f_car_bw, (float, int, np.ndarray)), \
        "f_car_bw must be a scalar or a numpy array"
    if isinstance(f_car_bw, (float, int)):
        f_car_bw = np.ones(f_car.shape) * f_car_bw

    assert isinstance(f_mod_bw, (float, int, np.ndarray)), \
        "f_mod_bw must be a scalar or a numpy array"
    if isinstance(f_mod_bw, (float, int)):
        f_mod_bw = np.ones(f_mod.shape) * f_mod_bw

    if decimate is None:
        def decim(x):
            return x
    else:
        assert decimate > 0, 'decimate must be positive'
        assert isinstance(decimate, int), \
            f'decimate must be an int, but got a {type(decimate)}'

        def decim(x):
            return x[..., ::decimate]  # Decimate the last axis

    if isinstance(n_perm_phasebin, str):
        assert n_perm_phasebin == 'full', \
                'n_perm_phasebin must be an integer or the str "full"'
        raise(NotImplementedError)
    elif isinstance(n_perm_phasebin, int):
        assert n_perm_phasebin >= 0, 'n_perm_phasebin must be positive'
        assert n_perm_phasebin < np.math.factorial(n_bins), \
            'n_perm_phasebin must be less than factorial(n_bins)'

    if n_perm_shift > 0:
        if min_shift is None:
            min_shift = int(len(s['a']) / 2)
        if max_shift is None:
            # Avoid shifting by so much that you'd run over the edge of the sig
            max_shift = len(s['a']) - np.max(lag)

        # Cluster analysis currently ignores multiple lags
        if len(lag) > 1:
            raise NotImplementedError
        else:
            i_lag = 0

    assert len(np.nonzero([n_perm_shift,
                           n_perm_signal,
                           n_perm_phasebin,
                           n_perm_phasebin_indiv,
                           perm_phasebin_flip])[0]) <= 1, \
        'Only use one type of permutation analysis at a time'
    if n_perm_shift:
        n_perm = n_perm_shift
    elif n_perm_phasebin:
        if n_perm_phasebin == 'full':
            n_perm = np.math.factorial(n_bins)
        else:
            n_perm = n_perm_phasebin
    elif n_perm_phasebin_indiv:
        n_perm = n_perm_phasebin_indiv
    elif n_perm_signal:
        assert s_a.ndim == 2, \
                'Permuting signals only works for signals with multiple epochs'
        n_perm = n_perm_signal
    elif perm_phasebin_flip:
        n_perm = 2 ** n_bins
    else:
        n_perm = 0

    assert diff_method in ('PD(AB)-PD(BA)', 'PD(AB-BA)', 'both'), \
        'diff_method "{diff_method}" not recognized'

    assert isinstance(min_cluster_size, int), 'min_cluster_size must be an int'
    assert min_cluster_size > 0, 'min_cluster_size must be a positive integer'

    # Initialize mutual information array
    # Dims: Permutation, LF freq, HF freq, CMI lag, direction, LF phase bin
    mi = np.full([n_perm + 1, len(f_mod), len(f_car), len(lag), 2, n_bins],
                 np.nan)
    # Initialize array to hold the number of observations in each bin
    counts = np.full([len(f_mod), n_bins], np.nan)

    for i_fm in range(len(f_mod)):
        # Compute the LF phase-difference of each signal
        fm = f_mod[i_fm]
        fm_bw = f_mod_bw[i_fm]
        if verbose:
            print(fm)
        filt = {sig: bp_filter(s[sig].T,
                               fm - (fm_bw / 2),
                               fm + (fm_bw / 2),
                               fs,
                               2).T
                for sig in 'ab'}
        phase = {sig: np.angle(hilbert(filt[sig], axis=0))
                 for sig in 'ab'}
        phase_diff = phase['a'] - phase['b']
        phase_diff = wrap_to_pi(phase_diff)
        phase_diff = np.digitize(phase_diff, phase_bins) - 1  # Binned
        # Append trials over time if data includes multiple epochs. Due to the
        # lag in the CMI, this will result in a small number of samples being
        # conditioned on data from a different epoch.
        if s_a.ndim == 2:
            phase_diff = np.ravel(phase_diff, 'F')
        # FIXME
        # Find a way to mark the epoch boundaries, and then excluding samples
        # from the CMI calculation if they are conditioned on samples from a
        # different epoch.

        for i_fc in range(len(f_car)):
            # Filter the HF signals
            fc = f_car[i_fc]
            fc_bw = f_car_bw[i_fc]
            filt = {sig: bp_filter(s[sig].T,
                                   fc - (fc_bw / 2),
                                   fc + (fc_bw / 2),
                                   fs, 2).T
                    for sig in 'ab'}
            # Make a 2D version of the signal with its Hilbert transform
            # This makes mutual information more informative
            h = {sig: hilbert(filt[sig], axis=0) for sig in 'ab'}
            sig_2d_orig = {sig: np.stack([np.real(h[sig]), np.imag(h[sig])])
                           for sig in 'ab'}

            for i_perm_sig in range(n_perm_signal + 1):

                # Randomize signals A and B within each epoch
                if i_perm_sig > 0:
                    for i_epoch in range(s_a.shape[1]):
                        if np.random.choice([True, False]):
                            tmp_a = copy.deepcopy(sig_2d_orig['a'][:, :,
                                                                   i_epoch])
                            tmp_b = copy.deepcopy(sig_2d_orig['b'][:, :,
                                                                   i_epoch])
                            sig_2d_orig['a'][:, :, i_epoch] = tmp_b
                            sig_2d_orig['b'][:, :, i_epoch] = tmp_a

                # Append epochs
                if s_a.ndim == 2:
                    sig_2d_append = {sig: np.reshape(sig_2d_orig[sig],
                                                     (2, -1),
                                                     order='F')
                                     for sig in 'ab'}
                else:
                    sig_2d_append = sig_2d_orig.copy()

                # Compute MI for each phase bin
                for phase_bin in np.unique(phase_diff):

                    phase_sel = phase_diff == phase_bin

                    def select_samps(x):
                        """
                        Helper function to select the samples with the right LF
                        phase difference and then decimate the signal.
                        """
                        return decim(x[:, phase_sel])

                    # Store the count of observations per phase bin
                    if i_fc == 0:
                        counts[i_fm, phase_bin] = np.sum(phase_sel)

                    # Randomly shift the HF time-series
                    for i_perm_shift in range(n_perm_shift + 1):
                        sig_2d = copy.deepcopy(sig_2d_append)
                        if i_perm_shift > 0:  # Don't shift the real data
                            # Shift signal A
                            sig_2d['a'] = np.roll(sig_2d['a'],
                                                  np.random.randint(min_shift,
                                                                    max_shift),
                                                  axis=1)
                        # Compute CMI in each direction
                        for i_lag in range(len(lag)):

                            def L(x):
                                """
                                Lag helper function
                                Because this rolls samples from the end to the
                                beginning, it will result in some samples being
                                counted in the MI calculation even though they
                                happened far apart in the real data. This will
                                only occur for a very small number of samples
                                (number of the lag), so it's negligible as long
                                as the length of the data is much larger than
                                the lag.
                                """
                                return np.roll(x, lag[i_lag], axis=1)

                            for i_direc, direc in enumerate('ab'):
                                if calc_type == 1:
                                    # Compute I(A;B|LA) and I(A;B|LB)
                                    i = gcmi.gccmi_ccc(
                                                select_samps(sig_2d['a']),
                                                select_samps(sig_2d['b']),
                                                select_samps(L(sig_2d[direc])))
                                elif calc_type == 2:
                                    # Compute I(LA;B|LB) and I(A;LB|LA)
                                    if direc == 'a':
                                        s1, s2 = ('a', 'b')
                                    else:
                                        s1, s2 = ('b', 'a')
                                    i = gcmi.gccmi_ccc(
                                                select_samps(L(sig_2d[s1])),
                                                select_samps(sig_2d[s2]),
                                                select_samps(L(sig_2d[s2])))
                                else:
                                    raise(NotImplementedError)

                                i_perm = max(i_perm_sig, i_perm_shift)
                                mi[i_perm,
                                   i_fm,
                                   i_fc,
                                   i_lag,
                                   i_direc,
                                   phase_bin] = i

    # Compute the permutation test by shuffling TE values across phase bins
    # Make a generator object to shuffle the phase bins
    if n_perm_phasebin_indiv:
        n_perm_phasebin = n_perm_phasebin_indiv
    if isinstance(n_perm_phasebin, int):
        perm_indices = (np.random.choice(n_bins, n_bins, False)
                        for _ in range(n_perm_phasebin))
    elif n_perm_phasebin == 'full':
        raise(NotImplementedError)

    # Shuffle the data for each permutation
    if n_perm_phasebin:
        for i_perm, perm_inx in enumerate(perm_indices):
            mi[i_perm + 1, ...] = mi[0:1, ..., perm_inx]
    elif n_perm_phasebin_indiv:
        for i_perm, perm_inx in enumerate(perm_indices):
            perm_inx_a = copy.copy(perm_inx)
            perm_inx_b = copy.copy(perm_inx)
            np.random.shuffle(perm_inx_b)
            mi[i_perm + 1, :, :, :, 0, :] = mi[0:1, :, :, :, 0, perm_inx_a]
            mi[i_perm + 1, :, :, :, 1, :] = mi[0:1, :, :, :, 1, perm_inx_b]
    elif perm_phasebin_flip:
        perm_indices = itertools.product([0, 1], repeat=n_bins)
        for i_perm, perm_inx in enumerate(perm_indices):
            for i_bin in range(n_bins):
                if perm_inx[i_bin]:  # Flip them if there's a 1
                    v0 = mi[0:1, :, :, :, 1, i_bin]
                    v1 = mi[0:1, :, :, :, 0, i_bin]
                else:
                    v0 = mi[0:1, :, :, :, 0, i_bin]
                    v1 = mi[0:1, :, :, :, 1, i_bin]
                mi[i_perm + 1, :, :, :, 0, i_bin] = v0
                mi[i_perm + 1, :, :, :, 1, i_bin] = v1
        # The first perm is the same as the real data
        mi = mi[1:, ...]
        n_perm -= 1

    # Compute a phase-dependence index for each combination of LF and HF
    mi_c = {}
    if method is not None:
        if diff_method in ('PD(AB)-PD(BA)', 'both'):
            mi_comod = mod_index(mi, method)
            mi_comod = {'a': mi_comod[..., 0],
                        'b': mi_comod[..., 1]}
            mi_comod['diff'] = mi_comod['a'] - mi_comod['b']
            mi_c['PD(AB)-PD(BA)'] = mi_comod
        if diff_method in ('PD(AB-BA)', 'both'):
            mi_diff_shape = list(mi.shape)
            mi_diff_shape[4] += 1  # One extra 'column' for the directions
            mi_diff = np.full(mi_diff_shape, np.nan)
            mi_diff[:, :, :, :, :2, :] = mi
            mi_diff[:, :, :, :, 2, :] = \
                mi[:, :, :, :, 0, :] - mi[:, :, :, :, 1, :]
            mi = mi_diff
            # Compute phase-dependence index for each combination of LF and HF
            mi_comod = mod_index(mi, method)
            # Get the difference between directions
            mi_comod = {'a': mi_comod[..., 0],
                        'b': mi_comod[..., 1],
                        'diff': mi_comod[..., 2]}
            mi_c['PD(AB-BA)'] = mi_comod

    # Get clusters and p-values
    if n_perm > 0:
        stat_info = {}
        for diff_meth, mi_comod in mi_c.items():
            # Threshold data for clustering by finding phase-lag TE differences
            # that are less than the percentiles defined by cluster alpha value
            quantiles = [100 * (cluster_alpha / 2),
                         100 * (1 - (cluster_alpha / 2))]
            thresh = np.percentile(mi_comod['diff'], quantiles)
            thresh_mi_comod = np.zeros(mi_comod['diff'].shape)
            thresh_mi_comod[mi_comod['diff'] < thresh[0]] = -1
            thresh_mi_comod[mi_comod['diff'] > thresh[1]] = 1

            # Cluster statistic: Summed absolute z-score
            z_mi_comod = stats.zscore(mi_comod['diff'], axis=None)

            # Only keep frequencies for which f_car > 2 * f_mod
            if ignore_corner:
                for i_fm, fm in enumerate(f_mod):
                    for i_fc, fc in enumerate(f_car):
                        if fc < (2 * fm):
                            thresh_mi_comod[:, i_fm, i_fc, ...] = 0

            def stat_fun(x):
                return np.sum(np.abs(x))

            # Find clusters for each permutation, including the empirical data
            clust_labels = np.full(thresh_mi_comod.shape, np.nan)
            cluster_stats = []
            for i_perm in range(n_perm + 1):
                # Find the clusters
                c_labs = measure.label(thresh_mi_comod[i_perm, :, :, i_lag])
                clust_labels[i_perm, :, :, i_lag] = c_labs
                # Get the cluster stat for each cluster
                perm_cluster_stat = []
                labels = clust_labels[i_perm, :, :, i_lag]
                for i_clust in range(1, int(np.max(labels)) + 1):
                    # Select the z-values in the cluster
                    x = z_mi_comod[i_perm, :, :, i_lag]
                    x = x[labels == i_clust]
                    # Only keep clusters greater than the threshold size
                    if x.size < min_cluster_size:
                        s = 0
                    else:
                        s = stat_fun(x)
                    perm_cluster_stat.append(s)
                cluster_stats.append(perm_cluster_stat)

            # Compute the p-value
            max_stat_per_perm = []
            for perm in cluster_stats:
                if len(perm) > 0:
                    max_stat_per_perm.append(np.max(perm))
                else:
                    max_stat_per_perm.append(0)
            max_stat_per_perm = np.array(max_stat_per_perm)
            cluster_thresh = np.percentile(max_stat_per_perm, [95])
            pval = np.mean(max_stat_per_perm[1:] > max_stat_per_perm[0])

            # Package the cluster stat results
            clust_stat_info = dict(labels=clust_labels[0, ...],
                                   stats=cluster_stats[0],
                                   cluster_thresh=cluster_thresh,
                                   max_per_perm=max_stat_per_perm,
                                   pval=pval)
            stat_info[diff_meth] = clust_stat_info

    else:
        stat_info = None

    if return_phase_bins:
        return mi_c, mi, stat_info
    else:
        return mi_c, stat_info


def cfc_phaselag_cmi_phase(s_a, s_b, fs, f_mod, f_car, cmi_lag, f_car_bw=5):
    """
    Compute conditional mutual information between two signals (one of which is
    lagged), conditioned on the phase difference of those signals.

    This doesn't work because knowing the phase difference doesn't actually
    help you predict A given B, but A is more predictive of B at different
    values of the phase difference.

    s_a, s_b : np.ndarray
        The signals
    cmi_lag : float, seq of floats
        The lags to test between the two variables. Should be positive numbers.

    """

    s = {'a': s_a, 'b': s_b}

    # Initialize mutual information array
    # Dims: LF freq, HF freq, CMI lag, direction
    mi = np.full([len(f_mod), len(f_car), len(cmi_lag), 2], np.nan)

    for i_fm, fm in enumerate(f_mod):
        print(fm)
        # Compute the LF phase-difference of each signal
        filt = {sig: bp_filter(s[sig].T, fm[0], fm[1], fs, 2).T
                for sig in 'ab'}
        phase = {sig: np.angle(hilbert(filt[sig], axis=0))
                 for sig in 'ab'}
        phase_diff = phase['a'] - phase['b']
        phase_diff = wrap_to_pi(phase_diff)
        # Make 2D signal out of the phase difference for use with GCMI
        phase_diff_2d = np.stack([np.sin(phase_diff), np.cos(phase_diff)])

        for i_fc, fc in enumerate(f_car):
            # Filter the HF signals
            filt = {sig: bp_filter(s[sig].T,
                                   fc - (f_car_bw / 2),
                                   fc + (f_car_bw / 2),
                                   fs, 2).T
                    for sig in 'ab'}
            # Make a 2D version of the signal with its Hilbert transform
            # This makes mutual information more informative
            h = {sig: hilbert(filt[sig]) for sig in 'ab'}
            sig_2d = {sig: np.stack([np.real(h[sig]), np.imag(h[sig])])
                      for sig in 'ab'}

            for i_lag, lag in enumerate(list(cmi_lag)):

                def L(x):
                    """ Lag function """
                    return np.roll(x, lag, axis=1)

                # Compute I(LA;B|PhaseDiff) and I(A;LB|PhaseDiff)
                mi_a = gcmi.gccmi_ccc(L(sig_2d['a']),
                                      sig_2d['b'],
                                      phase_diff_2d)
                mi_b = gcmi.gccmi_ccc(sig_2d['a'],
                                      L(sig_2d['b']),
                                      phase_diff_2d)
                mi[i_fm, i_fc, i_lag, :] = (mi_a, mi_b)

    return mi


def cfc_phaselag_mutualinfo(s_a, s_b, fs, f_mod, f_car,
                            f_car_bw=5, n_bins=18, method='sine psd'):
    """
    Compute CFC: HF mutual information as a function of LF phase lag.
    """

    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    s = {'a': s_a, 'b': s_b}

    # Initialize mutual information array: LF freq, HF freq, LF phase bin
    mi = np.full([len(f_mod), len(f_car), n_bins], np.nan)
    # Initialize array to hold the number of observations in each bin
    counts = np.full([len(f_mod), n_bins], np.nan)

    for i_fm, fm in enumerate(f_mod):
        print(fm)
        # Compute the LF phase-difference of each signal
        filt = {sig: bp_filter(s[sig].T, fm[0], fm[1], fs, 2).T
                for sig in 'ab'}
        phase = {sig: np.angle(hilbert(filt[sig], axis=0))
                 for sig in 'ab'}
        phase_diff = phase['a'] - phase['b']
        phase_diff = wrap_to_pi(phase_diff)
        phase_diff = np.digitize(phase_diff, phase_bins) - 1  # Binned
        # Append trials over time if data includes multiple trials
        phase_diff = np.ravel(phase_diff, 'F')

        for i_fc, fc in enumerate(f_car):
            # Filter the HF signals
            filt = {sig: bp_filter(s[sig].T,
                                   fc - (f_car_bw / 2),
                                   fc + (f_car_bw / 2),
                                   fs, 2).T
                    for sig in 'ab'}
            # Make a 2D version of the signal with its Hilbert transform
            # This makes mutual information more informative
            h = {sig: hilbert(filt[sig]) for sig in 'ab'}
            sig_2d = {sig: np.stack([np.real(h[sig]), np.imag(h[sig])])
                      for sig in 'ab'}

            # Compute MI for each phase bin
            for phase_bin in np.unique(phase_diff):
                phase_sel = phase_diff == phase_bin
                i = gcmi.gcmi_cc(sig_2d['a'][:, phase_sel],
                                 sig_2d['b'][:, phase_sel])
                mi[i_fm, i_fc, phase_bin] = i
                # Store the count of observations per phase bin
                if i_fc == 0:
                    counts[i_fm, phase_bin] = np.sum(phase_sel)

    # Compute a phase-dependence index for each combination of LF and HF
    mi_comod = mod_index(mi, method)

    return mi, mi_comod, counts


def cfc_modelcomp(s_a, s_b, fs, f_mod, f_car, n_cycles=5):
    """
    Compute CFC using model comparison between multiple models:
        - Individual phase of each signal
        - Phase-difference between the two signals
        - Phase-combination (product) of two signals
    """
    s = {'a': s_a, 'b': s_b}

    def fit_regression(x, y):
        """ Helper function to fit a regression
        """
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()
        keep_fields = ['bic', 'rsquared_adj']
        r = {field: getattr(results, field) for field in keep_fields}
        return r

    def regression_indiv(phase, amp_sig):
        """ Regression analysis predicting amplitude from individual phases
        """
        x = np.stack([np.sin(phase['a']),
                      np.cos(phase['a']),
                      np.sin(phase['b']),
                      np.cos(phase['b'])]).T
        results = fit_regression(x, amp_sig)
        return results

    def regression_phasediff(phase, amp_sig):
        """ Regression analysis going from individual phases and phase diff
        """
        phasediff = phase['a'] - phase['b']
        phasediff = wrap_to_pi(phasediff)
        x = np.stack([np.sin(phase['a']),
                      np.cos(phase['a']),
                      np.sin(phase['b']),
                      np.cos(phase['b']),
                      np.sin(phasediff),
                      np.cos(phasediff)]).T
        results = fit_regression(x, amp_sig)
        return results

    def regression_combined(phase, amp_sig):
        """ Regression from individual phases and combined phase
        """
        x = np.stack([np.sin(phase['a']),
                      np.cos(phase['a']),
                      np.sin(phase['b']),
                      np.cos(phase['b']),
                      np.sin(phase['a']) * np.sin(phase['b']),
                      np.sin(phase['a']) * np.cos(phase['b']),
                      np.cos(phase['a']) * np.sin(phase['b']),
                      np.cos(phase['a']) * np.cos(phase['b']),
                      ]).T
        results = fit_regression(x, amp_sig)
        return results

    # Get high frequency amplitude using a wavelet transform
    amp = {}
    amp['a'] = _wavelet_tfr(s_a, f_car, n_cycles, fs)
    amp['b'] = _wavelet_tfr(s_b, f_car, n_cycles, fs)
    # Append trials over time if data includes multiple trials
    # amp.shape: (time, carrier freq)
    for sig in 'ab':
        if amp[sig].ndim == 3:
            amp[sig] = np.concatenate(
                            [amp[sig][:, :, k]
                             for k in range(amp[sig].shape[2])],
                            axis=0)

    # Set up an object for the results
    # List of Phase-freq full of lists of Amp-freq full of dicts
    # Each dict is for amplitude in each signal (a and b)
    # Each of those will hold a dict of results for each regression analysis
    results = []

    # Run the models for each combination of phase-freq and amp-freq
    for i_fm, fm in enumerate(f_mod):
        print(fm)
        # Compute the LF phase of each signal
        filt = {sig: bp_filter(s[sig].T, fm[0], fm[1], fs, 2).T
                for sig in 'ab'}
        phase = {sig: np.angle(hilbert(filt[sig], axis=0))
                 for sig in 'ab'}
        # Append trials over time if data includes multiple trials
        for sig in 'ab':
            phase[sig] = np.ravel(phase[sig], 'F')

        results_fm = []  # Results for this fm
        # Run the loop for each amplitude frequency
        for i_fc, fc in enumerate(f_car):
            sig = 'b'  # FIXME Only look at HF power in the receiver
            amp_sig = np.squeeze(amp[sig][:, i_fc])
            res = {}
            # res['indiv'] = regression_indiv(phase, amp_sig)
            res['diff'] = regression_phasediff(phase, amp_sig)
            res['combined'] = regression_combined(phase, amp_sig)
            results_fm.append(res)
        results.append(results_fm)

    return results


def cfc_vonmises_2d(s_a, s_b, fs, f_mod, f_car, n_cycles=5, n_bins=18):
    """ How does phase in two signals predict gamma in each of the signals?
    Fit Von Mises functions to the HF amplitude, averaged within LF phase bins.
    Then fit a Von Mises function to the combined phase of the 2 signals.

    Start out by selecting one frequency for phase and one frequency for amp.
    """

    #  #### For testing
    # from comlag import *
    # from comlag import _wavelet, _wavelet_tfr, _buffer, _match_dims
    # f_mod = [[9, 11]]
    # f_car = [90]
    # n_cycles = 5
    # n_bins = 18

    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    phase_bin_centers = np.mean(np.stack([phase_bins[1:], phase_bins[:-1]]),
                                axis=0)

    def rsquare(x, y):
        return stats.pearsonr(x, y)[0] ** 2

    s = {'a': s_a,
         'b': s_b}

    # Get high frqeuency amplitude
    amp = {}
    for sig in 'ab':
        amp[sig] = _wavelet_tfr(s[sig], f_car, n_cycles, fs)
        # Append trials over time if data includes multiple trials
        # amp.shape: (time, carrier freq)
        if amp[sig].ndim == 3:
            amp[sig] = np.concatenate(
                            [amp[sig][:, :, k]
                             for k in range(amp[sig].shape[2])],
                            axis=0)

    # Compute the 2D-CFC
    sig_amp = 'b'  # Which signal to use for amplitude FIXME make work for both
    n_vonmises_params = 3
    n_vonmises_2d_params = 4
    fits = {}
    fits['a'] = np.full([len(f_mod), len(f_car), n_vonmises_params],
                        np.nan)
    fits['b'] = fits['a'].copy()
    fits['2d'] = np.full([len(f_mod), len(f_car), n_vonmises_2d_params],
                         np.nan)
    fits['2d_cont'] = fits['2d'].copy()
    rsq = {}
    rsq['a'] = np.full([len(f_mod), len(f_car)],
                       np.nan)
    rsq['b'] = rsq['a'].copy()
    rsq['2d'] = rsq['a'].copy()
    rsq['2d_cont'] = rsq['a'].copy()
    for i_fm, fm in enumerate(f_mod):

        # Get low frequency phase
        phase = {}
        phase_b = {}  # Binned
        for sig in 'ab':
            s_filt = bp_filter(s[sig].T, fm[0], fm[1], fs, 2).T
            phase[sig] = np.angle(hilbert(s_filt, axis=0))
            phase_b[sig] = np.digitize(phase[sig], phase_bins) - 1
            # Append trials over time if data includes multiple trials
            if phase[sig].ndim == 2:
                phase[sig] = np.ravel(phase[sig], 'F')
                phase_b[sig] = np.ravel(phase_b[sig], 'F')

        #  # Plot the HF amplitude as a function of phase in each signal
        # plt.clf()
        # plt.plot(phase['a'], amp['b'], 'o', alpha=0.2)
        # plt.plot(phase['b'], amp['b'], 'o', alpha=0.2)

        # Compute CFC for each carrier freq using KL divergence
        for i_fc, fc in enumerate(f_car):
            # Average HF amplitude per LF phase bin
            amplitude_dist = np.ones([n_bins, n_bins])  # 1 to avoid log(0)
            for bin_a, bin_b in itertools.product(range(n_bins),
                                                  range(n_bins)):
                keep_samps = (phase_b['a'] == bin_a) & (phase_b['b'] == bin_b)
                a = np.mean(amp[sig_amp][keep_samps, i_fc])
                amplitude_dist[bin_a, bin_b] = a

            # Fit separate Von Mises distribs for amp and phase in each sig
            bounds = ([1e-10, -np.pi, 0],  # Kappa, mu, scale
                      [np.inf, np.pi, np.inf])
            x = phase_bin_centers
            y_a = np.mean(amplitude_dist, axis=0)
            y_b = np.mean(amplitude_dist, axis=1)
            popt_a, _ = optimize.curve_fit(vonmises,
                                           x, y_a,
                                           bounds=bounds)
            popt_b, _ = optimize.curve_fit(vonmises,
                                           x, y_b,
                                           bounds=bounds)
            # Save the model fits
            fits['a'][i_fm, i_fc, :] = popt_a
            fits['b'][i_fm, i_fc, :] = popt_b
            # Save goodness-of-fit metrics
            rsq['a'][i_fm, i_fc] = rsquare(y_a, vonmises(x, *popt_a))
            rsq['b'][i_fm, i_fc] = rsquare(y_b, vonmises(x, *popt_b))
            # plt.clf()
            # plt.plot(x, y_a, '-r', label='Data sig$_a$')
            # plt.plot(x, vonmises(x, *popt_a), '--r', label='Fit sig$_a$')
            # plt.plot(x, y_b, '-b', label='Data sig$_b$')
            # plt.plot(x, vonmises(x, *popt_b), '--b', label='Fit sig$_b$')
            # plt.xlabel('LF Phase')
            # plt.ylabel('HF power')
            # plt.legend()
            # plt.savefig('/Users/geoff/Desktop/von_mises_fits.png')

            # Fit the 2d Van Mises function
            x_a = np.repeat(phase_bin_centers, phase_bin_centers.size)
            x_b = np.tile(phase_bin_centers, phase_bin_centers.size)
            x = np.stack([x_a, x_b])
            y = np.reshape(amplitude_dist, [-1], order='C')  # By rows
            bounds = ((1e-10, -np.pi, -np.pi, 0),  # kappa, mu1, mu2, scale
                      (np.inf, np.pi, np.pi, np.inf))
            popt_2d, _ = optimize.curve_fit(vm2d_helper, x, y,
                                            bounds=bounds)
            # Save the model fits
            fits['2d'][i_fm, i_fc, :] = popt_2d
            # Save goodness-of-fit metrics
            rsq['2d'][i_fm, i_fc] = rsquare(y, vm2d_helper(x, *popt_2d))
            # plt.clf()
            # p1, p2 = np.meshgrid(phase_bins[:-1], phase_bins[:-1])
            # plt.contourf(p1, p2, amplitude_dist)
            # plt.plot(popt_2d[2], popt_2d[1], 'ro')  # Plot 2D VonMises fit
            # plt.ylabel('$\\Phi(b)$')
            # plt.xlabel('$\\Phi(a)$')
            # plt.colorbar()
            # plt.savefig('/Users/geoff/Desktop/von_mises_2D_fits.png')

            # Fit a 2D Von Mises function after removing the influence of the
            # LF phase in each signal independently.
            amp_cont = amplitude_dist.copy()
            amp_cont -= np.tile(vonmises(phase_bin_centers, *popt_a),
                                [n_bins, 1])
            amp_cont -= np.tile(vonmises(phase_bin_centers, *popt_b),
                                [n_bins, 1]).T
            y = np.reshape(amp_cont, [-1], order='C')  # By rows
            popt_2d_cont, _ = optimize.curve_fit(vm2d_helper, x, y,
                                                 bounds=bounds)
            # Save the model fits
            fits['2d_cont'][i_fm, i_fc, :] = popt_2d_cont
            # Save goodness-of-fit metrics
            rsq['2d_cont'][i_fm, i_fc] = rsquare(y,
                                                 vm2d_helper(x, *popt_2d_cont))

    return fits, rsq


def vonmises(x, kappa, mu, scale):
    """ Von Mises distribution
    also check out scipy.stats.vonmises.fit()
    """
    y = stats.vonmises.pdf(x, kappa, mu) * scale
    return y


def vonmises2d(x1, x2, kappa, mu1, mu2, scale):
    """
    Fit a Von Mises distribution in two dimensions. There is a single peak
    centered at (mu1, mu2), with spread kappa. The peak falls off as you move
    away from this peak, and wraps around in phase of both input dimensions.

    Parameters
    ----------
    x1, x2 : np.ndarray
        Phase of each dimension. Bounded from (-pi, pi).
    kappa : float
        Spread of the Von Mises peak. Bounded from [0, inf].
    mu1, mu2 : float
        The center of the distribution in each dimension.
    scale : float
        Multiplier to scale the whole distribution.
    kappa : float
    """
    # Get the Euclidean distance to the peak phase after wrapping to (0, 2pi)

    def piwrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    # c = np.sqrt((piwrap(x1 - mu1) ** 2) + (piwrap(x2 - mu2) ** 2))
    # c = np.max(np.abs(np.stack([x1 - mu1, x2 - mu2])), axis=0)
    c1 = piwrap(x1 - mu1)
    c2 = piwrap(x2 - mu2)
    c = np.stack([c1, c2])
    # Take the largest of the two distances
    inx = np.argmax(np.abs(c), axis=0)
    c = np.array([c[inx[i], i] for i in range(len(c1))])
    # Fit to a Von Mises distrbution that's centered around 0
    y = vonmises(c, kappa, 0, 1) * scale
    return y


def vm2d_helper(x, kappa, mu1, mu2, scale):
    """ x is an array. First dim is for the different variables
    """
    y = vonmises2d(x[0, :], x[1, :], kappa, mu1, mu2, scale)
    return y


def psi_phaselag(s_a, s_b, fs, nfft, step_size=None, n_bins=10, psi_bw=10,
                 phase_diff_freq_lims=(2, 20)):
    """
    Compute phase slope index (PSI) as a function of phase lag
    """

    # psi_bw: Frequency smoothing of the PSI in Hz
    # phase_diff_freq_lims: Freqs at which phase diff is calculated
    fft_freqs = np.fft.fftfreq(nfft, 1 / fs)
    phase_diff_freq_inx = \
        (fft_freqs > phase_diff_freq_lims[0]) \
        & (fft_freqs < phase_diff_freq_lims[1])
    phase_diff_freq_inx = np.nonzero(phase_diff_freq_inx)[0]
    phase_bins = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)

    if step_size is None:
        step_size = int(nfft / 2)
    segment_onsets = np.arange(0, len(s_a), step_size)
    segment_onsets = segment_onsets[(segment_onsets + nfft) < len(s_a)]

    def _dft_helper(x):
        return np.fft.fft(np.hanning(len(x)) * x)

    # Arrays for the cross-spectral density
    # Dims: FFT frequency, LF phase bin, LF phase-diff frequency
    csd_ij = np.zeros([nfft, n_bins, nfft], dtype=np.complex128)
    csd_ii = csd_ij.copy()
    csd_jj = csd_ij.copy()
    phase_bin_counts = np.zeros([n_bins, nfft])  # LF phase bin, phase-diff frq
    for seg in segment_onsets:
        sl = slice(seg, (seg + nfft))
        z_i = _dft_helper(s_a[sl])  # Windowed DFT of this segment
        z_j = _dft_helper(s_b[sl])
        phase_diff = np.angle(z_i * np.conj(z_j))  # phase diff for each freq
        for i_freq in phase_diff_freq_inx:
            i_bin = np.nonzero(phase_bins < phase_diff[i_freq])[0].max()
            phase_bin_counts[i_bin, i_freq] += 1
            csd_ij[:, i_bin, i_freq] += z_i * np.conj(z_j)
            csd_ii[:, i_bin, i_freq] += z_i * np.conj(z_i)
            csd_jj[:, i_bin, i_freq] += z_j * np.conj(z_j)

    # Missing cells (no obs at that phase diff) show up as zero. Replace w/ nan
    csd_ij[csd_ij == 0] = np.nan
    csd_ii[csd_ij == 0] = np.nan
    csd_jj[csd_ij == 0] = np.nan

    # Only keep positive frequencies
    keep_freqs = fft_freqs >= 0
    csd_ij = csd_ij[keep_freqs, :, :][:, :, keep_freqs]
    csd_ii = csd_ii[keep_freqs, :, :][:, :, keep_freqs]
    csd_jj = csd_jj[keep_freqs, :, :][:, :, keep_freqs]
    phase_bin_counts = phase_bin_counts[:, keep_freqs]

    # Divide by the number of segments to get the CSD
    phase_bin_counts[phase_bin_counts == 0] = np.nan
    phase_bin_counts = np.tile(phase_bin_counts,
                               [np.sum(keep_freqs), 1, 1])
    csd_ij /= phase_bin_counts
    csd_ii /= phase_bin_counts
    csd_jj /= phase_bin_counts

    # Get complex coherency
    C_ij = csd_ij / np.sqrt(csd_ii * csd_jj)

    # Get the PSI for each frequency using a moving average
    Psi_ij = np.full([np.sum(keep_freqs), n_bins, np.sum(keep_freqs)], np.nan)
    kern = np.ones(np.sum(fft_freqs[:int(nfft/2)] <= psi_bw))  # For moving avg
    inner = np.conj(C_ij) * np.roll(C_ij, 1, axis=0)  # Inner part of PSI calc
    for i_bin in range(n_bins):
        for i_freq in phase_diff_freq_inx:
            Psi_ij[:, i_bin, i_freq] = np.imag(np.convolve(
                                                    inner[:, i_bin, i_freq],
                                                    kern,
                                                    'same'))

    # Compute the modulation index
    # PSD of a sine wave fit to the PSI as a function of phase-difference
    sine_psd = (np.abs(np.fft.fft(Psi_ij, axis=1)) ** 2) / n_bins
    mi_comod = sine_psd[:, 1, :]  # Take the freq matching the whole signal

    # Bundle the results together
    res = dict(Psi_ij=Psi_ij,
               mi_comod=mi_comod,
               freqs=fft_freqs[keep_freqs],
               phase_bins=phase_bins)
    return res


def pearson(a, b):
    """ Calculate the pearson correlation.
    This is faster than np.corrcoef
    """
    cov = np.dot(a - a.mean(), b - b.mean())
    r = cov / (a.size * a.std() * b.std())
    return r


def xcorr(a, b, max_lag):
    """ Normalized cross-correlation
    """

    def pearson(a, b):
        r = np.dot(a - a.mean(), b - b.mean()) / (a.size * a.std() * b.std())
        return r

    lags = range(-max_lag, max_lag)
    padding = np.zeros(max_lag)
    a = np.concatenate([padding, a, padding])
    b = np.concatenate([padding, b, padding])
    xc = []
    for lag in lags:
        x = pearson(a, np.roll(b, lag))
        xc.append(x)
    return lags, xc


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


def plot_filter_kernel(lowcut, highcut, fs, dur, **plot_kwargs):
    """
    Plot the filter kernels.

    Parameters
    ----------
    lowcut, highcut, fs : Passed on to bp_filter
    dur : float
        Duration of the impulse calculated in seconds.
    """
    import matplotlib.pyplot as plt
    impulse_len = int(dur * fs)  # samples
    t = np.arange(impulse_len) / fs  # Time vector in seconds
    t -= t.mean()

    impulse = np.zeros(impulse_len)
    impulse[impulse_len // 2] = 1

    plt.title(f"{lowcut:.2f}-{highcut:.2f} Hz")
    ir = bp_filter(impulse, lowcut, highcut, fs)
    plt.plot(t, ir, **plot_kwargs)
    return ir
