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

import itertools
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, fftconvolve
from scipy import stats, optimize
import statsmodels.api as sm


def wrap_to_pi(x):
    """ Wrap the input phase to the range [-pi, pi]
    """
    x = (x + np.pi) % (2 * np.pi) - np.pi
    return x

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
    start_points = range(0, x.shape[0] - n + 1, n - p) # Where each seg starts
    #assert len(x.shape) <= 2, 'Data must be 1- or 2-dimensional'
    if len(x.shape) == 1:
        result_shape = [n, len(start_points)]
    elif len(x.shape) == 2:
        result_shape = [n, x.shape[1], len(start_points)]
    elif len(x.shape) == 3:
        result_shape = [n, x.shape[1], x.shape[2], len(start_points)]
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
    s_a, s_b : ndarray (time,) or (time, trial)
        Signal arrays. If 2D, first dim must be time, and 2nd dim is trial.
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


def cfc_sine(s_a, s_b, fs, f_mod, f_car, n_cycles=5):
    """
    Compute CFC similar to Tort et al (2010, J Neurophysiol), but fitting the
    gamma amplitudes at each phase value to a sine-wave instead of comparing
    against a uniform distribution using KL distance.

    Parameters
    ----------
    s_a : ndarray (time,) or (time, trial)
        Signal array with the modualting signal. If 2D, first dim must be time,
        and 2nd dim is trial.
    s_b : ndarray (time,) or (time, trial)
        Signal array with the amplitude variations. Structure like s_a.
    fs : int,float
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
    #TODO test this with multichannel inputs

    # Get high frequency amplitude using a wavelet transform
    s_b_amp = _wavelet_tfr(s_b, f_car, n_cycles, fs)
    # Append trials over time if data includes multiple trials
    # s_b_amp shape: (time, carrier freq)
    if s_b.ndim == 2:
        s_b_amp = np.concatenate(
                    [s_b_amp[:,:,k] for k in range(s_b_amp.shape[2])],
                    axis=0)

    sin_amp = np.full([f_car.shape[0], f_mod.shape[0]], np.nan)
    sin_r = sin_amp.copy()
    for i_fm,fm in enumerate(f_mod):
        # Compute LF phase
        s_a_filt = bp_filter(s_a.T, fm[0], fm[1], fs, 2).T
        s_a_phase = np.angle(hilbert(s_a_filt, axis=0))
        # Append trials over time if data includes multiple trials
        if s_a_phase.ndim == 2:
            s_a_phase = np.ravel(s_a_phase, 'F')
        # Compute CFC for each carrier freq by checking for sinusoidal
        # modulation of HF power along with LF phase
        for i_fc,fc in enumerate(f_car):
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
    Compute CFC based on the low-frequency phase-difference between two signals.

    Compute CFC similar to Tort et al (2010, J Neurophysiol), but fitting the
    gamma amplitudes at each phase value to a sine-wave instead of comparing
    against a uniform distribution using KL distance.

    Parameters
    ----------
    s_a, s_b : ndarray (time,) or (time, trial)
        Signal arrays. If 2D, first dim must be time, and 2nd dim is trial.
    fs : int,float
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
    #TODO test this with multichannel inputs

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
                            [amp[sig][:,:,k] for k in range(amp[sig].shape[2])],
                            axis=0)

    # Initialize data structures to hold CFC values
    init_mat = np.full([f_car.shape[0], f_mod.shape[0]], np.nan)
    mi = {}
    for sig in ('a', 'b'):
        mi[sig] = {}
        for out in ('amp', 'r'):
            mi[sig][out] = init_mat.copy()

    # Compute the CFC
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
        # Compute CFC for each carrier freq by checking for sinusoidal
        # modulation of HF power along with LF phase
        for i_fc,fc in enumerate(f_car):
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
    x,_,_,_ = np.linalg.lstsq(a.T, b, rcond=None)
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
# plt.plot(s_a_phase, s_b_amp[:,i_fc], 'o', alpha=0.2)
# x_phase = np.linspace(-np.pi, np.pi, 100)
# plt.plot(x_phase, sine_helper(x_phase, popt[0], popt[1]))

# # Two ways to fit a sine wave
# # First simulate a signal
# k = 1000
# sim_amp = np.random.uniform(0.5, 2.0)
# sim_offset = np.random.uniform(0.0, 10.0)
# sim_phase = np.random.uniform(-np.pi, np.pi)
# sin_fnc = lambda a,p,x: a * np.sin(x + p)
# x = np.random.uniform(-np.pi, np.pi, k)
# y = sine_helper(x, sim_amp, sim_phase, sim_offset)
# y = y + np.random.normal(size=y.shape)
# plt.clf()
# plt.plot(x, y, 'o', alpha=0.2)
# # 1. Using optimize.curve_fit to estimate the amp and phase of a sine
# def sine_curve_fit(s_phase, s_amp):
#     popt, _ = curve_fit(sine_helper,
#                         s_phase,
#                         np.squeeze(s_amp))
#     return popt
# p_curve_fit = sine_curve_fit(x, y)
# phase_x = np.arange(-np.pi, np.pi, 1/fs) 
# plt.plot(phase_x, sine_helper(phase_x, *p_curve_fit),
#          linestyle='--')
# # 2. Using OLS
# # Fit a sine and a cosine, then calculate amp and phase
# def sine_ols(s_phase, s_amp):
#     a = np.stack([np.ones(s_phase.shape),
#                   np.sin(s_phase),
#                   np.cos(s_phase)])
#     b = np.squeeze(s_amp)
#     x,_,_,_ = np.linalg.lstsq(a.T, b, rcond=None)
#     z = np.complex(*x[1:])
#     p = [np.abs(z), np.angle(z), x[0]] # Amp, phase, offset
#     return p
# p_ols = sine_ols(x, y)
# plt.plot(phase_x, sine_helper(phase_x, *p_ols),
#          linestyle=':')
# # Compare the timing
# # curve_fit is about 5 times slower
# %timeit p_curve_fit = sine_curve_fit(x, y)
# %timeit p_ols = sine_ols(x, y)


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
                            [amp[sig][:,:,k] for k in range(amp[sig].shape[2])],
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

        results_fm = [] # Results for this fm
        # Run the loop for each amplitude frequency
        for i_fc,fc in enumerate(f_car):
            sig = 'b' #FIXME Only look at HF power in the receiver
            amp_sig = np.squeeze(amp[sig][:, i_fc])
            res = {}
            #res['indiv'] = regression_indiv(phase, amp_sig)
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

    # #### For testing
    # from comlag import *
    # from comlag import _wavelet, _wavelet_tfr, _buffer, _match_dims
    # f_mod = [[9, 11]]
    # f_car = [90]
    # n_cycles = 5
    # n_bins = 18

    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    phase_bin_centers = np.mean(np.stack([phase_bins[1:], phase_bins[:-1]]),
                                axis=0)

    rsquare = lambda x,y: stats.pearsonr(x,y)[0] ** 2
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
                            [amp[sig][:,:,k] for k in range(amp[sig].shape[2])],
                            axis=0)

    # Compute the 2D-CFC
    sig_amp = 'b' # Which signal to use for amplitude FIXME make work for both
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
    for i_fm,fm in enumerate(f_mod):

        # Get low frequency phase
        phase = {}
        phase_b = {} # Binned
        for sig in 'ab':
            s_filt = bp_filter(s[sig].T, fm[0], fm[1], fs, 2).T
            phase[sig] = np.angle(hilbert(s_filt, axis=0))
            phase_b[sig] = np.digitize(phase[sig], phase_bins) - 1
            # Append trials over time if data includes multiple trials
            if phase[sig].ndim == 2:
                phase[sig] = np.ravel(phase[sig], 'F')
                phase_b[sig] = np.ravel(phase_b[sig], 'F')

        # # Plot the HF amplitude as a function of phase in each signal
        # plt.clf()
        # plt.plot(phase['a'], amp['b'], 'o', alpha=0.2)
        # plt.plot(phase['b'], amp['b'], 'o', alpha=0.2)

        # Compute CFC for each carrier freq using KL divergence
        for i_fc,fc in enumerate(f_car):
            # Average HF amplitude per LF phase bin
            amplitude_dist = np.ones([n_bins, n_bins]) # 1 to avoid log(0)
            for bin_a, bin_b in itertools.product(range(n_bins), range(n_bins)):
                keep_samps = (phase_b['a'] == bin_a) & (phase_b['b'] == bin_b)
                a = np.mean(amp[sig_amp][keep_samps, i_fc])
                amplitude_dist[bin_a, bin_b] = a

            # Fit separate Von Mises distributions for amp and phase in each sig
            bounds = ([1e-10, -np.pi, 0], # Kappa, mu, scale
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
            y = np.reshape(amplitude_dist, [-1], order='C') # By rows
            bounds = ((1e-10, -np.pi, -np.pi, 0), # kappa, mu1, mu2, scale
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
            # plt.plot(popt_2d[2], popt_2d[1], 'ro') # Plot the 2D VonMises fit
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
            y = np.reshape(amp_cont, [-1], order='C') # By rows
            popt_2d_cont, _ = optimize.curve_fit(vm2d_helper, x, y,
                                                  bounds=bounds)
            # Save the model fits
            fits['2d_cont'][i_fm, i_fc, :] = popt_2d_cont
            # Save goodness-of-fit metrics
            rsq['2d_cont'][i_fm, i_fc] = rsquare(y, vm2d_helper(x, *popt_2d_cont))

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
    piwrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    # c = np.sqrt((piwrap(x1 - mu1) ** 2) + (piwrap(x2 - mu2) ** 2))
    #c = np.max(np.abs(np.stack([x1 - mu1, x2 - mu2])), axis=0)
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
    y = vonmises2d(x[0,:], x[1,:], kappa, mu1, mu2, scale)
    return y





'''
def NEW_METHOD(args): # FIXME Fill in args and function name
    """
    Given two signals s_a and s_b, does theta phase in the receiver correlate
    with gamma-band connectivity between the two signals?

    Parameters
    ----------
    s_a: ndarray
        Sender signal
    s_b : ndarray
        Receiver signal
    fs : int,float
        Sampling rate
    f_mod : list of lists (n, 2)
        Array with a row of cutoff frequencies for each bandpass filter for
        computing the phase of the modulation frequencies.
    f_car : list of lists (n, 2)
        Array with a row of cutoff frequencies for each bandpass filter for
        computing the amplitude at the carrier frequencies.
    n_cycles : int
        Number of cycles for the wavelet analysis to compute high-freq power
    n_bins : int
        Number of bins for low-frequency phase
    max_lag : int
        Maximum lag at which to compute cross-correlations of high-freq power

    Returns
    -------
    cm : ndarray (len(f_mod), len(f_car))
        Connectivity matrix
    """

    # Simulate some data
    import numpy as np
    from scipy.signal import hilbert
    from tqdm import tqdm
    import simulate
    dur = 20
    fs = 1000
    t, s_a, s_b = simulate.sim(dur=dur, fs=fs,
                            noise_amp=2.0,
                            signal_leakage=0.0,
                            gamma_lag_a_to_b=0.03,
                            common_noise_amp=0.0,
                            common_alpha_amp=0.0)
    from comlag import _wavelet_tfr, bp_filter
    # Which frequencies to calculate phase for
    f_mod_centers = np.logspace(np.log10(2), np.log10(30), 10)
    f_mod_width = f_mod_centers / 6
    f_mod = np.tile(f_mod_width, [2, 1]).T \
                * np.tile([-1, 1], [len(f_mod_centers), 1]) \
                + np.tile(f_mod_centers, [2, 1]).T
    # Which frequencies to calculate power for
    f_car = np.arange(30, 200, 5)
    # Other params
    n_cycles = 5
    n_bins = 10
    max_lag = 200

    # TODO
    # Instead of cross-correlating gamma power, try direct gamma time-courses


    sig = {'a': s_a, 'b': s_b}
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    # Get high frequency amplitude using a wavelet transform
    amp = {s: _wavelet_tfr(sig[s], f_car, n_cycles, fs) for s in sig.keys()}

    # Look at connectivity
    xcorr_data = np.zeros([n_bins, # LF phase
                          max_lag * 2, # Cross-corr lag
                          len(f_mod), # LF phase frequency,
                          len(f_car)]) # HF amp frequency
    phase_counts = np.zeros([n_bins, len(f_mod), len(f_car)])
    #for i_fm, fm in tqdm(enumerate(f_mod)):
    for i_fm in tqdm(range(len(f_mod))):
        fm = f_mod[i_fm]
        # Compute binned LF phase in each signal
        phase = {}
        for s in sig.keys():
            s_filt = bp_filter(sig[s].T, fm[0], fm[1], fs, 2).T
            s_phase = np.angle(hilbert(s_filt, axis=0))
            s_binned = np.digitize(s_phase, phase_bins) - 1
            phase[s] = s_binned
        # Look at connectivity in HF amplitude
        for i_fc, fc in enumerate(f_car):
            amp_a = amp['a'][:, i_fc, 0] #FIXME what's the 3rd dim?
            amp_b = amp['b'][:, i_fc, 0] #FIXME what's the 3rd dim?
            amp_a = (amp_a - np.mean(amp_a)) / (np.std(amp_a) * len(amp_a))
            amp_b = (amp_b - np.mean(amp_b)) / np.std(amp_b)
            for i_t in range(max_lag, len(sig['a']) - max_lag):
                lf_phase = phase['b'][i_t] # FIXME both signals insteaf of only 'b'
                phase_counts[lf_phase, i_fm, i_fc] += 1
                sl = slice(i_t - max_lag, i_t + max_lag)
                xc = np.correlate(amp_a[sl], amp_b[sl], mode='same') #FIXME make it work for 'valid'
                xcorr_data[lf_phase, :, i_fm, i_fc] += xc
            xcorr_data[lf_phase,:,i_fm,i_fc] /= len(t) # Normalize
            # # Compute the modulation index
            # # Doesn't work for xcorr because there are negative values - these are not proportions
            # # Kullback-Leibler divergence of the xcorr distribution vs uniform
            # x = np.ravel(xcorr_data[:, :, i_fm, i_fc])
            # x /= np.sum(x)
            # d_kl = np.sum(x * np.log(x * n_bins))
            # mi_mc = d_kl / np.log(n_bins)
            # mi[i_fc, i_fm] = mi_mc

            # New approach
            # for each timepoint
            #   check how strongly non-uniform the distribution across time is
            #       by testing for differences from 0? RMS?
            #           sqrt(mean((z - mean(z)) ** 2))
    # Get the RMS of cross-correlations within each phase-freq, amp-freq, and
    # time-point, across LF phase bins.
    xcorr_mean_over_bins = np.mean(xcorr_data, axis=0)
    rms_bin = np.sqrt(np.mean((xcorr_data - xcorr_mean_over_bins) ** 2, axis=0))
    rms_time = np.sqrt(np.mean((rms_bin - np.mean(rms_bin, axis=0)) ** 2, axis=0))
    rms_max = np.max(rms_bin, axis=0)


    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(t, s_a)
    plt.plot(t, s_b)
    plt.xlim(3, 4)
    
    plt.clf()
    plt.contourf(f_mod_centers, f_car, rms_max.T)
    plt.xlabel('Phase frequency (Hz)')
    plt.ylabel('Amplitude frequency (Hz)')
    plt.colorbar()


    i_fm = 5
    i_fc = 12
    plt.clf()
    plt.contourf(xcorr_data[:,:,i_fm,i_fc], levels=50)
    plt.xlabel('Lag (samp)')
    plt.ylabel('Phase bin')
    plt.title(f'Phase at {f_mod_centers[i_fm]:.1f} Hz, amp at {f_car[i_fc]} Hz')
    plt.colorbar()

    plt.clf()
    color_lim = np.max(np.abs(xcorr_data))
    lags = np.arange(-max_lag, max_lag) / fs
    for i_fc in range(len(f_car)): 
        plt.subplot(5, 7, i_fc+1) 
        x = xcorr_data[:,:,i_fm,i_fc]
        plt.contourf(lags, range(n_bins), x,
                     levels=np.linspace(-color_lim,
                                        color_lim, 50),
                     cmap='RdBu_r') 
        x_mean = np.mean(x, axis=0)
        rms = np.sqrt(np.mean((x - x_mean) ** 2, axis=0))
        plt.plot(lags, rms, '-k')
        plt.title(f'{f_car[i_fc]} Hz')
    plt.tight_layout()


    ## Pseudocode of the algorithm
    #max_lag = maximum_xcorr_lag
    #for freq_mod in freq_mods: # Modulation (LF phase) frequency
    #    for freq_car in freq_cars: # Carrier (HF amp) frequency
    #        xcorr_data = np.fill([len(phase_bins), max_lag * 2 + 1], np.nan)
    #        phase_bin_counts = np.zeros(len(phase_bins))
    #        for t in range(len(signal)): # For each timepoint
    #            lf_phase = get_lf_phase(t) # Binned LF phase
    #            phase_bin_counts[lf_phase] += 1
    #            xc = xcorr(hf_amp_a[t - max_lag, t + max_lag],
    #                       hf_amp_b[t - max_lag, t + max_lag])
    #            xcorr_data[lf_phase, :] += xc
    #        xcorr_data[lf_phase, :] /= phase_bin_counts[lf_phase] # Avg across segments
    '''

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
    pearson = lambda a,b:  np.dot(a - a.mean(), b - b.mean()) / (a.size * a.std() * b.std())
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




#############
# SKETCHPAD # 
#############

# Fit a single peak that wraps in phase with each signal 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for projection='3d'
import scipy


'''

# Simulate HF amplitude at two LF phases
k = int(1e3)
x1, x2 = (np.random.uniform(-np.pi, np.pi, size=k) for _ in range(2))
kappa = np.random.uniform(0.1, 5)
mu1, mu2 = (np.random.uniform(-np.pi, np.pi) for _ in range(2))
scale = np.random.normal(loc=5, scale=1)
noise_amp = 0.001
noise = np.random.normal(scale=noise_amp * scale, size=x1.shape)

x = np.stack([x1, x2])
y = vm2d_helper(x, kappa, mu1, mu2, scale)
y = y + noise
bounds = [(1e-10, np.inf), # kappa
          (-np.pi, np.pi), # mu
          (-np.pi, np.pi), # mu
          (0, np.inf)] # scale
popt, pcov = optimize.curve_fit(vm2d_helper, x, y)

plt.clf()
plt.subplot(1, 2, 1)
plt.plot(x1, y, 'o')
plt.plot(x2, y, 'o')

ax = plt.subplot(1, 2, 2, projection='3d')
ax.plot_trisurf(x1, x2, y, cmap='viridis')
ax.view_init(azim=0, elev=90)


msg = f"""
      Original
      k = {kappa:.2f}
      mu = {[mu1, mu2]}
      scale = {scale:.2f}
      
      Fitted
      k = {popt[0]:.2f}
      mu = {popt[1:3]}
      scale = {popt[3]:.2f}
      
      Error
      k = {kappa - popt[0]:.2f}
      mu = {np.array([mu1, mu2]) - popt[1:3]}
      scale = {scale - popt[3]:.2f}
      """
print(msg)




k = int(1e3)
x = np.random.uniform(-np.pi, np.pi, size=k)
kappa = np.random.uniform(0.1, 5)
mu = np.random.uniform(-np.pi, np.pi)
scale = np.random.normal(loc=5, scale=1)
noise_amp = 0.001
noise = np.random.normal(scale=noise_amp * scale, size=x1.shape)

y = vonmises(x, kappa, mu, scale)
y = y + noise

# Fit using curve_fit
bounds = [(1e-10, np.inf), # kappa
          (-np.pi, np.pi), # mu
          (0, np.inf)] # scale
popt_cf, pcov = optimize.curve_fit(vonmises, x, y)

# Fit using leastsq
def resid(p, x, y):
    return y - vonmises(x, *p)
popt_ls, pcov = optimize.leastsq(resid,
                                    [0.1, 0, 1],
                                    args=(x, y))
popt_ls[1] = piwrap(popt_ls[1])

# Plot the model fits
labels = ('True', 'CF', 'LS')
true_vals = (kappa, mu, scale)
point_types = ('*k', '+c', 'xm')
ylims = ([0, 10], [-3.5, 3.5], [0, 10])
plt.clf()
for i_param in range(3):
    plt.subplot(1, 3, i_param + 1)
    params = [true_vals[i_param],
              popt_cf[i_param],
              popt_ls[i_param]]
    for p,ptype,lab in zip(params, point_types, labels):
        plt.plot(0, p, ptype, label=lab)
    plt.ylim(ylims[i_param])
plt.legend()
plt.tight_layout()


msg = f"""
      Original
      k = {kappa:.2f}
      mu = {mu}
      scale = {scale:.2f}
      
      Fitted - curve_fit
      k = {popt_cf[0]:.2f}
      mu = {popt_cf[1]:.2f}
      scale = {popt_cf[2]:.2f}

      Fitted - leastsq
      k = {popt_ls[0]:.2f}
      mu = {piwrap(popt_ls[1]):.2f}
      scale = {popt_ls[2]:.2f}
      
      Error - curve_fit
      k = {kappa - popt_cf[0]:.2f}
      mu = {mu - popt_cf[1]:.2f}
      scale = {scale - popt_cf[2]:.2f}

      Error - leastsq
      k = {kappa - popt_ls[0]:.2f}
      mu = {mu - popt_ls[1]:.2f}
      scale = {scale - popt_ls[2]:.2f}
      """
print(msg)






'''
