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

    n_cycles = n_cycles * np.ones(len(f_carrier))

    # Set up the wavelets
    wavelets = [_wavelet(f, n, fs) for f,n in zip(f_carrier, n_cycles)]

    # Compute the power timecourse
    pwr = [abs(np.convolve(amp_sig, w, 'same')) ** 2 
                for w in wavelets]
    pwr = np.array(pwr).T

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
    x_fft = np.reshape(x_fft, [1024, 1, 194]) # Reshape to combine w/ power
    xspec = x_fft * np.conj(pwr_fft)
    # TODO For phase difference, maybe just slot in the phase difference
    # instead of x_fft? That would look like: 
    #   np.angle(a_fft * np.conj(b_fft))

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
    freq : int
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

