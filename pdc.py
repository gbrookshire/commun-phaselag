"""
Tools to look for phase dependent communication between two signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt
import statsmodels.api as sm


def bp_filter(data, lowcut, highcut, fs, order=2):
    """
    Bandpass filter the data

    Parameters
    ----------
    x : np.ndarray (1,)
        The data to be filtered
    lowcut : float
        The low-frequency cutoff of the filter
    highcut : float
        The high-frequency cutoff of the filter
    fs : float
        The sampling rate
    order : int
        The filter order

    Returns
    -------
    y : np.ndarray
        The filtered data
    """
    def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def wrap_to_pi(x):
    """ Wrap the input phase to the range [-pi, pi]
    """
    x = (x + np.pi) % (2 * np.pi) - np.pi
    return x


def mod_index(x, method):
    """
    Compute the modulation index given some dependent measure across phase bins

    Parameters
    ----------
    x : np.ndarray
        The dependent measure computed for each phase bin. The last dimension
        is phase bins.
    method : str
        The method used to compute the modulation index. Options are: tort,
        "sine psd", "sine amp", rsquared, vector (the vector length, inspired
        by the inter-trial phase consistency), "vector imag" (the imaginary
        part of the vector length), itlc (analogous to the inter-trial linear
        coherence, Delorme & Makeig, 2004).

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
        # Units: u^2 / Hz (for input unit u)
        sine_psd = (np.abs(np.fft.fft(x)) ** 2) / n_bins
        mi_comod = sine_psd[..., 1]  # Take the freq matching the whole signal

    elif method == 'sine amp':
        # Amplitude of a sine wave fit, normalized by sequence length
        # Units: u / Hz (for input uit u)
        sine_amp = np.abs(np.fft.fft(x)) / n_bins
        mi_comod = sine_amp[..., 1]

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


def pdc(**FILL_IN):  # FIXME
    """
    Compute the phase-dependent communication between two signals. This
    algorithm looks for transfer entropy in high-frequency activity that varies
    as a function of the low-frequency phase difference.

    Parameters
    ----------
    FILL IN

    Returns
    -------
    FILL IN
    """
    pass


def cluster_test(**FILL_IN):  # FIXME
    """
    Cluster-based permutation test to evaluate whether phase-dependent
    communication is higher than would be expected by chance.

    Parameters
    ----------
    FILL IN

    Returns
    -------
    FILL IN
    """
    pass


