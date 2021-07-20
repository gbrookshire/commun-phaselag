"""
Tools to look for phase dependent communication between two signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy import stats
import statsmodels.api as sm
from skimage import measure
import gcmi
from tqdm import tqdm


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


def mod_index(x, method='sine psd'):
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
    mi : np.ndarray
        Modulation index, returned as a comodulogram. Same shape as the input
        array x, but without the final dimension.
    """

    n_bins = x.shape[-1]
    phase_bins = np.linspace(-np.pi, np.pi, n_bins)

    if method == 'tort':
        assert x.ndim == 3, "Only implemented with 3-dimensional data"
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
        mi = d_kl / np.log(n_bins)

    elif method == 'sine psd':
        # PSD of a sine wave fit to the MI as a function of phase-difference
        # Units: u^2 / Hz (for input unit u)
        sine_psd = (np.abs(np.fft.fft(x)) ** 2) / n_bins
        mi = sine_psd[..., 1]  # Take the freq matching the whole signal

    elif method == 'sine amp':
        # Amplitude of a sine wave fit, normalized by sequence length
        # Units: u / Hz (for input uit u)
        sine_amp = np.abs(np.fft.fft(x)) / n_bins
        mi = sine_amp[..., 1]

    elif method == 'rsquared':
        # Find the R^2 of a sine wave fit to the MI by phase-lag
        assert x.ndim == 3, "Only implemented with 3-dimensional data"
        x_arr = np.stack([np.sin(phase_bins), np.cos(phase_bins)]).T
        mi = np.full(x.shape[:2], np.nan)
        for i_fm in range(x.shape[0]):
            for i_fc in range(x.shape[1]):
                y = x[i_fm, i_fc, :]
                y = y - np.mean(y)  # Remove mean to focus on sine-wave fits
                model = sm.OLS(y, x_arr)
                results = model.fit()
                rsq = results.rsquared
                mi[i_fm, i_fc] = rsq

    elif method == 'vector':
        # Look at mean vector length. Inspired by the ITPC
        theta = np.exp(1j * phase_bins)
        phase_vectors = x * theta
        mean_vector_length = np.abs(np.mean(phase_vectors, axis=-1))
        mi = mean_vector_length

    elif method == 'vector-imag':
        # Imaginary part of the mean vector length.
        # Inspired by the ITPC and imaginary coherence.
        theta = np.exp(1j * phase_bins)
        phase_vectors = x * theta
        mean_vector_length = np.abs(np.imag(np.mean(phase_vectors, axis=-1)))
        mi = mean_vector_length

    elif method == 'itlc':
        # Something like the inter-trial linear coherence (ITLC)
        # Delorme & Makeig (2004)
        F = x * np.exp(1j * phase_bins)
        n = n_bins
        num = np.sum(F, axis=-1)
        den = np.sqrt(n * np.sum(np.abs(F) ** 2, axis=-1))
        itlc = num / den
        itlc = np.abs(itlc)
        mi = itlc

    else:
        raise(Exception(f'method {method} not recognized'))

    return mi


def pdc(s_a, s_b, fs,
        lf_centers, lf_bandwidth,
        hf_centers, hf_bandwidth,
        lag, n_bins):
    """
    Compute the phase-dependent communication between two signals. This
    algorithm looks for transfer entropy in high-frequency activity that varies
    as a function of the low-frequency phase difference.

    Parameters
    ----------
    s_a, s_b : np.ndarray (time, ) or (time, trial)
        The two signals
    fs : scalar (int, float)
        The sampling rate of the signals
    lf_centers : list, nd.array
        The center frequencies of the low-frequency bandpass filters (in Hz)
    lf_bandwidth : scalar (int, float) or sequence (list or np.ndarray)
        The bandwidth of the low-frequency bandpass filters (in Hz)
    hf_centers : list, nd.array
        The center frequencies of the high-frequency bandpass filters (in Hz)
    hf_bandwidth : scalar (int, float) or sequence (list or np.ndarray)
        The bandwidth of the low-frequency bandpass filters (in Hz)
    lag : int
        The lag (in samples) to test between the two variables. Should be a
        positive integer.
    n_bins : int
        The number of phase-difference bins

    Returns
    -------
    res : dict
        Results of the analysis. Contains two items, one for each method of
        computing the difference. "PD(AB)-PD(BA)" shows the phase dependence
        (i.e. the modulation index) in one direction minus the phase dependence
        in the opposite direction. "PD(AB-BA)" shows the phase dependence of
        the difference in transfer entropy between the two directions, after
        calculating this difference separately for each low-frequency phase
        bin. Within each of those difference-method items is another
        dictionary. This sub-dictionary contains 3 comodulograms
    """
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    assert s_a.shape == s_b.shape, 'Data s_a and s_b must be the same shape'
    s = {'a': s_a, 'b': s_b}

    assert isinstance(hf_bandwidth, (float, int, np.ndarray)), \
        "hf_bandwidth must be a scalar or a numpy array"
    if isinstance(hf_bandwidth, (float, int)):
        hf_bandwidth = np.ones(hf_centers.shape) * hf_bandwidth

    assert isinstance(lf_bandwidth, (float, int, np.ndarray)), \
        "lf_bandwidth must be a scalar or a numpy array"
    if isinstance(lf_bandwidth, (float, int)):
        lf_bandwidth = np.ones(lf_centers.shape) * lf_bandwidth

    def L(x):
        """
        Lag helper function. Because this rolls samples from the end to the
        beginning, it will result in some samples being counted in the MI
        calculation even though they happened far apart in the real data. This
        will only occur for a very small number of samples (number of the lag),
        so it's negligible as long as the length of the data is much larger
        than the lag.
        """
        return np.roll(x, lag, axis=1)

    # Initialize mutual information array
    # Dims: LF freq, HF freq, direction, LF phase bin
    te = np.full([len(lf_centers), len(hf_centers), 2, n_bins],
                 np.nan)
    # Initialize array to hold the number of observations in each bin
    counts = np.full([len(lf_centers), n_bins], np.nan)

    for i_lf in tqdm(range(len(lf_centers)), desc='LF', leave=False):
        # Compute the LF phase-difference of each signal
        lf_c = lf_centers[i_lf]
        lf_bw = lf_bandwidth[i_lf]
        filt = {sig: bp_filter(s[sig].T,
                               lf_c - (lf_bw / 2),
                               lf_c + (lf_bw / 2),
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

        for i_hf in tqdm(range(len(hf_centers)), desc='HF', leave=False):
            # Filter the HF signals
            fc = hf_centers[i_hf]
            fc_bw = hf_bandwidth[i_hf]
            filt = {sig: bp_filter(s[sig].T,
                                   fc - (fc_bw / 2),
                                   fc + (fc_bw / 2),
                                   fs, 2).T
                    for sig in 'ab'}
            # Make a 2D version of the signal with its Hilbert transform
            # This makes mutual information more informative
            h = {sig: hilbert(filt[sig], axis=0) for sig in 'ab'}
            sig_2d = {sig: np.stack([np.real(h[sig]), np.imag(h[sig])])
                      for sig in 'ab'}

            # Append epochs
            if s_a.ndim == 2:
                sig_2d = {sig: np.reshape(sig_2d[sig],
                                          (2, -1),
                                          order='F')
                          for sig in 'ab'}

            # Compute MI for each phase bin
            for phase_bin in np.unique(phase_diff):

                # Get the samples with the desired phase difference
                phase_sel = phase_diff == phase_bin

                # Store the count of observations per phase bin
                if i_hf == 0:
                    counts[i_lf, phase_bin] = np.sum(phase_sel)

                # Compute CMI in each direction
                for i_direc, direc in enumerate('ab'):
                    # Compute I(LA;B|LB) and I(A;LB|LA)
                    if direc == 'a':
                        s1, s2 = ('a', 'b')
                    else:
                        s1, s2 = ('b', 'a')
                    i = gcmi.gccmi_ccc(L(sig_2d[s1])[:, phase_sel],
                                       sig_2d[s2][:, phase_sel],
                                       L(sig_2d[s2])[:, phase_sel])

                    # Store the transfer entropy value for this phase bin
                    te[i_lf, i_hf, i_direc, phase_bin] = i

    # Compute a phase-dependence modulation index for each combination of LF/HF
    res = {}
    res['a'] = mod_index(te[:, :, 0, :])
    res['b'] = mod_index(te[:, :, 1, :])
    res['PD(AB)-PD(BA)'] = res['a'] - res['b']
    res['PD(AB-BA)'] = mod_index(te[:, :, 0, :] - te[:, :, 1, :])

    return res


def cluster_test(emp, surr,
                 mask=None,
                 cluster_alpha=0.05,
                 cluster_stat='summed z-score',
                 min_cluster_size=1):
    """
    Cluster-based permutation test to evaluate whether phase-dependent
    communication is higher than would be expected by chance.

    Parameters
    ----------
    emp : np.ndarray
        The empirical data
    surr : np.ndarray
        The surrogate data. Must have the same shape as emp, plus one
        additional final dimension for each surrogate dataset. The dim for
        separate surrogate datasets comes first: [k, *emp.shape]
    mask : None, np.ndarray
        A boolean array of the same shape as emp, used to select observations
        to keep in the cluster analysis.
    cluster_alpha : float in the range [0, 1]
        The percentile used to select values to include in the clusters
    cluster_stat : str, function
        The function used to compute the cluster statistic. The default is
        "summed z-score", but any other function can be specified here.
    min_cluster_size : int
        The minimum size of a cluster retained in the analysis

    Returns
    -------
    FILL IN
    """
    assert emp.ndim == (surr.ndim - 1), \
        "Surrogate data must have one more dimension than empirical data"
    assert emp.shape == surr.shape[1:], \
        "Each surrogate dataset must have the same shape as empirical data"
    assert (mask is None) or (mask.dtype == bool), \
        "mask must be None or a boolean array"
    assert (mask is None) or (mask.shape == emp.shape), \
        "mask must be None or an array with the same shape as the emp data"
    # Combine the empirical and surrogate data
    x = np.concatenate([np.reshape(emp, [1, *emp.shape]), surr],
                       axis=0)
    # Apply the mask
    if mask is not None:
        x[mask] = np.nan
    # Threshold the data to find clusters
    quantiles = [100 * (cluster_alpha / 2),
                 100 * (1 - (cluster_alpha / 2))]
    thresh = np.nanpercentile(x, quantiles)
    # Separately mark positive and negative clusters
    thresh_x = np.zeros(x.shape)
    thresh_x[x < thresh[0]] = -1
    thresh_x[x > thresh[1]] = 1

    # Cluster statistic: Summed absolute z-score
    z_x = stats.zscore(x, axis=None)

    def stat_fun(x):
        return np.sum(np.abs(x))

    # Find clusters for each permutation, including the empirical data
    clust_labels = np.full(thresh_x.shape, np.nan)
    cluster_stats = []
    n_perm = surr.size[-1]
    for i_perm in range(n_perm + 1):
        # Find the clusters
        c_labs = measure.label(thresh_x[i_perm, ...])
        clust_labels[i_perm, ...] = c_labs
        # Get the cluster stat for each cluster
        perm_cluster_stat = []
        labels = clust_labels[i_perm, ...]
        for i_clust in range(1, int(np.max(labels)) + 1):
            # Select the z-values in the cluster
            x = z_x[i_perm, ...]
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

    return clust_stat_info
