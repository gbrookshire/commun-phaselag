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
