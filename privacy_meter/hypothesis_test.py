from typing import List

import numpy as np
from scipy.stats import norm


########################################################################################################################
# HYPOTHESIS TEST: THRESHOLDING
########################################################################################################################


def threshold_func(
        distribution: List[float],
        alpha: float
) -> float:
    """
    Function that returns the threshold as the alpha quantile of
    the provided distribution.

    Args:
        distribution: Sequence of values that form the distribution from which
        the threshold is computed.
        alpha: Quantile value that will be used to obtain the threshold from the
            distribution.

    Returns:
        threshold: alpha quantile of the provided distribution.
    """
    threshold = np.quantile(distribution, q=alpha, interpolation='lower')
    return threshold


########################################################################################################################
# HYPOTHESIS TEST: GAUSSIAN THRESHOLDING
########################################################################################################################


def gaussian_threshold_func(
        distribution: List[float],
        alpha: float
) -> float:
    """
    Function that returns the threshold as the alpha quantile of
    a Gaussian curve fit over the provided distribution.

    Args:
        distribution: Sequence of values that form the distribution from which
        the threshold is computed.
        alpha: Quantile value that will be used to obtain the threshold from the
            distribution.

    Returns:
        threshold: alpha quantile of the provided distribution.
    """
    loc, scale = norm.fit(distribution)
    threshold = norm.ppf(alpha, loc=loc, scale=scale)
    return threshold


########################################################################################################################
# HYPOTHESIS TEST: LINEAR INTERPOLATION THRESHOLDING
########################################################################################################################


def linear_itp_threshold_func(
        distribution: List[float],
        alpha: float
) -> float:
    """
    Function that returns the threshold as the alpha quantile of
    a linear interpolation curve fit over the provided distribution.

    Args:
        distribution: Sequence of values that form the distribution from which
        the threshold is computed.
        alpha: Quantile value that will be used to obtain the threshold from the
            distribution.

    Returns:
        threshold: alpha quantile of the provided distribution.
    """
    len_dist = len(distribution)
    if alpha >= np.float(1/len_dist):
        if alpha < 1 - np.float(1/len_dist):
            threshold = np.quantile(distribution, q=alpha, interpolation='linear')
        elif alpha == 1:
            threshold = 1000
        else:
            threshold = (1 - np.float(1/len_dist))/(1-alpha) * np.quantile(distribution,
                                                                           q=1-np.float(1/len_dist),
                                                                           interpolation='linear')
    else:
        threshold = np.float(len(distribution)) * alpha * np.quantile(distribution,
                                                                      q=np.float(1/len_dist),
                                                                      interpolation='linear')

    if threshold <= float(0.000001):
        threshold = -0.1

    return threshold
