from cmath import inf
from typing import List

import numpy as np
from scipy.stats import norm, lognorm
import math


########################################################################################################################
# HYPOTHESIS TEST: THRESHOLDING
########################################################################################################################


def threshold_func(
        distribution: List[float],
        alpha:  List[float],
        **kwargs
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
    threshold = np.quantile(distribution, q=alpha, interpolation='lower',**kwargs)
    return threshold


########################################################################################################################
# HYPOTHESIS TEST: LINEAR INTERPOLATION THRESHOLDING
########################################################################################################################


def linear_itp_threshold_func(
        distribution: List[float],
        alpha: List[float],
        signal_min=0,
        signal_max=1000,
        **kwargs
) -> float:
    """
    Function that returns the threshold as the alpha quantile of
    a linear interpolation curve fit over the provided distribution.
    Args:
        distribution: Sequence of values that form the distribution from which
        the threshold is computed. (Here we only consider positive signal values.)
        alpha: Quantile value that will be used to obtain the threshold from the
            distribution.
        signal_min: minimum of all possible signal values, default value is zero
        signal_max: maximum of all possible signal values, default vaule is 1000
    Returns:
        threshold: alpha quantile of the provided distribution.
    """
    
    if len(distribution.shape)>1:
        # for reference attacks
        distribution = np.concatenate([distribution,np.repeat(signal_min,distribution.shape[0]).reshape(-1,1)],axis=1)
        distribution = np.concatenate([distribution,np.repeat(signal_max,distribution.shape[0]).reshape(-1,1)],axis=1)
    else:
        distribution = np.append(distribution, signal_min)
        distribution = np.append(distribution, signal_max)
    threshold = np.quantile(distribution, q=alpha, method='linear',**kwargs)

    return threshold

########################################################################################################################
# HYPOTHESIS TEST: LOGIT RESCALE THRESHOLDING
########################################################################################################################


def logit_rescale_threshold_func(
        distribution: List[float],
        alpha: List[float],
        **kwargs,
) -> float:
    """
    Function that returns the threshold as the alpha quantile of a Gaussian fit
    over logit rescaling transform of the provided distribution
    Args:
        distribution: Sequence of values that form the distribution from which
        the threshold is computed. (Here we only consider positive signal values.)
        alpha: Quantile value that will be used to obtain the threshold from the
            distribution.
    Returns:
        threshold: alpha quantile of the provided distribution.
    """

    distribution = distribution+0.000001 # avoid nan
    distribution = np.log(np.divide(np.exp(- distribution), (1 - np.exp(- distribution))))
  
    if len(distribution.shape)>1:
        parameters = np.array([norm.fit(distribution[i]) for i in range(distribution.shape[0])])
        num_threshold = alpha.shape[0]
        num_points = distribution.shape[0]
        loc = parameters[:,0].reshape(-1,1).repeat(num_threshold,1)
        scale = parameters[:,1].reshape(-1,1).repeat(num_threshold,1)
        alpha = np.array(alpha).reshape(-1,1).repeat(num_points,1)
        threshold = norm.ppf(1-np.array(alpha),loc=loc.T,scale=scale.T)
    else:
        print('none')
        print(np.sum(distribution==-np.inf))
        loc,scale = norm.fit(distribution)
        threshold = norm.ppf(1 - np.array(alpha), loc=loc, scale=scale)
    
    threshold = np.log(np.exp(threshold) + 1) - threshold
    return threshold

########################################################################################################################
# HYPOTHESIS TEST: GAUSSIAN THRESHOLDING
########################################################################################################################


def gaussian_threshold_func(
        distribution: List[float],
        alpha: List[float],
        **kwargs,
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
    loc, scale = norm.fit(distribution,**kwargs,)
    threshold = norm.ppf(alpha, loc=loc, scale=scale)
    return threshold

########################################################################################################################
# HYPOTHESIS TEST: MIN LINEAR LOGIT RESCALE THRESHOLDING
########################################################################################################################


def min_linear_logit_threshold_func(
        distribution: List[float],
        alpha: List[float],
        signal_min=0,
        signal_max=1000,
        **kwargs,
) -> float:
    """
    Function that returns the threshold as the minimum of 1) alpha quantile of
    a linear interpolation curve fit over the provided distribution, and 2) alpha 
    quantile of a Gaussian fit over logit rescaling transform of the provided 
    distribution
    Args:
        distribution: Sequence of values that form the distribution from which
        the threshold is computed. (Here we only consider positive signal values.)
        alpha: Quantile value that will be used to obtain the threshold from the
            distribution.
        signal_min: minimum of all possible signal values, default value is zero
        signal_max: maximum of all possible signal values, default vaule is 1000
    Returns:
        threshold: alpha quantile of the provided distribution.
    """
    # distribution_linear = np.append(distribution, signal_min)
    # distribution_linear = np.append(distribution_linear, signal_max)
    # threshold_linear = np.quantile(distribution_linear, q=alpha, interpolation='linear',**kwargs,)

    threshold_linear = linear_itp_threshold_func(distribution,alpha,signal_min,signal_max,**kwargs)
    threshold_logit = logit_rescale_threshold_func(distribution,alpha,**kwargs)
    

    threshold = np.minimum(threshold_logit, threshold_linear)

    return threshold