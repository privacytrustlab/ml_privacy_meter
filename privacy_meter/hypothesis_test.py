from typing import List

import numpy as np


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
