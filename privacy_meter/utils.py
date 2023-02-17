import numpy as np


def flatten_array(arr):
    """
    Utility function to recursively flatten a list of lists.
    Each element in the list can be another list, tuple, set, or np.ndarray,
    and can have variable sizes.

    Args:
        arr: List of lists

    Returns:
        Flattened 1D np.ndarray version of arr.
    """
    flat_array = []
    for item in arr:
        if isinstance(item, (list, tuple, set, np.ndarray)):
            flat_array.extend(flatten_array(item))
        else:
            flat_array.append(item)
    return np.array(flat_array)


def default_quantile():
    """Return the default fprs

    Returns:
        arr: Numpy array, indicating the default fprs
    """
    return np.logspace(-5, 0, 100)
