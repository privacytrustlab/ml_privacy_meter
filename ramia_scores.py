import numpy as np


def get_topk(array, k):
    id = (-array).argsort(axis=1)[:, :k]
    return array[np.arange(array.shape[0])[:, None], id]

def get_bottomk(array, k):
    id = array.argsort(axis=1)[:, :k]
    return array[np.arange(array.shape[0])[:, None], id]


def trim_mia_scores(mia_scores: np.ndarray, trim_ratio: float, trim_direction: str) -> np.ndarray:
    """
    Trim the MIA scores to remove the samples that are not members.

    Args:
        mia_scores (np.ndarray): The MIA scores.
        trim_ratio (float): The ratio of samples to trim.
        trim_direction (str): The direction to trim the samples. Can be "none", "top", or "bottom".

    Returns:
        np.ndarray: The trimmed MIA score means.
    """
    if trim_direction not in ["none", "top", "bottom"]:
        raise ValueError(f"Invalid trim_direction: {trim_direction}")

    if trim_direction == "none":
        return mia_scores.mean(axis=1)

    if trim_direction == "top":
        return get_bottomk(mia_scores, int(trim_ratio * mia_scores.shape[1])).mean(axis=1)

    if trim_direction == "bottom":
        return get_topk(mia_scores, int(trim_ratio * mia_scores.shape[1])).mean(axis=1)
