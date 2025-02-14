import numpy as np
import torch


def sample_data_imputation(
    range_center, sample_size, num_missing_feats, col_range, is_categorical=False
):
    """
    Sample points with imputation for missing feature values.

    For a given range_center (which can be 1D or 2D), the function checks whether the last
    `num_missing_feats` columns contain exactly np.nan for each row (or for the 1D vector).
    If not, those entries are masked with np.nan.

    - For a 1D input, the function generates `sample_size` distinct imputed points.
    - For a 2D input, the function generates `sample_size` distinct imputed points for each row.

    The missing features (i.e. the last `num_missing_feats` columns) are imputed as follows:
      - If is_categorical is False (default), sample uniformly from the given ranges.
      - If is_categorical is True, sample an integer from low to high (inclusive) for each missing column.

    Args:
        range_center (np.ndarray or torch.Tensor): The data point(s) to be imputed.
            Can be a 1D array or a 2D array (with rows representing data points).
        sample_size (int): The number of distinct imputed points to generate.
            For a 2D input, this is the number of imputed candidates per row.
        num_missing_feats (int): The number of feature columns that should be treated as missing.
        col_range (list of tuple): A list of (min, max) tuples for each missing column.
        is_categorical (bool, optional): Whether to sample categorical (integer) values.
            If True, an integer is sampled from low to high (inclusive) for each missing feature.

    Returns:
        list:
            - For a 1D input, returns a list of imputed points (each as a torch.Tensor of dtype torch.float32).
            - For a 2D input, returns a list of lists, where each inner list contains `sample_size` imputed points
              (each as a torch.Tensor) corresponding to a row from range_center.
    """
    # Convert torch.Tensor to numpy array if necessary.
    if isinstance(range_center, torch.Tensor):
        range_center = range_center.numpy()

    # Precompute lower and upper bounds for missing features.
    # Note: these could be strings, so in the categorical branch we cast them to int.
    lows = np.array([rng[0] for rng in col_range])
    highs = np.array([rng[1] for rng in col_range])

    def impute_missing(candidate):
        """Helper to impute the missing features in the candidate vector."""
        if is_categorical:
            # Cast low and high to int before sampling.
            imputed = np.array(
                [
                    np.random.randint(int(low), int(high) + 1)
                    for low, high in zip(lows, highs)
                ]
            )
        else:
            # Sample continuous values uniformly.
            imputed = np.random.uniform(low=lows, high=highs)
        candidate[-num_missing_feats:] = imputed
        return candidate

    if range_center.ndim == 1:
        # 1D case: Ensure the last num_missing_feats entries are np.nan.
        if np.sum(np.isnan(range_center[-num_missing_feats:])) != num_missing_feats:
            range_center[-num_missing_feats:] = np.nan

        candidate_set = set()
        sampled_points = []
        while len(sampled_points) < sample_size:
            candidate = range_center.copy()
            candidate = impute_missing(candidate)
            candidate_tuple = tuple(candidate.tolist())
            if candidate_tuple not in candidate_set:
                candidate_set.add(candidate_tuple)
                sampled_points.append(torch.tensor(candidate, dtype=torch.float32))
        return sampled_points

    elif range_center.ndim == 2:
        # 2D case: For each row, generate sample_size distinct imputations.
        imputed_points = []  # list of lists: one list per row.
        for i in range(range_center.shape[0]):
            base_row = range_center[i, :].copy()  # Work on a copy of the row.
            # Ensure the last num_missing_feats entries are np.nan.
            if np.sum(np.isnan(base_row[-num_missing_feats:])) != num_missing_feats:
                base_row[-num_missing_feats:] = np.nan

            candidate_set = set()  # distinct candidates for the current row.
            row_samples = []
            while len(row_samples) < sample_size:
                candidate = base_row.copy()
                candidate = impute_missing(candidate)
                row_tuple = tuple(candidate.tolist())
                if row_tuple not in candidate_set:
                    candidate_set.add(row_tuple)
                    row_samples.append(torch.tensor(candidate, dtype=torch.float32))
            imputed_points.append(row_samples)
        return imputed_points

    else:
        raise ValueError("range_center must be either 1D or 2D.")
