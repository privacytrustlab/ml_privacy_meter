import pdb

import numpy as np
import torch


def sample_l2(range_center, radius, sample_size):
    """
    Sample points within the L2 range using Gaussian sampling with a scale
    dynamically adjusted for the sample size.
    Args:
        range_center (np.ndarray): The center of the L2 ball (N-dimensional point).
        radius (float): The radius of the L2 ball.
        sample_size (int): The number of points to sample.
    Returns:
        list: Points sampled within the L2 ball.
    """
    if type(range_center) == torch.Tensor:
        range_center = range_center.numpy()

    sampled_points = []
    if range_center.shape[0] == 1:
        range_center = range_center.squeeze(0)

    # Adjust scale based on radius and data size
    scale = radius / np.sqrt(len(range_center.reshape(-1)))

    while len(sampled_points) < sample_size:
        # Sample a candidate point from a multivariate normal distribution
        candidate = np.random.normal(loc=range_center, scale=scale, size=range_center.shape)
        # Check if the candidate point lies within the L2 ball
        if np.linalg.norm(candidate - range_center) <= radius:
            sampled_points.append(torch.tensor(candidate, dtype=torch.float32))

    return sampled_points

# Test
# range_center = np.random.rand(1, 28,28,3)# Center of the 3D ball
# print(range_center.shape)
# radius = 3.0  # L2 radius
# sample_size = 10  # Number of points to sample
#
# samples = sample_l2_gaussian(range_center, radius, sample_size)
#
# print("Sampled Points:\n", samples.shape)
#
# # Verify distances
# distances = [np.linalg.norm(sample - range_center) for sample in samples]
# print("Distances from center:\n", distances)
# print("Maximum distance:", np.max(distances))  # Should be <= radius
