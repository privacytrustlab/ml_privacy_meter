import pdb

import numpy as np
import torchvision.transforms as T
import pickle

# Define a mapping from string names to torchvision augmentation functions
augmentation_mapping = {
    "horizontal_flip": T.RandomHorizontalFlip(p=1.0),
    "vertical_flip": T.RandomVerticalFlip(p=1.0),
    "rotate": T.RandomRotation(degrees=30),
}


def sample_geometric(range_center, transformations_list, sample_size):
    """
    Sample points in the geometric range of the range_center.

    Args:
        range_center (np.ndarray): The center of the range.
        transformations_list (list): A list of strings representing the transformations to apply.
        sample_size (int): The number of samples to generate.

    Returns:
        np.ndarray: The samples in the geometric range.
    """
    # Initialize the samples list
    if len(transformations_list) == sample_size - 1:
        samples = [range_center] # Include the range_center as the first sample
    else:
        samples = []

    # Apply each transformation to the range_center
    for transformation in transformations_list:
        if transformation in augmentation_mapping:
            samples.append(augmentation_mapping[transformation](range_center))
        else:
            raise ValueError(f"Invalid transformation: {transformation}")

    return np.array(samples)


# Test
# with open("data/cifar10.pkl", "rb") as f:
#     dataset = pickle.load(f)
# first_image, _ = dataset[0] # First image and label
# print("Original Image Size:", first_image.shape)
#
# # Transformations to test
# transformations_list = ["horizontal_flip", "vertical_flip", "rotate"]
#
# # Generate augmented samples
# samples = sample_geometric(first_image, transformations_list, sample_size=3)
# print("Augmented Samples Size:", samples.shape)
