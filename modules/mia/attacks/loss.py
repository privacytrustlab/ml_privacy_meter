# import numpy as np

# def run_loss(target_signals: np.ndarray) -> np.ndarray:
#     """
#     Attack a target model using the LOSS attack.

#     Args:
#         target_signals (np.ndarray): Softmax value of all samples in the target model.

#     Returns:
#         np.ndarray: MIA score for all samples (a larger score indicates higher chance of being member).
#     """
#     mia_scores = -target_signals
#     return mia_scores