# api/utils.py

import numpy as np


def masks_from_results(results, threshold=0.5):
    """
    Ultralytics Results objesinden boolean mask üretir.

    Args:
        results: Ultralytics Results (results[0])
        threshold (float): mask threshold

    Returns:
        masks (np.ndarray | None): (N, H, W) boolean
    """

    if results.masks is None:
        return None

    # Tensor → NumPy
    masks = results.masks.data.cpu().numpy()

    # Threshold → boolean
    masks = masks > threshold

    return masks


