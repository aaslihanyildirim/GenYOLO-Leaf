# api/postprocess.py

import numpy as np
import cv2


def _ensure_mask_shape(mask, image_shape):
    """
    Maskeyi image shape ile zorla uyumlu hale getirir
    """
    H, W = image_shape[:2]

    if mask.shape[0] != H or mask.shape[1] != W:
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)

    return mask


def create_overlay(image, masks, alpha=0.5):
    overlay = image.copy()

    if masks is None or len(masks) == 0:
        return overlay

    for mask in masks:
        mask = _ensure_mask_shape(mask, image.shape)

        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        colored_mask = np.zeros_like(image, dtype=np.uint8)

        colored_mask[mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)

    return overlay


def extract_binary_mask(masks, image_shape):
    H, W = image_shape[:2]
    binary_mask = np.zeros((H, W), dtype=np.uint8)

    if masks is None or len(masks) == 0:
        return binary_mask

    for mask in masks:
        mask = _ensure_mask_shape(mask, image_shape)
        binary_mask[mask] = 255

    return binary_mask

