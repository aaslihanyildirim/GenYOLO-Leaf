# metrics.py
import numpy as np
import cv2

EPS = 1e-6


# -------------------------------------------------
# GT MASK HANDLING
# -------------------------------------------------

def normalize_gt_mask(gt_mask, target_shape):
    """
    GT mask'i güvenli hale getirir:
    - RGB instance mask
    - grayscale / binary mask
    - farklı boyutlar
    """
    if gt_mask is None:
        raise ValueError("GT mask is None")

    # Eğer RGB ise olduğu gibi al
    if gt_mask.ndim == 3 and gt_mask.shape[2] == 3:
        norm_mask = gt_mask

    # Grayscale / binary ise -> RGB'ye çevir
    elif gt_mask.ndim == 2:
        norm_mask = np.stack([gt_mask]*3, axis=-1)

    else:
        raise ValueError(f"Unsupported GT mask shape: {gt_mask.shape}")

    # Resize (NEAREST – instance bozulmasın)
    h, w = target_shape
    norm_mask = cv2.resize(
        norm_mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return norm_mask


def extract_gt_instances(gt_rgb_mask):
    """
    Renkli GT mask'ten instance'ları çıkarır.
    Her farklı renk = 1 instance
    """
    instances = []

    flat = gt_rgb_mask.reshape(-1, 3)
    colors = np.unique(flat, axis=0)

    for color in colors:
        # siyah arkaplanı atla
        if np.all(color == [0, 0, 0]):
            continue

        mask = np.all(gt_rgb_mask == color, axis=-1).astype(np.uint8)

        if mask.sum() > 0:
            instances.append(mask)

    return instances


# -------------------------------------------------
# IOU
# -------------------------------------------------

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + EPS)


# -------------------------------------------------
# MAIN METRIC FUNCTION
# -------------------------------------------------

def evaluate_instance_segmentation(gt_mask, pred_masks, iou_threshold=0.5):
    """
    gt_mask      : numpy (H,W) or (H,W,3)
    pred_masks   : list of (H,W) binary masks
    """

    if len(pred_masks) == 0:
        return {
            "mAP50": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "TP": 0,
            "FP": 0,
            "FN": 0,
        }

    h, w = pred_masks[0].shape
    gt_rgb = normalize_gt_mask(gt_mask, (h, w))
    gt_instances = extract_gt_instances(gt_rgb)

    matched_gt = set()
    TP = 0
    FP = 0

    for pred in pred_masks:
        best_iou = 0
        best_gt = -1

        for i, gt in enumerate(gt_instances):
            if i in matched_gt:
                continue

            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_iou >= iou_threshold:
            TP += 1
            matched_gt.add(best_gt)
        else:
            FP += 1

    FN = len(gt_instances) - TP

    precision = TP / (TP + FP + EPS)
    recall = TP / (TP + FN + EPS)
    mAP50 = precision  # single threshold approx

    return {
        "mAP50": float(mAP50),
        "precision": float(precision),
        "recall": float(recall),
        "TP": TP,
        "FP": FP,
        "FN": FN,
    }
