import cv2
import numpy as np


def resize_bio7_to_raw(bio7, raw_shape):
    """
    Resize prediction masks to full-resolution raw TIFF shape if needed.
    Uses nearest-neighbor interpolation to preserve labels/masks.
    """

    h, w = raw_shape[:2]
    out = {}

    for k, mask in bio7.items():
        if mask.shape == (h, w):
            out[k] = mask
        else:
            out[k] = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

    return out