"""
Localized star reduction module
Implements the formula: I_final = (M × I_erode) + ((1 - M) × I_original)
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def compute_final_image(original, eroded, mask):
    """
    Compute the final image by combining original and eroded via the mask.

    Formula: I_final = (M × I_erode) + ((1 - M) × I_original)

    Where mask is white (M=1) -> display eroded image (reduced stars)
    Where mask is black (M=0) -> display original image (preserved background)

    Parameters:
    - original: normalized original image (0-1)
    - eroded: normalized eroded image (0-1)
    - mask: smoothed normalized mask (0-1), white = stars

    Returns:
    - combined final image
    """
    # If image is color and mask is 2D, expand mask
    if original.ndim == 3 and mask.ndim == 2:
        mask_3d = np.stack([mask, mask, mask], axis=2)
    else:
        mask_3d = mask

    # Apply combination formula
    # Where M=1 (star) -> take eroded
    # Where M=0 (background) -> take original
    final = (mask_3d * eroded) + ((1 - mask_3d) * original)

    return final


def process_star_reduction(original, eroded, star_mask, gauss_sigma=2.0, mask_threshold=0.1):
    """
    Complete star reduction pipeline.

    Parameters:
    - original: normalized original image (0-1)
    - eroded: normalized eroded image (0-1)
    - star_mask: binary mask of stars (0-255)
    - gauss_sigma: sigma of gaussian blur for smoothing mask
    - mask_threshold: minimum threshold for mask

    Returns:
    - final image, smoothed mask
    """
    # Smooth the mask for smooth transitions
    mask_smooth = ndimage.gaussian_filter(star_mask.astype(np.float32) / 255.0, sigma=gauss_sigma)

    # Apply threshold
    mask_smooth = np.where(mask_smooth > mask_threshold, mask_smooth, 0)

    # Compute final image
    final = compute_final_image(original, eroded, mask_smooth)

    return final, mask_smooth
