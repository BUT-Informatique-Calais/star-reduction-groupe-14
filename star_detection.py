"""
Star detection module using DAOStarFinder
"""

from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def detect_stars(data, fwhm=3.0, threshold_sigma=5.5, radius=3.5):
    """
    Detect stars in an image and return a binary mask.

    Parameters:

    data: 2D numpy array (grayscale image)

    fwhm: Full Width at Half Maximum of the stars (default: 3.0)

    threshold_sigma: number of standard deviations above the background to detect a star (default: 5.5)

    radius: radius of the circles in the mask for each star (default: 3.5)

    Returns:

    mask: 2D numpy array with 255 for stars, 0 elsewhere

    sources: table of detected stars (or None if none found)
    """
    # If color image, convert to grayscale
    if data.ndim == 3:
        if data.shape[0] == 3:
            data_gray = data[0]
        elif data.shape[2] == 3:
            data_gray = np.mean(data, axis=2)
        else:
            data_gray = data[:, :, 0]
    else:
        data_gray = data

    # Calculate background statistics
    mean, median, std = sigma_clipped_stats(data_gray, sigma=3.0)

    # Create the detector
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)

    # Find the stars
    sources = daofind(data_gray - median)

    # Create an empty mask
    mask = np.zeros(data_gray.shape, dtype=np.uint8)

    if sources is None or len(sources) == 0:
        return mask, None

    # For each detected star, draw a circle in the mask
    for star in sources:
        x = int(star['xcentroid'])
        y = int(star['ycentroid'])

        if 0 <= y < data_gray.shape[0] and 0 <= x < data_gray.shape[1]:
            range_size = int(radius) + 1
            for dy in range(-range_size, range_size + 1):
                for dx in range(-range_size, range_size + 1):
                    new_y = y + dy
                    new_x = x + dx
                    # Vérifier si dans les limites et dans le rayon
                    if (0 <= new_y < data_gray.shape[0] and 0 <= new_x < data_gray.shape[1] and
                        np.sqrt(dx**2 + dy**2) <= radius):
                        mask[new_y, new_x] = 255

    return mask, sources


def smooth_mask(mask, sigma=2.0, threshold=0.1):
    """
    Apply a Gaussian blur to the mask for smooth transitions.

    Parameters:

    mask: binary mask (0-255)

    sigma: standard deviation of the Gaussian blur

    threshold: minimum threshold to keep values

    Returns:

    smoothed mask normalized between 0 and 1
    """
    # Normalize between 0 and 1
    mask_norm = mask.astype(np.float32) / 255.0

    # Apply Gaussian blur
    mask_smooth = ndimage.gaussian_filter(mask_norm, sigma=sigma)

    # Apply threshold to remove very low values
    mask_smooth = np.where(mask_smooth > threshold, mask_smooth, 0)

    return mask_smooth