from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def apply_erosion(data, kernel_size=3, iterations=2):
    """
    Apply morphological erosion on the image.

    Parameters:
    - data: numpy array (normalized image between 0 and 1)
    - kernel_size: size of the erosion kernel (default: 3)
    - iterations: number of erosion iterations (default: 2)

    Returns:
    - eroded image normalized between 0 and 1
    """
    # Convert to uint8 for OpenCV
    data_uint8 = (data * 255).astype(np.uint8)

    # Create the kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion
    eroded = cv.erode(data_uint8, kernel, iterations=iterations)

    # Convert back to normalized float
    eroded_norm = eroded.astype(np.float32) / 255.0

    return eroded_norm


def normalize_image(data):
    """
    Normalize an image between 0 and 1.

    Parameters:
    - data: numpy array

    Returns:
    - normalized image between 0 and 1
    """
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)

    if data_max > data_min:
        return (data - data_min) / (data_max - data_min)
    else:
        return data


def prepare_image(data):
    """
    Prepare the image for processing (normalization + transposition if needed).

    Parameters:
    - data: raw numpy array from FITS file

    Returns:
    - normalized image with correct dimensions
    """
    # Handle color images
    if data.ndim == 3:
        if data.shape[0] == 3:
            # If channels are first: (3, height, width)
            data = np.transpose(data, (1, 2, 0))
        # If already (height, width, 3), no change needed

    # Normalize
    return normalize_image(data)
