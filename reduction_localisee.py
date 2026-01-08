from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

# Load the original FITS image
hdul_original = fits.open('./examples/test_M31_linear.fits')
data_original = hdul_original[0].data

# Handle the dimensions (color or monochrome)
if data_original.ndim == 3 and data_original.shape[0] == 3:
    # Transpose from (3, height, width) to (height, width, 3)
    Ioriginal = np.transpose(data_original, (1, 2, 0))
elif data_original.ndim == 2:
    # Monochrome image – duplicate it on 3 channels to standardize
    Ioriginal = np.stack([data_original, data_original, data_original], axis=2)
else:
    Ioriginal = data_original

# Load the eroded FITS image
hdul_eroded = fits.open('./results/eroded.fits')
Ierode = hdul_eroded[0].data

# Normalize the data between 0 and 1
Ioriginal_norm = (Ioriginal - Ioriginal.min()) / (Ioriginal.max() - Ioriginal.min())
Ierode_norm = (Ierode - Ierode.min()) / (Ierode.max() - Ierode.min())


# Load the star mask
hdul_mask = fits.open('./results/star_mask.fits')
M_raw = hdul_mask[0].data


# Create a smooth mask with Gaussian blur
M = ndimage.gaussian_filter(M_raw.astype(np.float32) / 255.0, sigma=2.0) #sigma : contrôle l'étendue de l'effet du masque autour des étoiles
M = np.where(M > 0.5, M, 0)  # Remove very low values

# Extend the mask to the 3 color channels if necessary
if Ioriginal_norm.ndim == 3:
    M = np.stack([M, M, M], axis=2)

# Compute the final image by interpolation
Ifinal = (M * Ierode_norm) + ((1 - M) * Ioriginal_norm)


# Save the final image as FITS
fits.writeto('./results/image_finale.fits', Ifinal, overwrite=True)

# Save the final image as PNG
if Ifinal.ndim == 3:
    plt.imsave('./results/image_finale.png', Ifinal)
else:
    plt.imsave('./results/image_finale.png', Ifinal)

# Close the files
hdul_original.close()
hdul_eroded.close()
hdul_mask.close()