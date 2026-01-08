from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


def view_fits(filename):
    # Open and read the FITS file
    hdul = fits.open(filename)
    data = hdul[0].data

    plt.figure(figsize=(10, 8))

    if data.ndim == 3:
        # Color image 
        if data.shape[0] == 3:
            data = np.transpose(data, (1, 2, 0))
        # Normalize for display
        data_norm = (data - data.min()) / (data.max() - data.min())
        plt.imshow(data_norm)
    else:
        # Monochrome image
        plt.imshow(data, cmap='gray')

    plt.colorbar()
    plt.title(f'FITS: {filename}')
    plt.show()

    hdul.close()


# Example usage
view_fits('./results/image_finale.fits')
view_fits('./examples/test_M31_linear.fits')
view_fits('./examples/HorseHead.fits')
view_fits('./examples/test_M31_raw.fits')
