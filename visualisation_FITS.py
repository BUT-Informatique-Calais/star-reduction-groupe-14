from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


def view_fits(filename):
    """Visualiser un fichier FITS"""
    hdul = fits.open(filename)
    data = hdul[0].data

    plt.figure(figsize=(10, 8))

    if data.ndim == 3:
        # Image couleur - transposer si nécessaire
        if data.shape[0] == 3:
            data = np.transpose(data, (1, 2, 0))
        # Normaliser pour l'affichage
        data_norm = (data - data.min()) / (data.max() - data.min())
        plt.imshow(data_norm)
    else:
        # Image monochrome
        plt.imshow(data, cmap='gray')

    plt.colorbar()
    plt.title(f'FITS: {filename}')
    plt.show()

    hdul.close()


# Utilisation
view_fits('./results/image_finale.fits')
view_fits('./examples/test_M31_linear.fits')
