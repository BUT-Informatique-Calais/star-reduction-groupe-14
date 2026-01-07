# Groupe 14
# created on 04/01/2026
# last update : 06/01/2026

#modiifer quelque commentaire a 8h35 Alex F

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

# Charger l'image originale FITS
hdul_original = fits.open('./examples/test_M31_linear.fits')
data_original = hdul_original[0].data

# Gérer les dimensions (couleur ou monochrome)
if data_original.ndim == 3 and data_original.shape[0] == 3:
    # Transposer de (3, height, width) vers (height, width, 3)
    Ioriginal = np.transpose(data_original, (1, 2, 0))
elif data_original.ndim == 2:
    # Image monochrome - la dupliquer sur 3 canaux pour uniformiser
    Ioriginal = np.stack([data_original, data_original, data_original], axis=2)
else:
    Ioriginal = data_original

# Charger l'image érodée FITS
hdul_eroded = fits.open('./results/eroded.fits')
Ierode = hdul_eroded[0].data

# Normaliser les données entre 0 et 1
Ioriginal_norm = (Ioriginal - Ioriginal.min()) / (Ioriginal.max() - Ioriginal.min())
Ierode_norm = (Ierode - Ierode.min()) / (Ierode.max() - Ierode.min())


# charger le masque d'étoiles
hdul_mask = fits.open('./results/star_mask.fits')
M_raw = hdul_mask[0].data


# Créer un masque lissé avec flou gaussien
M = ndimage.gaussian_filter(M_raw.astype(np.float32), sigma=6.0)
M = M / M.max()  # Normaliser entre 0 et 1
M = np.where(M > 0.05, M, 0)

# Appliquer un deuxième lissage pour plus de douceur
M = ndimage.gaussian_filter(M, sigma=2.0)

# Étendre le masque pour les 3 canaux couleur si nécessaire
if Ioriginal_norm.ndim == 3:
    M = np.stack([M, M, M], axis=2)

# Calculer l'image finale par interpolation
Ifinal = (M * Ierode_norm) + ((1 - M) * Ioriginal_norm)


# Save the final image as FITS
fits.writeto('./results/image_finale.fits', Ifinal, overwrite=True)

# Save the final image as PNG
if Ifinal.ndim == 3:
    plt.imsave('./results/image_finale.png', Ifinal)
else:
    plt.imsave('./results/image_finale.png', Ifinal)

# Fermer les fichiers
hdul_original.close()
hdul_eroded.close()
hdul_mask.close()