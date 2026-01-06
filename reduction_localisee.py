# Groupe 14
# 04/01/2026
# Étape B : Réduction localisée

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

# charger l'image originale
original = Image.open('./results/original.png')
Ioriginal = np.array(original)

# si l'image est en couleur, prendre seulement le premier canal (prendre que le gris )
if Ioriginal.ndim == 3:
    Ioriginal = Ioriginal[:, :, 0] 


# charger l'image érodée
eroded = Image.open('./results/eroded.png')
Ierode = np.array(eroded)

# si l'image est en couleur, prendre seulement le premier canal
if Ierode.ndim == 3:
    Ierode = Ierode[:, :, 0]

# normaliser Ierode entre 0 et 1
Ierode = Ierode.astype(np.float32) / 255.0

# charger le masque d'étoiles
hdul_mask = fits.open('./results/star_mask.fits')
M_raw = hdul_mask[0].data


# 2. Créer un masque lissé avec flou gaussien
M = ndimage.gaussian_filter(M_raw.astype(np.float32) / 255.0, sigma=2.0)


# 3. Calculer l'image finale par interpolation
Ioriginal_float = Ioriginal.astype(np.float32) / 255.0
Ifinal = (M * Ierode) + ((1 - M) * Ioriginal_float)


# sauvegarder le masque lissé
M_save = (M * 255).astype(np.uint8)
plt.imsave('./results/masque_lisse.png', M_save, cmap='gray')

# sauvegarder l'image finale
Ifinal_save = (Ifinal * 255).astype(np.uint8)
plt.imsave('./results/image_finale.png', Ifinal_save, cmap='gray')
hdu_final = fits.PrimaryHDU(Ifinal_save)
hdu_final.writeto('./results/image_finale.fits', overwrite=True)

hdul_mask.close()
