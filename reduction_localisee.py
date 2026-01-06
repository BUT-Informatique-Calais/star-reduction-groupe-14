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

# si l'image est en couleur, prendre seulement le premier canal
if Ioriginal.ndim == 3:
    Ioriginal = Ioriginal[:, :, 0]

print(f"Image originale chargée: {Ioriginal.shape}")

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

print(f"Masque d'étoiles chargé: {M_raw.shape}")
print(f"Min M_raw: {M_raw.min()}, Max M_raw: {M_raw.max()}")

# 2. Créer un masque lissé avec flou gaussien
M = ndimage.gaussian_filter(M_raw.astype(np.float32) / 255.0, sigma=2.0)

print("Masque adouci avec flou gaussien")
print(f"Min M: {M.min()}, Max M: {M.max()}")

# 3. Calculer l'image finale par interpolation
# If inal = (M × Ierode) + ((1 - M) × Ioriginal)
Ioriginal_float = Ioriginal.astype(np.float32) / 255.0
Ifinal = (M * Ierode) + ((1 - M) * Ioriginal_float)

print("Image finale calculée par interpolation")

# sauvegarder le masque lissé
M_save = (M * 255).astype(np.uint8)
plt.imsave('./results/masque_lisse.png', M_save, cmap='gray')
print("Masque lissé sauvegardé: ./results/masque_lisse.png")

# sauvegarder l'image finale
Ifinal_save = (Ifinal * 255).astype(np.uint8)
plt.imsave('./results/image_finale.png', Ifinal_save, cmap='gray')
hdu_final = fits.PrimaryHDU(Ifinal_save)
hdu_final.writeto('./results/image_finale.fits', overwrite=True)
print("Image finale sauvegardée: ./results/image_finale.png")
print("Image finale FITS sauvegardée: ./results/image_finale.fits")

# fermer les fichiers
hdul_mask.close()
