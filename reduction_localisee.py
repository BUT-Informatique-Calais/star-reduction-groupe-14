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

# convertir en RGB (3 canaux) si nécessaire
if Ioriginal.ndim == 2:
    Ioriginal = np.stack([Ioriginal, Ioriginal, Ioriginal], axis=2)
elif Ioriginal.shape[2] == 4:  # Si RGBA, supprimer le canal alpha
    Ioriginal = Ioriginal[:, :, :3]

# charger l'image érodée
eroded = Image.open('./results/eroded.png')
Ierode = np.array(eroded)

# convertir en RGB (3 canaux) si nécessaire
if Ierode.ndim == 2:
    Ierode = np.stack([Ierode, Ierode, Ierode], axis=2)
elif Ierode.shape[2] == 4:  # Si RGBA, supprimer le canal alpha
    Ierode = Ierode[:, :, :3]

# normaliser Ierode entre 0 et 1
Ierode = Ierode.astype(np.float32) / 255.0

# charger le masque d'étoiles
hdul_mask = fits.open('./results/star_mask.fits')
M_raw = hdul_mask[0].data


# 2. Créer un masque lissé avec flou gaussien
M = ndimage.gaussian_filter(M_raw.astype(np.float32) / 255.0, sigma=2.0) #sigma : contrôle l'étendue de l'effet du masque autour des étoiles
M = np.where(M > 0.5, M, 0)  # Supprime les valeurs très faibles
# Étendre le masque pour les 3 canaux couleur
M = np.stack([M, M, M], axis=2)

# 3. Calculer l'image finale par interpolation
Ioriginal_float = Ioriginal.astype(np.float32) / 255.0
Ifinal = (M * Ierode) + ((1 - M) * Ioriginal_float)


# sauvegarder le masque lissé (prendre un seul canal)
M_save = (M[:,:,0] * 255).astype(np.uint8)
plt.imsave('./results/masque_lisse.png', M_save)

# sauvegarder l'image finale
Ifinal_save = (Ifinal * 255).astype(np.uint8)
plt.imsave('./results/image_finale.png', Ifinal_save)

hdul_mask.close()
