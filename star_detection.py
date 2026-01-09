"""
Module de détection d'étoiles utilisant DAOStarFinder
"""

from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def detect_stars(data, fwhm=3.0, threshold_sigma=5.5, radius=3.5):
    """
    Détecte les étoiles dans une image et retourne un masque binaire.

    Paramètres:
    - data: numpy array 2D (image en niveaux de gris)
    - fwhm: Full Width at Half Maximum des étoiles (défaut: 3.0)
    - threshold_sigma: nombre d'écarts-types au-dessus du fond pour détecter une étoile (défaut: 5.5)
    - radius: rayon des cercles dans le masque pour chaque étoile (défaut: 3.5)

    Retourne:
    - mask: numpy array 2D avec 255 pour les étoiles, 0 ailleurs
    - sources: table des étoiles détectées (ou None si aucune)
    """
    # Si image couleur, convertir en niveaux de gris
    if data.ndim == 3:
        if data.shape[0] == 3:
            data_gray = data[0]
        elif data.shape[2] == 3:
            data_gray = np.mean(data, axis=2)
        else:
            data_gray = data[:, :, 0]
    else:
        data_gray = data

    # Calculer les statistiques du fond
    mean, median, std = sigma_clipped_stats(data_gray, sigma=3.0)

    # Créer le détecteur
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)

    # Trouver les étoiles
    sources = daofind(data_gray - median)

    # Créer un masque vide
    mask = np.zeros(data_gray.shape, dtype=np.uint8)

    if sources is None or len(sources) == 0:
        return mask, None

    # Pour chaque étoile détectée, dessiner un cercle dans le masque
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
    Applique un flou gaussien au masque pour des transitions douces.

    Paramètres:
    - mask: masque binaire (0-255)
    - sigma: écart-type du flou gaussien
    - threshold: seuil minimum pour garder les valeurs

    Retourne:
    - masque lissé normalisé entre 0 et 1
    """
    # Normaliser entre 0 et 1
    mask_norm = mask.astype(np.float32) / 255.0

    # Appliquer le flou gaussien
    mask_smooth = ndimage.gaussian_filter(mask_norm, sigma=sigma)

    # Appliquer le seuil pour supprimer les valeurs très faibles
    mask_smooth = np.where(mask_smooth > threshold, mask_smooth, 0)

    return mask_smooth