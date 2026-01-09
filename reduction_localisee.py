"""
Module de réduction localisée d'étoiles
Implémente la formule: I_final = (M × I_erode) + ((1 - M) × I_original)
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def compute_final_image(original, eroded, mask):
    """
    Calcule l'image finale en combinant l'originale et l'érodée via le masque.

    Formule: I_final = (M × I_erode) + ((1 - M) × I_original)

    Où le masque est blanc (M=1) -> on affiche l'image érodée (étoiles réduites)
    Où le masque est noir (M=0) -> on affiche l'image originale (fond préservé)

    Paramètres:
    - original: image originale normalisée (0-1)
    - eroded: image érodée normalisée (0-1)
    - mask: masque lissé normalisé (0-1), blanc = étoiles

    Retourne:
    - image finale combinée
    """
    # Si l'image est en couleur et le masque en 2D, étendre le masque
    if original.ndim == 3 and mask.ndim == 2:
        mask_3d = np.stack([mask, mask, mask], axis=2)
    else:
        mask_3d = mask

    # Appliquer la formule de combinaison
    # Où M=1 (étoile) -> on prend l'érodée
    # Où M=0 (fond) -> on prend l'originale
    final = (mask_3d * eroded) + ((1 - mask_3d) * original)

    return final


def process_star_reduction(original, eroded, star_mask, gauss_sigma=2.0, mask_threshold=0.1):
    """
    Pipeline complet de réduction d'étoiles.

    Paramètres:
    - original: image originale normalisée (0-1)
    - eroded: image érodée normalisée (0-1)
    - star_mask: masque binaire des étoiles (0-255)
    - gauss_sigma: sigma du flou gaussien pour lisser le masque
    - mask_threshold: seuil minimum pour le masque

    Retourne:
    - image finale, masque lissé
    """
    # Lisser le masque pour des transitions douces
    mask_smooth = ndimage.gaussian_filter(star_mask.astype(np.float32) / 255.0, sigma=gauss_sigma)

    # Appliquer le seuil
    mask_smooth = np.where(mask_smooth > mask_threshold, mask_smooth, 0)

    # Calculer l'image finale
    final = compute_final_image(original, eroded, mask_smooth)

    return final, mask_smooth
