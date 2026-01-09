"""
Module d'érosion d'image pour réduction d'étoiles
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def apply_erosion(data, kernel_size=3, iterations=2):
    """
    Applique une érosion morphologique sur l'image.

    Paramètres:
    - data: numpy array (image normalisée entre 0 et 1)
    - kernel_size: taille du noyau d'érosion (défaut: 3)
    - iterations: nombre d'itérations d'érosion (défaut: 2)

    Retourne:
    - image érodée normalisée entre 0 et 1
    """
    # Convertir en uint8 pour OpenCV
    data_uint8 = (data * 255).astype(np.uint8)

    # Créer le noyau
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Appliquer l'érosion
    eroded = cv.erode(data_uint8, kernel, iterations=iterations)

    # Reconvertir en float normalisé
    eroded_norm = eroded.astype(np.float32) / 255.0

    return eroded_norm


def normalize_image(data):
    """
    Normalise une image entre 0 et 1.

    Paramètres:
    - data: numpy array

    Retourne:
    - image normalisée entre 0 et 1
    """
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)

    if data_max > data_min:
        return (data - data_min) / (data_max - data_min)
    else:
        return data


def prepare_image(data):
    """
    Prépare l'image pour le traitement (normalisation + transposition si nécessaire).

    Paramètres:
    - data: numpy array brut du fichier FITS

    Retourne:
    - image normalisée avec les dimensions correctes
    """
    # Gérer les images couleur
    if data.ndim == 3:
        if data.shape[0] == 3:  # (3, height, width) -> (height, width, 3)
            data = np.transpose(data, (1, 2, 0))

    # Normaliser
    return normalize_image(data)
