# Groupe 14
# 04/01/2026

from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# ouvrir le fichier fits
fits_file = './examples/test_M31_linear.fits'
hdul = fits.open(fits_file)

# afficher les infos
hdul.info()
header = hdul[0].header
data = hdul[0].data

# si l'image est en couleur, on prend juste un canal
if data.ndim == 3:
    if data.shape[0] == 3:
        data = data[0]
    else:
        data = data[:, :, 0]

# calculer le fond du ciel (en ignorant les étoiles)
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

print(f"\nStatistiques de l'image:")
print(f"  Moyenne: {mean:.2f}")
print(f"  Médiane: {median:.2f}")
print(f"  Écart-type: {std:.2f}")

#détection des étoiles avec DAOStarFinder

# fwhm : largeur typique d'une étoile en pixels
fwhm = 3.0

# combien de fois plus brillant que le bruit pour être une étoile
threshold = 5.5

# créer le détecteur
daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)

# trouver les étoiles
sources = daofind(data - median)

print(f"\nNombre d'étoiles détectées: {len(sources)}")

# créer le masque (image noire au départ)
mask = np.zeros(data.shape, dtype=np.uint8)

# rayon du cercle (diamètre de 3px = rayon de 1.5px)
radius = 3.5

# pour chaque étoile, dessiner un cercle blanc
for star in sources:
    x = int(star['xcentroid'])
    y = int(star['ycentroid'])
    
    # vérifier que les coordonnées sont dans l'image
    if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
        # créer un cercle autour de l'étoile
        range_size = int(radius) + 1
        for dy in range(-range_size, range_size + 1):
            for dx in range(-range_size, range_size + 1):
                new_y = y + dy
                new_x = x + dx
                # vérifier que le pixel est dans l'image et dans le cercle
                if (0 <= new_y < data.shape[0] and 0 <= new_x < data.shape[1] and
                    np.sqrt(dx**2 + dy**2) <= radius):
                    mask[new_y, new_x] = 255

#print(f"Pixels blancs (étoiles): {np.sum(mask == 255)}")

# sauvegarder le masque en png
plt.imsave('./results/star_mask.png', mask, cmap='gray')
print("Masque sauvegardé: ./results/star_mask.png")

# sauvegarder le masque en fits
hdu_mask = fits.PrimaryHDU(mask)
hdu_mask.writeto('./results/star_mask.fits', overwrite=True)
print("Masque FITS sauvegardé: ./results/star_mask.fits")

# fermer le fichier
hdul.close()