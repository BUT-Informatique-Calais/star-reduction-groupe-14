from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# Open and read the FITS file
fits_file = './examples/test_M31_linear.fits'
hdul = fits.open(fits_file)

# Display information about the file
hdul.info()

# Access the data from the primary HDU
data = hdul[0].data

# Access header information
header = hdul[0].header

# Handle both monochrome and color images
if data.ndim == 3:
    # Color image - need to transpose to (height, width, channels)
    if data.shape[0] == 3:  # If channels are first: (3, height, width)
        data = np.transpose(data, (1, 2, 0))
    # If already (height, width, 3), no change needed
    
    # Normalize the entire image to [0, 1] for matplotlib
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Save the data as a png image (no cmap for color images)
    plt.imsave('./results/original.png', data_normalized)

    # Pour OpenCV, garder l'ordre des canaux RGB (ne pas convertir en BGR)
    image = (data_normalized * 255).astype('uint8')

    # Définir le noyau et appliquer l'érosion
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv.erode(image, kernel, iterations=2)

    # Convertir back en float pour FITS (garder les couleurs originales)
    eroded_float = eroded_image.astype(np.float32) / 255.0

    # Sauvegarder PNG avec matplotlib (préserve les couleurs RGB)
    plt.imsave('./results/eroded.png', eroded_image / 255.0)

else:
    # Monochrome image
    plt.imsave('./results/original.png', data, cmap='gray')
    
    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

    # Apply erosion
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv.erode(image, kernel, iterations=2)

    # Convert back to float
    eroded_float = eroded_image.astype(np.float32) / 255.0



# Define a kernel for erosion
kernel = np.ones((3,3), np.uint8)
# Perform erosion
eroded_image = cv.erode(image, kernel, iterations=2)

# Convert back to RGB before saving (for color images only)
if data.ndim == 3:
    eroded_image = cv.cvtColor(eroded_image, cv.COLOR_BGR2RGB)

# Save the eroded image 
cv.imwrite('./results/eroded.png', eroded_image) #for easyer visualization
fits.writeto('./results/eroded.fits', eroded_float, overwrite=True)

# Close the file
hdul.close()

#TODO : trad commentaires