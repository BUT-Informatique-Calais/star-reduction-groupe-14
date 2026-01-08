from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# open fits file
fits_file = './examples/test_M31_linear.fits'
hdul = fits.open(fits_file)

# display info
hdul.info()
header = hdul[0].header
data = hdul[0].data

# if color image, convert to monochrome by taking one channel
if data.ndim == 3:
    if data.shape[0] == 3:
        data = data[0]
    else:
        data = data[:, :, 0]

# calculate background statistics
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

print(f"\nImage statistics:")
print(f"  Mean: {mean:.2f}")
print(f"  Median: {median:.2f}")
print(f"  Standard deviation: {std:.2f}")

# star detection parameters

# fwhm : typical full-width at half-maximum of the stars
fwhm = 3.0

# how many std above the background to consider a star
threshold = 5.5

# create detector
daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)

# find stars
sources = daofind(data - median)

print(f"\nNumber of stars detected: {len(sources)}")

# create an empty mask
mask = np.zeros(data.shape, dtype=np.uint8)

radius = 3.5

# for each detected star, draw a circle in the mask
for star in sources:
    x = int(star['xcentroid'])
    y = int(star['ycentroid'])
    
    if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
        range_size = int(radius) + 1
        for dy in range(-range_size, range_size + 1):
            for dx in range(-range_size, range_size + 1):
                new_y = y + dy
                new_x = x + dx
                # verify if within bounds and within radius
                if (0 <= new_y < data.shape[0] and 0 <= new_x < data.shape[1] and
                    np.sqrt(dx**2 + dy**2) <= radius):
                    mask[new_y, new_x] = 255

#print(f"Pixels blancs (étoiles): {np.sum(mask == 255)}")

# save mask as png
plt.imsave('./results/star_mask.png', mask, cmap='gray')
print("Saved mask as PNG: ./results/star_mask.png")

# save mask as fits
hdu_mask = fits.PrimaryHDU(mask)
hdu_mask.writeto('./results/star_mask.fits', overwrite=True)
print("Saved mask as FITS: ./results/star_mask.fits")

#close fits file
hdul.close()