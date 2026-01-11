[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zP0O23M7)

# Astronomical Star Reduction Application

## Project Overview

This application performs localized star reduction on astronomical FITS images. The goal is to reduce the intensity of stars while preserving background details such as nebulae and galaxies. This is particularly useful for revealing faint extended objects that are otherwise overwhelmed by bright point sources.

The application uses a combination of star detection algorithms (DAOStarFinder), morphological operations (erosion), and selective blending to achieve natural-looking results. A PyQt6 graphical interface allows users to interactively adjust parameters and visualize the before/after comparison.

## Authors
- Alex François
- Romain Théobald
- Willem Vanbaelinghem -- Dezitter

## Installation

### Requirements
- Python 3.8 or higher
- See `requirements.txt` for full dependency list

### Virtual Environment Setup

It is recommended to create a virtual environment before installing dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies Installation
```bash
pip install -r requirements.txt
```

The main dependencies include:
- PyQt6 (GUI interface)
- astropy (FITS file handling)
- photutils (star detection)
- opencv-python (morphological operations)
- numpy, scipy (numerical operations)
- matplotlib (visualization)

## Usage

### PyQt6 GUI Application (Recommended)
The main application provides an interactive interface for star reduction:

```bash
python app_pyqt6.py
```

**Features:**
- Load FITS images (`.fits`, `.fit`, `.FITS`)
- Detect stars automatically using DAOStarFinder
- Adjust detection parameters (FWHM, threshold, radius)
- Apply morphological erosion with adjustable kernel size and iterations
- Smooth the star mask with Gaussian blur
- View before/after comparison

### Individual Processing Scripts

**View FITS Files:**
```bash
python visualisation_FITS.py
```
Simple script to visualize FITS files using matplotlib. Edit the script to specify which file to view.

**Star Detection:**
```bash
python star_detection.py
```
Module containing the star detection algorithm using DAOStarFinder from photutils.

**Erosion Module:**
```bash
python erosion.py
```
Module containing morphological erosion functions using OpenCV.

**Localized Reduction:**
```bash
python reduction_localisee.py
```
Module implementing the selective blending formula: `I_final = (M × I_erode) + ((1 - M) × I_original)`

## Example Files
Example FITS files are located in the `examples/` directory. You can use these files to test the application:

- `examples/656nmos.fits` - Narrowband astronomical image
- `examples/g19_0.3-7.5keV.fits` - X-ray astronomical data
- `examples/HorseHead.fits` - Horsehead Nebula (monochrome)
- `examples/M-31Andromed220221022931.FITS` - Andromeda Galaxy (M31)
- `examples/M51_Lum.fit` - Whirlpool Galaxy (M51) luminance
- `examples/NGC 300-B.fit` - NGC 300 galaxy B-band
- `examples/orion_xray_low.fits` - Orion X-ray data
- `examples/test_M31_linear.fits` - M31 color image (linear stretch)
- `examples/test_M31_raw.fits` - M31 color image (raw data)

## Methods and Implementation

### 1. Star Detection
We use the **DAOStarFinder** algorithm from the `photutils` library, which is specifically designed for astronomical point source detection. The algorithm:
- Calculates background statistics using sigma-clipped statistics
- Detects sources above a threshold
- Returns positions of detected stars

Parameters:
- **FWHM** (Full Width at Half Maximum): Typical size of stars in pixels
- **Threshold**: Detection sensitivity in sigma units
- **Radius**: Size of circular mask around each star

### 2. Morphological Erosion
We apply **erosion** using OpenCV to reduce star brightness while preserving extended structures:
- Converts normalized image to uint8
- Applies erosion with a square kernel
- Multiple iterations can be performed for stronger reduction

Parameters:
- **Kernel size**: Size of erosion kernel
- **Iterations**: Number of erosion passes

### 3. Mask Smoothing
The binary star mask is smoothed using **Gaussian blur** to create gradual transitions:
- Prevents hard edges in the final image
- Creates natural-looking star reduction
- Uses scipy's `gaussian_filter`

Parameter:
- **Sigma**: Standard deviation of Gaussian kernel

### 4. Selective Blending
The final image combines original and eroded versions using the smoothed mask:

**Formula:** `I_final = (M × I_erode) + ((1 - M) × I_original)`

Where:
- **M = 1** (white in mask) → show eroded image (reduced stars)
- **M = 0** (black in mask) → show original image (preserved background)
- **0 < M < 1** → smooth transition between both

## Results

The application successfully achieves localized star reduction while preserving background details. Key results include:

### Successful Features:
- **Effective star detection**: DAOStarFinder reliably identifies point sources in various image types
- **Natural-looking reduction**: Gaussian smoothing of the mask prevents visible artifacts
- **Parameter flexibility**: Adjustable sliders allow fine-tuning for different images
- **Before/after comparison**: Split-screen visualization clearly shows the effect
- **Multiple format support**: Handles both monochrome and color FITS images
- **Export capabilities**: Saves all intermediate and final results in FITS and PNG formats

## Difficulties Encountered

### 1. Virtual Environment Inconsistencies
**Problem:** Different team members used different virtual environment configurations, leading to dependency conflicts.

**Solution:** 
- Added `venv/` to `.gitignore` to avoid committing environment files

### 2. Color vs Monochrome FITS Images
**Problem:** FITS files can have different data structures (2D grayscale vs 3D color, different axis ordering).

**Solution:**
- Implemented automatic dimension detection in all modules
- Convert color images to grayscale for star detection
- Expand 2D masks to 3D when needed for color images

### 3. Parameter Sensitivity
**Problem:** Default parameters don't work well for all image types (e.g., X-ray vs optical data).

**Solution:**
- Made all parameters adjustable through the interface

### 4. Dynamic Range and Normalization
**Problem:** FITS images have varying dynamic ranges (some with extreme outliers, negative values, etc.).

**Solution:**
- Used percentile-based normalization (0.5% to 99.5%) for display
- Handled NaN values properly with `np.nanmin` / `np.nanmax`
- Separate normalization for visualization vs processing

### 5. Mask Edge Artifacts
**Problem:** Hard edges in the binary star mask created visible boundaries in the final image.

**Solution:**
- Adjustable Gaussian blur to control the integrity of the mask edges

### 6. Interface Responsiveness
**Problem:** Processing large images in real time could freeze the interface.

**Solution:**
- Replace live reprocessing with a button that must be activated manually

