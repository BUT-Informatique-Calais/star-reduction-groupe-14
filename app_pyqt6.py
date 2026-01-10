"""
PyQt6 Interface for Astronomical Star Reduction - Before/After Comparison
"""

import sys
import numpy as np
from astropy.io import fits
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvasQTAgg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QGridLayout, QMessageBox,
    QSlider, QGroupBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, pyqtSignal

from star_detection import detect_stars, smooth_mask
from erosion import apply_erosion, prepare_image
from reduction_localisee import compute_final_image


class ImageCanvas(FigureCanvasQTAgg):
    """Custom matplotlib canvas for displaying images"""
    clicked = pyqtSignal(object)
    
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.patch.set_facecolor('black')
        self.image_data = None
        
        # Connect click event
        self.mpl_connect('button_press_event', self.on_click)

        # Display default message
        self.clear_display()
    
    def on_click(self, event):
        """Emit signal when clicked"""
        if event.inaxes is not None and self.image_data is not None:
            self.clicked.emit(self.image_data)
    
    def display_image(self, data, title, cmap='gray', vmin_percentile=0.5, vmax_percentile=99.5):
        """Display image with percentile normalization"""
        self.image_data = data
        self.ax.clear()
        
        # Normalize using percentiles (prevents extreme pixels from destroying contrast)
        if data.ndim == 3:
            data_min = np.nanpercentile(data, vmin_percentile)
            data_max = np.nanpercentile(data, vmax_percentile)
        else:
            data_min = np.nanpercentile(data, vmin_percentile)
            data_max = np.nanpercentile(data, vmax_percentile)

        if data_max > data_min:
            data_norm = np.clip((data - data_min) / (data_max - data_min), 0, 1)
        else:
            data_norm = data
        
        # Display image
        if data_norm.ndim == 3:
            self.ax.imshow(data_norm, origin='upper')
        else:
            self.ax.imshow(data_norm, cmap=cmap, origin='upper')

        self.ax.set_title(title, fontsize=10, color='white', fontweight='bold')
        self.ax.axis('off')
        self.fig.patch.set_facecolor('black')
        self.draw()

    def clear_display(self):
        """Show default message when no image is loaded"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'No image loaded',
                     ha='center', va='center', color='gray', fontsize=12,
                     transform=self.ax.transAxes)
        self.ax.axis('off')
        self.fig.patch.set_facecolor('black')
        self.draw()


class ZoomWindow(QMainWindow):
    """Zoom window for detailed image inspection"""
    def __init__(self, data, title):
        super().__init__()
        self.setWindowTitle(f"Zoom - {title}")
        self.setGeometry(100, 100, 900, 900)
        
        # Create larger canvas for zoom
        canvas = ImageCanvas(width=9, height=9, dpi=100)
        cmap = 'gray' if data.ndim == 2 else None
        canvas.display_image(data, title, cmap=cmap)

        # Setup layout
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


class ComparatorWindow(QMainWindow):
    """Side-by-side comparison with interactive slider"""
    def __init__(self, original, final):
        super().__init__()
        self.setWindowTitle("Comparator - Original vs Final")
        self.setGeometry(100, 100, 1000, 800)
        self.original = original
        self.final = final

        layout = QVBoxLayout()

        # Display canvas
        self.canvas = ImageCanvas(width=10, height=7, dpi=100)
        layout.addWidget(self.canvas, 1)

        # Slider for transition
        slider_container = QWidget()
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(60, 0, 60, 0)

        slider_label_left = QLabel("Original")
        slider_label_left.setFixedWidth(60)
        slider_layout.addWidget(slider_label_left)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.update_comparison)
        self.slider.setFixedHeight(20)
        slider_layout.addWidget(self.slider, 1)

        self.percent_label = QLabel("50%")
        self.percent_label.setFixedWidth(40)
        slider_layout.addWidget(self.percent_label)

        slider_label_right = QLabel("Final")
        slider_label_right.setFixedWidth(40)
        slider_layout.addWidget(slider_label_right)

        slider_container.setLayout(slider_layout)
        slider_container.setFixedHeight(50)
        layout.addWidget(slider_container)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setStyleSheet(f"background-color: #f0f0f0;")

        self.update_comparison()

    def update_comparison(self):
        """Update blend based on slider position"""
        value = self.slider.value()
        self.percent_label.setText(f"{value}%")

        # Create composite image by splitting left and right
        split = int(self.original.shape[1] * value / 100)
        if self.original.ndim == 3:
            composite = np.zeros_like(self.original)
            composite[:, :split] = self.original[:, :split]
            composite[:, split:] = self.final[:, split:]
        else:
            composite = np.zeros_like(self.original)
            composite[:, :split] = self.original[:, :split]
            composite[:, split:] = self.final[:, split:]

        self.canvas.display_image(composite, "")


class ReductionAstroApp(QMainWindow):
    """Main application for star reduction processing"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Star Reduction")
        self.setGeometry(50, 50, 1400, 850)

        # UI colors
        self.couleur_principale = "#2c3e50"
        self.couleur_bouton = "#3498db"
        self.couleur_texte = "#ecf0f1"
        
        # Store image data at different processing stages
        self.images_data = {
            'original': None,      # Normalized original image
            'original_raw': None,  # Raw image for star detection
            'erodee': None,        # Eroded image
            'masque_brut': None,   # Binary star mask
            'masque_lisse': None,  # Smoothed mask
            'finale': None         # Final processed image
        }
        self.nb_etoiles = 0

        # Keep zoom windows open
        self.zoom_windows = []
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Main central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()

        # Header with button to load an image
        header_layout = QHBoxLayout()

        header_label = QLabel("Réduction d'Étoiles")
        header_font = QFont("Arial", 16, QFont.Weight.Bold)
        header_label.setFont(header_font)
        header_label.setStyleSheet(f"color: {self.couleur_principale}; padding: 10px;")
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        btn_charger = QPushButton("Charger une image")
        btn_charger.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        btn_charger.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.couleur_bouton};
                color: {self.couleur_texte};
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2980b9;
            }}
        """)
        btn_charger.clicked.connect(self.charger_et_traiter)
        header_layout.addWidget(btn_charger)

        main_layout.addLayout(header_layout)
        
        # Image grid (1x2)
        images_layout = QGridLayout()
        
        # Original image canvas
        self.canvas_original = ImageCanvas()
        self.canvas_original.clicked.connect(lambda data: self.show_zoom(data, "Image Originale"))
        images_layout.addWidget(self.canvas_original, 0, 0)
        
        # Final image canvas
        self.canvas_finale = ImageCanvas()
        self.canvas_finale.clicked.connect(lambda data: self.show_zoom(data, "Image Finale"))
        images_layout.addWidget(self.canvas_finale, 0, 1)

        main_layout.addLayout(images_layout, 1)

        # Add controls panel
        controls_panel = self.create_controls_panel()
        main_layout.addWidget(controls_panel)

        # Info label (number of stars detected)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-style: italic;")
        main_layout.addWidget(self.info_label)

        # Button bar
        buttons_layout = QHBoxLayout()

        btn_retraiter = QPushButton("Retraiter")
        btn_retraiter.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        btn_retraiter.setStyleSheet(f"""
            QPushButton {{
                background-color: #27ae60;
                color: {self.couleur_texte};
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #229954;
            }}
        """)
        btn_retraiter.clicked.connect(self.retraiter)
        buttons_layout.addWidget(btn_retraiter)
        
        btn_erodee = QPushButton("Voir image érodée")
        btn_erodee.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        btn_erodee.setStyleSheet(f"""
            QPushButton {{
                background-color: #9b59b6;
                color: {self.couleur_texte};
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #8e44ad;
            }}
        """)
        btn_erodee.clicked.connect(self.show_erodee)
        buttons_layout.addWidget(btn_erodee)
        
        btn_masque = QPushButton("Voir masque d'étoiles")
        btn_masque.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        btn_masque.setStyleSheet(f"""
            QPushButton {{
                background-color: #e67e22;
                color: {self.couleur_texte};
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #d35400;
            }}
        """)
        btn_masque.clicked.connect(self.show_masque)
        buttons_layout.addWidget(btn_masque)

        btn_comparateur = QPushButton("Comparateur")
        btn_comparateur.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        btn_comparateur.setStyleSheet(f"""
            QPushButton {{
                background-color: #1abc9c;
                color: {self.couleur_texte};
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #16a085;
            }}
        """)
        btn_comparateur.clicked.connect(self.show_comparateur)
        buttons_layout.addWidget(btn_comparateur)

        btn_reinitialiser = QPushButton("Réinitialiser les sliders")
        btn_reinitialiser.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        btn_reinitialiser.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: #ecf0f1;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        btn_reinitialiser.clicked.connect(self.reinitialiser)
        buttons_layout.addWidget(btn_reinitialiser)
        
        buttons_layout.addStretch()
        main_layout.addLayout(buttons_layout)
        
        # Status bar
        self.statusBar().showMessage("En attente d'une image...")
        
        central_widget.setLayout(main_layout)
        
        # Global stylesheet
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #f0f0f0;
            }}
            QLabel {{
                color: #2c3e50;
            }}
        """)
    
    def create_panel(self, title, canvas):
        """Create panel with title and canvas"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Titre
        title_label = QLabel(title)
        title_label.setFixedHeight(50)
        title_font = QFont("Arial", 11, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"""
            background-color: {self.couleur_principale};
            color: {self.couleur_texte};
            padding: 8px;
            border-radius: 4px 4px 0 0;
        """)
        
        # Canvas
        layout.addWidget(title_label)
        layout.addWidget(canvas)
        
        panel.setLayout(layout)
        panel.setStyleSheet("""
            QWidget {
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
        """)
        
        return panel

    def create_controls_panel(self):
        """Create control panel with all sliders"""
        controls_group = QGroupBox("Paramètres de traitement")
        controls_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.couleur_principale};
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)

        layout = QVBoxLayout()

        # Star detection parameters
        detection_label = QLabel("- Détection d'étoiles -")
        detection_label.setStyleSheet("font-weight: bold; color: #2980b9;")
        layout.addWidget(detection_label)

        # FWHM (star size - typical stellar Full Width at Half Maximum)
        fwhm_layout = QHBoxLayout()
        fwhm_layout.addWidget(QLabel("FWHM (taille étoiles):"))
        self.fwhm_slider = QSlider(Qt.Orientation.Horizontal)
        self.fwhm_slider.setMinimum(5)
        self.fwhm_slider.setMaximum(200)
        self.fwhm_slider.setValue(12)
        self.fwhm_slider.valueChanged.connect(self.on_slider_change)
        self.fwhm_label = QLabel("1.2")
        fwhm_layout.addWidget(self.fwhm_slider)
        fwhm_layout.addWidget(self.fwhm_label)
        layout.addLayout(fwhm_layout)

        # Detection threshold (sensitivity in sigma units)
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Seuil détection :"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(10)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(25)
        self.threshold_slider.valueChanged.connect(self.on_slider_change)
        self.threshold_label = QLabel("2.5")
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        layout.addLayout(threshold_layout)

        # Mask radius (pixels around each detected star)
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Rayon masque étoiles:"))
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setMinimum(10)
        self.radius_slider.setMaximum(150)
        self.radius_slider.setValue(36)
        self.radius_slider.valueChanged.connect(self.on_slider_change)
        self.radius_label = QLabel("3.6")
        radius_layout.addWidget(self.radius_slider)
        radius_layout.addWidget(self.radius_label)
        layout.addLayout(radius_layout)

        # Erosion parameters
        erosion_label = QLabel("- Érosion -")
        erosion_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        layout.addWidget(erosion_label)

        # Erosion kernel size (higher = stronger erosion)
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("Taille du kernel:"))
        self.kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_slider.setMinimum(1)
        self.kernel_slider.setMaximum(10)
        self.kernel_slider.setValue(1)
        self.kernel_slider.valueChanged.connect(self.on_slider_change)
        self.kernel_label = QLabel("3x3")
        kernel_layout.addWidget(self.kernel_slider)
        kernel_layout.addWidget(self.kernel_label)
        layout.addLayout(kernel_layout)

        # Number of erosion iterations (higher = more erosion)
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Itérations:"))
        self.iter_slider = QSlider(Qt.Orientation.Horizontal)
        self.iter_slider.setMinimum(1)
        self.iter_slider.setMaximum(10)
        self.iter_slider.setValue(1)
        self.iter_slider.valueChanged.connect(self.on_slider_change)
        self.iter_label = QLabel("1")
        iter_layout.addWidget(self.iter_slider)
        iter_layout.addWidget(self.iter_label)
        layout.addLayout(iter_layout)

        # Mask smoothing parameters
        mask_label = QLabel("- Lissage du masque -")
        mask_label.setStyleSheet("font-weight: bold; color: #8e44ad;")
        layout.addWidget(mask_label)

        # Gaussian blur for smooth mask transitions
        gauss_layout = QHBoxLayout()
        gauss_layout.addWidget(QLabel("Flou gaussien :"))
        self.gauss_slider = QSlider(Qt.Orientation.Horizontal)
        self.gauss_slider.setMinimum(10)
        self.gauss_slider.setMaximum(100)
        self.gauss_slider.setValue(18)
        self.gauss_slider.valueChanged.connect(self.on_slider_change)
        self.gauss_label = QLabel("1.8")
        gauss_layout.addWidget(self.gauss_slider)
        gauss_layout.addWidget(self.gauss_label)
        layout.addLayout(gauss_layout)

        # Mask threshold (binary threshold after smoothing)
        seuil_layout = QHBoxLayout()
        seuil_layout.addWidget(QLabel("Seuil masque:"))
        self.seuil_slider = QSlider(Qt.Orientation.Horizontal)
        self.seuil_slider.setMinimum(1)
        self.seuil_slider.setMaximum(100)
        self.seuil_slider.setValue(54)
        self.seuil_slider.valueChanged.connect(self.on_slider_change)
        self.seuil_label = QLabel("0.54")
        seuil_layout.addWidget(self.seuil_slider)
        seuil_layout.addWidget(self.seuil_label)
        layout.addLayout(seuil_layout)

        # Display parameters
        display_label = QLabel("- Affichage -")
        display_label.setStyleSheet("font-weight: bold; color: #e67e22;")
        layout.addWidget(display_label)

        # Image contrast control (percentile normalization)
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contraste (percentile):"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setMinimum(900)  # 90.0%
        self.contrast_slider.setMaximum(1000)  # 100.0%
        self.contrast_slider.setValue(995)  # 99.5%
        self.contrast_slider.valueChanged.connect(self.on_contrast_change)
        self.contrast_label = QLabel("99.5%")
        contrast_layout.addWidget(self.contrast_slider)
        contrast_layout.addWidget(self.contrast_label)
        layout.addLayout(contrast_layout)

        controls_group.setLayout(layout)
        return controls_group

    def on_slider_change(self):
        """Update labels when slider changes and reprocess image"""
        # Update all slider value labels
        self.fwhm_label.setText(f"{self.fwhm_slider.value() / 10.0:.1f}")
        self.threshold_label.setText(f"{self.threshold_slider.value() / 10.0:.1f}")
        self.radius_label.setText(f"{self.radius_slider.value() / 10.0:.1f}")

        # Map kernel slider values to kernel sizes
        sizes = {1: "3x3", 2: "5x5", 3: "7x7", 4: "9x9", 5: "11x11", 6: "13x13", 7: "15x15", 8: "17x17", 9: "19x19", 10: "21x21"}
        self.kernel_label.setText(sizes[self.kernel_slider.value()])
        self.iter_label.setText(str(self.iter_slider.value()))
        self.gauss_label.setText(f"{self.gauss_slider.value() / 10.0:.1f}")
        self.seuil_label.setText(f"{self.seuil_slider.value() / 100.0:.2f}")


    def on_contrast_change(self):
        """Update display contrast based on slider"""
        # Calculate percentile from slider value
        percentile = self.contrast_slider.value() / 10.0
        self.contrast_label.setText(f"{percentile:.1f}%")

        # Redisplay images with new contrast
        if self.images_data['original'] is not None:
            self.canvas_original.display_image(
                self.images_data['original'],
                "Originale",
                cmap='gray' if self.images_data['original'].ndim == 2 else None,
                vmax_percentile=percentile
            )

        if self.images_data['finale'] is not None:
            self.canvas_finale.display_image(
                self.images_data['finale'],
                "Finale",
                cmap='gray' if self.images_data['finale'].ndim == 2 else None,
                vmax_percentile=percentile
            )

    def charger_et_traiter(self):
        """Load FITS file and process it"""
        # Open file dialog
        chemin, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner une image FITS",
            "examples",
            "FITS Files (*.fits *.fit *.FITS)"
        )
        
        if not chemin:
            return
        
        self.statusBar().showMessage("Chargement et traitement en cours...")

        try:
            # Load FITS file
            with fits.open(chemin) as hdul:
                data_raw = hdul[0].data.astype(float)

            # Store raw data for star detection
            self.images_data['original_raw'] = data_raw.copy()

            # Prepare image (normalize + transpose if needed)
            data_norm = prepare_image(data_raw)

            # Display original image
            self.images_data['original'] = data_norm
            self.canvas_original.display_image(data_norm, "Originale")

            # Process the image
            self.traiter_image()

        except Exception as e:
            # Show error message
            QMessageBox.critical(self, "Erreur", f"Erreur lors du traitement:\n{str(e)}")
            self.statusBar().showMessage("Erreur")

    def traiter_image(self):
        """Apply star reduction algorithm"""
        try:
            # Get stored data
            data_norm = self.images_data['original']
            data_raw = self.images_data['original_raw']

            # Retrieve all parameters from sliders
            fwhm = self.fwhm_slider.value() / 10.0
            threshold_sigma = self.threshold_slider.value() / 10.0
            radius = self.radius_slider.value() / 10.0

            # Map slider values to kernel sizes
            kernel_sizes = {1: 3, 2: 5, 3: 7, 4: 9, 5: 11, 6: 13, 7: 15, 8: 17, 9: 19, 10: 21}
            kernel_size = kernel_sizes[self.kernel_slider.value()]
            iterations = self.iter_slider.value()

            gauss_sigma = self.gauss_slider.value() / 10.0
            mask_threshold = self.seuil_slider.value() / 100.0

            # STEP 1: Detect stars (create binary mask)
            masque_brut, sources = detect_stars(
                data_raw,
                fwhm=fwhm,
                threshold_sigma=threshold_sigma,
                radius=radius
            )
            self.images_data['masque_brut'] = masque_brut

            # Count detected stars
            if sources is not None:
                self.nb_etoiles = len(sources)
            else:
                self.nb_etoiles = 0

            # STEP 2: Apply gaussian blur to smooth the mask
            masque_lisse = smooth_mask(masque_brut, sigma=gauss_sigma, threshold=mask_threshold)
            self.images_data['masque_lisse'] = masque_lisse

            # STEP 3: Create eroded image
            # Apply morphological erosion to reduce stars
            data_erodee = apply_erosion(data_norm, kernel_size=kernel_size, iterations=iterations)
            self.images_data['erodee'] = data_erodee

            # STEP 4: Compute final image
            # Formula: I_final = (M × I_erode) + ((1 - M) × I_original)
            # Blend original and eroded images using smooth mask
            image_finale = compute_final_image(data_norm, data_erodee, masque_lisse)
            self.images_data['finale'] = image_finale

            # Display final processed image
            self.canvas_finale.display_image(image_finale, "Finale")

            # Update status bar
            self.statusBar().showMessage(f"Traitement terminé - {self.nb_etoiles} étoiles détectées")
            self.info_label.setText(f"Étoiles détectées: {self.nb_etoiles}")

        except Exception as e:
            # Show error message
            QMessageBox.critical(self, "Erreur", f"Erreur lors du traitement:\n{str(e)}")
            self.statusBar().showMessage("Erreur")
    
    def retraiter(self):
        """Reprocess image with updated parameters"""
        if self.images_data['original'] is None:
            return

        self.statusBar().showMessage("Retraitement en cours...")
        self.traiter_image()

    def show_erodee(self):
        """Display eroded image in zoom window"""
        if self.images_data['erodee'] is None:
            QMessageBox.warning(self, "Attention", "Veuillez charger une image d'abord")
            return
        
        # Open zoom window with eroded image
        self.show_zoom(self.images_data['erodee'], "Image Érodée")

    def show_masque(self):
        """Display star mask in zoom window"""
        if self.images_data['masque_lisse'] is None:
            QMessageBox.warning(self, "Attention", "Veuillez charger une image d'abord")
            return

        # Open zoom window with mask
        self.show_zoom(self.images_data['masque_lisse'], f"Masque d'étoiles ({self.nb_etoiles} détectées)")

    def show_zoom(self, data, title):
        """Open zoom window for detailed image inspection"""
        if data is None:
            QMessageBox.warning(self, "Attention", "Aucune image à afficher")
            return
        
        # Create and show zoom window
        zoom_window = ZoomWindow(data, title)
        zoom_window.show()
        
        # Keep reference to prevent garbage collection
        self.zoom_windows.append(zoom_window)
    
    def show_comparateur(self):
        """Open comparison window with slider"""
        if self.images_data['original'] is None or self.images_data['finale'] is None:
            QMessageBox.warning(self, "Attention", "Veuillez charger et traiter une image d'abord")
            return

        # Create comparison window
        comparator = ComparatorWindow(self.images_data['original'], self.images_data['finale'])
        comparator.show()
        self.zoom_windows.append(comparator)

    def reinitialiser(self):
        """Reset all sliders to default values"""
        # Reset all sliders to defaults
        self.fwhm_slider.setValue(12)
        self.threshold_slider.setValue(25)
        self.radius_slider.setValue(36)
        self.kernel_slider.setValue(1)
        self.iter_slider.setValue(1)
        self.gauss_slider.setValue(18)
        self.seuil_slider.setValue(54)
        self.contrast_slider.setValue(995)

        # Reprocess if image is loaded
        if self.images_data['original'] is not None:
            self.statusBar().showMessage("Réinitialisation des paramètres...")
        else:
            self.statusBar().showMessage("En attente d'une image...")


def main():
    """Start the application"""
    app = QApplication(sys.argv)
    window = ReductionAstroApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
