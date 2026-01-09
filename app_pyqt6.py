"""
Interface PyQt6 pour la réduction astro - Avant/Après
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
    clicked = pyqtSignal(object)
    
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.patch.set_facecolor('black')
        self.image_data = None
        
        # Binder le clic
        self.mpl_connect('button_press_event', self.on_click)

        # Afficher le message par défaut
        self.clear_display()
    
    def on_click(self, event):
        """Émet le signal quand on clique"""
        if event.inaxes is not None and self.image_data is not None:
            self.clicked.emit(self.image_data)
    
    def display_image(self, data, title, cmap='viridis'):
        """Affiche une image"""
        self.image_data = data
        
        self.ax.clear()
        
        # Normaliser les données
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        if data_max > data_min:
            data_norm = (data - data_min) / (data_max - data_min)
        else:
            data_norm = data
        
        # Afficher
        if data_norm.ndim == 3:
            self.ax.imshow(data_norm, origin='upper')
        else:
            self.ax.imshow(data_norm, cmap=cmap, origin='upper')
        
        self.ax.set_title(title, fontsize=10, color='white', fontweight='bold')
        self.ax.axis('off')
        self.fig.patch.set_facecolor('black')
        self.draw()

    def clear_display(self):
        """Affiche le message par défaut quand aucune image n'est chargée"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Aucune image chargée',
                     ha='center', va='center', color='gray', fontsize=12,
                     transform=self.ax.transAxes)
        self.ax.axis('off')
        self.fig.patch.set_facecolor('black')
        self.draw()


class ZoomWindow(QMainWindow):
    """Fenêtre de zoom"""
    def __init__(self, data, title):
        super().__init__()
        self.setWindowTitle(f"Zoom - {title}")
        self.setGeometry(100, 100, 900, 900)
        
        # Créer le canvas
        canvas = ImageCanvas(width=9, height=9, dpi=100)
        canvas.display_image(data, title, cmap='viridis' if data.ndim == 3 or len(data.shape) == 2 else 'hot')
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


class ComparatorWindow(QMainWindow):
    """Fenêtre simple de comparaison"""
    def __init__(self, original, final):
        super().__init__()
        self.setWindowTitle("Comparateur - Original vs Final")
        self.setGeometry(100, 100, 1000, 800)
        self.original = original
        self.final = final

        layout = QVBoxLayout()

        # Canvas pour l'image de comparaison
        self.canvas = ImageCanvas(width=10, height=7, dpi=100)
        layout.addWidget(self.canvas, 1)

        # Slider en bas avec spacing pour aligner avec l'image
        slider_container = QWidget()
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(60, 0, 60, 0)  # Margins pour aligner avec l'image

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
        """Met à jour la comparaison selon le slider"""
        value = self.slider.value()
        self.percent_label.setText(f"{value}%")

        # Créer l'image composite
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Réduction Astro")
        self.setGeometry(50, 50, 1400, 850)

        # Styles
        self.couleur_principale = "#2c3e50"
        self.couleur_bouton = "#3498db"
        self.couleur_texte = "#ecf0f1"
        
        # Données
        self.images_data = {
            'original': None,      # Image originale normalisée
            'original_raw': None,  # Image brute pour la détection
            'erodee': None,        # Image érodée
            'masque_brut': None,   # Masque binaire des étoiles
            'masque_lisse': None,  # Masque lissé
            'finale': None         # Image finale
        }
        self.nb_etoiles = 0

        # Garder les fenêtres de zoom ouvertes
        self.zoom_windows = []
        self.init_ui()
    
    def init_ui(self):
        """Initialise l'interface"""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        
        # En-tête
        header_label = QLabel("Réduction d'Étoiles")
        header_font = QFont("Arial", 16, QFont.Weight.Bold)
        header_label.setFont(header_font)
        header_label.setStyleSheet(f"color: {self.couleur_principale}; padding: 10px;")
        main_layout.addWidget(header_label)
        
        # Zone des images (grille 1x2)
        images_layout = QGridLayout()
        
        # Image originale
        self.canvas_original = ImageCanvas()
        self.canvas_original.clicked.connect(lambda data: self.show_zoom(data, "Image Originale"))
        images_layout.addWidget(self.create_panel("IMAGE ORIGINALE", self.canvas_original), 0, 0)
        
        # Image finale
        self.canvas_finale = ImageCanvas()
        self.canvas_finale.clicked.connect(lambda data: self.show_zoom(data, "Image Finale"))
        images_layout.addWidget(self.create_panel("IMAGE FINALE (étoiles réduites)", self.canvas_finale), 0, 1)

        main_layout.addLayout(images_layout, 1)

        # Ajouter le panneau de contrôles
        controls_panel = self.create_controls_panel()
        main_layout.addWidget(controls_panel)

        # Label d'information (nbr d'étoiles détectées quand photo chargée)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-style: italic;")
        main_layout.addWidget(self.info_label)

        # Barre de boutons
        buttons_layout = QHBoxLayout()
        
        btn_charger = QPushButton("Charger et traiter")
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
        buttons_layout.addWidget(btn_charger)
        
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
        
        # Barre de statut
        self.statusBar().showMessage("En attente d'une image...")
        
        central_widget.setLayout(main_layout)
        
        # Style global
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #f0f0f0;
            }}
            QLabel {{
                color: #2c3e50;
            }}
        """)
    
    def create_panel(self, title, canvas):
        """Crée un panel avec titre et canvas"""
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
        """Crée le panneau de contrôles avec sliders"""
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

        # --- Paramètres de détection d'étoiles ---
        detection_label = QLabel("- Détection d'étoiles -")
        detection_label.setStyleSheet("font-weight: bold; color: #2980b9;")
        layout.addWidget(detection_label)

        # FWHM (taille typique des étoiles)
        fwhm_layout = QHBoxLayout()
        fwhm_layout.addWidget(QLabel("FWHM (taille étoiles):"))
        self.fwhm_slider = QSlider(Qt.Orientation.Horizontal)
        self.fwhm_slider.setMinimum(10)
        self.fwhm_slider.setMaximum(100)
        self.fwhm_slider.setValue(30)
        self.fwhm_slider.valueChanged.connect(self.on_slider_change)
        self.fwhm_label = QLabel("3.0")
        fwhm_layout.addWidget(self.fwhm_slider)
        fwhm_layout.addWidget(self.fwhm_label)
        layout.addLayout(fwhm_layout)

        # Threshold (sensibilité de détection)
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Seuil détection :"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(10)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(55)
        self.threshold_slider.valueChanged.connect(self.on_slider_change)
        self.threshold_label = QLabel("5.5")
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        layout.addLayout(threshold_layout)

        # Rayon du masque
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Rayon masque étoiles:"))
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setMinimum(10)
        self.radius_slider.setMaximum(150)
        self.radius_slider.setValue(35)
        self.radius_slider.valueChanged.connect(self.on_slider_change)
        self.radius_label = QLabel("3.5")
        radius_layout.addWidget(self.radius_slider)
        radius_layout.addWidget(self.radius_label)
        layout.addLayout(radius_layout)

        # --- Paramètres d'érosion ---
        erosion_label = QLabel("- Érosion -")
        erosion_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        layout.addWidget(erosion_label)

        # Kernel size
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

        # Itérations
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Itérations:"))
        self.iter_slider = QSlider(Qt.Orientation.Horizontal)
        self.iter_slider.setMinimum(1)
        self.iter_slider.setMaximum(10)
        self.iter_slider.setValue(2)
        self.iter_slider.valueChanged.connect(self.on_slider_change)
        self.iter_label = QLabel("2")
        iter_layout.addWidget(self.iter_slider)
        iter_layout.addWidget(self.iter_label)
        layout.addLayout(iter_layout)

        # --- Paramètres du masque ---
        mask_label = QLabel("- Lissage du masque -")
        mask_label.setStyleSheet("font-weight: bold; color: #8e44ad;")
        layout.addWidget(mask_label)

        # Flou gaussien
        gauss_layout = QHBoxLayout()
        gauss_layout.addWidget(QLabel("Flou gaussien (sigma):"))
        self.gauss_slider = QSlider(Qt.Orientation.Horizontal)
        self.gauss_slider.setMinimum(10)
        self.gauss_slider.setMaximum(100)
        self.gauss_slider.setValue(20)
        self.gauss_slider.valueChanged.connect(self.on_slider_change)
        self.gauss_label = QLabel("2.0")
        gauss_layout.addWidget(self.gauss_slider)
        gauss_layout.addWidget(self.gauss_label)
        layout.addLayout(gauss_layout)

        # Seuil du masque
        seuil_layout = QHBoxLayout()
        seuil_layout.addWidget(QLabel("Seuil masque:"))
        self.seuil_slider = QSlider(Qt.Orientation.Horizontal)
        self.seuil_slider.setMinimum(1)
        self.seuil_slider.setMaximum(100)
        self.seuil_slider.setValue(10)
        self.seuil_slider.valueChanged.connect(self.on_slider_change)
        self.seuil_label = QLabel("0.10")
        seuil_layout.addWidget(self.seuil_slider)
        seuil_layout.addWidget(self.seuil_label)
        layout.addLayout(seuil_layout)

        controls_group.setLayout(layout)
        return controls_group

    def on_slider_change(self):
        """Appelé quand un slider change, met à jour les labels"""
        # Mettre à jour tous les labels
        self.fwhm_label.setText(f"{self.fwhm_slider.value() / 10.0:.1f}")
        self.threshold_label.setText(f"{self.threshold_slider.value() / 10.0:.1f}")
        self.radius_label.setText(f"{self.radius_slider.value() / 10.0:.1f}")

        sizes = {1: "3x3", 2: "5x5", 3: "7x7", 4: "9x9", 5: "11x11", 6: "13x13", 7: "15x15", 8: "17x17", 9: "19x19", 10: "21x21"}
        self.kernel_label.setText(sizes[self.kernel_slider.value()])
        self.iter_label.setText(str(self.iter_slider.value()))
        self.gauss_label.setText(f"{self.gauss_slider.value() / 10.0:.1f}")
        self.seuil_label.setText(f"{self.seuil_slider.value() / 100.0:.2f}")

        # Retraiter automatiquement si une image est chargée
        if self.images_data['original'] is not None:
            self.retraiter()
    
    def charger_et_traiter(self):
        """Charge et traite une image"""
        chemin, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner une image FITS",
            "examples",
            "FITS Files (*.fits *.fit)"
        )
        
        if not chemin:
            return
        
        self.statusBar().showMessage("Chargement et traitement en cours...")

        try:
            # Charger l'image
            with fits.open(chemin) as hdul:
                data_raw = hdul[0].data.astype(float)

            # Garder les données brutes pour la détection
            self.images_data['original_raw'] = data_raw.copy()

            # Préparer l'image (normalisation + transposition)
            data_norm = prepare_image(data_raw)

            # Afficher l'original
            self.images_data['original'] = data_norm
            self.canvas_original.display_image(data_norm, "Originale")

            # Traiter l'image
            self.traiter_image()

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du traitement:\n{str(e)}")
            self.statusBar().showMessage("Erreur")

    def traiter_image(self):
        """Applique l'algorithme de réduction d'étoiles"""
        try:
            data_norm = self.images_data['original']
            data_raw = self.images_data['original_raw']

            # Récupérer les paramètres
            fwhm = self.fwhm_slider.value() / 10.0
            threshold_sigma = self.threshold_slider.value() / 10.0
            radius = self.radius_slider.value() / 10.0

            kernel_sizes = {1: 3, 2: 5, 3: 7, 4: 9, 5: 11, 6: 13, 7: 15, 8: 17, 9: 19, 10: 21}
            kernel_size = kernel_sizes[self.kernel_slider.value()]
            iterations = self.iter_slider.value()

            gauss_sigma = self.gauss_slider.value() / 10.0
            mask_threshold = self.seuil_slider.value() / 100.0

            # --- ÉTAPE 1: Détecter les étoiles (masque binaire) ---
            masque_brut, sources = detect_stars(
                data_raw,
                fwhm=fwhm,
                threshold_sigma=threshold_sigma,
                radius=radius
            )
            self.images_data['masque_brut'] = masque_brut

            if sources is not None:
                self.nb_etoiles = len(sources)
            else:
                self.nb_etoiles = 0

            # --- ÉTAPE 2: Appliquer le flou gaussien sur le masque ---
            masque_lisse = smooth_mask(masque_brut, sigma=gauss_sigma, threshold=mask_threshold)
            self.images_data['masque_lisse'] = masque_lisse

            # --- ÉTAPE 3: Créer l'image érodée ---
            data_erodee = apply_erosion(data_norm, kernel_size=kernel_size, iterations=iterations)
            self.images_data['erodee'] = data_erodee

            # --- ÉTAPE 4: Calculer l'image finale ---
            # I_final = (M × I_erode) + ((1 - M) × I_original)
            image_finale = compute_final_image(data_norm, data_erodee, masque_lisse)
            self.images_data['finale'] = image_finale

            # Afficher l'image finale
            self.canvas_finale.display_image(image_finale, "Finale (étoiles réduites)")

            # Mettre à jour le statut
            self.statusBar().showMessage(f"Traitement terminé - {self.nb_etoiles} étoiles détectées")
            self.info_label.setText(
                f"Étoiles détectées: {self.nb_etoiles}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du traitement:\n{str(e)}")
            self.statusBar().showMessage("Erreur")
    
    def retraiter(self):
        """Retraite l'image avec les nouveaux paramètres"""
        if self.images_data['original'] is None:
            return

        self.statusBar().showMessage("Retraitement en cours...")
        self.traiter_image()

    def show_erodee(self):
        """Affiche la fenêtre avec l'image érodée"""
        if self.images_data['erodee'] is None:
            QMessageBox.warning(self, "Attention", "Veuillez charger une image d'abord")
            return
        
        self.show_zoom(self.images_data['erodee'], "Image Érodée")

    def show_masque(self):
        """Affiche la fenêtre avec le masque d'étoiles"""
        if self.images_data['masque_lisse'] is None:
            QMessageBox.warning(self, "Attention", "Veuillez charger une image d'abord")
            return

        self.show_zoom(self.images_data['masque_lisse'], f"Masque d'étoiles ({self.nb_etoiles} détectées)")

    def show_zoom(self, data, title):
        """Affiche une fenêtre de zoom"""
        if data is None:
            QMessageBox.warning(self, "Attention", "Aucune image à afficher")
            return
        
        zoom_window = ZoomWindow(data, title)
        zoom_window.show()
        
        # Garder une référence pour éviter le garbage collection
        self.zoom_windows.append(zoom_window)
    
    def show_comparateur(self):
        """Affiche la fenêtre de comparaison"""
        if self.images_data['original'] is None or self.images_data['finale'] is None:
            QMessageBox.warning(self, "Attention", "Veuillez charger et traiter une image d'abord")
            return

        comparator = ComparatorWindow(self.images_data['original'], self.images_data['finale'])
        comparator.show()
        self.zoom_windows.append(comparator)

    def reinitialiser(self):
        """Réinitialise les sliders"""
        # Remettre les sliders à leurs valeurs par défaut
        self.fwhm_slider.setValue(30)
        self.threshold_slider.setValue(55)
        self.radius_slider.setValue(35)
        self.kernel_slider.setValue(1)
        self.iter_slider.setValue(2)
        self.gauss_slider.setValue(20)
        self.seuil_slider.setValue(10)

        # Si une image est chargée, la retraiter avec les valeurs par défaut
        if self.images_data['original'] is not None:
            self.statusBar().showMessage("Réinitialisation des paramètres...")
        else:
            self.statusBar().showMessage("En attente d'une image...")


def main():
    app = QApplication(sys.argv)
    window = ReductionAstroApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
