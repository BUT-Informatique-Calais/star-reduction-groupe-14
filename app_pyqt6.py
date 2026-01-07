"""
Interface PyQt6 pour la réduction astro - Avant/Après
"""

import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
import cv2
from scipy import ndimage
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QGridLayout, QMessageBox, QStatusBar
)
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt, pyqtSignal


class ImageCanvas(FigureCanvas):
    """Canvas matplotlib intégré dans PyQt6"""
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


class ReductionAstroApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Réduction Astro - Avant/Après")
        self.setGeometry(50, 50, 1400, 850)
        
        # Styles
        self.couleur_principale = "#2c3e50"
        self.couleur_bouton = "#3498db"
        self.couleur_texte = "#ecf0f1"
        
        # Données
        self.images_data = {
            'original': None,
            'erodee': None,
            'masque': None,
            'finale': None
        }
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
        header_label = QLabel("Réduction Astro - Visualisation Avant/Après")
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
        images_layout.addWidget(self.create_panel("IMAGE FINALE", self.canvas_finale), 0, 1)
        
        main_layout.addLayout(images_layout, 1)
        
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
        
        btn_reinitialiser = QPushButton("Réinitialiser")
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
    
    def charger_et_traiter(self):
        """Charge et traite une image"""
        chemin, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner une image FITS",
            "examples",
            "FITS Files (*.fits *.fit);;All Files (*)"
        )
        
        if not chemin:
            return
        
        self.statusBar().showMessage("Traitement en cours...")
        
        try:
            # Charger l'image
            with fits.open(chemin) as hdul:
                data = hdul[0].data.astype(float)
            
            # Gérer les dimensions
            if data.ndim == 3:
                if data.shape[0] == 3:
                    data = np.transpose(data, (1, 2, 0))
            
            # Normaliser
            data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
            
            # Afficher l'original
            self.images_data['original'] = data_norm
            self.canvas_original.display_image(data_norm, "Originale")
            
            # Érosion
            if data_norm.ndim == 3:
                data_uint8 = (data_norm * 255).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                data_eroded = cv2.erode(data_uint8, kernel, iterations=2)
                data_result_eroded = data_eroded.astype(float) / 255.0
            else:
                data_uint8 = (data_norm * 255).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                data_eroded = cv2.erode(data_uint8, kernel, iterations=2)
                data_result_eroded = data_eroded.astype(float) / 255.0
            
            # Afficher érodée (stockage uniquement, pas d'affichage)
            self.images_data['erodee'] = data_result_eroded
            
            # Créer le masque
            if data_norm.ndim == 3:
                data_norm_gray = np.mean(data_norm, axis=2)
                data_eroded_gray = np.mean(data_result_eroded, axis=2)
            else:
                data_norm_gray = data_norm
                data_eroded_gray = data_result_eroded
            
            masque_brut = np.abs(data_norm_gray - data_eroded_gray)
            masque_lisse = ndimage.gaussian_filter(masque_brut.astype(np.float32), sigma=6.0)
            masque_norm = masque_lisse / (np.nanmax(masque_lisse) + 1e-8)
            masque_norm = np.where(masque_norm > 0.05, masque_norm, 0)
            masque_final = ndimage.gaussian_filter(masque_norm, sigma=2.0)
            
            # Stocker le masque (pas d'affichage)
            self.images_data['masque'] = masque_final
            
            # Image finale
            if data_norm.ndim == 3:
                masque_3d = np.stack([masque_final, masque_final, masque_final], axis=2)
                image_finale = data_norm * (1 - masque_3d) + data_result_eroded * masque_3d
            else:
                image_finale = data_norm * (1 - masque_final) + data_result_eroded * masque_final
            
            # Afficher finale
            self.images_data['finale'] = image_finale
            self.canvas_finale.display_image(image_finale, "Finale")
            
            self.statusBar().showMessage("✓ Traitement terminé")
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du traitement:\n{str(e)}")
            self.statusBar().showMessage("✗ Erreur")
    
    def show_erodee(self):
        """Affiche la fenêtre avec l'image érodée"""
        if self.images_data['erodee'] is None:
            QMessageBox.warning(self, "Attention", "Veuillez charger une image d'abord")
            return
        
        self.show_zoom(self.images_data['erodee'], "Image Érodée")
    
    def show_zoom(self, data, title):
        """Affiche une fenêtre de zoom"""
        if data is None:
            QMessageBox.warning(self, "Attention", "Aucune image à afficher")
            return
        
        zoom_window = ZoomWindow(data, title)
        zoom_window.show()
        
        # Garder une référence pour éviter le garbage collection
        self.zoom_windows.append(zoom_window)
    
    def reinitialiser(self):
        """Réinitialise l'interface"""
        self.images_data = {
            'original': None,
            'erodee': None,
            'masque': None,
            'finale': None
        }
        
        for canvas in [self.canvas_original, self.canvas_finale]:
            canvas.ax.clear()
            canvas.ax.text(0.5, 0.5, 'Aucune image chargée', 
                          ha='center', va='center', color='gray', fontsize=12,
                          transform=canvas.ax.transAxes)
            canvas.ax.axis('off')
            canvas.draw()
        
        self.statusBar().showMessage("En attente d'une image...")


def main():
    app = QApplication(sys.argv)
    window = ReductionAstroApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
