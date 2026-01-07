"""
Interface graphique simple pour visualiser l'avant/après du traitement d'images FITS
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import cv2
from scipy import ndimage


class InterfaceAstro:
    def __init__(self, root):
        self.root = root
        self.root.title("Réduction Astro - Avant/Après")
        self.root.geometry("1400x800")
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.image_avant = None
        self.image_apres = None
        self.chemin_avant = None
        self.chemin_apres = None
        self.taille_image = (500, 500)  # Taille fixe pour toutes les images
        
        # Style
        self.couleur_principale = "#2c3e50"
        self.couleur_bouton = "#3498db"
        self.couleur_texte = "#ecf0f1"
        
        # Frame principal
        self.create_header()
        self.create_main_content()
        self.create_footer()
    
    def create_header(self):
        """Crée l'en-tête de l'application"""
        header = tk.Frame(self.root, bg=self.couleur_principale, height=80)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        
        titre = tk.Label(
            header,
            text="Réduction Astro - Visualisation Avant/Après",
            font=("Arial", 18, "bold"),
            bg=self.couleur_principale,
            fg=self.couleur_texte
        )
        titre.pack(pady=15)
    
    def create_main_content(self):
        """Crée le contenu principal avec les 4 images - Layout 2x2"""
        # Frame principal pour les images
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Ligne du haut (Original et Finale)
        top_frame = tk.Frame(main_frame, bg="#f0f0f0")
        top_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Colonne HAUT GAUCHE (Image originale)
        self.frame_original = tk.Frame(top_frame, bg="white", relief=tk.RAISED, bd=2)
        self.frame_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.frame_original.pack_propagate(False)
        
        titre_original = tk.Label(
            self.frame_original,
            text="IMAGE ORIGINALE",
            font=("Arial", 12, "bold"),
            bg=self.couleur_principale,
            fg=self.couleur_texte
        )
        titre_original.pack(fill=tk.X)
        
        self.canvas_original = tk.Canvas(
            self.frame_original,
            bg="black",
            highlightthickness=0,
            width=self.taille_image[0],
            height=self.taille_image[1]
        )
        self.canvas_original.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Colonne HAUT DROITE (Image finale)
        self.frame_finale = tk.Frame(top_frame, bg="white", relief=tk.RAISED, bd=2)
        self.frame_finale.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.frame_finale.pack_propagate(False)
        
        titre_finale = tk.Label(
            self.frame_finale,
            text="IMAGE FINALE",
            font=("Arial", 12, "bold"),
            bg=self.couleur_principale,
            fg=self.couleur_texte
        )
        titre_finale.pack(fill=tk.X)
        
        self.canvas_finale = tk.Canvas(
            self.frame_finale,
            bg="black",
            highlightthickness=0,
            width=self.taille_image[0],
            height=self.taille_image[1]
        )
        self.canvas_finale.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Ligne du bas (Érodée et Masque)
        bottom_frame = tk.Frame(main_frame, bg="#f0f0f0")
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Colonne BAS GAUCHE (Image érodée)
        self.frame_erodee = tk.Frame(bottom_frame, bg="white", relief=tk.RAISED, bd=2)
        self.frame_erodee.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.frame_erodee.pack_propagate(False)
        
        titre_erodee = tk.Label(
            self.frame_erodee,
            text="IMAGE ÉRODÉE",
            font=("Arial", 12, "bold"),
            bg=self.couleur_principale,
            fg=self.couleur_texte
        )
        titre_erodee.pack(fill=tk.X)
        
        self.canvas_erodee = tk.Canvas(
            self.frame_erodee,
            bg="black",
            highlightthickness=0,
            width=self.taille_image[0],
            height=self.taille_image[1]
        )
        self.canvas_erodee.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Colonne BAS DROITE (Masque)
        self.frame_masque = tk.Frame(bottom_frame, bg="white", relief=tk.RAISED, bd=2)
        self.frame_masque.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.frame_masque.pack_propagate(False)
        
        titre_masque = tk.Label(
            self.frame_masque,
            text="MASQUE D'ÉTOILES",
            font=("Arial", 12, "bold"),
            bg=self.couleur_principale,
            fg=self.couleur_texte
        )
        titre_masque.pack(fill=tk.X)
        
        self.canvas_masque = tk.Canvas(
            self.frame_masque,
            bg="black",
            highlightthickness=0,
            width=self.taille_image[0],
            height=self.taille_image[1]
        )
        self.canvas_masque.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Label de statut
        self.label_statut = tk.Label(
            main_frame,
            text="En attente d'une image...",
            font=("Arial", 9),
            bg="#ecf0f1",
            fg="#7f8c8d"
        )
        self.label_statut.pack(fill=tk.X, padx=5, pady=5)
        
        # Stockage des données d'images pour le zoom
        self.images_data = {
            'original': None,
            'erodee': None,
            'masque': None,
            'finale': None
        }
        
        # Stockage des canvas matplotlib pour les bindings
        self.canvas_mpl = {
            'original': None,
            'erodee': None,
            'masque': None,
            'finale': None
        }
        
        # Binding des clics sur les canvas
        self.canvas_original.bind("<Button-1>", lambda e: self.zoom_image(self.images_data['original'], "Image Originale"))
        self.canvas_erodee.bind("<Button-1>", lambda e: self.zoom_image(self.images_data['erodee'], "Image Érodée"))
        self.canvas_masque.bind("<Button-1>", lambda e: self.zoom_image(self.images_data['masque'], "Masque d'Étoiles"))
        self.canvas_finale.bind("<Button-1>", lambda e: self.zoom_image(self.images_data['finale'], "Image Finale"))
    
    def create_footer(self):
        """Crée le pied de page avec les boutons d'action"""
        footer = tk.Frame(self.root, bg=self.couleur_principale, height=60)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        footer.pack_propagate(False)
        
        # Frame pour les boutons
        btn_frame = tk.Frame(footer, bg=self.couleur_principale)
        btn_frame.pack(pady=10)
        
        btn_charger = tk.Button(
            btn_frame,
            text="Charger et traiter",
            command=self.charger_et_traiter,
            bg=self.couleur_bouton,
            fg=self.couleur_texte,
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            padx=15,
            pady=8
        )
        btn_charger.pack(side=tk.LEFT, padx=5)
        
        btn_reinitialiser = tk.Button(
            btn_frame,
            text="Réinitialiser",
            command=self.reinitialiser,
            bg="#e74c3c",
            fg=self.couleur_texte,
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            padx=15,
            pady=8
        )
        btn_reinitialiser.pack(side=tk.LEFT, padx=5)
    
    def charger_image_fits(self):
        """Ouvre un dialogue pour charger une image FITS"""
        fichier = filedialog.askopenfilename(
            title="Sélectionner une image FITS",
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")],
            initialdir="examples"
        )
        return fichier
    
    def afficher_image_fits(self, chemin, canvas):
        """Affiche une image FITS sur un canvas"""
        if not chemin:
            return
        
        try:
            # Charger l'image FITS
            with fits.open(chemin) as hdul:
                data = hdul[0].data.astype(float)
            
            # Gérer les images en couleur
            if data.ndim == 3:
                if data.shape[0] == 3:
                    data = np.transpose(data, (1, 2, 0))
            
            # Normaliser l'image entre 0 et 1
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            if data_max > data_min:
                data_norm = (data - data_min) / (data_max - data_min)
            else:
                data_norm = data
            
            # Stocker pour le zoom
            self.images_data['original'] = data_norm
            
            # Créer une figure matplotlib
            fig = Figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            
            # Afficher l'image
            if data_norm.ndim == 3:
                im = ax.imshow(data_norm, origin='upper')
            else:
                im = ax.imshow(data_norm, cmap='viridis', origin='upper')
            
            ax.set_title(os.path.basename(chemin), fontsize=10, color="white")
            ax.axis('off')
            fig.patch.set_facecolor('black')
            
            # Intégrer dans tkinter
            for widget in canvas.winfo_children():
                widget.destroy()
            
            canvas_tk = FigureCanvasTkAgg(fig, master=canvas)
            canvas_tk.draw()
            canvas_widget = canvas_tk.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            
            # Binder le clic sur le canvas matplotlib
            canvas_tk.mpl_connect('button_press_event', lambda e: self.zoom_image(self.images_data['original'], "Image Originale"))
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image:\n{str(e)}")
    
    def charger_et_traiter(self):
        """Charge une image et applique automatiquement le traitement complet"""
        chemin = self.charger_image_fits()
        if not chemin:
            return
        
        self.chemin_avant = chemin
        self.label_statut.config(text="Traitement en cours...", fg="#e67e22")
        self.root.update()
        
        try:
            # 1. Afficher l'image originale
            self.afficher_image_fits(chemin, self.canvas_original)
            
            # 2. Charger et appliquer l'érosion
            with fits.open(chemin) as hdul:
                data = hdul[0].data.astype(float)
            
            # Gérer les images en couleur ou monochrome
            if data.ndim == 3:
                if data.shape[0] == 3:
                    data = np.transpose(data, (1, 2, 0))
            
            # Appliquer l'érosion
            data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
            
            if data.ndim == 3:
                data_uint8 = (data_norm * 255).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                data_eroded = cv2.erode(data_uint8, kernel, iterations=2)
                data_result_eroded = data_eroded.astype(float) / 255.0
            else:
                data_uint8 = (data_norm * 255).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                data_eroded = cv2.erode(data_uint8, kernel, iterations=2)
                data_result_eroded = data_eroded.astype(float) / 255.0
            
            # Afficher l'image érodée
            self.afficher_image_traitee(data_result_eroded, self.canvas_erodee, "Après érosion")
            
            # 3. Créer le masque d'étoiles (différence entre original et érodé)
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
            
            # Afficher le masque
            self.afficher_image_traitee(masque_final, self.canvas_masque, "Masque")
            
            # 4. Appliquer la réduction localisée
            if data.ndim == 3:
                masque_3d = np.stack([masque_final, masque_final, masque_final], axis=2)
                image_finale = data_norm * (1 - masque_3d) + data_result_eroded * masque_3d
            else:
                image_finale = data_norm * (1 - masque_final) + data_result_eroded * masque_final
            
            # Afficher l'image finale
            self.afficher_image_traitee(image_finale, self.canvas_finale, "Image finale")
            
            self.label_statut.config(text="✓ Traitement terminé", fg="#27ae60")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du traitement:\n{str(e)}")
            self.label_statut.config(text="✗ Erreur", fg="#e74c3c")
            import traceback
            traceback.print_exc()
    
    def afficher_image_traitee(self, data, canvas, titre):
        """Affiche une image traitée (numpy array) sur le canvas spécifié"""
        try:
            # Normaliser les données entre 0 et 1
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            if data_max > data_min:
                data_norm = (data - data_min) / (data_max - data_min)
            else:
                data_norm = data
            
            # Stocker les données pour le zoom
            if "original" in titre.lower():
                self.images_data['original'] = data_norm
                key = 'original'
            elif "érod" in titre.lower():
                self.images_data['erodee'] = data_norm
                key = 'erodee'
            elif "masque" in titre.lower():
                self.images_data['masque'] = data_norm
                key = 'masque'
            elif "final" in titre.lower():
                self.images_data['finale'] = data_norm
                key = 'finale'
            else:
                key = None
            
            # Créer une figure matplotlib
            fig = Figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            
            # Afficher l'image
            if data_norm.ndim == 3:
                im = ax.imshow(data_norm, origin='upper')
            else:
                im = ax.imshow(data_norm, cmap='viridis', origin='upper')
            
            ax.set_title(titre, fontsize=10, color="white")
            ax.axis('off')
            fig.patch.set_facecolor('black')
            
            # Intégrer dans tkinter
            for widget in canvas.winfo_children():
                widget.destroy()
            
            canvas_tk = FigureCanvasTkAgg(fig, master=canvas)
            canvas_tk.draw()
            canvas_widget = canvas_tk.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            
            # Binder le clic sur le canvas matplotlib
            if key:
                canvas_tk.mpl_connect('button_press_event', lambda e: self.zoom_image(self.images_data[key], titre))
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher l'image:\n{str(e)}")
    
    def zoom_image(self, data, titre):
        """Affiche une image en plein écran avec zoom"""
        if data is None:
            messagebox.showwarning("Attention", "Aucune image à afficher")
            return
        
        # Créer une nouvelle fenêtre
        zoom_window = tk.Toplevel(self.root)
        zoom_window.title(f"Zoom - {titre}")
        zoom_window.geometry("900x900")
        
        # Normaliser les données
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        if data_max > data_min:
            data_norm = (data - data_min) / (data_max - data_min)
        else:
            data_norm = data
        
        # Créer la figure
        fig = Figure(figsize=(9, 9), dpi=100)
        ax = fig.add_subplot(111)
        
        # Afficher l'image
        if data_norm.ndim == 3:
            im = ax.imshow(data_norm, origin='upper')
        else:
            im = ax.imshow(data_norm, cmap='viridis', origin='upper')
        
        ax.set_title(titre, fontsize=14, color="white", pad=15)
        ax.axis('off')
        fig.patch.set_facecolor('black')
        
        # Intégrer le canvas dans la fenêtre
        canvas_zoom = FigureCanvasTkAgg(fig, master=zoom_window)
        canvas_zoom.draw()
        canvas_zoom.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Ajouter un bouton de fermeture
        btn_close = tk.Button(
            zoom_window,
            text="Fermer",
            command=zoom_window.destroy,
            bg=self.couleur_principale,
            fg=self.couleur_texte,
            font=("Arial", 10, "bold")
        )
        btn_close.pack(pady=10)
    
    def charger_avant(self):
        """Charge l'image AVANT"""
        self.charger_et_traiter()
    
    def charger_apres(self):
        """Deprecated - Gardé pour compatibilité"""
        pass
    
    def reinitialiser(self):
        """Réinitialise l'interface"""
        self.chemin_avant = None
        self.chemin_apres = None
        
        # Vider les canvas
        for canvas in [self.canvas_original, self.canvas_erodee, self.canvas_masque, self.canvas_finale]:
            canvas.delete("all")
            canvas.create_text(
                canvas.winfo_width()//2,
                canvas.winfo_height()//2,
                text="Aucune image chargée",
                fill="gray",
                font=("Arial", 12)
            )
        
        self.label_statut.config(text="En attente d'une image...", fg="#7f8c8d")


def main():
    root = tk.Tk()
    app = InterfaceAstro(root)
    root.mainloop()


if __name__ == "__main__":
    main()
