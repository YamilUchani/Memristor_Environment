"""
neuromorphic_lab.main
=====================
Punto de entrada principal para la interfaz Tkinter de NeuromorphicLab.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk

# Asegurar import de neurolab
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from neurolab.gui.styles import COLORS
from neurolab.gui.tabs.validation_tab import ValidationTab
from neurolab.gui.tabs.plasticidad_tab import PlasticidadTab


class NeuromorphicLab(tk.Tk):
    """Aplicación Principal NeuromorphicLab Tkinter."""

    def __init__(self):
        super().__init__()
        self.title("Neuromorphic Lab — Plataforma de Validación y Simulación")
        self.geometry("1280x850")
        self.configure(bg=COLORS['bg_panel'])

        self._build_ui()

    def _build_ui(self):
        # Estilo del Notebook
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=COLORS['bg_panel'], borderwidth=0)
        style.configure('TNotebook.Tab', background=COLORS['bg_dark'], foreground=COLORS['text'], padding=[12, 6])
        style.map('TNotebook.Tab', background=[('selected', COLORS['bg_panel'])], foreground=[('selected', COLORS['success'])])

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # --- Pestaña: Plasticidad Sináptica ---
        self.plasticity_tab = PlasticidadTab(self.notebook)
        self.notebook.add(self.plasticity_tab, text='  ⚡ Plasticidad (Fase 3)  ')

        # --- Pestaña: Validación Papers ---
        self.validation_tab = ValidationTab(self.notebook)
        self.notebook.add(self.validation_tab, text='  📊 Validación Papers  ')


def main():
    app = NeuromorphicLab()
    app.mainloop()


if __name__ == '__main__':
    main()
