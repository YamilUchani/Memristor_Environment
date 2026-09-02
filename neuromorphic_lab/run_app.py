"""
Script ejecutor de la interfaz gráfica Neuromorphic Lab.
Uso: python run_app.py
"""

import sys
import os

# Asegurar que el paquete neurolab se encuentre en el PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from neurolab.gui.app import main

if __name__ == "__main__":
    main()
