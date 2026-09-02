"""
Script ejecutor de la interfaz gráfica Neuromorphic Lab.

Permite arrancar la GUI desde la raíz del repositorio SIN necesidad de
moverse a la carpeta neuromorphic_lab/:

    python run_app.py

También funciona desde dentro de la carpeta:
    cd neuromorphic_lab && python run_app.py
"""

import os
import sys

# Ruta absoluta a la carpeta que contiene el paquete `neurolab`
_NEUROLAB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuromorphic_lab")

# Asegurar que el paquete `neurolab` esté en el PYTHONPATH
if _NEUROLAB_DIR not in sys.path:
    sys.path.insert(0, _NEUROLAB_DIR)

from neurolab.gui.app import main  # noqa: E402

if __name__ == "__main__":
    main()