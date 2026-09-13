import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuromorphic_lab"))

from PySide6.QtWidgets import QApplication
from neurolab.gui.main_window import MainWindow

app = QApplication.instance() or QApplication(sys.argv)
win = MainWindow()

# La ventana principal usa docks (ya no hay QTabWidget principal).
# run_simulation() despacha a cada simulación según el dock visible.
win.run_simulation()

# Ejecutar de forma explícita los 3 motores de simulación para validación headless
win._run_memristor_simulation()
win._run_neuron_simulation()
win._run_hybrid_simulation()

print("[OK] MainWindow initialized and all 3 simulations executed successfully.")

