import sys
import os

# Asegurar que neuromorphic_lab/ esté en el PYTHONPATH (ejecución desde la raíz)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuromorphic_lab"))

from PySide6.QtWidgets import QApplication
from neurolab.gui.main_window import MainWindow
import time

app = QApplication(sys.argv)
win = MainWindow()
win.show()

# Cambiar a pestaña 3
win.tabs.setCurrentIndex(2)

# Abrir ventana de memristor
win.memristor_window.show()

# Cambiar señal a Sinusoidal
win.hybrid_signal_panel.combo_waveform.setCurrentText("Sinusoidal")

# Forzar ejecucion
win.run_simulation()

# Imprimir stats
v_drop = win.memristor_window.plot_canvas.ax_vi_t.lines[0].get_ydata()
print("V_drop max:", max(v_drop), "min:", min(v_drop))
print("First 10 values:", v_drop[:10])
sys.exit(0)
