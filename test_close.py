import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Asegurar que importamos del lugar correcto
sys.path.insert(0, os.path.abspath("neuromorphic_lab"))

from neurolab.gui.app import main
from neurolab.gui.main_window import MainWindow

app = QApplication(sys.argv)
window = MainWindow()
window.show()

def close_app():
    window.close()
    app.quit()

QTimer.singleShot(1000, close_app)
sys.exit(app.exec())
