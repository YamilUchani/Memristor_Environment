import sys
import os
import signal
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from neurolab.gui.main_window import MainWindow

def main():
    """Punto de entrada para la ejecución de la GUI neurolab."""
    # Permitir que el sistema maneje SIGINT (Ctrl+C) limpiamente sin traceback de C++ eventFilter
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # PySide6 >= 6.10 ya no empaqueta fuentes; si no hay QT_QPA_FONTDIR, apuntar al
    # directorio de fuentes del SO para evitar el aviso "Cannot find font directory".
    if os.name == "nt" and not os.environ.get("QT_QPA_FONTDIR"):
        _fonts = r"C:\Windows\Fonts"
        if os.path.isdir(_fonts):
            os.environ["QT_QPA_FONTDIR"] = _fonts

    app = QApplication(sys.argv)

    # Timer periódico para permitir que Python procese eventos de interrupción de teclado
    sigint_timer = QTimer()
    sigint_timer.start(200)
    sigint_timer.timeout.connect(lambda: None)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
