import sys
import os
import signal
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QtMsgType, QMessageLogContext, qInstallMessageHandler, qFormatLogMessage
from neurolab.gui.main_window import MainWindow


def _qt_message_handler(mode: QtMsgType, context: QMessageLogContext, message: str):
    """Filtra avisos benignos de Qt y re-emite el resto con el formato por defecto."""
    # "This plugin does not support propagateSizeHints()" lo emite QPlatformWindow en
    # los plugins de plataforma (windows/offscreen) cuando el layout pide propagar
    # hints de tamaño; es inofensivo y solo ensucia la consola.
    if "propagateSizeHints" in message:
        return
    formatted = qFormatLogMessage(mode, context, message).rstrip("\n")
    sys.stderr.write(formatted + "\n")
    sys.stderr.flush()


def main():
    """Punto de entrada para la ejecución de la GUI neurolab."""
    print("Neuromorphic Lab » inicializando interfaz gráfica...")
    print("  (puede tardar unos segundos en la primera carga)")

    # Permitir que el sistema maneje SIGINT (Ctrl+C) limpiamente sin traceback de C++ eventFilter
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Filtro de avisos benignos de Qt (antes de crear QApplication)
    qInstallMessageHandler(_qt_message_handler)

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

    # En Windows la ventana puede quedar detrás de la consola: traerla al frente
    window.raise_()
    window.activateWindow()

    print("✅ Neuromorphic Lab » ventana abierta.")
    print("   La consola queda 'bloqueada' mientras la ventana esté abierta (es normal).")
    print("   Para salir: cierra la ventana o presiona Ctrl+C.")
    sys.stdout.flush()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
