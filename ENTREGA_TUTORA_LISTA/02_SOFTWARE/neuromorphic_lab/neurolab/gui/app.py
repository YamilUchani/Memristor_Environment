import sys
import os
import signal
import atexit
import ctypes
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QtMsgType, QMessageLogContext, qInstallMessageHandler, qFormatLogMessage
from neurolab.gui.main_window import MainWindow


def _save_and_restore_console():
    """Guarda los modos de consola de Windows (STDIN / STDOUT) y los restablece al salir."""
    if os.name != "nt":
        return lambda: None
    try:
        kernel32 = ctypes.windll.kernel32
        h_in = kernel32.GetStdHandle(-10)   # STD_INPUT_HANDLE
        h_out = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        
        mode_in = ctypes.c_ulong()
        mode_out = ctypes.c_ulong()
        
        got_in = (kernel32.GetConsoleMode(h_in, ctypes.byref(mode_in)) != 0)
        got_out = (kernel32.GetConsoleMode(h_out, ctypes.byref(mode_out)) != 0)
        
        orig_in = mode_in.value if got_in else None
        orig_out = mode_out.value if got_out else None

        def restore():
            try:
                if got_in and orig_in is not None:
                    kernel32.SetConsoleMode(h_in, orig_in)
                if got_out and orig_out is not None:
                    kernel32.SetConsoleMode(h_out, orig_out)
            except Exception:
                pass

        atexit.register(restore)
        return restore
    except Exception:
        return lambda: None


def _qt_message_handler(mode: QtMsgType, context: QMessageLogContext, message: str):
    """Filtra avisos benignos de Qt y re-emite el resto con el formato por defecto."""
    if "propagateSizeHints" in message or "does not support raise()" in message:
        return
    formatted = qFormatLogMessage(mode, context, message).rstrip("\n")
    sys.stderr.write(formatted + "\n")
    sys.stderr.flush()


def main():
    """Punto de entrada para la ejecución de la GUI neurolab."""
    restore_console = _save_and_restore_console()

    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

    print("Neuromorphic Lab » inicializando interfaz gráfica...", flush=True)
    print("  (puede tardar unos segundos en la primera carga)", flush=True)

    def _sigint_handler(sig, frame):
        restore_console()
        try:
            app_inst = QApplication.instance()
            if app_inst:
                app_inst.quit()
        except Exception:
            pass
        os._exit(0)

    # Permitir que Ctrl+C termine la aplicación y libere la consola inmediatamente
    signal.signal(signal.SIGINT, _sigint_handler)

    # Filtro de avisos benignos de Qt (antes de crear QApplication)
    qInstallMessageHandler(_qt_message_handler)

    # PySide6 >= 6.10 ya no empaqueta fuentes; si no hay QT_QPA_FONTDIR, apuntar al
    # directorio de fuentes del SO para evitar el aviso "Cannot find font directory".
    if os.name == "nt" and not os.environ.get("QT_QPA_FONTDIR"):
        _fonts = r"C:\Windows\Fonts"
        if os.path.isdir(_fonts):
            os.environ["QT_QPA_FONTDIR"] = _fonts

    if os.name == "nt" and os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        if os.environ.get("NEUROLAB_HEADLESS") == "1":
            print("[AVISO] Modo headless forzado (NEUROLAB_HEADLESS=1): ventana invisible.", flush=True)
        else:
            del os.environ["QT_QPA_PLATFORM"]
            print("[AVISO] Se ignoró QT_QPA_PLATFORM=offscreen de esta sesión: la ventana", flush=True)
            print("       sería invisible. Para forzar modo headless usa NEUROLAB_HEADLESS=1.", flush=True)

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

    print("[OK] Neuromorphic Lab » ventana abierta.", flush=True)

    exit_code = 0
    try:
        exit_code = app.exec()
    finally:
        restore_console()
        sys.stdout.flush()
        sys.stderr.flush()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
