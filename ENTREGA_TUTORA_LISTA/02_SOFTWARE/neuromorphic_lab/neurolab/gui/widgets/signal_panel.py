from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QComboBox,
    QPushButton, QCheckBox, QLabel
)
from PySide6.QtCore import Signal
import numpy as np
from neurolab.gui.widgets.arrow_spinbox import ArrowDoubleSpinBox, ArrowSpinBox

# Límite de seguridad para la señal: evita colapso de memoria con duraciones/dts
# extremos (e.g. 10 000 s @ 0.001 ms -> 1e10 pasos). El dt se escala hacia arriba.
MAX_SIGNAL_STEPS = 100_000

class SignalPanel(QWidget):
    """
    Panel de configuración para la fuente de voltaje de excitación experimental.
    Utiliza botones independientes de flechas (▲ y ▼) fuera de la barra de texto.
    """
    run_simulation_requested = Signal()
    param_changed = Signal()

    def __init__(self, parent=None, mode="voltage"):
        super().__init__(parent)
        self.mode = mode
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        group = QGroupBox("5. 📡 Configuración Experimental de Señal")
        form = QFormLayout()

        # Interruptor de Actualización en Tiempo Real
        self.chk_realtime = QCheckBox("⚡ Simulación en Tiempo Real")
        self.chk_realtime.setChecked(True)
        self.chk_realtime.setStyleSheet("font-weight: bold; color: #a6e3a1;")

        self.combo_waveform = QComboBox()
        self.combo_waveform.addItems([
            "Sinusoidal",
            "Triangular",
            "Pulsos Cuadrados",
            "Tren de Pulsos (Unipolar)",
            "Tren de Pulsos en Ráfagas (Burst)",
            "Corriente Constante"
        ])

        # Controles numéricos con flechas independientes
        if self.mode == "current":
            self.spin_v0 = ArrowDoubleSpinBox(value=1.0, min_val=-1000.0, max_val=1000.0, step=1.0, suffix=" µA")
            amp_label = "Amplitud I_0:"
        else:
            self.spin_v0 = ArrowDoubleSpinBox(value=1.0, min_val=-1000.0, max_val=1000.0, step=1.0, suffix=" V")
            amp_label = "Amplitud V_0:"
        self.spin_f0 = ArrowDoubleSpinBox(value=100.0, min_val=0.001, max_val=10000.0, step=1.0, decimals=3, suffix=" Hz")
        
        # Parámetros de Ráfaga (Burst)
        self.spin_t_on = ArrowDoubleSpinBox(value=1.0, min_val=0.001, max_val=1000.0, step=0.1, decimals=3, suffix=" s")
        self.spin_t_off = ArrowDoubleSpinBox(value=1.0, min_val=0.001, max_val=1000.0, step=0.1, decimals=3, suffix=" s")
        self.lbl_t_on = QLabel("Tiempo ON (T_on):")
        self.lbl_t_off = QLabel("Tiempo OFF (T_off):")

        self.spin_duration = ArrowDoubleSpinBox(value=0.05, min_val=0.0001, max_val=10000.0, step=0.01, decimals=4, suffix=" s")
        self.spin_dt_ms = ArrowDoubleSpinBox(value=0.001, min_val=0.0001, max_val=10.0, step=0.0005, decimals=5, suffix=" ms")

        form.addRow(self.chk_realtime)
        form.addRow("Forma de Onda:", self.combo_waveform)
        form.addRow(amp_label, self.spin_v0)
        form.addRow("Frecuencia f_0:", self.spin_f0)
        form.addRow(self.lbl_t_on, self.spin_t_on)
        form.addRow(self.lbl_t_off, self.spin_t_off)
        form.addRow("Duración Total:", self.spin_duration)
        form.addRow("Paso Temporal Δt:", self.spin_dt_ms)
        group.setLayout(form)
        layout.addWidget(group)

        # Botón de Simulación
        self.btn_run = QPushButton("▶ EJECUTAR SIMULACIÓN")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #11111b;
                font-weight: bold;
                font-size: 13px;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
        """)
        self.btn_run.clicked.connect(self.run_simulation_requested.emit)
        layout.addWidget(self.btn_run)

        self.setLayout(layout)
        self._connect_signals()

        self.combo_waveform.currentTextChanged.connect(self._update_burst_visibility)
        self._update_burst_visibility(self.combo_waveform.currentText())

    def _update_burst_visibility(self, text: str):
        """Muestra u oculta los campos de T_on y T_off según la forma de onda elegida."""
        is_burst = (text == "Tren de Pulsos en Ráfagas (Burst)")
        self.lbl_t_on.setVisible(is_burst)
        self.spin_t_on.setVisible(is_burst)
        self.lbl_t_off.setVisible(is_burst)
        self.spin_t_off.setVisible(is_burst)

    def _connect_signals(self):
        """Conecta cambios de la fuente de señal para emitir param_changed."""
        self.combo_waveform.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_v0.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_f0.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_t_on.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_t_off.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_duration.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_dt_ms.valueChanged.connect(lambda *_: self.param_changed.emit())

    @property
    def is_realtime_enabled(self) -> bool:
        """Devuelve True si el modo de actualización en tiempo real está activo."""
        return self.chk_realtime.isChecked()

    def to_dict(self) -> dict:
        """Exporta la configuración de señal a un diccionario JSON."""
        return {
            "realtime_enabled": self.chk_realtime.isChecked(),
            "waveform": self.combo_waveform.currentText(),
            "v0": self.spin_v0.value(),
            "f0": self.spin_f0.value(),
            "t_on": self.spin_t_on.value(),
            "t_off": self.spin_t_off.value(),
            "duration": self.spin_duration.value(),
            "dt_ms": self.spin_dt_ms.value(),
        }

    def from_dict(self, data: dict):
        """Carga la configuración de señal desde un diccionario JSON de forma silenciosa e instantánea."""
        was_blocked = self.signalsBlocked()
        self.blockSignals(True)
        try:
            if "realtime_enabled" in data:
                self.chk_realtime.setChecked(bool(data["realtime_enabled"]))
            if "waveform" in data:
                idx = self.combo_waveform.findText(data["waveform"])
                if idx >= 0:
                    self.combo_waveform.setCurrentIndex(idx)
            if "v0" in data: self.spin_v0.setValue(data["v0"])
            if "f0" in data: self.spin_f0.setValue(data["f0"])
            if "t_on" in data: self.spin_t_on.setValue(data["t_on"])
            if "t_off" in data: self.spin_t_off.setValue(data["t_off"])
            if "duration" in data: self.spin_duration.setValue(data["duration"])
            if "dt_ms" in data: self.spin_dt_ms.setValue(data["dt_ms"])
        finally:
            self.blockSignals(was_blocked)

        self._update_burst_visibility(self.combo_waveform.currentText())
        self.param_changed.emit()

    def reset_defaults(self):
        """Restablece los parámetros por defecto de la señal experimental del Paper Strukov 2008."""
        self.chk_realtime.setChecked(True)
        self.combo_waveform.setCurrentIndex(0)
        self.spin_v0.setValue(1.0)
        self.spin_f0.setValue(100.0)      # 100 Hz
        self.spin_t_on.setValue(1.0)
        self.spin_t_off.setValue(1.0)
        self.spin_duration.setValue(0.05)  # 50 ms -> 5 ciclos completos a 100 Hz
        self.spin_dt_ms.setValue(0.001)    # 0.001 ms -> 1 µs (50 000 pasos)

    def generate_voltage_signal(self):
        """
        Genera el vector de tiempo t y el vector de voltaje v según la configuración.

        Returns:
            tuple (t, v, dt_sec)
        """
        v0 = self.spin_v0.value()
        f0 = self.spin_f0.value()
        duration = self.spin_duration.value()
        dt_sec = self.spin_dt_ms.value() * 1e-3  # ms a segundos

        steps = int(duration / dt_sec)
        if steps < 2:
            steps = 2
            dt_sec = duration / float(steps)
        if steps > MAX_SIGNAL_STEPS:
            # Límite de seguridad: no generar más de MAX_SIGNAL_STEPS pasos (evita colapso)
            dt_sec = duration / float(MAX_SIGNAL_STEPS)
            steps = MAX_SIGNAL_STEPS

        t = np.linspace(0.0, duration, steps)
        waveform = self.combo_waveform.currentText()

        if waveform == "Sinusoidal":
            v = v0 * np.sin(2.0 * np.pi * f0 * t)
        elif waveform == "Triangular":
            phase = (t * f0) % 1.0
            v = np.where(phase < 0.5, 4.0 * v0 * phase - v0, 3.0 * v0 - 4.0 * v0 * phase)
        elif waveform == "Pulsos Cuadrados":
            # Pulso cuadrado unipolar (50% de ciclo de trabajo) de 0 a v0
            v = np.where(np.sin(2.0 * np.pi * f0 * t) >= 0, v0, 0.0)
        elif waveform == "Tren de Pulsos (Unipolar)":
            # Pulso positivo estrecho (20% de ciclo de trabajo) de 0 a v0
            phase = (t * f0) % 1.0
            v = np.where(phase < 0.2, v0, 0.0)
        elif waveform == "Tren de Pulsos en Ráfagas (Burst)":
            # Ráfaga de pulsos: durante T_on se emiten pulsos (20% duty cycle) a frec f0, durante T_off silencio (0 V)
            t_on = self.spin_t_on.value()
            t_off = self.spin_t_off.value()
            t_period = t_on + t_off
            if t_period <= 0:
                t_period = 1.0
            burst_phase = (t % t_period)
            is_on = (burst_phase < t_on)
            pulse_phase = (t * f0) % 1.0
            is_pulse = (pulse_phase < 0.2)
            v = np.where(is_on & is_pulse, v0, 0.0)
        elif waveform == "Corriente Constante":
            v = np.full(steps, v0)
        else:
            v = v0 * np.sin(2.0 * np.pi * f0 * t)

        return t, v, dt_sec

