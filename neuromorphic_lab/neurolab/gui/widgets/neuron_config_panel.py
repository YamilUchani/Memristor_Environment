from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLabel, QComboBox, QHBoxLayout
)
from PySide6.QtCore import Signal
from neurolab.neurons import LIFConfig, LIFNeuron
from neurolab.gui.widgets.arrow_spinbox import ArrowDoubleSpinBox

class NeuronConfigPanel(QWidget):
    """Panel de configuración en tiempo real para los parámetros LIF."""
    param_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        group = QGroupBox("Parámetros LIF")
        layout = QFormLayout()

        self.spin_c_m = ArrowDoubleSpinBox(value=100.0, min_val=0.001, max_val=999999.0, step=1.0, suffix="")
        self.combo_c_unit = QComboBox()
        self.combo_c_unit.addItems(["pF", "nF", "uF", "mF"])
        self.combo_c_unit.setCurrentText("nF")
        lay_c = QHBoxLayout()
        lay_c.setContentsMargins(0,0,0,0)
        lay_c.addWidget(self.spin_c_m)
        lay_c.addWidget(self.combo_c_unit)

        self.spin_r_leak = ArrowDoubleSpinBox(value=1.0, min_val=0.001, max_val=999999.0, step=1.0, suffix="")
        self.combo_r_unit = QComboBox()
        self.combo_r_unit.addItems(["Ω", "kΩ", "MΩ", "GΩ"])
        self.combo_r_unit.setCurrentText("MΩ")
        lay_r = QHBoxLayout()
        lay_r.setContentsMargins(0,0,0,0)
        lay_r.addWidget(self.spin_r_leak)
        lay_r.addWidget(self.combo_r_unit)
        self.spin_v_rest = ArrowDoubleSpinBox(value=0.0, min_val=-100.0, max_val=100.0, step=0.1, suffix=" V")
        self.spin_v_th = ArrowDoubleSpinBox(value=0.95, min_val=-100.0, max_val=100.0, step=0.05, suffix=" V")
        self.spin_v_reset = ArrowDoubleSpinBox(value=0.15, min_val=-100.0, max_val=100.0, step=0.05, suffix=" V")
        self.spin_t_ref = ArrowDoubleSpinBox(value=2.0, min_val=0.0, max_val=1000.0, step=1.0, suffix=" ms")

        self.lbl_tau = QLabel("Constante τ: — | I_th: —")
        self.lbl_tau.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold;")

        self.spin_c_m.valueChanged.connect(self._on_change)
        self.spin_r_leak.valueChanged.connect(self._on_change)
        self.spin_v_rest.valueChanged.connect(self._on_change)
        self.spin_v_th.valueChanged.connect(self._on_change)
        self.spin_v_reset.valueChanged.connect(self._on_change)
        self.spin_t_ref.valueChanged.connect(self._on_change)

        self.combo_c_unit.currentTextChanged.connect(self._on_change)
        self.combo_r_unit.currentTextChanged.connect(self._on_change)

        layout.addRow("Capacitancia (C_m):", lay_c)
        layout.addRow("Resistencia Fuga (R):", lay_r)
        layout.addRow("Potencial Reposo (V_rest):", self.spin_v_rest)
        layout.addRow("Potencial Umbral (V_th):", self.spin_v_th)
        layout.addRow("Potencial Reset (V_reset):", self.spin_v_reset)
        layout.addRow("Periodo Refractario (t_ref):", self.spin_t_ref)
        layout.addRow("Info de Red:", self.lbl_tau)
        
        group.setLayout(layout)
        main_layout.addWidget(group)
        main_layout.addStretch()
        self.setLayout(main_layout)

        # Recalcular la etiqueta τ/I_th con los valores iniciales reales del panel
        self._on_change()

    def _get_c_multiplier(self) -> float:
        unit = self.combo_c_unit.currentText()
        if unit == "pF": return 1e-12
        if unit == "nF": return 1e-9
        if unit in ["uF", "µF"]: return 1e-6
        if unit == "mF": return 1e-3
        return 1e-9

    def _get_r_multiplier(self) -> float:
        unit = self.combo_r_unit.currentText()
        if unit == "Ω": return 1.0
        if unit == "kΩ": return 1e3
        if unit == "MΩ": return 1e6
        if unit == "GΩ": return 1e9
        return 1e3

    def _on_change(self):
        c_farads = self.spin_c_m.value() * self._get_c_multiplier()
        r_ohms = self.spin_r_leak.value() * self._get_r_multiplier()
        tau_ms = r_ohms * c_farads * 1e3
        
        # I_th teórica
        try:
            v = self.spin_v_th.value() - self.spin_v_rest.value()
            i_th_uA = (v / r_ohms) * 1e6
            self.lbl_tau.setText(f"Constante τ: {tau_ms:.1f} ms | I_th: {i_th_uA:.2f} µA")
        except ZeroDivisionError:
            self.lbl_tau.setText(f"Constante τ: {tau_ms:.1f} ms | I_th: N/A")
            
        self.param_changed.emit()

    def build_neuron(self) -> LIFNeuron:
        cfg = LIFConfig(
            c_m = self.spin_c_m.value() * self._get_c_multiplier(),
            r_leak = self.spin_r_leak.value() * self._get_r_multiplier(),
            v_rest = self.spin_v_rest.value(),
            v_th = self.spin_v_th.value(),
            v_reset = self.spin_v_reset.value(),
            t_ref = self.spin_t_ref.value() * 1e-3
        )
        return LIFNeuron(cfg)
