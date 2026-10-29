import json
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLabel, QComboBox, QHBoxLayout, QPushButton,
    QFileDialog, QMessageBox
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

        self.spin_r_series = ArrowDoubleSpinBox(value=100.0, min_val=0.001, max_val=999999.0, step=1.0, suffix="")
        self.combo_rs_unit = QComboBox()
        self.combo_rs_unit.addItems(["Ω", "kΩ", "MΩ", "GΩ"])
        self.combo_rs_unit.setCurrentText("kΩ")
        lay_rs = QHBoxLayout()
        lay_rs.setContentsMargins(0,0,0,0)
        lay_rs.addWidget(self.spin_r_series)
        lay_rs.addWidget(self.combo_rs_unit)

        self.spin_r_leak = ArrowDoubleSpinBox(value=1.0, min_val=0.001, max_val=999999.0, step=1.0, suffix="")
        self.combo_r_unit = QComboBox()
        self.combo_r_unit.addItems(["Ω", "kΩ", "MΩ", "GΩ"])
        self.combo_r_unit.setCurrentText("MΩ")
        lay_r = QHBoxLayout()
        lay_r.setContentsMargins(0,0,0,0)
        lay_r.addWidget(self.spin_r_leak)
        lay_r.addWidget(self.combo_r_unit)
        self.spin_v_rest = ArrowDoubleSpinBox(value=0.0, min_val=-100.0, max_val=100.0, step=0.1, suffix=" V")
        self.spin_v_th = ArrowDoubleSpinBox(value=2.5, min_val=-100.0, max_val=100.0, step=0.05, suffix=" V")
        self.spin_v_reset = ArrowDoubleSpinBox(value=0.0, min_val=-100.0, max_val=100.0, step=0.05, suffix=" V")
        self.spin_t_ref = ArrowDoubleSpinBox(value=2.0, min_val=0.0, max_val=1000.0, step=1.0, suffix=" ms")

        self.lbl_tau = QLabel("Constante τ_m: — | τ_eq: —")
        self.lbl_tau.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold;")

        self.spin_c_m.valueChanged.connect(self._on_change)
        self.spin_r_series.valueChanged.connect(self._on_change)
        self.spin_r_leak.valueChanged.connect(self._on_change)
        self.spin_v_rest.valueChanged.connect(self._on_change)
        self.spin_v_th.valueChanged.connect(self._on_change)
        self.spin_v_reset.valueChanged.connect(self._on_change)
        self.spin_t_ref.valueChanged.connect(self._on_change)

        self.combo_c_unit.currentTextChanged.connect(self._on_change)
        self.combo_rs_unit.currentTextChanged.connect(self._on_change)
        self.combo_r_unit.currentTextChanged.connect(self._on_change)

        layout.addRow("Capacitancia (C_m):", lay_c)
        layout.addRow("Resistencia Serie (R_S):", lay_rs)
        layout.addRow("Resistencia Fuga (R_leak):", lay_r)
        layout.addRow("Potencial Reposo (V_rest):", self.spin_v_rest)
        layout.addRow("Potencial Umbral (V_th):", self.spin_v_th)
        layout.addRow("Potencial Reset (V_reset):", self.spin_v_reset)
        layout.addRow("Periodo Refractario (t_ref):", self.spin_t_ref)
        layout.addRow("Info de Circuito:", self.lbl_tau)

        # Botones de Guardar / Cargar Perfil LIF
        btn_save = QPushButton("💾 Guardar Config LIF")
        btn_save.setStyleSheet("""
            QPushButton { background-color: #313244; color: #89b4fa; font-weight: bold; padding: 6px; border-radius: 4px; border: 1px solid #45475a; }
            QPushButton:hover { background-color: #45475a; color: #b4befe; }
        """)
        btn_save.clicked.connect(self._save_lif_config_file)

        btn_load = QPushButton("📂 Cargar Config LIF")
        btn_load.setStyleSheet("""
            QPushButton { background-color: #313244; color: #a6e3a1; font-weight: bold; padding: 6px; border-radius: 4px; border: 1px solid #45475a; }
            QPushButton:hover { background-color: #45475a; color: #b4befe; }
        """)
        btn_load.clicked.connect(self._load_lif_config_file)

        lay_io_btns = QHBoxLayout()
        lay_io_btns.setContentsMargins(0, 4, 0, 4)
        lay_io_btns.addWidget(btn_save)
        lay_io_btns.addWidget(btn_load)
        layout.addRow(lay_io_btns)

        # Preset del circuito hardware real (LM393 + 2N7000)
        btn_hw = QPushButton("🧩 Preset Guía Hardware (LM393/2N7000)")
        btn_hw.setStyleSheet("""
            QPushButton { background-color: #cba6f7; color: #11111b; font-weight: bold; padding: 8px; border-radius: 6px; }
            QPushButton:hover { background-color: #b4befe; }
        """)
        btn_hw.setToolTip(
            "Aplica los valores del circuito real de la tesis:\n"
            "C_m = 100 nF, R_S = 100 kΩ, R_leak = 1 MΩ, V_th = +2.5 V, V_reset = 0 V (a masa vía 2N7000),\n"
            "t_ref = 2 ms."
        )
        btn_hw.clicked.connect(self._apply_hardware_preset)
        layout.addRow(btn_hw)
        
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

    def _get_rs_multiplier(self) -> float:
        unit = self.combo_rs_unit.currentText()
        if unit == "Ω": return 1.0
        if unit == "kΩ": return 1e3
        if unit == "MΩ": return 1e6
        if unit == "GΩ": return 1e9
        return 1e3

    def _get_r_multiplier(self) -> float:
        unit = self.combo_r_unit.currentText()
        if unit == "Ω": return 1.0
        if unit == "kΩ": return 1e3
        if unit == "MΩ": return 1e6
        if unit == "GΩ": return 1e9
        return 1e6

    def _on_change(self):
        c_farads = self.spin_c_m.value() * self._get_c_multiplier()
        rs_ohms = self.spin_r_series.value() * self._get_rs_multiplier()
        rleak_ohms = self.spin_r_leak.value() * self._get_r_multiplier()
        
        # 1. tau_m pasivo de membrana (olvido/fuga sin entrada): tau_m = R_leak * C_m
        tau_leak_ms = rleak_ohms * c_farads * 1e3
        
        # 2. tau_eq equivalente durante la carga con V_IN: tau_eq = (R_S || R_leak) * C_m
        r_eq = (rs_ohms * rleak_ohms) / (rs_ohms + rleak_ohms)
        tau_eq_ms = r_eq * c_farads * 1e3
        
        # Voltaje asintótico asumiendo Vin = 5V
        try:
            v_inf_5v = (rleak_ohms / (rs_ohms + rleak_ohms)) * 5.0
            self.lbl_tau.setText(f"τ_m (Fuga): {tau_leak_ms:.1f} ms | τ_eq (Carga): {tau_eq_ms:.1f} ms | V_inf(5V): {v_inf_5v:.2f} V")
        except ZeroDivisionError:
            self.lbl_tau.setText(f"τ_m: {tau_leak_ms:.1f} ms | τ_eq: N/A")
            
        self.param_changed.emit()

    def _apply_hardware_preset(self):
        """
        Carga los parámetros del circuito real de la guía hardware (LM393 + 2N7000):
        C_m = 100 nF, R_S = 100 kΩ, R_leak = 1 MΩ, V_th = +2.5 V, V_reset = 0 V, t_ref = 2 ms.
        """
        self.combo_c_unit.setCurrentText("nF")
        self.spin_c_m.setValue(100.0)
        self.combo_rs_unit.setCurrentText("kΩ")
        self.spin_r_series.setValue(100.0)
        self.combo_r_unit.setCurrentText("MΩ")
        self.spin_r_leak.setValue(1.0)
        self.spin_v_rest.setValue(0.0)
        self.spin_v_th.setValue(2.5)
        self.spin_v_reset.setValue(0.0)
        self.spin_t_ref.setValue(2.0)
        self._on_change()

    def set_signal_panel(self, signal_panel):
        """Asocia el panel de señal para incluir su estado en el archivo JSON."""
        self._signal_panel = signal_panel

    def to_dict(self) -> dict:
        data = {
            "c_m_value": self.spin_c_m.value(),
            "c_m_unit": self.combo_c_unit.currentText(),
            "r_series_value": self.spin_r_series.value(),
            "r_series_unit": self.combo_rs_unit.currentText(),
            "r_leak_value": self.spin_r_leak.value(),
            "r_leak_unit": self.combo_r_unit.currentText(),
            "v_rest": self.spin_v_rest.value(),
            "v_th": self.spin_v_th.value(),
            "v_reset": self.spin_v_reset.value(),
            "t_ref": self.spin_t_ref.value(),
        }
        if hasattr(self, '_signal_panel') and self._signal_panel is not None:
            data["signal_panel"] = self._signal_panel.to_dict()
        return data

    def from_dict(self, data: dict):
        was_blocked = self.signalsBlocked()
        self.blockSignals(True)
        try:
            if "c_m_value" in data: self.spin_c_m.setValue(data["c_m_value"])
            if "c_m_unit" in data: self.combo_c_unit.setCurrentText(data["c_m_unit"])
            if "r_series_value" in data: self.spin_r_series.setValue(data["r_series_value"])
            if "r_series_unit" in data: self.combo_rs_unit.setCurrentText(data["r_series_unit"])
            if "r_leak_value" in data: self.spin_r_leak.setValue(data["r_leak_value"])
            if "r_leak_unit" in data: self.combo_r_unit.setCurrentText(data["r_leak_unit"])
            if "v_rest" in data: self.spin_v_rest.setValue(data["v_rest"])
            if "v_th" in data: self.spin_v_th.setValue(data["v_th"])
            if "v_reset" in data: self.spin_v_reset.setValue(data["v_reset"])
            if "t_ref" in data: self.spin_t_ref.setValue(data["t_ref"])
            if "signal_panel" in data and hasattr(self, '_signal_panel') and self._signal_panel is not None:
                self._signal_panel.from_dict(data["signal_panel"])
        finally:
            self.blockSignals(was_blocked)

        self._on_change()


    def _save_lif_config_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Configuración de Neurona LIF", "configs/lif_config.json", "Archivos JSON (*.json)"
        )
        if file_path:
            payload = {
                "_meta": {
                    "type": "lif_neuron_config",
                    "saved_at": datetime.now().isoformat(timespec="seconds")
                }
            }
            payload.update(self.to_dict())
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            QMessageBox.information(self, "Guardado Exitoso", f"Configuración LIF guardada en:\n{file_path}")

    def _load_lif_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Configuración de Neurona LIF", "configs/", "Archivos JSON (*.json)"
        )
        if file_path and Path(file_path).exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.from_dict(data)
                QMessageBox.information(self, "Carga Exitosa", f"Configuración LIF cargada desde:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error al Cargar", f"No se pudo cargar la configuración:\n{str(e)}")

    def build_neuron(self) -> LIFNeuron:
        cfg = LIFConfig(
            c_m = self.spin_c_m.value() * self._get_c_multiplier(),
            r_series = self.spin_r_series.value() * self._get_rs_multiplier(),
            r_leak = self.spin_r_leak.value() * self._get_r_multiplier(),
            v_rest = self.spin_v_rest.value(),
            v_th = self.spin_v_th.value(),
            v_reset = self.spin_v_reset.value(),
            t_ref = self.spin_t_ref.value() * 1e-3
        )
        return LIFNeuron(cfg)
