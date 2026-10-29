"""
neurolab/gui/read_write_config_dialog.py
=========================================
Diálogo para configurar explícitamente lectura y escritura.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QComboBox,
    QDialogButtonBox, QGroupBox, QCheckBox
)
from PySide6.QtCore import Qt

from neurolab.crossbar.addressing import ReadConfig, WriteConfig


class ReadWriteConfigDialog(QDialog):
    """
    Diálogo para configurar los modos de lectura y escritura.
    
    Permite al usuario definir explícitamente:
    - Voltajes de lectura
    - Umbral de programación
    - Esquema de escritura (row, 1T1R, V2, V3)
    - Número y duración de pulsos
    """
    
    def __init__(self, parent=None,
                 read_cfg: ReadConfig = None,
                 write_cfg: WriteConfig = None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Configurar Lectura y Escritura")
        self.setMinimumWidth(500)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', sans-serif; }
            QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 10px; font-weight: bold; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QLabel { color: #a6adc8; }
            QComboBox, QDoubleSpinBox, QSpinBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px; }
            QPushButton { background-color: #89b4fa; color: #11111b; font-weight: bold; padding: 6px 14px; border-radius: 4px; }
            QPushButton:hover { background-color: #b4befe; }
        """)
        
        self.read_cfg = read_cfg or ReadConfig()
        self.write_cfg = write_cfg or WriteConfig()
        
        self._build_ui()
        self._load_values()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        # ============================================================
        # GRUPO 1: LECTURA
        # ============================================================
        grp_read = QGroupBox("📖 Modo Lectura")
        grp_read.setStyleSheet("QGroupBox { color: #89b4fa; font-weight: bold; }")
        form_read = QFormLayout(grp_read)
        
        self.spin_V_read = QDoubleSpinBox()
        self.spin_V_read.setRange(0.0, 5.0)
        self.spin_V_read.setSingleStep(0.05)
        self.spin_V_read.setDecimals(3)
        self.spin_V_read.setSuffix(" V")
        form_read.addRow("V_read (voltaje lectura):", self.spin_V_read)
        
        self.spin_V_col_read = QDoubleSpinBox()
        self.spin_V_col_read.setRange(-5.0, 5.0)
        self.spin_V_col_read.setSingleStep(0.05)
        self.spin_V_col_read.setDecimals(3)
        self.spin_V_col_read.setSuffix(" V")
        form_read.addRow("V_col (voltaje columnas):", self.spin_V_col_read)
        
        self.spin_t_settle = QDoubleSpinBox()
        self.spin_t_settle.setRange(0.0001, 10.0)
        self.spin_t_settle.setSingleStep(0.001)
        self.spin_t_settle.setDecimals(4)
        self.spin_t_settle.setSuffix(" s")
        form_read.addRow("t_settle (establecimiento):", self.spin_t_settle)
        
        layout.addWidget(grp_read)
        
        # ============================================================
        # GRUPO 2: ESCRITURA
        # ============================================================
        grp_write = QGroupBox("⚡ Modo Escritura")
        grp_write.setStyleSheet("QGroupBox { color: #f38ba8; font-weight: bold; }")
        form_write = QFormLayout(grp_write)
        
        self.combo_scheme = QComboBox()
        self.combo_scheme.addItem("V/2 (Cruz de medio pulso)", "V2")
        self.combo_scheme.addItem("1T1R (Aislamiento directo)", "1T1R")
        self.combo_scheme.addItem("Escritura por Fila Completa", "row")
        self.combo_scheme.addItem("V/3 (Tres niveles de tensión)", "V3")
        self.combo_scheme.currentIndexChanged.connect(self._on_scheme_changed)
        form_write.addRow("Esquema:", self.combo_scheme)
        
        self.spin_V_program = QDoubleSpinBox()
        self.spin_V_program.setRange(0.0, 10.0)
        self.spin_V_program.setSingleStep(0.1)
        self.spin_V_program.setDecimals(3)
        self.spin_V_program.setSuffix(" V")
        self.spin_V_program.valueChanged.connect(self._update_V_preview)
        form_write.addRow("V_program (voltaje objetivo):", self.spin_V_program)
        
        self.lbl_V_row_preview = QLabel("+1.000 V")
        self.lbl_V_row_preview.setStyleSheet("color: #f9e2af; font-weight: bold;")
        form_write.addRow("V_row (fila activa):", self.lbl_V_row_preview)
        
        self.lbl_V_col_preview = QLabel("-1.000 V")
        self.lbl_V_col_preview.setStyleSheet("color: #f9e2af; font-weight: bold;")
        form_write.addRow("V_col (columna activa):", self.lbl_V_col_preview)
        
        self.spin_V_th = QDoubleSpinBox()
        self.spin_V_th.setRange(0.0, 5.0)
        self.spin_V_th.setSingleStep(0.05)
        self.spin_V_th.setDecimals(3)
        self.spin_V_th.setSuffix(" V")
        self.spin_V_th.valueChanged.connect(self._validate)
        form_write.addRow("V_th (umbral conmutación):", self.spin_V_th)
        
        self.spin_n_pulses = QSpinBox()
        self.spin_n_pulses.setRange(1, 1000)
        form_write.addRow("N° pulsos:", self.spin_n_pulses)
        
        self.spin_t_pulse = QDoubleSpinBox()
        self.spin_t_pulse.setRange(0.0001, 1.0)
        self.spin_t_pulse.setSingleStep(0.0001)
        self.spin_t_pulse.setDecimals(4)
        self.spin_t_pulse.setSuffix(" s")
        form_write.addRow("Duración pulso:", self.spin_t_pulse)
        
        self.chk_c2c = QCheckBox("Habilitar variabilidad dinámica C2C")
        form_write.addRow("", self.chk_c2c)
        
        self.spin_c2c_sigma = QDoubleSpinBox()
        self.spin_c2c_sigma.setRange(0.0, 0.5)
        self.spin_c2c_sigma.setSingleStep(0.01)
        self.spin_c2c_sigma.setDecimals(3)
        form_write.addRow("σ C2C:", self.spin_c2c_sigma)
        
        layout.addWidget(grp_write)
        
        # ============================================================
        # VALIDACIÓN
        # ============================================================
        self.lbl_status = QLabel()
        self.lbl_status.setStyleSheet("font-size: 11px; padding: 6px;")
        layout.addWidget(self.lbl_status)
        
        # ============================================================
        # BOTONES
        # ============================================================
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.spin_V_read.valueChanged.connect(self._validate)
    
    def _load_values(self):
        """Carga valores actuales en los widgets."""
        self.spin_V_read.setValue(self.read_cfg.V_read)
        self.spin_V_col_read.setValue(self.read_cfg.V_col)
        self.spin_t_settle.setValue(self.read_cfg.t_settle)
        
        idx = self.combo_scheme.findData(self.write_cfg.scheme)
        if idx >= 0:
            self.combo_scheme.setCurrentIndex(idx)
            
        self.spin_V_program.setValue(self.write_cfg.V_program)
        self.spin_V_th.setValue(self.write_cfg.V_th)
        self.spin_n_pulses.setValue(self.write_cfg.n_pulses)
        self.spin_t_pulse.setValue(self.write_cfg.t_pulse)
        self.chk_c2c.setChecked(self.write_cfg.c2c_enabled)
        self.spin_c2c_sigma.setValue(self.write_cfg.c2c_sigma)
        
        self._update_V_preview()
        self._validate()
    
    def _on_scheme_changed(self):
        """Actualiza preview al cambiar esquema."""
        self._update_V_preview()
    
    def _update_V_preview(self):
        """Calcula y muestra V_row y V_col según esquema."""
        V = self.spin_V_program.value()
        scheme = self.combo_scheme.currentData() or "V2"
        
        if scheme == 'V2':
            V_row, V_col = +V / 2.0, -V / 2.0
        elif scheme == 'V3':
            V_row, V_col = +2.0 * V / 3.0, -V / 3.0
        elif scheme == '1T1R':
            V_row, V_col = V, 0.0
        else:  # 'row'
            V_row, V_col = V, 0.0
        
        self.lbl_V_row_preview.setText(f"{V_row:+.3f} V")
        self.lbl_V_col_preview.setText(f"{V_col:+.3f} V")
        self._validate()
    
    def _validate(self):
        """Valida coherencia de parámetros."""
        errors = []
        warnings = []
        
        V_read = self.spin_V_read.value()
        V_th = self.spin_V_th.value()
        V_program = self.spin_V_program.value()
        
        if V_read >= V_th:
            errors.append(f"❌ V_read ({V_read:.2f}V) ≥ V_th ({V_th:.2f}V) → lectura destructiva")
        
        if V_program <= V_th:
            errors.append(f"❌ V_program ({V_program:.2f}V) ≤ V_th ({V_th:.2f}V) → no programa")
        
        if abs(self.spin_V_col_read.value()) >= V_th:
            errors.append(f"❌ V_col lectura ({self.spin_V_col_read.value():.2f}V) ≥ V_th")
        
        scheme = self.combo_scheme.currentData() or "V2"
        if scheme == 'V2' and (V_program / 2.0) >= V_th:
            warnings.append(f"⚠️ Half-select ({V_program/2.0:.2f}V) ≥ V_th ({V_th:.2f}V) → las celdas half-selected sufrirán disturbios de escritura")
        
        if errors:
            self.lbl_status.setText("\n".join(errors))
            self.lbl_status.setStyleSheet("color: #f38ba8; padding: 6px; background-color: #31222b; border-radius: 4px;")
        elif warnings:
            self.lbl_status.setText("\n".join(warnings))
            self.lbl_status.setStyleSheet("color: #f9e2af; padding: 6px; background-color: #2b291d; border-radius: 4px;")
        else:
            self.lbl_status.setText("✅ Configuración válida")
            self.lbl_status.setStyleSheet("color: #a6e3a1; padding: 6px; background-color: #1e292d; border-radius: 4px;")
    
    def get_read_config(self) -> ReadConfig:
        """Retorna ReadConfig actualizado."""
        return ReadConfig(
            V_read=self.spin_V_read.value(),
            V_th=self.spin_V_th.value(),
            t_settle=self.spin_t_settle.value(),
            V_col=self.spin_V_col_read.value(),
        )
    
    def get_write_config(self) -> WriteConfig:
        """Retorna WriteConfig actualizado."""
        return WriteConfig(
            V_program=self.spin_V_program.value(),
            V_th=self.spin_V_th.value(),
            scheme=self.combo_scheme.currentData() or "V2",
            n_pulses=self.spin_n_pulses.value(),
            t_pulse=self.spin_t_pulse.value(),
            c2c_enabled=self.chk_c2c.isChecked(),
            c2c_sigma=self.spin_c2c_sigma.value(),
        )
