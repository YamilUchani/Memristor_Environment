"""
neurolab.gui.widgets.config_panel
====================================
Panel de Control interactivo para la configuración en 5 secciones del Memristor.

Incluye:
  - Todos los parámetros estocásticos detallados (Seed, C2C, D2D, Ruido, Ventanas)
  - Guardado/Carga de perfiles JSON via ProfileManager (directorio configs/ por defecto)
  - Diálogo de Presets del Sistema (lista de perfiles incluidos con el simulador)
  - Metadata de versión en exportaciones JSON
"""
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLineEdit,
    QCheckBox, QComboBox, QPushButton, QLabel, QFileDialog,
    QHBoxLayout, QMessageBox, QDialog, QListWidget, QListWidgetItem,
    QDialogButtonBox
)
from PySide6.QtCore import Signal, Qt
from neurolab.devices.config import DeviceConfig, ElectricalConfig, StrukovConfig
from neurolab.core.memristor import Memristor
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.realism import (
    BiolekWindowModifier, JoglekarWindowModifier,
    D2DVariabilityModifier, C2CVariabilityModifier, ThermalNoiseModifier
)
from neurolab.gui.widgets.arrow_spinbox import ArrowDoubleSpinBox, ArrowSpinBox
from neurolab.io.profile_manager import ProfileManager


class SystemPresetsDialog(QDialog):
    """
    Diálogo modal que muestra los perfiles JSON disponibles en el directorio configs/.
    Permite seleccionar y cargar un preset del sistema con un doble click o botón.
    """

    def __init__(self, profile_manager: ProfileManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📋 Presets del Sistema")
        self.setMinimumSize(420, 300)
        self.selected_path: Path | None = None
        self._pm = profile_manager

        layout = QVBoxLayout()

        lbl = QLabel("Selecciona un preset para cargarlo:")
        lbl.setStyleSheet("color: #cdd6f4; font-weight: bold; margin-bottom: 4px;")
        layout.addWidget(lbl)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #89b4fa;
                color: #11111b;
                border-radius: 3px;
            }
            QListWidget::item:hover {
                background-color: #313244;
            }
        """)

        self._profiles: list[Path] = profile_manager.list_profiles()
        for path in self._profiles:
            display = profile_manager.get_profile_display_name(path)
            item = QListWidgetItem(f"  {display}")
            item.setToolTip(str(path))
            self.list_widget.addItem(item)

        if not self._profiles:
            self.list_widget.addItem("  (No hay perfiles en configs/)")

        self.list_widget.itemDoubleClicked.connect(self._accept)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Cargar Perfil")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.setStyleSheet("color: #cdd6f4;")
        layout.addWidget(buttons)

        self.setLayout(layout)

    def _accept(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self._profiles):
            self.selected_path = self._profiles[row]
            self.accept()
        else:
            QMessageBox.warning(self, "Sin Selección", "Selecciona un perfil de la lista.")


class ConfigPanel(QWidget):
    """
    Panel de Control interactivo para la configuración del Memristor.

    Secciones:
        1. 🪪 Identidad del Dispositivo
        2. ⚡ Parámetros Eléctricos Universales
        3. 🔬 Parámetros Físicos (Strukov 2008)
        4. 🛠️  Modificadores de Realismo y Ventanas
    """
    preset_requested = Signal()
    param_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signal_panel = None
        self._profile_manager = ProfileManager()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # ── Barra de botones superior ────────────────────────────────────────
        toolbar_layout = QHBoxLayout()

        btn_preset = QPushButton("⚡ Perfil Paper Strukov")
        btn_preset.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #11111b;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        btn_preset.clicked.connect(self._load_strukov_paper_preset)

        btn_system = QPushButton("📋 Presets del Sistema")
        btn_system.setStyleSheet("""
            QPushButton {
                background-color: #cba6f7;
                color: #11111b;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #f5c2e7; }
        """)
        btn_system.clicked.connect(self._open_system_presets)

        btn_save = QPushButton("💾 Guardar JSON")
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #11111b;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #94e2d5; }
        """)
        btn_save.clicked.connect(self.save_profile_dialog)

        btn_load = QPushButton("📂 Cargar JSON")
        btn_load.setStyleSheet("""
            QPushButton {
                background-color: #fab387;
                color: #11111b;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #f9e2af; }
        """)
        btn_load.clicked.connect(self.load_profile_dialog)

        toolbar_layout.addWidget(btn_preset)
        toolbar_layout.addWidget(btn_system)
        toolbar_layout.addWidget(btn_save)
        toolbar_layout.addWidget(btn_load)
        main_layout.addLayout(toolbar_layout)

        # ── 1. Identidad y Tipo de Dispositivo ──────────────────────────────
        group_a = QGroupBox("1. 🪪 Identidad del Dispositivo")
        layout_a = QFormLayout()

        self.txt_device_name = QLineEdit("Strukov TiO2")
        self.txt_device_family = QLineEdit("oxide_memristor")
        self.combo_model_name = QComboBox()
        self.combo_model_name.addItems(["strukov", "ideal_normalized"])

        layout_a.addRow("Nombre Dispositivo:", self.txt_device_name)
        layout_a.addRow("Familia Física:", self.txt_device_family)
        layout_a.addRow("Modelo Matemático:", self.combo_model_name)
        group_a.setLayout(layout_a)
        main_layout.addWidget(group_a)

        # ── 2. Parámetros Eléctricos Universales ────────────────────────────
        group_b = QGroupBox("2. ⚡ Parámetros Eléctricos Universales")
        layout_b = QFormLayout()

        self.spin_r_on = ArrowDoubleSpinBox(value=100.0, min_val=1.0, max_val=1e6, step=10.0, suffix=" Ω")
        self.spin_r_off = ArrowDoubleSpinBox(value=16000.0, min_val=10.0, max_val=1e9, step=500.0, suffix=" Ω")
        self.spin_x0 = ArrowDoubleSpinBox(value=0.1, min_val=0.0, max_val=1.0, step=0.05, decimals=2)

        self.lbl_g_info = QLabel("G_ON: 10.0 mS | G_OFF: 62.5 µS")
        self.lbl_g_info.setStyleSheet("color: #a6adc8; font-size: 11px;")

        self.spin_r_on.valueChanged.connect(self._update_g_labels)
        self.spin_r_off.valueChanged.connect(self._update_g_labels)

        layout_b.addRow("Resistencia R_ON:", self.spin_r_on)
        layout_b.addRow("Resistencia R_OFF:", self.spin_r_off)
        layout_b.addRow("Estado Inicial x_0:", self.spin_x0)
        layout_b.addRow("Conductancias:", self.lbl_g_info)
        group_b.setLayout(layout_b)
        main_layout.addWidget(group_b)

        # ── 3. Parámetros Físicos del Modelo (Strukov) ──────────────────────
        group_e = QGroupBox("3. 🔬 Parámetros Físicos (Strukov 2008)")
        layout_e = QFormLayout()

        self.spin_D_nm = ArrowDoubleSpinBox(value=10.0, min_val=1.0, max_val=500.0, step=1.0, suffix=" nm")
        self.spin_mu_v = ArrowDoubleSpinBox(value=1e-14, min_val=1e-16, max_val=1e-10, step=1e-15, decimals=15)

        layout_e.addRow("Espesor Capa D:", self.spin_D_nm)
        layout_e.addRow("Movilidad µ_v:", self.spin_mu_v)
        group_e.setLayout(layout_e)
        main_layout.addWidget(group_e)

        # ── 4. Modificadores de Realismo y Ventanas ──────────────────────────
        group_d = QGroupBox("4. 🛠️ Modificadores de Realismo y Ventanas")
        layout_d = QFormLayout()

        self.combo_realism_mode = QComboBox()
        self.combo_realism_mode.addItems([
            "Modo 1 — Ideal (Strukov Puro)",
            "Modo 2 — Extendido (Ventana Biolek)",
            "Modo 3 — Realista Estocástico (Biolek + C2C + Ruido)",
            "Personalizado"
        ])
        self.combo_realism_mode.currentIndexChanged.connect(self._on_realism_mode_changed)

        # Reproducibilidad Científica (SEED)
        self.spin_seed = ArrowSpinBox(value=42, min_val=0, max_val=999999, step=1)
        self.spin_seed.spin.setToolTip("Semilla fija para reproducibilidad científica estricta")

        # Ventanas de Frontera
        self.combo_window_type = QComboBox()
        self.combo_window_type.addItems(["Biolek", "Joglekar", "Sin Ventana"])
        self.spin_biolek_p = ArrowSpinBox(value=5, min_val=1, max_val=20, step=1)

        # C2C — Ornstein-Uhlenbeck
        self.chk_c2c = QCheckBox("Variabilidad Ciclo a Ciclo (C2C - Ornstein-Uhlenbeck)")
        self.chk_c2c.setChecked(False)
        self.spin_c2c_sigma = ArrowDoubleSpinBox(value=0.05, min_val=0.001, max_val=0.5, step=0.01, decimals=3)
        self.spin_c2c_theta = ArrowDoubleSpinBox(value=1.0, min_val=0.1, max_val=10.0, step=0.1, decimals=2)

        # D2D — Device-to-Device
        self.chk_d2d = QCheckBox("Variabilidad Dispositivo a Dispositivo (D2D)")
        self.chk_d2d.setChecked(False)
        self.spin_d2d_sigma = ArrowDoubleSpinBox(value=0.05, min_val=0.001, max_val=0.5, step=0.01, decimals=3)

        # Ruido Térmico de Lectura
        self.chk_noise = QCheckBox("Ruido Térmico de Lectura")
        self.chk_noise.setChecked(False)
        self.spin_noise_std = ArrowDoubleSpinBox(value=1e-6, min_val=0.0, max_val=1e-2, step=1e-6, decimals=7, suffix=" A")

        layout_d.addRow("Modo de Presets:", self.combo_realism_mode)
        layout_d.addRow("🔑 Semilla (Seed):", self.spin_seed)
        layout_d.addRow("Tipo de Ventana:", self.combo_window_type)
        layout_d.addRow("Exponente Ventana (p):", self.spin_biolek_p)
        layout_d.addRow(self.chk_c2c)
        layout_d.addRow("  ↳ Volatilidad C2C (σ):", self.spin_c2c_sigma)
        layout_d.addRow("  ↳ Retorno Media (θ):", self.spin_c2c_theta)
        layout_d.addRow(self.chk_d2d)
        layout_d.addRow("  ↳ Dispersión D2D (σ):", self.spin_d2d_sigma)
        layout_d.addRow(self.chk_noise)
        layout_d.addRow("  ↳ Nivel Ruido (std):", self.spin_noise_std)
        group_d.setLayout(layout_d)
        main_layout.addWidget(group_d)

        main_layout.addStretch()
        self.setLayout(main_layout)
        self._update_g_labels()
        self._connect_signals()

    # ── Lógica de presets de modo ────────────────────────────────────────────

    def _on_realism_mode_changed(self, index: int):
        """Conmuta automáticamente las casillas de realismo según el modo seleccionado."""
        if index == 0:   # Modo 1 — Ideal
            self.combo_window_type.setCurrentText("Sin Ventana")
            self.chk_c2c.setChecked(False)
            self.chk_d2d.setChecked(False)
            self.chk_noise.setChecked(False)
        elif index == 1:  # Modo 2 — Extendido
            self.combo_window_type.setCurrentText("Biolek")
            self.chk_c2c.setChecked(False)
            self.chk_d2d.setChecked(False)
            self.chk_noise.setChecked(False)
        elif index == 2:  # Modo 3 — Realista Estocástico
            self.combo_window_type.setCurrentText("Biolek")
            self.chk_c2c.setChecked(True)
            self.chk_d2d.setChecked(True)
            self.chk_noise.setChecked(True)

    def _connect_signals(self):
        """Conecta los cambios de cualquier campo para emitir param_changed."""
        self.txt_device_name.textChanged.connect(lambda *_: self.param_changed.emit())
        self.txt_device_family.textChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_model_name.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_r_on.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_r_off.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_x0.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_D_nm.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_mu_v.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_seed.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_window_type.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_biolek_p.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.chk_c2c.toggled.connect(lambda *_: self.param_changed.emit())
        self.spin_c2c_sigma.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_c2c_theta.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.chk_d2d.toggled.connect(lambda *_: self.param_changed.emit())
        self.spin_d2d_sigma.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.chk_noise.toggled.connect(lambda *_: self.param_changed.emit())
        self.spin_noise_std.valueChanged.connect(lambda *_: self.param_changed.emit())

    def _update_g_labels(self):
        r_on = self.spin_r_on.value()
        r_off = self.spin_r_off.value()
        g_on_ms = (1.0 / r_on) * 1e3 if r_on > 0 else 0
        g_off_us = (1.0 / r_off) * 1e6 if r_off > 0 else 0
        self.lbl_g_info.setText(f"G_ON: {g_on_ms:.2f} mS | G_OFF: {g_off_us:.2f} µS")

    def set_signal_panel(self, signal_panel):
        """Conecta la instancia de SignalPanel para unificar la gestión del perfil completo."""
        self.signal_panel = signal_panel

    # ── Carga del preset del paper ───────────────────────────────────────────

    def _load_strukov_paper_preset(self):
        self.txt_device_name.setText("Strukov TiO2 (Paper Fig 2b)")
        self.spin_r_on.setValue(100.0)
        self.spin_r_off.setValue(16000.0)
        self.spin_x0.setValue(0.1)
        self.spin_D_nm.setValue(10.0)
        self.spin_mu_v.setValue(1e-14)
        self.spin_seed.setValue(42)
        self.combo_window_type.setCurrentText("Sin Ventana")
        self.chk_c2c.setChecked(False)
        self.chk_d2d.setChecked(False)
        self.chk_noise.setChecked(False)
        self.combo_realism_mode.setCurrentIndex(0)

        if self.signal_panel is not None:
            self.signal_panel.reset_defaults()

        self.param_changed.emit()

    # ── Diálogo de Presets del Sistema ──────────────────────────────────────

    def _open_system_presets(self):
        """Abre el diálogo de lista de perfiles del sistema."""
        dlg = SystemPresetsDialog(self._profile_manager, parent=self)
        if dlg.exec() == QDialog.Accepted and dlg.selected_path is not None:
            try:
                data = self._profile_manager.load(dlg.selected_path)
                self.from_dict(data)
                self.param_changed.emit()
                QMessageBox.information(
                    self, "Preset Cargado",
                    f"✓ Preset del sistema cargado:\n{self._profile_manager.get_profile_display_name(dlg.selected_path)}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error al Cargar Preset", f"No se pudo cargar el preset:\n{str(e)}")

    # ── Serialización ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Exporta la configuración actual (dispositivo + señal) como diccionario JSON serializable."""
        data = {
            "device_name": self.txt_device_name.text(),
            "device_family": self.txt_device_family.text(),
            "model_name": self.combo_model_name.currentText(),
            "r_on": self.spin_r_on.value(),
            "r_off": self.spin_r_off.value(),
            "initial_state": self.spin_x0.value(),
            "D_nm": self.spin_D_nm.value(),
            "mu_v": self.spin_mu_v.value(),
            "seed": self.spin_seed.value(),
            "realism_mode_index": self.combo_realism_mode.currentIndex(),
            "window_type": self.combo_window_type.currentText(),
            "biolek_p": self.spin_biolek_p.value(),
            "enable_c2c": self.chk_c2c.isChecked(),
            "c2c_sigma": self.spin_c2c_sigma.value(),
            "c2c_theta": self.spin_c2c_theta.value(),
            "enable_d2d": self.chk_d2d.isChecked(),
            "d2d_sigma": self.spin_d2d_sigma.value(),
            "enable_noise": self.chk_noise.isChecked(),
            "noise_std": self.spin_noise_std.value(),
        }

        if self.signal_panel is not None:
            data["signal"] = self.signal_panel.to_dict()

        return data

    def from_dict(self, data: dict):
        """Importa y carga la configuración desde un diccionario JSON."""
        if "device_name" in data:
            self.txt_device_name.setText(data["device_name"])
        if "device_family" in data:
            self.txt_device_family.setText(data["device_family"])
        if "model_name" in data:
            idx = self.combo_model_name.findText(data["model_name"])
            if idx >= 0:
                self.combo_model_name.setCurrentIndex(idx)
        if "r_on" in data:
            self.spin_r_on.setValue(float(data["r_on"]))
        if "r_off" in data:
            self.spin_r_off.setValue(float(data["r_off"]))
        if "initial_state" in data:
            self.spin_x0.setValue(float(data["initial_state"]))
        if "D_nm" in data:
            self.spin_D_nm.setValue(float(data["D_nm"]))
        if "mu_v" in data:
            self.spin_mu_v.setValue(float(data["mu_v"]))
        if "seed" in data:
            self.spin_seed.setValue(int(data["seed"]))
        if "realism_mode_index" in data:
            self.combo_realism_mode.setCurrentIndex(int(data["realism_mode_index"]))
        if "window_type" in data:
            idx = self.combo_window_type.findText(data["window_type"])
            if idx >= 0:
                self.combo_window_type.setCurrentIndex(idx)
        if "biolek_p" in data:
            self.spin_biolek_p.setValue(int(data["biolek_p"]))
        if "enable_c2c" in data:
            self.chk_c2c.setChecked(bool(data["enable_c2c"]))
        if "c2c_sigma" in data:
            self.spin_c2c_sigma.setValue(float(data["c2c_sigma"]))
        if "c2c_theta" in data:
            self.spin_c2c_theta.setValue(float(data["c2c_theta"]))
        if "enable_d2d" in data:
            self.chk_d2d.setChecked(bool(data["enable_d2d"]))
        if "d2d_sigma" in data:
            self.spin_d2d_sigma.setValue(float(data["d2d_sigma"]))
        if "enable_noise" in data:
            self.chk_noise.setChecked(bool(data["enable_noise"]))
        if "noise_std" in data:
            self.spin_noise_std.setValue(float(data["noise_std"]))

        if "signal" in data and self.signal_panel is not None:
            self.signal_panel.from_dict(data["signal"])

    # ── Diálogos de Guardar/Cargar ───────────────────────────────────────────

    def save_profile_dialog(self):
        """Abre diálogo para guardar el perfil actual a un archivo .json."""
        default_dir = str(self._profile_manager.configs_dir)
        device_name = self.txt_device_name.text().strip() or "perfil_memristor"
        default_filename = str(self._profile_manager.configs_dir / f"{device_name}.json")

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Perfil de Memristor",
            default_filename,
            "Archivos JSON (*.json)"
        )
        if file_path:
            try:
                data = self.to_dict()
                saved_path = self._profile_manager.save(
                    data,
                    profile_name=device_name,
                    file_path=file_path
                )
                QMessageBox.information(
                    self, "Perfil Guardado",
                    f"✓ Perfil guardado con éxito en:\n{saved_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error al Guardar", f"No se pudo guardar el archivo:\n{str(e)}")

    def load_profile_dialog(self):
        """Abre diálogo para cargar un perfil de memristor desde un archivo .json."""
        default_dir = str(self._profile_manager.configs_dir)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Perfil de Memristor",
            default_dir,
            "Archivos JSON (*.json)"
        )
        if file_path:
            try:
                data = self._profile_manager.load(file_path)
                self.from_dict(data)
                QMessageBox.information(
                    self, "Perfil Cargado",
                    f"✓ Perfil cargado exitosamente desde:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error al Cargar", f"Archivo JSON no válido:\n{str(e)}")

    # ── Construcción del Memristor ────────────────────────────────────────────

    def build_memristor(self) -> Memristor:
        """Construye una instancia de Memristor según la configuración actual del panel."""
        identity = DeviceConfig(
            name=self.txt_device_name.text(),
            family=self.txt_device_family.text(),
            model=self.combo_model_name.currentText()
        )

        electrical = ElectricalConfig(
            r_on=self.spin_r_on.value(),
            r_off=self.spin_r_off.value(),
            initial_state=self.spin_x0.value()
        )

        strukov_config = StrukovConfig(
            D=self.spin_D_nm.value() * 1e-9,   # nm → m
            mu_v=self.spin_mu_v.value()
        )

        seed = self.spin_seed.value()
        modifiers = []

        # 1. Ventana de Frontera
        win_type = self.combo_window_type.currentText()
        if win_type == "Biolek":
            modifiers.append(BiolekWindowModifier(p=self.spin_biolek_p.value()))
        elif win_type == "Joglekar":
            modifiers.append(JoglekarWindowModifier(p=self.spin_biolek_p.value()))

        # 2. Variabilidad D2D (ahora con efecto real vía modify_dxdt)
        if self.chk_d2d.isChecked():
            modifiers.append(D2DVariabilityModifier(variability_std=self.spin_d2d_sigma.value(), seed=seed))

        # 3. Variabilidad C2C (Ornstein-Uhlenbeck)
        if self.chk_c2c.isChecked():
            modifiers.append(C2CVariabilityModifier(
                sigma=self.spin_c2c_sigma.value(),
                theta=self.spin_c2c_theta.value(),
                seed=seed
            ))

        # 4. Ruido Térmico de Lectura
        if self.chk_noise.isChecked():
            modifiers.append(ThermalNoiseModifier(noise_std=self.spin_noise_std.value(), seed=seed))

        return Memristor(
            math_model=StrukovMathModel(),
            electrical=electrical,
            identity=identity,
            model_config=strukov_config,
            modifiers=modifiers
        )
