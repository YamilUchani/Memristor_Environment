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


MATERIAL_MOBILITY_MAP = {
    "TiO₂ (Dióxido de Titanio - Strukov 2008)": {
        "mu_v": 1.0e-14,
        "family": "TiO2_oxide",
        "info": "TiO₂: Movilidad iónica alta (1×10⁻¹⁴ m²/V·s) — Conmutación rápida (Strukov et al., 2008)."
    },
    "TaOₓ (Óxido de Tántalo - Industrial)": {
        "mu_v": 1.0e-15,
        "family": "TaOx_oxide",
        "info": "TaOₓ: Movilidad iónica intermedia (1×10⁻¹⁵ m²/V·s) — Alta durabilidad (TSMC/Panasonic)."
    },
    "HfO₂ (Óxido de Hafnio - CMOS LIF 2025)": {
        "mu_v": 1.0e-16,
        "family": "HfO2_oxide",
        "info": "HfO₂: Movilidad iónica moderada (1×10⁻¹⁶ m²/V·s) — Deriva óptima para Neurona LIF (Wang et al., 2025)."
    },
    "WOₓ (Óxido de Tungsteno - Sináptico)": {
        "mu_v": 5.0e-15,
        "family": "WOx_oxide",
        "info": "WOₓ: Movilidad iónica sináptica (5×10⁻¹⁵ m²/V·s) — Conmutación analógica suave."
    },
    "NbOₓ (Óxido de Niobio - Mott Switch)": {
        "mu_v": 1.0e-17,
        "family": "NbOx_oxide",
        "info": "NbOₓ: Movilidad iónica ultra-lenta (1×10⁻¹⁷ m²/V·s) — Emulación de conmutación por umbral."
    },
    "Personalizado": {
        "mu_v": None,
        "family": "custom_oxide",
        "info": "Personalizado: Movilidad iónica ajustada manualmente por el usuario."
    }
}


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

        btn_volatile_preset = QPushButton("🧠 Preset Volátil")
        btn_volatile_preset.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #11111b;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #94e2d5; }
        """)
        btn_volatile_preset.clicked.connect(self._load_volatile_preset)

        btn_system = QPushButton("📋 Presets Sistema")
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
                background-color: #94e2d5;
                color: #11111b;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #89dceb; }
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
        toolbar_layout.addWidget(btn_volatile_preset)
        toolbar_layout.addWidget(btn_system)
        toolbar_layout.addWidget(btn_save)
        toolbar_layout.addWidget(btn_load)
        main_layout.addLayout(toolbar_layout)

        # ── 1. Identidad y Material del Dispositivo ─────────────────────────
        group_a = QGroupBox("1. 🪪 Identidad del Dispositivo y Material")
        layout_a = QFormLayout()

        self.txt_device_name = QLineEdit("Strukov TiO2")
        self.combo_material = QComboBox()
        self.combo_material.addItems(list(MATERIAL_MOBILITY_MAP.keys()))

        self.txt_device_family = QLineEdit("TiO2_oxide")
        self.combo_model_name = QComboBox()
        self.combo_model_name.addItems(["strukov", "ideal_normalized"])

        layout_a.addRow("Nombre Dispositivo:", self.txt_device_name)
        layout_a.addRow("Material Memristivo:", self.combo_material)
        layout_a.addRow("Familia Física:", self.txt_device_family)
        layout_a.addRow("Modelo Matemático:", self.combo_model_name)
        group_a.setLayout(layout_a)
        main_layout.addWidget(group_a)

        # ── 2. Parámetros Eléctricos Universales ────────────────────────────
        group_b = QGroupBox("2. ⚡ Parámetros Eléctricos Universales")
        layout_b = QFormLayout()

        self.spin_r_on = ArrowDoubleSpinBox(value=100.0, min_val=0.001, max_val=999999.0, step=1.0, suffix="")
        self.combo_ron_unit = QComboBox()
        self.combo_ron_unit.addItems(["Ω", "kΩ", "MΩ", "GΩ"])
        self.combo_ron_unit.setCurrentText("Ω")
        lay_ron = QHBoxLayout()
        lay_ron.setContentsMargins(0, 0, 0, 0)
        lay_ron.addWidget(self.spin_r_on)
        lay_ron.addWidget(self.combo_ron_unit)

        self.spin_r_off = ArrowDoubleSpinBox(value=16.0, min_val=0.001, max_val=999999.0, step=1.0, suffix="")
        self.combo_roff_unit = QComboBox()
        self.combo_roff_unit.addItems(["Ω", "kΩ", "MΩ", "GΩ"])
        self.combo_roff_unit.setCurrentText("kΩ")
        lay_roff = QHBoxLayout()
        lay_roff.setContentsMargins(0, 0, 0, 0)
        lay_roff.addWidget(self.spin_r_off)
        lay_roff.addWidget(self.combo_roff_unit)

        self.spin_x0 = ArrowDoubleSpinBox(value=0.10, min_val=0.0, max_val=1.0, step=0.01, decimals=2)

        self.lbl_g_info = QLabel("G_ON: 10.00 mS | G_OFF: 62.50 µS")
        self.lbl_g_info.setStyleSheet("color: #a6adc8; font-size: 11px;")

        self.spin_r_on.valueChanged.connect(self._update_g_labels)
        self.combo_ron_unit.currentTextChanged.connect(self._update_g_labels)
        self.spin_r_off.valueChanged.connect(self._update_g_labels)
        self.combo_roff_unit.currentTextChanged.connect(self._update_g_labels)

        layout_b.addRow("Resistencia R_ON:", lay_ron)
        layout_b.addRow("Resistencia R_OFF:", lay_roff)
        layout_b.addRow("Estado Inicial x_0:", self.spin_x0)
        layout_b.addRow("Conductancias:", self.lbl_g_info)
        group_b.setLayout(layout_b)
        main_layout.addWidget(group_b)

        # ── 3. Parámetros Físicos del Modelo (Strukov) ──────────────────────
        group_e = QGroupBox("3. 🔬 Parámetros Físicos del Material (Strukov 2008)")
        layout_e = QFormLayout()

        self.spin_D_nm = ArrowDoubleSpinBox(value=10.0, min_val=1.0, max_val=500.0, step=1.0, suffix=" nm")
        self.spin_mu_v = ArrowDoubleSpinBox(value=1e-14, min_val=1e-20, max_val=1e-8, step=1e-16, decimals=18)

        self.lbl_material_info = QLabel("TiO₂: Movilidad iónica alta (1×10⁻¹⁴ m²/V·s) — Conmutación rápida (Strukov et al., 2008).")
        self.lbl_material_info.setStyleSheet("color: #a6adc8; font-size: 11px; font-style: italic;")
        self.lbl_material_info.setWordWrap(True)

        layout_e.addRow("Espesor Capa D:", self.spin_D_nm)
        layout_e.addRow("Movilidad µ_v:", self.spin_mu_v)
        layout_e.addRow("Característica del Material:", self.lbl_material_info)
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
            "Modo 4 — Memristor Volátil (Decaimiento Difusivo)",
            "Personalizado"
        ])
        self.combo_realism_mode.currentIndexChanged.connect(self._on_realism_mode_changed)

        # Reproducibilidad Científica (SEED)
        self.spin_seed = ArrowSpinBox(value=42, min_val=0, max_val=999999, step=1)
        self.spin_seed.spin.setToolTip("Semilla fija para reproducibilidad científica estricta")

        self.combo_window_type = QComboBox()
        self.combo_window_type.addItems(["Biolek", "Joglekar", "Sin Ventana"])
        # Auditoría v2026.09.11: max_val limitado a 5 (antes 20).
        # p > 5 hace que f(x) ≈ 0 en los bordes, "congelando" el memristor
        # y bloqueando la conmutación. No es físicamente realista para óxidos neurómorficos.
        self.spin_biolek_p = ArrowSpinBox(value=2, min_val=1, max_val=5, step=1)
        self.spin_biolek_p.spin.setToolTip(
            "Exponente p de la función de ventana de Joglekar/Biolek.\n"
            "Rango recomendado: 1 a 3.\n"
            "Valores altos (p > 5) hacen f(x) ≈ 0 cerca de los bordes,\n"
            "bloqueando la conmutación del memristor ('memristor congelado').\n"
            "Para óxidos neurocomputacionales (TaOx, HfO2), p = 1 o p = 2."
        )

        # C2C — Ornstein-Uhlenbeck
        self.chk_c2c = QCheckBox("Variabilidad Ciclo a Ciclo (C2C - Ornstein-Uhlenbeck)")
        self.chk_c2c.setChecked(False)
        self.spin_c2c_sigma = ArrowDoubleSpinBox(value=0.05, min_val=0.001, max_val=0.5, step=0.01, decimals=3)
        self.spin_c2c_theta = ArrowDoubleSpinBox(value=1.0, min_val=0.1, max_val=10.0, step=0.1, decimals=2)

        # D2D — Device-to-Device
        self.chk_d2d = QCheckBox("Variabilidad Dispositivo a Dispositivo (D2D)")
        self.chk_d2d.setChecked(False)
        self.spin_d2d_sigma = ArrowDoubleSpinBox(value=0.05, min_val=0.0, max_val=0.5, step=0.005, decimals=4)
        self.chk_show_d2d_ensemble = QCheckBox("  ↳ 👥 Graficar Población Monte Carlo")
        self.chk_show_d2d_ensemble.setChecked(True)
        self.chk_show_d2d_ensemble.setEnabled(False)
        self.chk_show_d2d_ensemble.setToolTip("Muestra un abanico de memristores fabricados para visualizar la dispersión D2D")
        self.chk_show_d2d_ensemble.toggled.connect(lambda *_: self.param_changed.emit())

        self.spin_d2d_count = ArrowSpinBox(value=6, min_val=2, max_val=20, step=1)
        self.spin_d2d_count.setEnabled(False)
        self.spin_d2d_count.spin.setToolTip("Número de celdas a simular en la población Monte Carlo (2 a 20)")
        self.spin_d2d_count.valueChanged.connect(lambda *_: self.param_changed.emit())

        # Ruido Térmico de Lectura
        self.chk_noise = QCheckBox("Ruido Térmico de Lectura")
        self.chk_noise.setChecked(False)
        self.spin_noise_std = ArrowDoubleSpinBox(value=1e-6, min_val=0.0, max_val=1e-2, step=1e-6, decimals=7, suffix=" A")

        # ── Modo Volátil / Difusivo (Periódo Refractario Natural) ────────────────────
        self.chk_volatile = QCheckBox(
            "🧠 Modo Volátil (Memristor Difusivo) — Periódo Refractario Natural"
        )
        self.chk_volatile.setChecked(False)
        self.chk_volatile.setToolTip(
            "Activa el término de relajación volátil:\n"
            "  dx/dt_total = dx/dt_strukov + (-(x - x_eq) / τ_relax)\n\n"
            "Simula memristores difusivos (Ag/SiO₂, NbOx, VO₂) cuyo filamento\n"
            "conductor se disuelve espontáneamente al retirar el estímulo.\n\n"
            "USO: permite que el memristor en serie vuelva a R_OFF tras cada\n"
            "estímulo, habilitando operación como SENSOR repetible sin RESET externo."
        )

        self.spin_tau_relax = ArrowDoubleSpinBox(
            value=0.05, min_val=0.001, max_val=2.0, step=0.01, decimals=3, suffix=" s"
        )
        self.spin_tau_relax.setEnabled(False)
        self.spin_tau_relax.spin.setToolTip(
            "τ_relax: Tiempo de relajación del memristor volátil.\n"
            "Determina cuánto tarda x en volver a x_eq tras un estímulo."
        )

        # x_eq: estado de equilibrio bajo la relajación volátil. Independiente de
        # x_0 (estado inicial). Si x_eq < x_0, el memristor se relaja hacia OFF.
        self.spin_x_eq = ArrowDoubleSpinBox(
            value=0.05, min_val=0.0, max_val=1.0, step=0.01, decimals=3, suffix=""
        )
        self.spin_x_eq.setEnabled(False)
        self.spin_x_eq.spin.setToolTip(
            "x_eq: estado de equilibrio bajo la relajación volátil.\n"
            "Es adónde tiende x cuando no hay excitación.\n"
            "Debe ser MENOR que x_0 para que el memristor se relaje hacia OFF."
        )

        self.lbl_volatile_info = QLabel()
        self.lbl_volatile_info.setWordWrap(True)
        self.lbl_volatile_info.setStyleSheet(
            "color: #94e2d5; font-size: 10px; background-color: #1a2a2a; "
            "border: 1px solid #94e2d5; border-radius: 3px; padding: 3px;"
        )
        self.lbl_volatile_info.setVisible(False)

        def _update_volatile_info():
            if not self.chk_volatile.isChecked():
                self.lbl_volatile_info.setVisible(False)
                return
            tau = self.spin_tau_relax.value()
            x_eq = self.spin_x_eq.value() if hasattr(self, "spin_x_eq") else self.spin_x0.value()
            r_on = self.get_r_on_ohms()
            r_off = self.get_r_off_ohms()
            self.lbl_volatile_info.setText(
                f"🧠 Volátil activo | τ_relax = {tau*1e3:.0f} ms | "
                f"x_eq = {x_eq:.2f} | Rango: {r_on/1e3:.1f}–{r_off/1e3:.0f} kΩ"
            )
            self.lbl_volatile_info.setVisible(True)

        self._update_volatile_info = _update_volatile_info

        def _on_volatile_toggled(checked):
            self.spin_tau_relax.setEnabled(checked)
            self.spin_x_eq.setEnabled(checked)
            _update_volatile_info()
            self.param_changed.emit()

        self.chk_volatile.toggled.connect(_on_volatile_toggled)
        self.spin_tau_relax.valueChanged.connect(lambda *_: (_update_volatile_info(), self.param_changed.emit()))
        self.spin_x_eq.valueChanged.connect(lambda *_: (_update_volatile_info(), self.param_changed.emit()))

        layout_d.addRow("Modo de Presets:", self.combo_realism_mode)
        layout_d.addRow("🔑 Semilla (Seed):", self.spin_seed)
        layout_d.addRow("Tipo de Ventana:", self.combo_window_type)
        layout_d.addRow("Exponente Ventana (p):", self.spin_biolek_p)
        layout_d.addRow(self.chk_c2c)
        layout_d.addRow("  ↳ Volatilidad C2C (σ):", self.spin_c2c_sigma)
        layout_d.addRow("  ↳ Retorno Media (θ):", self.spin_c2c_theta)
        layout_d.addRow(self.chk_d2d)
        layout_d.addRow("  ↳ Dispersión D2D (σ):", self.spin_d2d_sigma)
        layout_d.addRow(self.chk_show_d2d_ensemble)
        layout_d.addRow("  ↳ Nº Celdas Población:", self.spin_d2d_count)
        layout_d.addRow(self.chk_noise)
        layout_d.addRow("  ↳ Nivel Ruido (std):", self.spin_noise_std)
        layout_d.addRow(self.chk_volatile)
        layout_d.addRow("  ↳ Tiempo Relajación (τ):", self.spin_tau_relax)
        layout_d.addRow("  ↳ Estado Equilibrio (x_eq):", self.spin_x_eq)
        layout_d.addRow(self.lbl_volatile_info)
        group_d.setLayout(layout_d)
        main_layout.addWidget(group_d)

        # ── 5. Datos de Validación Experimental (CSV) ─────────────────────────
        group_csv = QGroupBox("5. 📊 Datos de Validación Experimental (CSV)")
        layout_csv = QVBoxLayout()

        self.btn_csv_toggle = QPushButton("📊 Validación CSV: ACTIVA")
        self.btn_csv_toggle.setCheckable(True)
        self.btn_csv_toggle.setChecked(True)
        self.btn_csv_toggle.setMinimumHeight(34)
        self.btn_csv_toggle.clicked.connect(self._on_csv_toggle_clicked)
        self._update_csv_toggle_style()

        layout_csv.addWidget(self.btn_csv_toggle)
        group_csv.setLayout(layout_csv)
        main_layout.addWidget(group_csv)

        # ── 6. Gráficas Visibles ──────────────────────────────────────────────────
        group_plots = QGroupBox("6. 📈 Gráficas Visibles")
        layout_plots = QVBoxLayout()
        
        self.chk_plot_iv = QCheckBox("Curva I-V (Histéresis)")
        self.chk_plot_vt_it = QCheckBox("Señales V(t) e I(t)")
        self.chk_plot_x = QCheckBox("Estado Interno x(t)")
        self.chk_plot_r = QCheckBox("Resistencia R(t)")
        self.chk_plot_g = QCheckBox("Conductancia G(t)")
        self.chk_plot_p = QCheckBox("Potencia P(t)")
        
        for chk in [self.chk_plot_iv, self.chk_plot_vt_it, self.chk_plot_x, self.chk_plot_r, self.chk_plot_g, self.chk_plot_p]:
            chk.setChecked(True)
            layout_plots.addWidget(chk)
            chk.toggled.connect(lambda *_: self.param_changed.emit())
            
        group_plots.setLayout(layout_plots)
        main_layout.addWidget(group_plots)

        main_layout.addStretch()
        self.setLayout(main_layout)
        self._update_g_labels()
        self._connect_signals()

    @property
    def is_csv_validation_enabled(self) -> bool:
        """Devuelve True si se deben superponer los datos de validación CSV."""
        return self.btn_csv_toggle.isChecked()

    @property
    def active_plots(self) -> list[str]:
        """Devuelve una lista con los identificadores de las gráficas seleccionadas."""
        active = []
        if self.chk_plot_iv.isChecked(): active.append("iv")
        if self.chk_plot_vt_it.isChecked(): active.append("vt_it")
        if self.chk_plot_x.isChecked(): active.append("x")
        if self.chk_plot_r.isChecked(): active.append("r")
        if self.chk_plot_g.isChecked(): active.append("g")
        if self.chk_plot_p.isChecked(): active.append("p")
        return active

    def _on_csv_toggle_clicked(self):
        """Actualiza el estilo del toggle al cambiar estado y dispara param_changed."""
        self._update_csv_toggle_style()
        self.param_changed.emit()

    def _update_csv_toggle_style(self):
        """Aplica el estilo visual al botón toggle según el estado ON/OFF."""
        if self.btn_csv_toggle.isChecked():
            self.btn_csv_toggle.setText("📊 Validación CSV: ACTIVA")
            self.btn_csv_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #f9e2af;
                    color: #11111b;
                    font-weight: bold;
                    border-radius: 5px;
                    border: 2px solid #e6a817;
                }
                QPushButton:hover { background-color: #eba836; }
            """)
        else:
            self.btn_csv_toggle.setText("📊 Validación CSV: INACTIVA")
            self.btn_csv_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #313244;
                    color: #585b70;
                    font-weight: bold;
                    border-radius: 5px;
                    border: 2px solid #45475a;
                }
                QPushButton:hover { background-color: #45475a; color: #cdd6f4; }
            """)

    # ── Lógica de presets de modo ────────────────────────────────────────────

    def _on_realism_mode_changed(self, index: int):
        """Conmuta automáticamente las casillas de realismo según el modo seleccionado."""
        if index == 0:   # Modo 1 — Ideal
            self.combo_window_type.setCurrentText("Sin Ventana")
            self.chk_c2c.setChecked(False)
            self.chk_d2d.setChecked(False)
            self.chk_noise.setChecked(False)
            self.chk_volatile.setChecked(False)
        elif index == 1:  # Modo 2 — Extendido
            self.combo_window_type.setCurrentText("Biolek")
            self.chk_c2c.setChecked(False)
            self.chk_d2d.setChecked(False)
            self.chk_noise.setChecked(False)
            self.chk_volatile.setChecked(False)
        elif index == 2:  # Modo 3 — Realista Estocástico
            self.combo_window_type.setCurrentText("Biolek")
            self.chk_c2c.setChecked(True)
            self.chk_d2d.setChecked(True)
            self.chk_noise.setChecked(True)
            self.chk_volatile.setChecked(False)
        elif index == 3:  # Modo 4 — Memristor Volátil
            self.combo_window_type.setCurrentText("Sin Ventana")
            self.chk_c2c.setChecked(False)
            self.chk_d2d.setChecked(False)
            self.chk_noise.setChecked(False)
            self.chk_volatile.setChecked(True)
            self.spin_tau_relax.setValue(1.0)
            self.spin_x_eq.setValue(0.05)
            self.spin_x0.setValue(0.99)

    def _connect_signals(self):
        """Conecta los cambios de cualquier campo para emitir param_changed."""
        self.txt_device_name.textChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_material.currentIndexChanged.connect(self._on_material_changed)
        self.txt_device_family.textChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_model_name.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_r_on.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_ron_unit.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_r_off.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_roff_unit.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_x0.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_D_nm.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_mu_v.valueChanged.connect(self._on_mu_v_spin_changed)
        self.spin_seed.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_window_type.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_biolek_p.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.chk_c2c.toggled.connect(lambda *_: self.param_changed.emit())
        self.spin_c2c_sigma.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_c2c_theta.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.chk_d2d.toggled.connect(self._on_d2d_toggled)
        self.spin_d2d_sigma.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.chk_noise.toggled.connect(lambda *_: self.param_changed.emit())
        # Volatile: ya conectado arriba en los lambdas de su propia sección

    def _on_material_changed(self, index: int):
        mat_name = self.combo_material.currentText()
        if mat_name in MATERIAL_MOBILITY_MAP:
            data = MATERIAL_MOBILITY_MAP[mat_name]
            if data["mu_v"] is not None:
                self.spin_mu_v.blockSignals(True)
                self.spin_mu_v.setValue(data["mu_v"])
                self.spin_mu_v.blockSignals(False)
            self.txt_device_family.setText(data["family"])
            self.lbl_material_info.setText(data["info"])
        self.param_changed.emit()

    def _on_mu_v_spin_changed(self, val: float):
        matched = False
        for mat_name, data in MATERIAL_MOBILITY_MAP.items():
            if data["mu_v"] is not None and abs(data["mu_v"] - val) < 1e-20:
                self.combo_material.blockSignals(True)
                self.combo_material.setCurrentText(mat_name)
                self.combo_material.blockSignals(False)
                self.lbl_material_info.setText(data["info"])
                matched = True
                break
        if not matched:
            self.combo_material.blockSignals(True)
            self.combo_material.setCurrentText("Personalizado")
            self.combo_material.blockSignals(False)
            self.lbl_material_info.setText(f"Personalizado: µ_v = {val:.2e} m²/V·s")
        self.param_changed.emit()

    def _on_d2d_toggled(self, checked: bool):
        self.chk_show_d2d_ensemble.setEnabled(checked)
        self.spin_d2d_count.setEnabled(checked)
        self.param_changed.emit()

    def get_r_on_ohms(self) -> float:
        unit = self.combo_ron_unit.currentText()
        if unit == "Ω": return self.spin_r_on.value()
        if unit == "kΩ": return self.spin_r_on.value() * 1e3
        if unit == "MΩ": return self.spin_r_on.value() * 1e6
        if unit == "GΩ": return self.spin_r_on.value() * 1e9
        return self.spin_r_on.value()

    def get_r_off_ohms(self) -> float:
        unit = self.combo_roff_unit.currentText()
        if unit == "Ω": return self.spin_r_off.value()
        if unit == "kΩ": return self.spin_r_off.value() * 1e3
        if unit == "MΩ": return self.spin_r_off.value() * 1e6
        if unit == "GΩ": return self.spin_r_off.value() * 1e9
        return self.spin_r_off.value() * 1e3

    def _update_g_labels(self):
        r_on = self.get_r_on_ohms()
        r_off = self.get_r_off_ohms()
        g_on_ms = (1.0 / r_on) * 1e3 if r_on > 0 else 0
        g_off_us = (1.0 / r_off) * 1e6 if r_off > 0 else 0
        self.lbl_g_info.setText(f"G_ON: {g_on_ms:.2f} mS | G_OFF: {g_off_us:.2f} µS")

    def set_signal_panel(self, signal_panel):
        """Conecta la instancia de SignalPanel para unificar la gestión del perfil completo."""
        self.signal_panel = signal_panel

    def _load_strukov_paper_preset(self):
        self.txt_device_name.setText("Strukov TiO2 (Paper Fig 2b)")
        self.combo_material.setCurrentText("TiO₂ (Dióxido de Titanio - Strukov 2008)")
        self.spin_r_on.setValue(100.0)
        self.combo_ron_unit.setCurrentText("Ω")
        self.spin_r_off.setValue(16.0)
        self.combo_roff_unit.setCurrentText("kΩ")
        self.spin_x0.setValue(0.10)
        self.spin_D_nm.setValue(10.0)
        self.spin_mu_v.setValue(1e-14)
        self.spin_seed.setValue(42)
        self.combo_window_type.setCurrentText("Sin Ventana")
        self.chk_c2c.setChecked(False)
        self.chk_d2d.setChecked(False)
        self.chk_noise.setChecked(False)
        self.chk_volatile.setChecked(False)
        self.combo_realism_mode.setCurrentIndex(0)
        self.btn_csv_toggle.setChecked(True)
        self._update_csv_toggle_style()

        if self.signal_panel is not None:
            idx = self.signal_panel.combo_waveform.findText("Sinusoidal")
            if idx >= 0:
                self.signal_panel.combo_waveform.setCurrentIndex(idx)
            self.signal_panel.spin_v0.setValue(1.0)
            self.signal_panel.spin_f0.setValue(0.5)      # 0.5 Hz — escala temporal real del paper
            self.signal_panel.spin_duration.setValue(6.0)  # 6 s — 0.6 unidades × t_0 = 6 s
            self.signal_panel.spin_dt_ms.setValue(1.0)    # 1 ms — 6000 pasos

        self.param_changed.emit()

    def _load_volatile_preset(self):
        """Preset para Memristor Volátil (Decaimiento Difusivo)."""
        self.txt_device_name.setText("Memristor Volátil HfO2")
        self.combo_material.setCurrentText("HfO₂ (Óxido de Hafnio - CMOS LIF 2025)")
        self.spin_r_on.setValue(1.0)
        self.combo_ron_unit.setCurrentText("kΩ")
        self.spin_r_off.setValue(1.0)
        self.combo_roff_unit.setCurrentText("MΩ")
        self.spin_x0.setValue(0.99)
        self.spin_D_nm.setValue(10.0)
        self.spin_mu_v.setValue(1e-16)
        self.spin_seed.setValue(42)
        self.combo_window_type.setCurrentText("Sin Ventana")
        self.chk_c2c.setChecked(False)
        self.chk_d2d.setChecked(False)
        self.chk_noise.setChecked(False)
        self.chk_volatile.setChecked(True)
        self.spin_tau_relax.setValue(1.0)
        self.spin_x_eq.setValue(0.05)
        self.combo_realism_mode.setCurrentIndex(3)

        if self.signal_panel is not None:
            idx = self.signal_panel.combo_waveform.findText("Tren de Pulsos (Unipolar)")
            if idx >= 0:
                self.signal_panel.combo_waveform.setCurrentIndex(idx)
            self.signal_panel.spin_v0.setValue(5.0)
            self.signal_panel.spin_f0.setValue(40.0)
            self.signal_panel.spin_duration.setValue(5.0)
            self.signal_panel.spin_dt_ms.setValue(0.01)

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
            "material": self.combo_material.currentText(),
            "device_family": self.txt_device_family.text(),
            "model_name": self.combo_model_name.currentText(),
            "r_on": self.spin_r_on.value(),
            "r_on_unit": self.combo_ron_unit.currentText(),
            "r_off": self.spin_r_off.value(),
            "r_off_unit": self.combo_roff_unit.currentText(),
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
            "enable_volatile": self.chk_volatile.isChecked(),
            "tau_relax": self.spin_tau_relax.value(),
            "x_eq": self.spin_x_eq.value() if hasattr(self, "spin_x_eq") else 0.05,
            "enable_csv_validation": self.btn_csv_toggle.isChecked(),
        }

        if self.signal_panel is not None:
            data["signal"] = self.signal_panel.to_dict()

        return data

    def from_dict(self, data: dict):
        """Importa y carga la configuración desde un diccionario JSON."""
        if "device_name" in data:
            self.txt_device_name.setText(data["device_name"])
        if "material" in data:
            idx = self.combo_material.findText(data["material"])
            if idx >= 0:
                self.combo_material.setCurrentIndex(idx)
        if "device_family" in data:
            self.txt_device_family.setText(data["device_family"])
        if "model_name" in data:
            idx = self.combo_model_name.findText(data["model_name"])
            if idx >= 0:
                self.combo_model_name.setCurrentIndex(idx)

        if "r_on" in data:
            self.spin_r_on.setValue(float(data["r_on"]))
        if "r_on_unit" in data:
            self.combo_ron_unit.setCurrentText(str(data["r_on_unit"]))
        elif "r_on" in data and float(data["r_on"]) >= 1000:
            val = float(data["r_on"])
            if val >= 1e6:
                self.spin_r_on.setValue(val / 1e6)
                self.combo_ron_unit.setCurrentText("MΩ")
            else:
                self.spin_r_on.setValue(val / 1e3)
                self.combo_ron_unit.setCurrentText("kΩ")

        if "r_off" in data:
            self.spin_r_off.setValue(float(data["r_off"]))
        if "r_off_unit" in data:
            self.combo_roff_unit.setCurrentText(str(data["r_off_unit"]))
        elif "r_off" in data and float(data["r_off"]) >= 1000:
            val = float(data["r_off"])
            if val >= 1e6:
                self.spin_r_off.setValue(val / 1e6)
                self.combo_roff_unit.setCurrentText("MΩ")
            else:
                self.spin_r_off.setValue(val / 1e3)
                self.combo_roff_unit.setCurrentText("kΩ")

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
        if "enable_volatile" in data:
            self.chk_volatile.setChecked(bool(data["enable_volatile"]))
        if "tau_relax" in data:
            self.spin_tau_relax.setValue(float(data["tau_relax"]))
        if "x_eq" in data and hasattr(self, "spin_x_eq"):
            self.spin_x_eq.setValue(float(data["x_eq"]))
        if "enable_csv_validation" in data:
            self.btn_csv_toggle.setChecked(bool(data["enable_csv_validation"]))
            self._update_csv_toggle_style()

        if "signal" in data and self.signal_panel is not None:
            self.signal_panel.from_dict(data["signal"])

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

    def build_memristor(self, seed_override: Optional[int] = None) -> Memristor:
        """Construye una instancia de Memristor según la configuración actual del panel."""
        identity = DeviceConfig(
            name=self.txt_device_name.text(),
            family=self.txt_device_family.text(),
            model=self.combo_model_name.currentText()
        )

        electrical = ElectricalConfig(
            r_on=self.get_r_on_ohms(),
            r_off=self.get_r_off_ohms(),
            initial_state=self.spin_x0.value()
        )

        strukov_config = StrukovConfig(
            D=self.spin_D_nm.value() * 1e-9,   # nm → m
            mu_v=self.spin_mu_v.value()
        )

        seed = seed_override if seed_override is not None else self.spin_seed.value()
        modifiers = []

        # 1. Ventana de Frontera
        win_type = self.combo_window_type.currentText()
        if win_type == "Biolek":
            modifiers.append(BiolekWindowModifier(p=self.spin_biolek_p.value()))
        elif win_type == "Joglekar":
            modifiers.append(JoglekarWindowModifier(p=self.spin_biolek_p.value()))

        # 2. Variabilidad D2D (dispersión estática en R_on, R_off y conmutación)
        if self.chk_d2d.isChecked():
            d2d_mod = D2DVariabilityModifier(variability_std=self.spin_d2d_sigma.value(), seed=seed)
            electrical = d2d_mod.apply_to_electrical(electrical)
            modifiers.append(d2d_mod)

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

        # 5. Modo Volátil / Difusivo (Término de Relajación — Periodo Refractario Natural)
        if self.chk_volatile.isChecked():
            from neurolab.devices.realism.volatile import VolatileDecayModifier
            # x_eq = equilibrio bajo relajación. Independiente de x_0 (estado inicial).
            # Si no hay control en la GUI, usa 0.05 (memristor difusivo típico).
            x_eq = float(self.spin_x_eq.value()) if hasattr(self, "spin_x_eq") else 0.05
            modifiers.append(VolatileDecayModifier(
                tau_relax=self.spin_tau_relax.value(),
                x0_override=x_eq,
            ))

        return Memristor(
            math_model=StrukovMathModel(),
            electrical=electrical,
            identity=identity,
            model_config=strukov_config,
            modifiers=modifiers,
            clip_x=(win_type != "Sin Ventana")
        )
