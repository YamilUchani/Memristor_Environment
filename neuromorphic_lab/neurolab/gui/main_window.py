"""
neurolab.gui.main_window
=========================
Ventana Principal de la aplicación Neuromorphic Lab.

Características:
  - Loop de simulación con debounce (100 ms) para actualización en tiempo real
  - Autoguardado de sesión al cerrar la ventana (configs/last_session.json)
  - Restauración automática de última sesión al arrancar
  - Título dinámico con el nombre del dispositivo activo
"""
import os
import time
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea,
    QStatusBar, QMessageBox, QSplitter, QTabWidget, QGroupBox, QLabel, QDockWidget,
    QComboBox, QFormLayout, QApplication, QTextBrowser, QPushButton, QRadioButton
)
from PySide6.QtCore import Qt, QTimer
from neurolab.gui.widgets.config_panel import ConfigPanel
from neurolab.gui.widgets.signal_panel import SignalPanel
from neurolab.gui.widgets.plot_canvas import MplCanvas
from neurolab.gui.widgets.neuron_config_panel import NeuronConfigPanel
from neurolab.gui.widgets.neuron_plot_canvas import NeuronMplCanvas
from neurolab.gui.widgets.hybrid_plot_canvas import HybridMplCanvas
from neurolab.gui.widgets.arrow_spinbox import ArrowDoubleSpinBox, NoUnfocusedWheelEventFilter
from neurolab.gui.widgets.synapse_config_panel import SynapseConfigPanel
from neurolab.gui.widgets.synapse_plot_canvas import SynapseMplCanvas
from neurolab.io.profile_manager import ProfileManager
from neurolab.io.validation_loader import ValidationDataLoader
from neurolab.core.validation_metrics import compute_all_metrics
from neurolab.circuits.hybrid import MemristorLIFCircuit, ResistorLIFCircuit
from neurolab.core.lif_validation import compute_lif_validation_metrics
from neurolab.gui.crossbar_view import CrossbarView


class MainWindow(QMainWindow):
    """Ventana Principal de la aplicación Neuromorphic Lab."""

    @property
    def config_panel(self) -> ConfigPanel:
        """Devuelve el ConfigPanel del Memristor actualmente activo en la sub-pestaña de Pestaña 1."""
        return self.memristor_subtabs.currentWidget() if hasattr(self, 'memristor_subtabs') else self.config_panel_1

    def __init__(self):
        super().__init__()
        self._profile_manager = ProfileManager()
        self._validation_loader = ValidationDataLoader()
        self._active_profile_name: str = "Sin Perfil"

        # Filtro global: ignora rueda del ratón en controles numéricos o listas sin clic previo
        self._wheel_filter = NoUnfocusedWheelEventFilter()
        app_instance = QApplication.instance()
        if app_instance is not None:
            app_instance.installEventFilter(self._wheel_filter)

        self.setWindowTitle("Neuromorphic Lab — Simulador Universal de Dispositivos Memristivos")
        self.resize(1280, 800)

        # Temporizador debounce para suavizar actualizaciones en tiempo real
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(100)
        self.debounce_timer.timeout.connect(self.run_simulation)

        # Timer de simulación en tiempo real continuo (Pestaña 3)
        self.rt_timer = QTimer(self)
        self.rt_timer.setInterval(30)  # ~33 FPS
        self.rt_timer.timeout.connect(self._rt_timer_tick)
        self._rt_is_active = False
        self._rt_v_applied = 0.0
        self._rt_t = 0.0
        self._rt_mem = None
        self._rt_neuron = None
        self._rt_refractory_left = 0.0
        self._rt_t_hist = []
        self._rt_v_in_hist = []
        self._rt_v_m_hist = []
        self._rt_r_s_hist = []
        self._rt_r_leak_hist = []
        self._rt_i_in_hist = []
        self._rt_spike_times = []

        self._apply_dark_theme()
        self.init_ui()
        # Simulación inicial ÚNICA: con la sesión restaurada si existe, o con
        # los valores por defecto en caso contrario (evita simular dos veces).
        if not self._restore_last_session():
            self.run_simulation()

    # ── Tema Visual ──────────────────────────────────────────────────────────

    def _apply_dark_theme(self):
        """Aplica un tema visual oscuro moderno a toda la interfaz Qt."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
                padding-right: 24px;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #89b4fa;
            }
            QDoubleSpinBox::up-button, QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 22px; height: 13px;
                background-color: #45475a;
                border-left: 1px solid #313244;
                border-bottom: 1px solid #313244;
                border-top-right-radius: 4px;
            }
            QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover {
                background-color: #89b4fa;
            }
            QDoubleSpinBox::down-button, QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 22px; height: 13px;
                background-color: #45475a;
                border-left: 1px solid #313244;
                border-bottom-right-radius: 4px;
            }
            QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {
                background-color: #89b4fa;
            }
            QCheckBox {
                color: #cdd6f4;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px; height: 16px;
            }
            QStatusBar {
                background-color: #181825;
                color: #a6adc8;
            }
            QScrollArea { border: none; }
            QSplitter::handle:horizontal {
                background-color: #45475a;
                width: 6px;
                margin: 0px 2px;
                border-radius: 3px;
            }
            QSplitter::handle:horizontal:hover {
                background-color: #89b4fa;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 4px;
                background-color: #1e1e2e;
            }
            QTabBar::tab {
                background-color: #313244;
                color: #a6adc8;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #89b4fa;
                color: #11111b;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background-color: #45475a;
            }
        """)

    # ── Inicialización UI ────────────────────────────────────────────────────

    def init_ui(self):
        self.setDockNestingEnabled(True)

        # Usamos un QWidget vacío como central para que los docks ocupen el espacio principal
        self.setCentralWidget(None)

        # === Menú Superior ===
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: #181825;
                color: #cdd6f4;
            }
            QMenuBar::item:selected {
                background-color: #313244;
            }
            QMenu {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
            }
            QMenu::item:selected {
                background-color: #89b4fa;
                color: #11111b;
            }
        """)
        vista_menu = menu_bar.addMenu("🖥️ Vista")
        
        act_tabify = vista_menu.addAction("📑 Agrupar todo en Pestañas")
        act_tabify.triggered.connect(self._action_tabify_all)
        
        act_tile = vista_menu.addAction("🪟 Dividir Pantalla (Mosaico)")
        act_tile.triggered.connect(self._action_tile_all)

        # === Pestaña 1: Memristor ===
        tab_memristor = QWidget()
        layout_memristor = QHBoxLayout(tab_memristor)
        layout_memristor.setContentsMargins(0, 0, 0, 0)
        
        splitter_m = QSplitter(Qt.Horizontal)
        controls_container_m = QWidget()
        controls_layout_m = QVBoxLayout(controls_container_m)
        # Sub-pestañas para configurar Memristores (Strukov Ideal, HfO₂ Serie Neurona, Prezioso 2014)
        self.memristor_subtabs = QTabWidget()
        self.config_panel_1 = ConfigPanel()   # Strukov Ideal
        self.config_panel_2 = ConfigPanel()   # HfO₂ Serie Neurona (Fusiona Volátil + Híbrido)
        self.config_panel_3 = ConfigPanel()   # Prezioso 2014

        # Subpestaña 1: Strukov Ideal (TiO₂)
        self.config_panel_1.txt_device_name.setText("Memristor No Volátil (Strukov)")
        self.config_panel_1.combo_material.setCurrentText("TiO₂ (Dióxido de Titanio - Strukov 2008)")
        self.config_panel_1.spin_r_on.setValue(100.0)
        self.config_panel_1.combo_ron_unit.setCurrentText("Ω")
        self.config_panel_1.spin_r_off.setValue(16.0)
        self.config_panel_1.combo_roff_unit.setCurrentText("kΩ")
        self.config_panel_1.spin_x0.setValue(0.10)
        self.config_panel_1.spin_D_nm.setValue(10.0)
        self.config_panel_1.spin_mu_v.setValue(1e-14)
        self.config_panel_1.combo_realism_mode.setCurrentIndex(0)  # Modo 1 — Ideal (Sin Ventana, Strukov 2008)
        self.config_panel_1.combo_window_type.setCurrentText("Sin Ventana")
        self.config_panel_1.spin_biolek_p.setValue(5)
        self.config_panel_1.spin_seed.setValue(42)

        # Subpestaña 2: HfO₂ Serie Neurona (fusiona Volátil + Híbrido)
        self.config_panel_2.txt_device_name.setText("HfO₂ Serie Neurona")
        self.config_panel_2.combo_material.setCurrentText("HfO₂ (Óxido de Hafnio - CMOS LIF 2025)")
        self.config_panel_2.combo_model_name.setCurrentText("strukov")
        self.config_panel_2.spin_r_on.setValue(1.0)
        self.config_panel_2.combo_ron_unit.setCurrentText("kΩ")
        self.config_panel_2.spin_r_off.setValue(1.0)
        self.config_panel_2.combo_roff_unit.setCurrentText("MΩ")
        self.config_panel_2.spin_x0.setValue(0.05)
        self.config_panel_2.spin_D_nm.setValue(10.0)
        self.config_panel_2.spin_mu_v.setValue(1e-14)
        self.config_panel_2.combo_realism_mode.setCurrentIndex(3)  # Modo 4 — Memristor Volátil (Decaimiento Difusivo)
        self.config_panel_2.combo_window_type.setCurrentText("Biolek")
        self.config_panel_2.spin_biolek_p.setValue(2)
        self.config_panel_2.spin_seed.setValue(42)
        self.config_panel_2.chk_c2c.setChecked(False)
        self.config_panel_2.chk_d2d.setChecked(False)
        self.config_panel_2.chk_noise.setChecked(False)
        self.config_panel_2.chk_volatile.setChecked(True)
        self.config_panel_2.btn_csv_toggle.setChecked(False)
        self.config_panel_2._update_csv_toggle_style()
        self.config_panel_2.spin_tau_relax.setValue(0.5)
        if hasattr(self.config_panel_2, "spin_x_eq"):
            self.config_panel_2.spin_x_eq.setValue(0.05)

        # Subpestaña 3: Prezioso 2014 (memristor_prezioso.json)
        self.config_panel_3.txt_device_name.setText("Prezioso 2014 (Al₂O₃/TiO₂-x)")
        self.config_panel_3.combo_material.setCurrentText("TiO₂ (Dióxido de Titanio - Strukov 2008)")
        self.config_panel_3.combo_model_name.setCurrentText("prezioso")
        self.config_panel_3.spin_r_on.setValue(4.75)
        self.config_panel_3.combo_ron_unit.setCurrentText("kΩ")
        self.config_panel_3.spin_r_off.setValue(1.0)
        self.config_panel_3.combo_roff_unit.setCurrentText("MΩ")
        self.config_panel_3.spin_x0.setValue(0.05)
        self.config_panel_3.spin_D_nm.setValue(30.0)
        self.config_panel_3.spin_mu_v.setValue(1e-12)
        self.config_panel_3.combo_realism_mode.setCurrentIndex(1)  # Modo 2 — Ventana Biolek
        self.config_panel_3.combo_window_type.setCurrentText("Biolek")
        self.config_panel_3.spin_biolek_p.setValue(1)
        self.config_panel_3.spin_seed.setValue(42)

        self.signal_panel = SignalPanel()
        self.config_panel_1.set_signal_panel(self.signal_panel)
        self.config_panel_2.set_signal_panel(self.signal_panel)
        self.config_panel_3.set_signal_panel(self.signal_panel)

        # Cargar perfiles JSON tras enlazar signal_panel
        cfg_struk_path = self._profile_manager.configs_dir / "strukov_ideal.json"
        if cfg_struk_path.exists():
            try:
                data_struk = self._profile_manager.load(str(cfg_struk_path))
                self.config_panel_1.from_dict(data_struk)
            except Exception:
                pass

        cfg_hfo2_path = self._profile_manager.configs_dir / "memristor_hfo2_neuron.json"
        if cfg_hfo2_path.exists():
            try:
                data_hfo2 = self._profile_manager.load(str(cfg_hfo2_path))
                self.config_panel_2.from_dict(data_hfo2)
            except Exception:
                pass

        cfg_prez_path = self._profile_manager.configs_dir / "memristor_prezioso.json"
        if cfg_prez_path.exists():
            try:
                data_prez = self._profile_manager.load(str(cfg_prez_path))
                self.config_panel_3.from_dict(data_prez)
            except Exception:
                pass

        self.memristor_subtabs.addTab(self.config_panel_1, "🔬 1 · Strukov Ideal (TiO₂)")
        self.memristor_subtabs.addTab(self.config_panel_2, "🧠 2 · HfO₂ Serie Neurona")
        self.memristor_subtabs.addTab(self.config_panel_3, "📊 3 · Prezioso 2014")
        self.memristor_subtabs.setCurrentIndex(1)  # Sub-Pestaña 2 (HfO₂ Neurona) activa por defecto
        self._on_subtab_changed(1)  # Forzar carga de señal y parámetros de HfO2 por defecto

        controls_layout_m.addWidget(self.memristor_subtabs)
        controls_layout_m.addWidget(self.signal_panel)
        
        scroll_m = QScrollArea()
        scroll_m.setWidget(controls_container_m)
        scroll_m.setWidgetResizable(True)
        scroll_m.setMinimumWidth(350)
        
        self.plot_canvas = MplCanvas(self)
        
        splitter_m.addWidget(scroll_m)
        splitter_m.addWidget(self.plot_canvas)
        splitter_m.setSizes([460, 820])
        splitter_m.setCollapsible(0, False)
        layout_memristor.addWidget(splitter_m)
        
        # Panel de métricas cuantitativas de validación
        self.metrics_label = QLabel()
        self.metrics_label.setTextFormat(Qt.RichText)
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setAlignment(Qt.AlignTop)
        self.metrics_label.setStyleSheet("""
            QLabel {
                background-color: #181825;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 6px;
                font-size: 11px;
                color: #cdd6f4;
            }
        """)
        self.metrics_label.setText(
            "<p style='color:#585b70; text-align:center;'>"
            "Activa la Validación CSV y ejecuta la simulación<br>"
            "para ver las métricas cuantitativas."
            "</p>"
        )
        controls_layout_m.addWidget(self.metrics_label)
        controls_layout_m.setStretch(0, 0)  # config panel
        controls_layout_m.setStretch(1, 0)  # signal panel
        controls_layout_m.setStretch(2, 1)  # metrics panel expande

        # === Pestaña 1: Memristor (Dock) ===
        self.dock_memristor = QDockWidget("🔬 Simulador de Memristor", self)
        self.dock_memristor.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_memristor.setWidget(tab_memristor)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_memristor)

        # === Pestaña 2: Neurona LIF ===
        tab_neuron = QWidget()
        layout_neuron = QHBoxLayout(tab_neuron)
        layout_neuron.setContentsMargins(0, 0, 0, 0)
        
        splitter_n = QSplitter(Qt.Horizontal)
        controls_container_n = QWidget()
        controls_layout_n = QVBoxLayout(controls_container_n)
        
        self.neuron_config_panel = NeuronConfigPanel()
        
        # Preconfiguración de la Neurona LIF
        self.neuron_config_panel.combo_c_unit.setCurrentText("nF")
        self.neuron_config_panel.spin_c_m.setValue(100.0)
        self.neuron_config_panel.combo_rs_unit.setCurrentText("kΩ")
        self.neuron_config_panel.spin_r_series.setValue(100.0)
        self.neuron_config_panel.combo_r_unit.setCurrentText("MΩ")
        self.neuron_config_panel.spin_r_leak.setValue(1.0)
        self.neuron_config_panel.spin_v_rest.setValue(0.0)
        self.neuron_config_panel.spin_v_th.setValue(1.0)
        self.neuron_config_panel.spin_v_reset.setValue(0.0)
        self.neuron_config_panel.spin_t_ref.setValue(0.0)
        self.neuron_config_panel._on_change()

        # SignalPanel en modo fuente de voltaje (V_IN)
        self.neuron_signal_panel = SignalPanel(mode="voltage")
        self.neuron_config_panel.set_signal_panel(self.neuron_signal_panel)

        cfg_lif_path = self._profile_manager.configs_dir / "lif_config.json"
        if cfg_lif_path.exists():
            try:
                import json
                with open(cfg_lif_path, "r", encoding="utf-8") as f:
                    data_lif = json.load(f)
                self.neuron_config_panel.from_dict(data_lif)
            except Exception:
                pass

        
        controls_layout_n.addWidget(self.neuron_config_panel)
        controls_layout_n.addWidget(self.neuron_signal_panel)
        
        scroll_n = QScrollArea()
        scroll_n.setWidget(controls_container_n)
        scroll_n.setWidgetResizable(True)
        scroll_n.setMinimumWidth(350)
        
        self.neuron_plot_canvas = NeuronMplCanvas(self)
        
        splitter_n.addWidget(scroll_n)
        splitter_n.addWidget(self.neuron_plot_canvas)
        splitter_n.setSizes([460, 820])
        splitter_n.setCollapsible(0, False)
        layout_neuron.addWidget(splitter_n)
        
        # === Pestaña 2: Neurona LIF (Dock) ===
        self.dock_neuron = QDockWidget("🧠 Simulador Neurona LIF", self)
        self.dock_neuron.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_neuron.setWidget(tab_neuron)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_neuron)

        # === Pestaña 3: Híbrido Memristor-LIF ===
        tab_hybrid = QWidget()
        layout_hybrid = QHBoxLayout(tab_hybrid)
        layout_hybrid.setContentsMargins(0, 0, 0, 0)

        splitter_h = QSplitter(Qt.Horizontal)
        controls_container_h = QWidget()
        controls_layout_h = QVBoxLayout(controls_container_h)

        # QGroupBox: Ubicación y Selección de Memristores en el Circuito LIF
        group_mode = QGroupBox("📍 Ubicación y Selección de Memristores en LIF")
        lay_mode = QVBoxLayout(group_mode)

        form_combos = QFormLayout()
        self.combo_mem_rs = QComboBox()
        self.combo_mem_rs.addItems(["Memristor 1 (Strukov Ideal)", "Memristor 2 (HfO₂ Serie Neurona)", "Memristor 3 (Prezioso 2014)"])

        self.combo_mem_rs.setCurrentIndex(1)  # Default: Memristor 2 (HfO₂ Serie Neurona)

        self.combo_mem_rs.setStyleSheet("color: #cdd6f4; background-color: #181825; border: 1px solid #45475a; padding: 4px; border-radius: 4px;")

        lbl_rs = QLabel("Memristor en Serie (R_S):")
        lbl_rs.setStyleSheet("color: #89b4fa; font-weight: bold;")

        form_combos.addRow(lbl_rs, self.combo_mem_rs)
        lay_mode.addLayout(form_combos)

        self.rb_mem_series = QRadioButton("1. Memristor en Serie (R_S) — Modo Sensor")
        self.rb_mem_none   = QRadioButton("2. Sin Memristores (LIF pasivo — validación paper)")
        self.rb_mem_series.setChecked(True)

        for rb in (self.rb_mem_series, self.rb_mem_none):
            rb.setStyleSheet("color: #cdd6f4; font-weight: bold; padding: 3px;")
            lay_mode.addWidget(rb)
            rb.toggled.connect(self._on_param_changed)

        self.combo_mem_rs.currentIndexChanged.connect(self._on_param_changed)

        # QGroupBox: Memristor Activo (Pestaña 1)
        group_mem = QGroupBox("🔬 Memristor Activo (Pestaña 1)")
        lay_mem = QVBoxLayout(group_mem)
        self.lbl_hybrid_mem_info = QLabel()
        self.lbl_hybrid_mem_info.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        self.lbl_hybrid_mem_info.setWordWrap(True)
        btn_edit_mem = QPushButton("✏️ Modificar Memristor en Pestaña 1")
        btn_edit_mem.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #89b4fa;
                border: 1px solid #45475a;
                padding: 5px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45475a; }
        """)
        btn_edit_mem.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        lay_mem.addWidget(self.lbl_hybrid_mem_info)
        lay_mem.addWidget(btn_edit_mem)

        # QGroupBox: Neurona LIF Activa (Pestaña 2)
        group_neu = QGroupBox("🧠 Neurona LIF Activa (Pestaña 2)")
        lay_neu = QVBoxLayout(group_neu)
        self.lbl_hybrid_neuron_info = QLabel()
        self.lbl_hybrid_neuron_info.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        self.lbl_hybrid_neuron_info.setWordWrap(True)
        btn_edit_neu = QPushButton("✏️ Modificar Neurona LIF en Pestaña 2")
        btn_edit_neu.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #a6e3a1;
                border: 1px solid #45475a;
                padding: 5px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45475a; }
        """)
        btn_edit_neu.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        lay_neu.addWidget(self.lbl_hybrid_neuron_info)
        lay_neu.addWidget(btn_edit_neu)

        # === QGroupBox: Modo de Ejecución / Estímulo ===
        group_stim_mode = QGroupBox("⚡ Modo de Ejecución de la Simulación")
        lay_stim_mode = QVBoxLayout(group_stim_mode)

        self.rb_mode_batch = QRadioButton("📊 Modo Lote (Señal Programada V(t))")
        self.rb_mode_realtime = QRadioButton("🔴 Modo Tiempo Real (Estímulo Manual Presionar / Soltar)")
        self.rb_mode_batch.setChecked(True)

        for rb in (self.rb_mode_batch, self.rb_mode_realtime):
            rb.setStyleSheet("color: #cdd6f4; font-weight: bold; padding: 3px;")
            lay_stim_mode.addWidget(rb)
            rb.toggled.connect(self._on_hybrid_execution_mode_changed)

        # === QGroupBox: Control Manual en Tiempo Real ===
        self.group_realtime = QGroupBox("🔴 Control Manual en Tiempo Real (Memristor Volátil)")
        lay_rt = QVBoxLayout(self.group_realtime)

        form_rt = QFormLayout()
        self.spin_rt_vstim = ArrowDoubleSpinBox(value=5.0, min_val=0.1, max_val=10.0, step=0.5, suffix=" V")
        self.spin_rt_vstim.spin.setToolTip("Voltaje V_0 de los pulsos aplicados al presionar")
        lbl_vstim = QLabel("Voltaje de Pulso (V_0):")
        lbl_vstim.setStyleSheet("color: #89b4fa; font-weight: bold;")
        form_rt.addRow(lbl_vstim, self.spin_rt_vstim)

        self.spin_rt_fstim = ArrowDoubleSpinBox(value=40.0, min_val=1.0, max_val=200.0, step=5.0, suffix=" Hz")
        self.spin_rt_fstim.spin.setToolTip("Frecuencia del tren de pulsos en tiempo real")
        lbl_fstim = QLabel("Frecuencia (f_0):")
        lbl_fstim.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        form_rt.addRow(lbl_fstim, self.spin_rt_fstim)

        self.spin_rt_duty = ArrowDoubleSpinBox(value=50.0, min_val=10.0, max_val=90.0, step=5.0, suffix=" %")
        self.spin_rt_duty.spin.setToolTip("Porcentaje de ciclo de trabajo (duty cycle)")
        lbl_duty = QLabel("Ancho de Pulso (Duty):")
        lbl_duty.setStyleSheet("color: #f9e2af; font-weight: bold;")
        form_rt.addRow(lbl_duty, self.spin_rt_duty)

        lay_rt.addLayout(form_rt)

        self.btn_push_hold = QPushButton("🔴 MANTENER PRESIONADO PARA GENERAR TREN DE PULSOS (V_IN)")
        self.btn_push_hold.setMinimumHeight(50)
        self.btn_push_hold.setCursor(Qt.PointingHandCursor)
        self.btn_push_hold.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #f38ba8;
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #f38ba8;
                border-radius: 6px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
        """)
        self.btn_push_hold.pressed.connect(self._on_rt_button_pressed)
        self.btn_push_hold.released.connect(self._on_rt_button_released)
        lay_rt.addWidget(self.btn_push_hold)

        self.lbl_rt_status = QLabel("💤 REPOSO: V_IN = 0.00 V (Relajación Volátil: R_S sube espontáneamente)")
        self.lbl_rt_status.setWordWrap(True)
        self.lbl_rt_status.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #a6e3a1;
                font-size: 11px;
                padding: 6px;
                border: 1px solid #45475a;
                border-radius: 4px;
                background-color: #181825;
            }
        """)
        lay_rt.addWidget(self.lbl_rt_status)

        btn_rt_reset = QPushButton("🔄 Reiniciar Tiempo Real")
        btn_rt_reset.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #89b4fa;
                border: 1px solid #45475a;
                padding: 5px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45475a; }
        """)
        btn_rt_reset.clicked.connect(self._reset_realtime_buffers)
        lay_rt.addWidget(btn_rt_reset)

        self.group_realtime.setVisible(False)

        # Panel de Fuente de Señal para la entrada al circuito híbrido
        self.hybrid_signal_panel = SignalPanel(mode="voltage")
        idx = self.hybrid_signal_panel.combo_waveform.findText("Tren de Pulsos (Unipolar)")
        if idx >= 0:
            self.hybrid_signal_panel.combo_waveform.setCurrentIndex(idx)
        self.hybrid_signal_panel.spin_v0.setValue(5.0)
        self.hybrid_signal_panel.spin_f0.setValue(40.0)
        self.hybrid_signal_panel.spin_duration.setValue(5.0)
        self.hybrid_signal_panel.spin_dt_ms.setValue(0.01)

        controls_layout_h.addWidget(group_mode)
        controls_layout_h.addWidget(group_mem)
        controls_layout_h.addWidget(group_neu)
        controls_layout_h.addWidget(group_stim_mode)
        controls_layout_h.addWidget(self.group_realtime)
        controls_layout_h.addWidget(self.hybrid_signal_panel)

        scroll_h = QScrollArea()
        scroll_h.setWidget(controls_container_h)
        scroll_h.setWidgetResizable(True)
        scroll_h.setMinimumWidth(380)

        self.hybrid_plot_canvas = NeuronMplCanvas(self)

        splitter_h.addWidget(scroll_h)
        splitter_h.addWidget(self.hybrid_plot_canvas)
        splitter_h.setSizes([460, 820])
        splitter_h.setCollapsible(0, False)
        layout_hybrid.addWidget(splitter_h)

        # === Pestaña 3: Híbrido Memristor-LIF (Dock) ===
        self.dock_hybrid = QDockWidget("🧠+🔬 Híbrido Memristor-LIF", self)
        self.dock_hybrid.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_hybrid.setWidget(tab_hybrid)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_hybrid)

        # === Pestaña 4: Plasticidad Sináptica (Fase 3) ===
        tab_synapse = QWidget()
        layout_synapse = QHBoxLayout(tab_synapse)
        layout_synapse.setContentsMargins(0, 0, 0, 0)

        splitter_s = QSplitter(Qt.Horizontal)
        controls_container_s = QWidget()
        controls_layout_s = QVBoxLayout(controls_container_s)

        self.synapse_config_panel = SynapseConfigPanel()
        controls_layout_s.addWidget(self.synapse_config_panel)

        scroll_s = QScrollArea()
        scroll_s.setWidget(controls_container_s)
        scroll_s.setWidgetResizable(True)
        scroll_s.setMinimumWidth(380)

        self.synapse_plot_canvas = SynapseMplCanvas(self)

        splitter_s.addWidget(scroll_s)
        splitter_s.addWidget(self.synapse_plot_canvas)
        splitter_s.setSizes([460, 820])
        splitter_s.setCollapsible(0, False)
        layout_synapse.addWidget(splitter_s)

        self.synapse_config_panel.simulation_requested.connect(
            lambda exp_type, sim_data: self.synapse_plot_canvas.plot_experiment(exp_type, sim_data)
        )
        self.config_panel_1.param_changed.connect(self._sync_memristor_to_synapse)
        self.config_panel_2.param_changed.connect(self._sync_memristor_to_synapse)
        self.config_panel_3.param_changed.connect(self._sync_memristor_to_synapse)
        self.memristor_subtabs.currentChanged.connect(lambda *_: self._sync_memristor_to_synapse())

        self.dock_synapses = QDockWidget("⚡ Plasticidad Sináptica (Fase 3)", self)
        self.dock_synapses.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_synapses.setWidget(tab_synapse)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_synapses)

        # === Pestaña 5: Documentación Formal ===
        tab_docs = QWidget()
        layout_docs = QVBoxLayout(tab_docs)
        layout_docs.setContentsMargins(10, 10, 10, 10)
        
        self.docs_browser = QTextBrowser()
        self.docs_browser.setOpenExternalLinks(True)
        self.docs_browser.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        
        # Cargar el archivo Markdown
        doc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "Modelo_Fisico_Strukov_2008.md")
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                md_content = f.read()
                self.docs_browser.setMarkdown(md_content)
        except Exception as e:
            self.docs_browser.setHtml(f"<h2 style='color:#f38ba8;'>Error al cargar documentación:</h2><p>{e}</p>")
            
        layout_docs.addWidget(self.docs_browser)
        
        # === Pestaña 5: Crossbar 1x1, 2x2 & 4x4 (Fase 4) (Dock) ===
        self.crossbar_view = CrossbarView(self)
        self.dock_crossbar = QDockWidget("🔷 Crossbar 1×1, 2×2 & 4×4 (Fase 4)", self)
        self.dock_crossbar.setAllowedAreas(Qt.AllDockWidgetAreas)

        self.dock_crossbar.setWidget(self.crossbar_view)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_crossbar)

        # === Pestaña 6: Documentación Formal (Dock) ===
        self.dock_docs = QDockWidget("📖 Fundamentos Físicos", self)
        self.dock_docs.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_docs.setWidget(tab_docs)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_docs)

        # Apilar las pestañas (tabify) por defecto
        self.tabifyDockWidget(self.dock_memristor, self.dock_neuron)
        self.tabifyDockWidget(self.dock_neuron, self.dock_hybrid)
        self.tabifyDockWidget(self.dock_hybrid, self.dock_synapses)
        self.tabifyDockWidget(self.dock_synapses, self.dock_crossbar)
        self.tabifyDockWidget(self.dock_crossbar, self.dock_docs)
        self.dock_memristor.raise_()

        # Conectar visibilidad del dock de plasticidad para simular al activarse
        self.dock_synapses.visibilityChanged.connect(
            lambda visible: self.synapse_config_panel.run_simulation() if visible else None
        )

        # === Conexión de Señales ===
        # Modificar botones de "Modificar en Pestaña X" para que traigan el dock al frente
        btn_edit_mem.clicked.disconnect()
        btn_edit_neu.clicked.disconnect()
        btn_edit_mem.clicked.connect(lambda: self.dock_memristor.raise_())
        btn_edit_neu.clicked.connect(lambda: self.dock_neuron.raise_())

        self.signal_panel.run_simulation_requested.connect(self.run_simulation)
        self.config_panel_1.param_changed.connect(self._on_param_changed)
        self.config_panel_2.param_changed.connect(self._on_param_changed)
        self.config_panel_3.param_changed.connect(self._on_param_changed)
        self.memristor_subtabs.currentChanged.connect(self._on_subtab_changed)
        self.signal_panel.param_changed.connect(self._on_param_changed)

        self.neuron_signal_panel.run_simulation_requested.connect(self.run_simulation)
        self.neuron_config_panel.param_changed.connect(self._on_param_changed)
        self.neuron_signal_panel.param_changed.connect(self._on_param_changed)

        self.hybrid_signal_panel.run_simulation_requested.connect(self.run_simulation)
        self.hybrid_signal_panel.param_changed.connect(self._on_param_changed)

        # Reemplazamos tabs.currentChanged por detectar visibilidad si fuera necesario, 
        # pero con DockWidgets cada panel puede notificar cambios, y actualizaremos
        # solo los visibles.
        
        # Conectar cambios de visibilidad de los docks para correr la simulación al mostrarse
        self.dock_memristor.visibilityChanged.connect(lambda visible: self.run_simulation() if visible else None)
        self.dock_neuron.visibilityChanged.connect(lambda visible: self.run_simulation() if visible else None)
        self.dock_hybrid.visibilityChanged.connect(self._on_hybrid_dock_visibility_changed)

        # Barra de Estado
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo. Modo en tiempo real activo.")

        # Nota: la simulación inicial la ejecuta el constructor (una sola vez)
        # después de intentar restaurar la última sesión.

    # ── Funciones del Menú Vista ──────────────────────────────────────────────
    
    def _action_tabify_all(self):
        """Agrupa todos los docks en un solo espacio con pestañas."""
        self.tabifyDockWidget(self.dock_memristor, self.dock_neuron)
        self.tabifyDockWidget(self.dock_neuron, self.dock_hybrid)
        self.tabifyDockWidget(self.dock_hybrid, self.dock_docs)
        self.dock_memristor.raise_()
        
    def _action_tile_all(self):
        """Divide la pantalla para mostrar los paneles más importantes a la vez."""
        # Colocamos Memristor a la izquierda
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_memristor)
        # Híbrido a la derecha
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_hybrid)
        # Neurona abajo del memristor
        self.splitDockWidget(self.dock_memristor, self.dock_neuron, Qt.Vertical)
        # Documentos abajo del híbrido (si se quiere ver) o agrupado
        self.tabifyDockWidget(self.dock_hybrid, self.dock_docs)
        self.dock_hybrid.raise_()

    # ── Gestión de Sesión ────────────────────────────────────────────────────

    def _restore_last_session(self) -> bool:
        """
        Restaura la última sesión guardada (configs/last_session.json).
        Devuelve True si la sesión se restauró correctamente; False si no
        existe o está corrupta (la app arranca con valores por defecto).
        No ejecuta la simulación: el llamador decide cuándo simular, de modo
        que el arranque ejecute la simulación inicial una sola vez.
        """
        try:
            data = self._profile_manager.load_last_session()
            if data is None:
                return False
            self.config_panel_1.from_dict(data)
            device_name = data.get("device_name", "Última Sesión")
            self._update_window_title(device_name)
            self.status_bar.showMessage(f"✓ Sesión restaurada: '{device_name}'")
            return True
        except Exception:
            # Sesión corrupta: ignorar silenciosamente
            return False

    def _save_last_session(self):
        """Guarda la configuración actual como última sesión (silencioso, sin diálogo)."""
        try:
            data = self.config_panel.to_dict()
            self._profile_manager.save_last_session(data)
        except Exception:
            pass  # Nunca interrumpir el cierre por un error de guardado

    def _update_window_title(self, device_name: str = ""):
        """Actualiza el título de la ventana con el nombre del dispositivo activo."""
        if device_name:
            self.setWindowTitle(
                f"Neuromorphic Lab — {device_name}"
            )
        else:
            self.setWindowTitle("Neuromorphic Lab — Simulador Universal de Dispositivos Memristivos")

    def _sync_memristor_to_synapse(self):
        """Sincroniza los parámetros del memristor activo de Pestaña 1 hacia Plasticidad (Pestaña 4) y Crossbar (Pestaña 5)."""
        if not hasattr(self, 'memristor_subtabs'):
            return

        active_idx = self.memristor_subtabs.currentIndex()
        if active_idx == 0:
            panel = self.config_panel_1
        elif active_idx == 1:
            panel = self.config_panel_2
        else:
            panel = self.config_panel_3

        # 1. Sincronizar Pestaña 4: Plasticidad Sináptica
        if hasattr(self, 'synapse_config_panel'):
            sp = self.synapse_config_panel
            if hasattr(sp, 'memristor_subtabs'):
                sp.memristor_subtabs.blockSignals(True)
                sp.memristor_subtabs.setCurrentIndex(active_idx)
                sp.memristor_subtabs.blockSignals(False)

            target_cp = sp.get_active_config_panel()
            target_cp.txt_device_name.setText(panel.txt_device_name.text())
            target_cp.combo_model_name.setCurrentText(panel.combo_model_name.currentText())
            target_cp.spin_r_on.setValue(panel.spin_r_on.value())
            target_cp.combo_ron_unit.setCurrentText(panel.combo_ron_unit.currentText())
            target_cp.spin_r_off.setValue(panel.spin_r_off.value())
            target_cp.combo_roff_unit.setCurrentText(panel.combo_roff_unit.currentText())
            target_cp.spin_x0.setValue(panel.spin_x0.value())
            target_cp.combo_window_type.setCurrentText(panel.combo_window_type.currentText())
            target_cp.chk_volatile.setChecked(panel.chk_volatile.isChecked())
            target_cp.spin_tau_relax.setValue(panel.spin_tau_relax.value())

            sp.run_simulation()

        # 2. Sincronizar Pestaña 5: Crossbar View (1x1, 2x2, 4x4)
        if hasattr(self, 'crossbar_view') and hasattr(self.crossbar_view, 'elements'):
            model_str = panel.combo_model_name.currentText().lower()
            dev_name = panel.txt_device_name.text()
            r_on = panel.get_r_on_ohms()
            r_off = panel.get_r_off_ohms()
            x0 = float(panel.spin_x0.value())
            is_vol = panel.chk_volatile.isChecked()
            tau_rel = float(panel.spin_tau_relax.value())

            for elem in self.crossbar_view.elements.values():
                if hasattr(elem, 'params'):
                    elem.params['model'] = model_str
                    elem.params['name'] = dev_name
                    elem.params['RON'] = r_on
                    elem.params['ROFF'] = r_off
                    elem.params['x0'] = x0
                    elem.params['is_volatile'] = is_vol
                    elem.params['enable_volatile'] = is_vol
                    elem.params['volatile_tau_relax'] = tau_rel
                    elem.params['tau_relax'] = tau_rel
                    if hasattr(elem, 'on_params_changed'):
                        elem.on_params_changed()

            if hasattr(self.crossbar_view, 'canvas') and self.crossbar_view.canvas:
                self.crossbar_view.canvas.update()

    # ── Eventos Qt ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Intercepta el cierre para autoguardar la sesión actual."""
        self._save_last_session()
        event.accept()

    # ── Lógica de Simulación ─────────────────────────────────────────────────

    def _on_param_changed(self):
        """Disparado ante cualquier cambio de parámetro si la opción en tiempo real está activa."""
        if hasattr(self, 'rb_mode_realtime') and self.rb_mode_realtime.isChecked():
            idx_rs = self.combo_mem_rs.currentIndex()
            self._rt_mem = self._get_memristor_by_index(idx_rs, seed_offset=0)
            self._rt_neuron = self.neuron_config_panel.build_neuron()

        if self.signal_panel.is_realtime_enabled or self.neuron_signal_panel.is_realtime_enabled or self.hybrid_signal_panel.is_realtime_enabled:
            self.debounce_timer.start(180)

    def run_simulation(self):
        """Ejecuta la simulación únicamente para las vistas visibles en pantalla."""
        if hasattr(self, 'dock_memristor') and self.dock_memristor.isVisible() and not self.dock_memristor.visibleRegion().isEmpty():
            self._run_memristor_simulation()
        if hasattr(self, 'dock_neuron') and self.dock_neuron.isVisible() and not self.dock_neuron.visibleRegion().isEmpty():
            self._run_neuron_simulation()
        if hasattr(self, 'dock_hybrid') and self.dock_hybrid.isVisible() and not self.dock_hybrid.visibleRegion().isEmpty():
            self._run_hybrid_simulation()

    def _run_memristor_simulation(self):
        """Ejecuta la simulación usando la configuración activa del memristor y la señal."""
        try:
            start_time = time.time()

            # 1. Construir memristor y obtener señal
            memristor = self.config_panel.build_memristor()
            t, v_signal, dt = self.signal_panel.generate_voltage_signal()
            steps = len(t)

            i_out = np.zeros(steps)
            x_state = np.zeros(steps)
            r_hist = np.zeros(steps)
            g_hist = np.zeros(steps)

            # Bug #2 fix: Inyectar el dt real de simulación al modificador C2C
            from neurolab.devices.realism.c2c import C2CVariabilityModifier
            for mod in memristor.modifiers:
                if isinstance(mod, C2CVariabilityModifier):
                    mod.set_simulation_dt(dt)

            # 2. Bucle de integración temporal (Euler Explícito)
            for step_idx in range(steps):
                v_inst = v_signal[step_idx]
                i_inst = memristor.step(voltage=v_inst, dt=dt)
                i_out[step_idx] = i_inst
                x_state[step_idx] = memristor.x
                r_hist[step_idx] = memristor.resistance
                g_hist[step_idx] = memristor.conductance


            # 2b. Población Monte Carlo de Dispositivos (D2D Ensemble)
            ensemble_data = []
            if self.config_panel.chk_d2d.isChecked() and self.config_panel.chk_show_d2d_ensemble.isChecked():
                base_seed = self.config_panel.spin_seed.value()
                num_ensemble = self.config_panel.spin_d2d_count.value()
                for dev_i in range(1, num_ensemble + 1):
                    dev_k = self.config_panel.build_memristor(seed_override=base_seed + dev_i * 101)
                    i_k = np.zeros(steps)
                    x_k = np.zeros(steps)
                    r_k = np.zeros(steps)
                    g_k = np.zeros(steps)
                    for s_idx in range(steps):
                        v_k = v_signal[s_idx]
                        i_k[s_idx] = dev_k.step(voltage=v_k, dt=dt)
                        x_k[s_idx] = dev_k.x
                        r_k[s_idx] = dev_k.resistance
                        g_k[s_idx] = dev_k.conductance
                    ensemble_data.append({
                        "v_drop": v_signal,
                        "i_out": i_k,
                        "x_state": x_k,
                        "r_hist": r_k,
                        "g_hist": g_k
                    })

            elapsed = time.time() - start_time

            # 3. Cargar datos de validación CSV si el interruptor está activo
            val_data = None
            es_hfo2_neurona = "hfo" in self.config_panel.txt_device_name.text().lower() if hasattr(self.config_panel, 'txt_device_name') else False
            if self.config_panel.is_csv_validation_enabled and not es_hfo2_neurona:
                subtab_idx = self.memristor_subtabs.currentIndex() if hasattr(self, 'memristor_subtabs') else 0
                dev_name = self.config_panel.txt_device_name.text().lower() if hasattr(self.config_panel, 'txt_device_name') else ""
                if subtab_idx == 2 or "prezioso" in dev_name:
                    val_data = self._validation_loader.load_all(dataset="prezioso")
                else:
                    val_data = self._validation_loader.load_all(dataset="default")

            # 4. Graficar resultados
            self.plot_canvas.plot_results(
                t, v_signal, v_signal, i_out, x_state, r_hist, g_hist,
                val_data=val_data,
                active_plots=self.config_panel.active_plots,
                ensemble_data=ensemble_data if ensemble_data else None
            )

            # 5. Métricas cuantitativas de validación
            if val_data is not None:
                metrics = compute_all_metrics(
                    t_sim=t,
                    i_sim_mA=i_out * 1e3,
                    x_sim=x_state,
                    r_sim_kohm=r_hist / 1e3,
                    g_sim_us=g_hist * 1e6,
                    val_data=val_data,
                    v_sim=v_signal
                )
                self._update_metrics_panel(metrics)
            else:
                self.metrics_label.setText(
                    "<p style='color:#585b70; text-align:center;'>"
                    "Activa la Validación CSV para ver las métricas cuantitativas."
                    "</p>"
                )

            # 6. Actualizar título y barra de estado
            device_name = memristor.identity.device_name
            self._update_window_title(device_name)

            r_min, r_max = np.min(r_hist), np.max(r_hist)
            msg = (
                f"✓ Simulación Memristor ejecutada en {elapsed:.3f}s | Pasos: {steps:,} | "
                f"Dispositivo: '{device_name}' | "
                f"R_min: {r_min:.1f} Ω, R_max: {r_max:.1f} Ω"
            )
            self.status_bar.showMessage(msg)

        except Exception as e:
            QMessageBox.critical(self, "Error de Simulación", f"Ocurrió un error al ejecutar la simulación del Memristor:\n{str(e)}")
            self.status_bar.showMessage("❌ Error en la simulación.")
            
    def _update_metrics_panel(self, metrics: list) -> None:
        """Construye y muestra la tabla HTML de métricas cuantitativas de validación."""
        if not metrics:
            self.metrics_label.setText(
                "<p style='color:#585b70; text-align:center;'>Sin métricas disponibles.</p>"
            )
            return

        header = (
            "<table width='100%' cellspacing='0' cellpadding='0'>"
            "<tr style='background-color:#313244;'>"
            "<th style='padding:4px 6px; text-align:left; color:#89b4fa; font-size:11px;'>Magnitud</th>"
            "<th style='padding:4px 6px; text-align:right; color:#89b4fa; font-size:11px;'>MAE</th>"
            "<th style='padding:4px 6px; text-align:right; color:#89b4fa; font-size:11px;'>RMSE</th>"
            "<th style='padding:4px 6px; text-align:right; color:#89b4fa; font-size:11px;'>Err.Rel.Máx</th>"
            "<th style='padding:4px 6px; text-align:right; color:#89b4fa; font-size:11px;'>R²</th>"
            "</tr>"
        )
        rows = "".join(m.to_html_row() for m in metrics)
        footer = "</table>"

        # Leyenda de color
        legend = (
            "<p style='font-size:10px; color:#585b70; margin-top:4px;'>"
            "<span style='color:#a6e3a1;'>■</span> Excelente &nbsp;"
            "<span style='color:#f9e2af;'>■</span> Aceptable &nbsp;"
            "<span style='color:#f38ba8;'>■</span> Revisar"
            "</p>"
        )

        title = "<p style='color:#89b4fa; font-weight:bold; margin-bottom:4px;'>📊 Métricas de Validación CSV</p>"
        self.metrics_label.setText(title + header + rows + footer + legend)

    def _run_neuron_simulation(self):
        """Ejecuta la simulación de la Neurona LIF en tiempo real impulsada por voltaje V_IN y R_S."""
        try:
            start_time = time.time()
            
            neuron = self.neuron_config_panel.build_neuron()
            t, v_signal, dt = self.neuron_signal_panel.generate_voltage_signal()
            steps = len(t)

            v_m_hist = np.zeros(steps)
            i_in_hist = np.zeros(steps)
            r_leak_hist = np.full(steps, neuron.config.r_leak)
            r_series_hist = np.full(steps, neuron.config.r_series)
            
            for step_idx in range(steps):
                v_m_hist[step_idx] = neuron.v_membrane
                # Corriente inyectada por la fuente a través de R_S: I_in = (V_IN - V_m) / R_S
                i_in_hist[step_idx] = (v_signal[step_idx] - neuron.v_membrane) / neuron.config.r_series
                neuron.step(voltage_input=v_signal[step_idx], dt=dt)
                
            elapsed = time.time() - start_time
            
            # ── Validación Matemática Analítica LIF (Sección E) ──
            lif_val = compute_lif_validation_metrics(t, v_m_hist, v_signal, neuron.config, is_voltage_input=True)

            self.neuron_plot_canvas.plot_results(
                t, v_signal, v_m_hist, r_leak_hist, neuron.spike_times, 
                neuron.config.v_th, neuron.config.v_reset,
                neuron.config.v_rest, i_in=i_in_hist, r_series_hist=r_series_hist,
                val_metrics=lif_val
            )
            
            n_spikes = len(neuron.spike_times)
            freq = n_spikes / t[-1] if t[-1] > 0 else 0
            
            self._update_window_title("Simulador LIF (Fuente Voltaje V_IN + R_S)")
            
            msg = (f"✓ Simulación LIF ejecutada en {elapsed:.3f}s | Pasos: {steps:,} | "
                   f"Spikes: {n_spikes} ({freq:.1f} Hz) | "
                   f"Validación Teórica: MAE={lif_val['mae']*1e3:.2f} mV, RMSE={lif_val['rmse']*1e3:.2f} mV, R²={lif_val['r2']:.4f} | VALIDADO ANALÍTICAMENTE ✅")
            self.status_bar.showMessage(msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error de Simulación", f"Error en la simulación LIF:\n{str(e)}")
            self.status_bar.showMessage("❌ Error en la simulación LIF.")

    def _on_subtab_changed(self, index: int):
        """Al cambiar de subpestaña, actualiza el panel activo y la fuente de señal con su perfil JSON."""
        json_map = {
            0: "strukov_ideal.json",
            1: "memristor_hfo2_neuron.json",
            2: "memristor_prezioso.json"
        }
        json_file = json_map.get(index)
        if json_file:
            path = self._profile_manager.configs_dir / json_file
            if path.exists():
                try:
                    data = self._profile_manager.load(str(path))
                    panel = self.memristor_subtabs.widget(index)
                    if hasattr(panel, "from_dict"):
                        panel.from_dict(data)
                except Exception:
                    pass
        self.run_simulation()

    def _get_panel_by_index(self, index: int) -> ConfigPanel:
        """Devuelve el ConfigPanel correspondiente al índice de sub-pestaña."""
        if index == 1:
            return self.config_panel_2   # HfO₂ Serie Neurona
        elif index == 2:
            return self.config_panel_3   # Prezioso 2014
        else:
            return self.config_panel_1   # Strukov Ideal

    def _get_memristor_by_index(self, index: int, seed_offset: int = 0) -> Memristor:
        """Devuelve una instancia de Memristor configurado según el índice de la sub-pestaña (0, 1, 2)."""
        panel = self._get_panel_by_index(index)

        if seed_offset != 0:
            base_seed = panel.spin_seed.value()
            return panel.build_memristor(seed_override=base_seed + seed_offset)
        return panel.build_memristor()

    def _update_hybrid_summary(self):
        """Actualiza las tarjetas informativas de la Pestaña 3 mostrando los valores exactos de los paneles."""
        try:
            idx_rs = self.combo_mem_rs.currentIndex()

            panel_rs = self._get_panel_by_index(idx_rs)
            
            mem_rs = self._get_memristor_by_index(idx_rs)
            neu = self.neuron_config_panel.build_neuron()
            
            mode_series = self.rb_mem_series.isChecked()
            mode_none = self.rb_mem_none.isChecked()

            def _fmt_r(val_ohm: float) -> str:
                if val_ohm >= 1e6:
                    return f"{val_ohm/1e6:.2f} MΩ"
                elif val_ohm >= 1e3:
                    return f"{val_ohm/1e3:.2f} kΩ"
                else:
                    return f"{val_ohm:.1f} Ω"

            mat_rs_name = panel_rs.combo_material.currentText().split(' ')[0]
            
            if mode_none:
                mem_text = (
                    f"• <b>Resistencia Serie:</b> Fija ({_fmt_r(neu.config.r_series)})<br/>"
                    f"• <b>Circuito LIF Pasivo</b> (validación paper)"
                )
            else:
                mem_text = (
                    f"• <b>Resistencia Serie:</b> Sustituida por Memristor {idx_rs+1} "
                    f"(R<sub>ini</sub> = {_fmt_r(mem_rs.resistance)})<br/>"
                    f"• <b>Memristor {idx_rs+1} ({mat_rs_name}):</b> "
                    f"R<sub>ON</sub> = {_fmt_r(panel_rs.get_r_on_ohms())} | "
                    f"R<sub>OFF</sub> = {_fmt_r(panel_rs.get_r_off_ohms())}"
                )
            
            cfg = neu.config
            neu_text = (
                f"• <b>Capacitancia C<sub>m</sub>:</b> {cfg.c_m*1e9:.2f} nF<br/>"
                f"• <b>Resistencia Serie Fija (LIF P2):</b> {_fmt_r(cfg.r_series)}<br/>"
                f"• <b>Resistencia Fuga Fija (LIF P2):</b> {_fmt_r(cfg.r_leak)}<br/>"
                f"• <b>Umbral V<sub>th</sub>:</b> {cfg.v_th:.2f} V | <b>Reset V<sub>reset</sub>:</b> {cfg.v_reset:.2f} V"
            )
            
            self.lbl_hybrid_mem_info.setText(mem_text)
            self.lbl_hybrid_neuron_info.setText(neu_text)
        except Exception:
            pass

    def _run_hybrid_simulation(self):
        """
        Ejecuta la simulación Híbrida (Pestaña 3) con los gráficos de Neurona LIF.
        Selecciona dinámicamente qué Memristor de Pestaña 1 se asigna a R_S y a R_leak.
        """
        try:
            start_time = time.time()
            
            # Actualizar tarjetas informativas en la Pestaña 3
            self._update_hybrid_summary()
            
            # 1. Obtener Memristor(es) de Pestaña 1 y Neurona LIF de Pestaña 2 según selección
            idx_rs = self.combo_mem_rs.currentIndex()

            mem1 = self._get_memristor_by_index(idx_rs, seed_offset=0)
            neuron = self.neuron_config_panel.build_neuron()
            
            # 2. Generar señal de entrada V_IN(t)
            t, v_signal, dt = self.hybrid_signal_panel.generate_voltage_signal()
            steps = len(t)

            v_m_hist = np.zeros(steps)
            i_in_hist = np.zeros(steps)
            r_leak_hist = np.zeros(steps)
            r_series_hist = np.zeros(steps)

            # Determinar modo seleccionado
            mode_series = self.rb_mem_series.isChecked()
            mode_none = self.rb_mem_none.isChecked()

            r_s_base = neuron.config.r_series
            r_leak_base = neuron.config.r_leak
            c_m = neuron.config.c_m
            v_rest = neuron.config.v_rest
            v_th = neuron.config.v_th
            v_reset = neuron.config.v_reset
            t_ref = neuron.config.t_ref

            # Bug #2 fix: Inyectar el dt real de simulación a los modificadores C2C
            from neurolab.devices.realism.c2c import C2CVariabilityModifier
            for mem in (mem1,):
                for mod in mem.modifiers:
                    if isinstance(mod, C2CVariabilityModifier):
                        mod.set_simulation_dt(dt)

            refractory_left = 0.0

            for step_idx in range(steps):
                v_in = v_signal[step_idx]
                v_m = neuron.v_membrane
                
                # --- Cálculo de Resistencias e Inyección según el modo ---
                if mode_series:
                    v_drop_s = max(v_in - v_m, 0.0)
                    i_in_val = mem1.step(v_drop_s, dt)
                    r_s_curr = max(mem1.resistance, mem1.electrical.r_on)
                    r_leak_curr = r_leak_base
                    i_leak_val = (v_m - v_rest) / r_leak_curr if r_leak_curr > 0 else 0.0
                else:  # mode_none
                    r_s_curr = r_s_base
                    r_leak_curr = r_leak_base
                    i_in_val = max(v_in - v_m, 0.0) / r_s_curr if r_s_curr > 0 else 0.0
                    i_leak_val = (v_m - v_rest) / r_leak_curr if r_leak_curr > 0 else 0.0

                # Guardar historiales
                v_m_hist[step_idx] = v_m
                i_in_hist[step_idx] = i_in_val
                r_series_hist[step_idx] = r_s_curr
                r_leak_hist[step_idx] = r_leak_curr

                # --- Integración Física de la Neurona LIF ---
                neuron.t += dt
                neuron.has_spiked = False

                if refractory_left > 0.0:
                    refractory_left -= dt
                    dv_refrac = (-i_leak_val / c_m) * dt
                    neuron.v_membrane += dv_refrac
                    if neuron.v_membrane < v_reset:
                        neuron.v_membrane = v_reset
                else:
                    dv = ((i_in_val - i_leak_val) / c_m) * dt
                    neuron.v_membrane += dv
                    
                    if neuron.v_membrane >= v_th:
                        neuron.has_spiked = True
                        neuron.spike_times.append(neuron.t)
                        neuron.v_membrane = v_reset
                        if t_ref > 0.0:
                            refractory_left = t_ref


            elapsed = time.time() - start_time

            # 3. Graficar en el NeuronMplCanvas de la Neurona LIF
            self.hybrid_plot_canvas.plot_results(
                t=t,
                signal_in=v_signal,
                v_m=v_m_hist,
                r_m_hist=r_leak_hist,
                spike_times=neuron.spike_times,
                v_th=v_th,
                v_reset=v_reset,
                v_rest=v_rest,
                i_in=i_in_hist,
                r_series_hist=r_series_hist
            )

            n_spikes = len(neuron.spike_times)
            freq = n_spikes / t[-1] if t[-1] > 0 else 0

            mode_str = f"Serie (Memristor {idx_rs + 1})" if mode_series else "LIF pasivo"
            self._update_window_title(f"Neuromorphic Lab — Híbrido: {mode_str}")

            msg = (f"✓ Simulación Híbrida ejecutada en {elapsed:.3f}s | Config: {mode_str} | "
                   f"Spikes (P2): {n_spikes} ({freq:.1f} Hz)")
            self.status_bar.showMessage(msg)

        except Exception as e:
            QMessageBox.critical(self, "Error de Simulación Híbrida", f"Error en la simulación Híbrida:\n{str(e)}")
            self.status_bar.showMessage("❌ Error en la simulación Híbrida.")

    # ── Lógica de Tiempo Real Interactivo (Pestaña 3) ─────────────────────────

    def _on_hybrid_dock_visibility_changed(self, visible: bool):
        if not visible:
            if hasattr(self, 'rt_timer') and self.rt_timer.isActive():
                self.rt_timer.stop()
        else:
            if hasattr(self, 'rb_mode_realtime') and self.rb_mode_realtime.isChecked():
                self._start_realtime_mode()
            else:
                self.run_simulation()

    def _on_hybrid_execution_mode_changed(self):
        """Conmuta entre simulación por lote y modo tiempo real interactivo en Pestaña 3."""
        if self.rb_mode_realtime.isChecked():
            self.hybrid_signal_panel.setVisible(False)
            self.group_realtime.setVisible(True)
            self._start_realtime_mode()
        else:
            self.group_realtime.setVisible(False)
            self.hybrid_signal_panel.setVisible(True)
            self._stop_realtime_mode()
            self.run_simulation()

    def _start_realtime_mode(self):
        """Inicializa los objetos de simulación en tiempo real y enciende el timer."""
        idx_rs = self.combo_mem_rs.currentIndex()
        self._rt_mem = self._get_memristor_by_index(idx_rs, seed_offset=0)
        self._rt_neuron = self.neuron_config_panel.build_neuron()
        self._reset_realtime_buffers()
        self._rt_is_active = True
        self.rt_timer.start(30)

    def _stop_realtime_mode(self):
        """Detiene el timer de tiempo real."""
        self._rt_is_active = False
        if hasattr(self, 'rt_timer'):
            self.rt_timer.stop()

    def _reset_realtime_buffers(self):
        """Limpia los historiales de tiempo real y resetea el reloj a t = 0.0 s."""
        idx_rs = self.combo_mem_rs.currentIndex()
        self._rt_mem = self._get_memristor_by_index(idx_rs, seed_offset=0)
        self._rt_neuron = self.neuron_config_panel.build_neuron()
        self._rt_t = 0.0
        self._rt_refractory_left = 0.0
        self._rt_button_is_pressed = False

        self._rt_t_hist = []
        self._rt_v_in_hist = []
        self._rt_v_m_hist = []
        self._rt_r_s_hist = []
        self._rt_r_leak_hist = []
        self._rt_i_in_hist = []
        self._rt_spike_times = []

        if hasattr(self, 'btn_push_hold'):
            self.btn_push_hold.setText("🔴 MANTENER PRESIONADO PARA GENERAR TREN DE PULSOS (V_IN)")
            self.btn_push_hold.setStyleSheet("""
                QPushButton {
                    background-color: #313244;
                    color: #f38ba8;
                    font-weight: bold;
                    font-size: 13px;
                    border: 2px solid #f38ba8;
                    border-radius: 6px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #45475a;
                }
            """)
        if hasattr(self, 'lbl_rt_status'):
            self.lbl_rt_status.setText("💤 REPOSO: V_IN = 0.00 V (Relajación Volátil: R_S sube espontáneamente)")
            self.lbl_rt_status.setStyleSheet("font-weight: bold; color: #a6e3a1; font-size: 11px; padding: 6px; border: 1px solid #45475a; border-radius: 4px; background-color: #181825;")

    def _on_rt_button_pressed(self):
        self._rt_button_is_pressed = True
        v_stim = self.spin_rt_vstim.value()
        f_stim = self.spin_rt_fstim.value()
        duty_stim = self.spin_rt_duty.value()
        self.btn_push_hold.setText(f"⚡ GENERANDO TREN DE PULSOS: +{v_stim:.2f} V @ {f_stim:.0f} Hz ({duty_stim:.0f}% Duty) ⚡")
        self.btn_push_hold.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #11111b;
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #ff79c6;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        self.lbl_rt_status.setText(f"⚡ TREN DE PULSOS ACTIVO: +{v_stim:.2f} V @ {f_stim:.0f} Hz (Generando pulsos mientras mantengas presionado)")
        self.lbl_rt_status.setStyleSheet("font-weight: bold; color: #f38ba8; font-size: 11px; padding: 6px; border: 1px solid #f38ba8; border-radius: 4px; background-color: #181825;")

    def _on_rt_button_released(self):
        self._rt_button_is_pressed = False
        self.btn_push_hold.setText("🔴 MANTENER PRESIONADO PARA GENERAR TREN DE PULSOS (V_IN)")
        self.btn_push_hold.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #f38ba8;
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #f38ba8;
                border-radius: 6px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
        """)
        self.lbl_rt_status.setText("💤 REPOSO: V_IN = 0.00 V (Relajación Volátil: R_S sube espontáneamente)")
        self.lbl_rt_status.setStyleSheet("font-weight: bold; color: #a6e3a1; font-size: 11px; padding: 6px; border: 1px solid #45475a; border-radius: 4px; background-color: #181825;")

    def _rt_timer_tick(self):
        """Paso continuo de simulación e integración en tiempo real (~33 FPS)."""
        if not getattr(self, '_rt_is_active', False) or self._rt_mem is None or self._rt_neuron is None:
            return

        if hasattr(self, 'dock_hybrid') and self.dock_hybrid.isHidden():
            self.rt_timer.stop()
            return

        mode_series = self.rb_mem_series.isChecked()
        neu = self._rt_neuron
        mem1 = self._rt_mem

        r_s_base = neu.config.r_series
        r_leak_base = neu.config.r_leak
        c_m = neu.config.c_m
        v_rest = neu.config.v_rest
        v_th = neu.config.v_th
        v_reset = neu.config.v_reset
        t_ref = neu.config.t_ref

        # Parámetros del tren de pulsos en tiempo real
        v_stim = self.spin_rt_vstim.value()
        f_stim = self.spin_rt_fstim.value()
        duty_frac = self.spin_rt_duty.value() / 100.0
        is_pressed = getattr(self, '_rt_button_is_pressed', False)

        # 30 sub-pasos por tick (0.001s = 1ms por sub-paso = 0.030s simulados por tick de 30ms)
        N_SUBSTEPS = 30
        dt_sub = 0.001

        for _ in range(N_SUBSTEPS):
            self._rt_t += dt_sub
            v_m = neu.v_membrane

            # Cálculo de voltaje instantáneo del tren de pulsos si el botón está presionado
            if is_pressed and f_stim > 0:
                phase = (self._rt_t * f_stim) % 1.0
                v_in = v_stim if phase < duty_frac else 0.0
            else:
                v_in = 0.0

            if mode_series:
                v_drop_s = max(v_in - v_m, 0.0)
                i_in_val = mem1.step(v_drop_s, dt_sub)
                r_s_curr = mem1.resistance
                r_leak_curr = r_leak_base
                i_leak_val = (v_m - v_rest) / r_leak_curr if r_leak_curr > 0 else 0.0
            else:
                r_s_curr = r_s_base
                r_leak_curr = r_leak_base
                i_in_val = max(v_in - v_m, 0.0) / r_s_curr if r_s_curr > 0 else 0.0
                i_leak_val = (v_m - v_rest) / r_leak_curr if r_leak_curr > 0 else 0.0

            neu.t += dt_sub
            neu.has_spiked = False

            if self._rt_refractory_left > 0.0:
                self._rt_refractory_left -= dt_sub
                dv_refrac = (-i_leak_val / c_m) * dt_sub
                neu.v_membrane += dv_refrac
            else:
                dv = ((i_in_val - i_leak_val) / c_m) * dt_sub
                neu.v_membrane += dv
                if neu.v_membrane >= v_th:
                    neu.has_spiked = True
                    self._rt_spike_times.append(self._rt_t)
                    neu.v_membrane = v_reset
                    if t_ref > 0.0:
                        self._rt_refractory_left = t_ref

            self._rt_t_hist.append(self._rt_t)
            self._rt_v_in_hist.append(v_in)
            self._rt_v_m_hist.append(neu.v_membrane)
            self._rt_r_s_hist.append(r_s_curr)
            self._rt_r_leak_hist.append(r_leak_curr)
            self._rt_i_in_hist.append(i_in_val)

        WINDOW_SIZE = 10.0
        t_min = max(0.0, self._rt_t - WINDOW_SIZE)

        while self._rt_t_hist and self._rt_t_hist[0] < t_min:
            self._rt_t_hist.pop(0)
            self._rt_v_in_hist.pop(0)
            self._rt_v_m_hist.pop(0)
            self._rt_r_s_hist.pop(0)
            self._rt_r_leak_hist.pop(0)
            self._rt_i_in_hist.pop(0)

        self._rt_spike_times = [st for st in self._rt_spike_times if st >= t_min]

        if len(self._rt_t_hist) > 1:
            t_arr = np.array(self._rt_t_hist)
            v_in_arr = np.array(self._rt_v_in_hist)
            v_m_arr = np.array(self._rt_v_m_hist)
            r_s_arr = np.array(self._rt_r_s_hist)
            r_leak_arr = np.array(self._rt_r_leak_hist)
            i_in_arr = np.array(self._rt_i_in_hist)

            self.hybrid_plot_canvas.plot_results(
                t=t_arr,
                signal_in=v_in_arr,
                v_m=v_m_arr,
                r_m_hist=r_leak_arr,
                spike_times=self._rt_spike_times,
                v_th=v_th,
                v_reset=v_reset,
                v_rest=v_rest,
                i_in=i_in_arr,
                r_series_hist=r_s_arr
            )

            r_s_kohm = r_s_arr[-1] / 1e3
            n_spikes = len(self._rt_spike_times)
            state_str = f"TREN DE PULSOS ({f_stim:.0f} Hz)" if is_pressed else "REPOSO (Relajación Volátil)"
            msg = (f"🔴 Tiempo Real Activo [{state_str}] | "
                   f"R_S actual: {r_s_kohm:.1f} kΩ | "
                   f"Spikes ventana (10s): {n_spikes}")
            self.status_bar.showMessage(msg)
