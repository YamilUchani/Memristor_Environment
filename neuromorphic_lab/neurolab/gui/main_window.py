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
from neurolab.io.profile_manager import ProfileManager
from neurolab.io.validation_loader import ValidationDataLoader
from neurolab.core.validation_metrics import compute_all_metrics
from neurolab.circuits.hybrid import MemristorLIFCircuit, ResistorLIFCircuit
from neurolab.core.lif_validation import compute_lif_validation_metrics


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
        # Sub-pestañas para configurar múltiples Memristores (Memristor 1, Memristor 2, Memristor 3)
        self.memristor_subtabs = QTabWidget()
        self.config_panel_1 = ConfigPanel()
        self.config_panel_2 = ConfigPanel()
        self.config_panel_3 = ConfigPanel()

        self.config_panel_1.txt_device_name.setText("Memristor No Volátil (Strukov)")
        self.config_panel_1.combo_material.setCurrentText("TiO₂ (Dióxido de Titanio - Strukov 2008)")
        self.config_panel_1.spin_r_on.setValue(100.0)
        self.config_panel_1.combo_ron_unit.setCurrentText("Ω")
        self.config_panel_1.spin_r_off.setValue(16.0)
        self.config_panel_1.combo_roff_unit.setCurrentText("kΩ")
        self.config_panel_1.spin_x0.setValue(0.10)
        self.config_panel_1.spin_D_nm.setValue(10.0)
        self.config_panel_1.spin_mu_v.setValue(1e-14)
        self.config_panel_1.combo_realism_mode.setCurrentIndex(1)  # Modo 2 — Extendido (Ventana Biolek)
        self.config_panel_1.combo_window_type.setCurrentText("Biolek")
        self.config_panel_1.spin_biolek_p.setValue(5)
        self.config_panel_1.spin_seed.setValue(42)

        self.config_panel_2.txt_device_name.setText("Memristor Serie Alternativo")
        self.config_panel_2.combo_material.setCurrentText("HfO₂ (Óxido de Hafnio - CMOS LIF 2025)")
        self.config_panel_2.spin_r_on.setValue(10.0)
        self.config_panel_2.combo_ron_unit.setCurrentText("kΩ")
        self.config_panel_2.spin_r_off.setValue(1.0)
        self.config_panel_2.combo_roff_unit.setCurrentText("MΩ")
        self.config_panel_2.spin_x0.setValue(0.05)
        self.config_panel_2.spin_D_nm.setValue(10.0)
        self.config_panel_2.spin_mu_v.setValue(1e-16)
        self.config_panel_2.combo_realism_mode.setCurrentIndex(3)  # Modo 4 — Memristor Volátil
        self.config_panel_2.combo_window_type.setCurrentText("Biolek")
        self.config_panel_2.spin_biolek_p.setValue(3)
        self.config_panel_2.spin_seed.setValue(42)
        self.config_panel_2.chk_volatile.setChecked(True)
        self.config_panel_2.spin_tau_relax.setValue(0.5)
        if hasattr(self.config_panel_2, "spin_x_eq"):
            self.config_panel_2.spin_x_eq.setValue(0.05)

        cfg_vol_path = self._profile_manager.configs_dir / "memristor_volatile.json"
        if cfg_vol_path.exists():
            try:
                data_vol = self._profile_manager.load(str(cfg_vol_path))
                self.config_panel_2.from_dict(data_vol)
            except Exception:
                pass

        # Subpestaña 3: Memristor Híbrido (lif_config_mem.json)
        self.config_panel_3.txt_device_name.setText("Memristor Serie (Híbrido)")
        self.config_panel_3.combo_material.setCurrentText("HfO₂ (Óxido de Hafnio - CMOS LIF 2025)")
        self.config_panel_3.spin_r_on.setValue(10.0)
        self.config_panel_3.combo_ron_unit.setCurrentText("kΩ")
        self.config_panel_3.spin_r_off.setValue(1.0)
        self.config_panel_3.combo_roff_unit.setCurrentText("MΩ")
        self.config_panel_3.spin_x0.setValue(0.10)
        self.config_panel_3.spin_D_nm.setValue(10.0)
        self.config_panel_3.spin_mu_v.setValue(1e-16)
        self.config_panel_3.combo_realism_mode.setCurrentIndex(1)  # Modo 2 — Extendido (Ventana Biolek)
        self.config_panel_3.combo_window_type.setCurrentText("Biolek")
        self.config_panel_3.spin_biolek_p.setValue(3)
        self.config_panel_3.spin_seed.setValue(42)

        cfg_3_path = self._profile_manager.configs_dir / "lif_config_mem.json"
        if cfg_3_path.exists():
            try:
                data_3 = self._profile_manager.load(str(cfg_3_path))
                self.config_panel_3.from_dict(data_3)
            except Exception:
                pass

        self.signal_panel = SignalPanel()
        # Default signal for Memristor Tab (Sinusoidal by default)
        idx = self.signal_panel.combo_waveform.findText("Sinusoidal")
        if idx >= 0:
            self.signal_panel.combo_waveform.setCurrentIndex(idx)
        self.signal_panel.spin_v0.setValue(1.0)
        self.signal_panel.spin_f0.setValue(0.5)
        self.signal_panel.spin_duration.setValue(8.0)
        self.signal_panel.spin_dt_ms.setValue(0.1)

        self.config_panel_1.set_signal_panel(self.signal_panel)
        self.config_panel_2.set_signal_panel(self.signal_panel)
        self.config_panel_3.set_signal_panel(self.signal_panel)

        self.memristor_subtabs.addTab(self.config_panel_1, "🔬 Memristor 1 (No Volátil)")
        self.memristor_subtabs.addTab(self.config_panel_2, "🔬 Memristor 2 (Serie - Alternativo)")
        self.memristor_subtabs.addTab(self.config_panel_3, "🔬 Memristor 3 (Híbrido)")
        self.memristor_subtabs.setCurrentIndex(1)  # Sub-Pestaña 2 (Volátil) activa por defecto

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

        cfg_lif_path = self._profile_manager.configs_dir / "lif_config.json"
        if cfg_lif_path.exists():
            try:
                import json
                with open(cfg_lif_path, "r", encoding="utf-8") as f:
                    data_lif = json.load(f)
                self.neuron_config_panel.from_dict(data_lif)
            except Exception:
                pass

        # SignalPanel en modo fuente de voltaje (V_IN)
        self.neuron_signal_panel = SignalPanel(mode="voltage")
        self.neuron_config_panel.set_signal_panel(self.neuron_signal_panel)
        
        # === PRECONFIGURACIÓN AUTOMÁTICA NEURONA LIF ===
        idx = self.neuron_signal_panel.combo_waveform.findText("Tren de Pulsos (Unipolar)")
        if idx >= 0:
            self.neuron_signal_panel.combo_waveform.setCurrentIndex(idx)
        self.neuron_signal_panel.spin_v0.setValue(3.0)
        self.neuron_signal_panel.spin_f0.setValue(100.0)
        self.neuron_signal_panel.spin_duration.setValue(2.0)
        self.neuron_signal_panel.spin_dt_ms.setValue(0.001)
        
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
        self.combo_mem_rs.addItems(["Memristor 1 (Sub-Pestaña 1)", "Memristor 2 (Sub-Pestaña 2)", "Memristor 3 (Sub-Pestaña 3)"])

        self.combo_mem_rs.setCurrentIndex(2)  # Default: Memristor 3 (Serie - Híbrido)

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

        # === Pestaña 4: Documentación Formal ===
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
        
        # === Pestaña 4: Documentación Formal (Dock) ===
        self.dock_docs = QDockWidget("📖 Fundamentos Físicos", self)
        self.dock_docs.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_docs.setWidget(tab_docs)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_docs)

        # Apilar las pestañas (tabify) por defecto
        self.tabifyDockWidget(self.dock_memristor, self.dock_neuron)
        self.tabifyDockWidget(self.dock_neuron, self.dock_hybrid)
        self.tabifyDockWidget(self.dock_hybrid, self.dock_docs)
        self.dock_memristor.raise_()

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
        self.memristor_subtabs.currentChanged.connect(self.run_simulation)
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
        self.dock_hybrid.visibilityChanged.connect(lambda visible: self.run_simulation() if visible else None)

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
            self.config_panel.from_dict(data)
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

    # ── Eventos Qt ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Intercepta el cierre para autoguardar la sesión actual."""
        self._save_last_session()
        event.accept()

    # ── Lógica de Simulación ─────────────────────────────────────────────────

    def _on_param_changed(self):
        """Disparado ante cualquier cambio de parámetro si la opción en tiempo real está activa."""
        # Se ejecuta un temporizador global que procesará todas las vistas visibles
        if self.signal_panel.is_realtime_enabled or self.neuron_signal_panel.is_realtime_enabled or self.hybrid_signal_panel.is_realtime_enabled:
            self.debounce_timer.start(180)

    def run_simulation(self):
        """Ejecuta la simulación para todos los docks que estén visibles actualmente."""
        if hasattr(self, 'dock_memristor') and not self.dock_memristor.visibleRegion().isEmpty():
            self._run_memristor_simulation()
        if hasattr(self, 'dock_neuron') and not self.dock_neuron.visibleRegion().isEmpty():
            self._run_neuron_simulation()
        if hasattr(self, 'dock_hybrid') and not self.dock_hybrid.visibleRegion().isEmpty():
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
            if self.config_panel.is_csv_validation_enabled:
                val_data = self._validation_loader.load_all()

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
                    val_data=val_data
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
            
            self.neuron_plot_canvas.plot_results(
                t, v_signal, v_m_hist, r_leak_hist, neuron.spike_times, 
                neuron.config.v_th, neuron.config.v_reset,
                neuron.config.v_rest, i_in=i_in_hist, r_series_hist=r_series_hist
            )
            
            # ── Validación Matemática Analítica LIF (Sección E) ──
            lif_val = compute_lif_validation_metrics(t, v_m_hist, v_signal, neuron.config, is_voltage_input=True)
            
            n_spikes = len(neuron.spike_times)
            freq = n_spikes / t[-1] if t[-1] > 0 else 0
            
            self._update_window_title("Simulador LIF (Fuente Voltaje V_IN + R_S)")
            
            msg = (f"✓ Simulación LIF ejecutada en {elapsed:.3f}s | Pasos: {steps:,} | "
                   f"Spikes: {n_spikes} ({freq:.1f} Hz) | "
                   f"Validación Teórica: MAE={lif_val['mae']*1e3:.2f} mV, RMSE={lif_val['rmse']*1e3:.2f} mV, R²={lif_val['r2']:.4f}")
            self.status_bar.showMessage(msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error de Simulación", f"Error en la simulación LIF:\n{str(e)}")
            self.status_bar.showMessage("❌ Error en la simulación LIF.")

    def _get_panel_by_index(self, index: int) -> ConfigPanel:
        """Devuelve el ConfigPanel correspondiente al índice de sub-pestaña."""
        if index == 1:
            return self.config_panel_2
        elif index == 2:
            return self.config_panel_3
        else:
            return self.config_panel_1

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
                    r_s_curr = mem1.resistance
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
