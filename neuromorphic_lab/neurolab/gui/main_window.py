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
import time
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea,
    QStatusBar, QMessageBox, QSplitter, QTabWidget, QTabBar,
    QGroupBox, QFormLayout, QComboBox, QDialog, QPushButton,
    QDoubleSpinBox, QLabel, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, Signal
from neurolab.gui.widgets.config_panel import ConfigPanel
from neurolab.gui.widgets.signal_panel import SignalPanel
from neurolab.gui.widgets.plot_canvas import MplCanvas
from neurolab.gui.widgets.neuron_config_panel import NeuronConfigPanel
from neurolab.gui.widgets.neuron_plot_canvas import NeuronMplCanvas
from neurolab.gui.widgets.hybrid_plot_canvas import HybridMplCanvas
from neurolab.gui.widgets.arrow_spinbox import ArrowDoubleSpinBox
from neurolab.io.profile_manager import ProfileManager
from neurolab.circuits.hybrid import MemristorLIFCircuit, ResistorLIFCircuit


class MemristorConfigWindow(QDialog):
    """Ventana secundaria flotante para monitorizar el Memristor acoplado."""
    param_changed = Signal()  # Se emite para notificar a la Pestaña 3
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Monitor del Memristor (Strukov) - Acoplado")
        self.setMinimumSize(1000, 700)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        splitter_m = QSplitter(Qt.Horizontal)
        controls_container_m = QWidget()
        controls_layout_m = QVBoxLayout(controls_container_m)
        
        self.config_panel = ConfigPanel()
        
        controls_layout_m.addWidget(self.config_panel)
        
        scroll_m = QScrollArea()
        scroll_m.setWidget(controls_container_m)
        scroll_m.setWidgetResizable(True)
        scroll_m.setMinimumWidth(350)
        
        self.plot_canvas = MplCanvas(self)
        
        splitter_m.addWidget(scroll_m)
        splitter_m.addWidget(self.plot_canvas)
        splitter_m.setSizes([350, 930])
        splitter_m.setCollapsible(0, False)
        
        main_layout.addWidget(splitter_m)
        
        # Conexión de señal
        self.config_panel.param_changed.connect(self.param_changed.emit)
        
        self.config_panel._load_strukov_paper_preset()

    def update_graphs(self, t, v_in, v_drop, i_out, x_state, r_hist, g_hist):
        """Actualiza las gráficas con los resultados del loop híbrido principal."""
        try:
            self.plot_canvas.plot_results(t, v_in, v_drop, i_out, x_state, r_hist, g_hist)
        except Exception as e:
            import traceback
            traceback.print_exc()
            pass


class MainWindow(QMainWindow):
    """Ventana Principal de la aplicación Neuromorphic Lab."""

    def __init__(self):
        super().__init__()
        self._profile_manager = ProfileManager()
        self._active_profile_name: str = "Sin Perfil"

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
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # === Pestaña 1: Memristor ===
        tab_memristor = QWidget()
        layout_memristor = QHBoxLayout(tab_memristor)
        layout_memristor.setContentsMargins(0, 0, 0, 0)
        
        splitter_m = QSplitter(Qt.Horizontal)
        controls_container_m = QWidget()
        controls_layout_m = QVBoxLayout(controls_container_m)
        
        self.config_panel = ConfigPanel()
        self.signal_panel = SignalPanel()
        self.config_panel.set_signal_panel(self.signal_panel)
        
        controls_layout_m.addWidget(self.config_panel)
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
        
        self.tabs.addTab(tab_memristor, "🔬 Simulador de Memristor")

        # === Pestaña 2: Neurona LIF ===
        tab_neuron = QWidget()
        layout_neuron = QHBoxLayout(tab_neuron)
        layout_neuron.setContentsMargins(0, 0, 0, 0)
        
        splitter_n = QSplitter(Qt.Horizontal)
        controls_container_n = QWidget()
        controls_layout_n = QVBoxLayout(controls_container_n)
        
        self.neuron_config_panel = NeuronConfigPanel()
        # Reutilizamos SignalPanel en modo corriente
        self.neuron_signal_panel = SignalPanel(mode="current")
        
        # === PRECONFIGURACIÓN AUTOMÁTICA NEURONA LIF ===
        # Configuramos la señal para generar un pulso de 5ms a 2uA
        # Tren de Pulsos (Unipolar) tiene 20% de duty cycle. 20% de 25ms = 5ms. 1/0.025 = 40 Hz.
        idx = self.neuron_signal_panel.combo_waveform.findText("Tren de Pulsos (Unipolar)")
        if idx >= 0:
            self.neuron_signal_panel.combo_waveform.setCurrentIndex(idx)
        self.neuron_signal_panel.spin_v0.setValue(20.0)  # 20 µA > I_th=0.95 µA -> ~1 spike por pulso con C=100nF, R=1MΩ
        self.neuron_signal_panel.spin_f0.setValue(40.0) # 40 Hz -> 25ms periodo -> 5ms pulso
        self.neuron_signal_panel.spin_duration.setValue(0.1) # 100 ms de simulación (alcanza para 4 pulsos)
        
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
        
        self.tabs.addTab(tab_neuron, "🧠 Simulador Neurona LIF")

        # === Pestaña 3: Híbrido Memristor-LIF ===
        tab_hybrid = QWidget()
        layout_hybrid = QHBoxLayout(tab_hybrid)
        layout_hybrid.setContentsMargins(0, 0, 0, 0)
        
        splitter_h = QSplitter(Qt.Horizontal)
        controls_container_h = QWidget()
        controls_layout_h = QVBoxLayout(controls_container_h)
        
        # --- Paneles de Configuración ---
        self.hybrid_neuron_panel = NeuronConfigPanel()
        # En modo "Igualar a Pestaña 2" (clamp) el panel se interpreta como CORRIENTE
        # en µA (mismo significado que la Pestaña 2): 20 µA @ 40 Hz por defecto.
        self.hybrid_signal_panel = SignalPanel(mode="current")  # I_objetivo en µA (clamp) / V (directo)
        self.hybrid_signal_panel.combo_waveform.setCurrentText("Tren de Pulsos (Unipolar)")
        self.hybrid_signal_panel.spin_v0.setValue(20.0)  # 20 µA (== Pestaña 2)
        self.hybrid_signal_panel.spin_f0.setValue(40.0)  # 40 Hz
        self.hybrid_signal_panel.spin_duration.setValue(0.1)  # 100 ms

        # --- Selector de Modo de Acoplamiento ---
        group_coupling = QGroupBox("Modo de Acoplamiento (Sinapsis)")
        layout_coupling = QVBoxLayout()
        self.combo_coupling_mode = QComboBox()
        self.combo_coupling_mode.addItems([
            "Memristor Dinámico (Ventana Secundaria)",
            "Resistencia Fija (R_leak de Neurona)"
        ])
        self.combo_coupling_mode.setStyleSheet("""
            QComboBox { background-color: #313244; color: #a6e3a1; font-weight: bold; padding: 6px; }
        """)
        layout_coupling.addWidget(self.combo_coupling_mode)

        # Resistencia de acople fija (solo aplica al modo "Resistencia Fija")
        self.spin_r_fixed = QDoubleSpinBox()
        self.spin_r_fixed.setRange(0.1, 100000.0)
        self.spin_r_fixed.setValue(10.0)
        self.spin_r_fixed.setDecimals(1)
        self.spin_r_fixed.setSingleStep(1.0)
        self.spin_r_fixed.setSuffix(" kΩ")
        self.spin_r_fixed.setToolTip(
            "Resistencia de acople en modo Resistencia Fija.\n"
            "Con 'Igualar a Pestaña 2' activo: V_fuente = V_m + I_objetivo·R_syn,\n"
            "así que R_syn solo define la escala V↔I (el resultado es idéntico a Pestaña 2).\n"
            "Sin clamp: I_syn = (V_fuente − V_m)/R_syn (R baja ⇒ más ráfagas)."
        )
        layout_coupling.addWidget(self.spin_r_fixed)

        # Clamp de corriente: equivalencia EXACTA con la Pestaña 2 mediante voltaje
        self.chk_current_clamp = QCheckBox("⚡ Igualar a Pestaña 2 (clamp de corriente vía voltaje)")
        self.chk_current_clamp.setChecked(True)
        self.chk_current_clamp.setStyleSheet("font-weight: bold; color: #a6e3a1;")
        self.chk_current_clamp.setToolTip(
            "ACTIVO (por defecto): V_fuente(t) = V_m + I_objetivo(t)·R_syn. La neurona\n"
            "recibe EXACTAMENTE la misma corriente que la Pestaña 2 ⇒ misma V(t) y\n"
            "los mismos spikes. La diferencia es que la señal se entrega como VOLTAJE\n"
            "a través de la resistencia sináptica R_syn (fija y visible en el gráfico ②).\n\n"
            "INACTIVO: V_fuente se aplica directamente (I_syn = (V_fuente − V_m)/R_syn),\n"
            "con la realimentación de V_m que produce ráfagas por pulso."
        )
        layout_coupling.addWidget(self.chk_current_clamp)

        # Nota física
        label_hint_coupling = QLabel(
            "Con el clamp activo, I_syn ≡ I_objetivo(t) ⇒ resultado idéntico a la\n"
            "Pestaña 2 (mismos spikes y V(t)). La diferencia está en cómo se entrega\n"
            "la señal: aquí es VOLTAJE a través de R_syn (gráfico ① V_fuente y ② R_syn).\n"
            "Desactívalo para ver el efecto de la realimentación de V_m (ráfagas)."
        )
        label_hint_coupling.setWordWrap(True)
        label_hint_coupling.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout_coupling.addWidget(label_hint_coupling)
        group_coupling.setLayout(layout_coupling)
        
        # Ventana Flotante del Memristor
        self.memristor_window = MemristorConfigWindow(self)
        
        self.btn_config_memristor = QPushButton("⚙️ Configurar Memristor Acoplado")
        self.btn_config_memristor.setStyleSheet("""
            QPushButton { background-color: #f38ba8; color: #11111b; font-weight: bold; padding: 10px; border-radius: 6px; }
            QPushButton:hover { background-color: #eba0ac; }
            QPushButton:disabled { background-color: #45475a; color: #a6adc8; }
        """)
        self.btn_config_memristor.clicked.connect(self.memristor_window.show)
        
        controls_layout_h.addWidget(group_coupling)
        controls_layout_h.addWidget(self.btn_config_memristor)
        controls_layout_h.addWidget(self.hybrid_neuron_panel)
        controls_layout_h.addWidget(self.hybrid_signal_panel)
        
        scroll_h = QScrollArea()
        scroll_h.setWidget(controls_container_h)
        scroll_h.setWidgetResizable(True)
        scroll_h.setMinimumWidth(350)
        
        self.hybrid_plot_canvas = HybridMplCanvas(self)
        
        splitter_h.addWidget(scroll_h)
        splitter_h.addWidget(self.hybrid_plot_canvas)
        splitter_h.setSizes([460, 820])
        splitter_h.setCollapsible(0, False)
        layout_hybrid.addWidget(splitter_h)
        
        self.tabs.addTab(tab_hybrid, "🧠+🔬 Híbrido Memristor-LIF")

        # === Conexión de Señales ===
        self.signal_panel.run_simulation_requested.connect(self.run_simulation)
        self.config_panel.param_changed.connect(self._on_param_changed)
        self.signal_panel.param_changed.connect(self._on_param_changed)

        self.neuron_signal_panel.run_simulation_requested.connect(self.run_simulation)
        self.neuron_config_panel.param_changed.connect(self._on_param_changed)
        self.neuron_signal_panel.param_changed.connect(self._on_param_changed)

        self.hybrid_signal_panel.run_simulation_requested.connect(self.run_simulation)
        self.hybrid_neuron_panel.param_changed.connect(self._on_param_changed)
        self.hybrid_signal_panel.param_changed.connect(self._on_param_changed)
        self.memristor_window.param_changed.connect(self._on_param_changed)
        self.combo_coupling_mode.currentIndexChanged.connect(self._on_param_changed)
        self.combo_coupling_mode.currentIndexChanged.connect(self._on_coupling_mode_changed)
        
        self.tabs.currentChanged.connect(self.run_simulation)

        # Barra de Estado
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo. Modo en tiempo real activo.")

        # Nota: la simulación inicial la ejecuta el constructor (una sola vez)
        # después de intentar restaurar la última sesión.

    def _on_coupling_mode_changed(self, index: int):
        """Activa/Desactiva el botón y la ventana del memristor según el modo."""
        if index == 0:
            self.btn_config_memristor.setEnabled(True)
            self.spin_r_fixed.setEnabled(False)
            self.chk_current_clamp.setEnabled(False)
            self.status_bar.showMessage("Modo de Acoplamiento: Memristor Dinámico activado.")
        else:
            # En Resistencia Fija no hay memristor: deshabilitar el botón y ocultar su monitor
            self.btn_config_memristor.setEnabled(False)
            self.memristor_window.hide()
            self.spin_r_fixed.setEnabled(True)
            self.chk_current_clamp.setEnabled(True)
            self.status_bar.showMessage("Modo de Acoplamiento: Resistencia Fija activado.")

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
        idx = self.tabs.currentIndex()
        if idx == 0 and self.signal_panel.is_realtime_enabled:
            self.debounce_timer.start(50)
        elif idx == 1 and self.neuron_signal_panel.is_realtime_enabled:
            self.debounce_timer.start(50)
        elif idx == 2 and self.hybrid_signal_panel.is_realtime_enabled:
            self.debounce_timer.start(50)

    def run_simulation(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            self._run_memristor_simulation()
        elif idx == 1:
            self._run_neuron_simulation()
        else:
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

            # 2. Bucle de integración temporal (Euler Explícito)
            for step_idx in range(steps):
                v_inst = v_signal[step_idx]
                i_inst = memristor.step(voltage=v_inst, dt=dt)
                i_out[step_idx] = i_inst
                x_state[step_idx] = memristor.x
                r_hist[step_idx] = memristor.resistance
                g_hist[step_idx] = memristor.conductance

            elapsed = time.time() - start_time

            # 3. Graficar resultados
            self.plot_canvas.plot_results(t, v_signal, v_signal, i_out, x_state, r_hist, g_hist)

            # 4. Actualizar título y barra de estado
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
            
    def _run_neuron_simulation(self):
        """Ejecuta la simulación de la Neurona LIF en tiempo real."""
        try:
            start_time = time.time()
            
            neuron = self.neuron_config_panel.build_neuron()
            t, i_signal_raw, dt = self.neuron_signal_panel.generate_voltage_signal()
            steps = len(t)

            # La señal generada por SignalPanel actúa como corriente en uA
            # La neurona procesa la entrada en Amperios
            i_signal = i_signal_raw * 1e-6
            
            v_m_hist = np.zeros(steps)
            r_m_hist = np.full(steps, neuron.config.r_leak)
            
            for step_idx in range(steps):
                v_m_hist[step_idx] = neuron.v_membrane
                neuron.step(current_input=i_signal[step_idx], dt=dt)
                
            elapsed = time.time() - start_time
            
            self.neuron_plot_canvas.plot_results(
                t, i_signal, v_m_hist, r_m_hist, neuron.spike_times, 
                neuron.config.v_th, neuron.config.v_reset,
                neuron.config.v_rest
            )
            
            n_spikes = len(neuron.spike_times)
            freq = n_spikes / t[-1] if t[-1] > 0 else 0
            
            self._update_window_title("Simulador LIF (Standalone)")
            
            msg = (f"✓ Simulación LIF ejecutada en {elapsed:.3f}s | Pasos: {steps:,} | "
                   f"Spikes: {n_spikes} ({freq:.1f} Hz)")
            self.status_bar.showMessage(msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error de Simulación", f"Error en la simulación LIF:\n{str(e)}")
            self.status_bar.showMessage("❌ Error en la simulación LIF.")

    def _run_hybrid_simulation(self):
        """Ejecuta la simulación acoplada o fija dependiendo del modo seleccionado."""
        try:
            start_time = time.time()

            neuron = self.hybrid_neuron_panel.build_neuron()

            coupling_mode = self.combo_coupling_mode.currentIndex()
            is_memristor_mode = (coupling_mode == 0)

            # El acoplamiento se resuelve con los circuitos reales de neurolab.circuits.hybrid
            # (incluyen rectificador de diodo: bloquean corriente inversa cuando V_in < V_m).
            if is_memristor_mode:
                self._coupled_memristor = self.memristor_window.config_panel.build_memristor()
                circuit = MemristorLIFCircuit(memristor=self._coupled_memristor, neuron=neuron)
            else:
                r_fixed_ohm = self.spin_r_fixed.value() * 1e3
                circuit = ResistorLIFCircuit(r_input=r_fixed_ohm, neuron=neuron)

            t, v_in, dt = self.hybrid_signal_panel.generate_voltage_signal()
            steps = len(t)
            v_source_hist = np.zeros(steps)
            v_m_hist = np.zeros(steps)
            r_syn_hist = np.zeros(steps)
            i_in_hist = np.zeros(steps)

            if is_memristor_mode:
                x_state_hist = np.zeros(steps)
                g_hist = np.zeros(steps)
                v_drop_hist = np.zeros(steps)

            # Clamp de corriente (solo Resistencia Fija): con el checkbox activo,
            # V_fuente(t) = V_m + I_objetivo(t)·R_syn  -> la corriente inyectada es
            # EXACTAMENTE I_objetivo(t), idéntica a la Pestaña 2 (misma V(t), mismos
            # spikes). La diferencia es que la señal se entrega como VOLTAJE.
            use_current_clamp = (not is_memristor_mode) and self.chk_current_clamp.isChecked()

            for step_idx in range(steps):
                if use_current_clamp:
                    # Señal del panel en µA -> I_objetivo en A (igual interpretación
                    # que la Pestaña 2, que multiplica la señal por 1e-6).
                    i_target = v_in[step_idx] * 1e-6
                    v_source = neuron.v_membrane + i_target * r_fixed_ohm
                    res = circuit.step(v_source, dt)
                    v_source_hist[step_idx] = v_source
                else:
                    res = circuit.step(v_in[step_idx], dt)
                    v_source_hist[step_idx] = v_in[step_idx]

                if is_memristor_mode:
                    r_syn_hist[step_idx] = res["r_M"]
                    x_state_hist[step_idx] = res["x"]
                    g_hist[step_idx] = self._coupled_memristor.conductance
                    v_drop_hist[step_idx] = res["v_M"]
                    i_in_hist[step_idx] = res["i_M"]
                else:
                    r_syn_hist[step_idx] = circuit.r_input
                    i_in_hist[step_idx] = res["i_in"]
                v_m_hist[step_idx] = neuron.config.v_th if res["has_spiked"] else res["v_m"]
                    
            elapsed = time.time() - start_time
            
            self.hybrid_plot_canvas.plot_results(
                t=t,
                v_source=v_source_hist,
                i_in=i_in_hist,
                v_m=v_m_hist,
                r_syn_hist=r_syn_hist,
                spike_times=neuron.spike_times,
                v_th=neuron.config.v_th,
                v_reset=neuron.config.v_reset,
                v_rest=neuron.config.v_rest
            )
            
            if is_memristor_mode:
                self.memristor_window.update_graphs(t, v_in, v_drop_hist, i_in_hist, x_state_hist, r_syn_hist, g_hist)
            
            n_spikes = len(neuron.spike_times)
            freq = n_spikes / t[-1] if t[-1] > 0 else 0
            
            title = "Híbrido: Memristor + LIF (Acoplado)" if is_memristor_mode else "LIF Standalone (Resistencia Fija)"
            self._update_window_title(title)
            
            msg = (f"✓ Simulación ejecutada en {elapsed:.3f}s | Pasos: {steps:,} | "
                   f"Spikes: {n_spikes} ({freq:.1f} Hz)")
            if not is_memristor_mode and steps > 0 and np.max(i_in_hist) > 0:
                msg += f" | I_syn max: {np.max(i_in_hist)*1e6:.0f} µA"
            self.status_bar.showMessage(msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error de Simulación", f"Error en la simulación Híbrida:\n{str(e)}")
            self.status_bar.showMessage("❌ Error en la simulación Híbrida.")
