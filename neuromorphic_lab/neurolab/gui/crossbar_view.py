"""
neurolab.gui.crossbar_view
==========================
Panel principal de la pestaña Crossbar (PySide6 / Qt6).
LAYOUT MATRICIAL 2D con ANIMACIÓN EN TIEMPO REAL:
- Fila horizontal (PRE)
- Columna vertical (POST)
- Flujo dinámico de partículas y destellos de Spikes
"""

from typing import Dict, Any, Optional
import math
import random
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QMessageBox, QTabWidget
)
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QCursor
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer

from neurolab.gui.styles import COLORS, FONTS, CANVAS, get_qcolor
from neurolab.gui.crossbar_elements import (
    SensorElement, MemristorElement, VolatileMemristorElement,
    NeuronElement, ActuatorElement, VisualElement
)
from neurolab.gui.config_dialogs import ConfigDialog
from neurolab.gui.base_view import BaseCrossbarCanvas
from neurolab.gui.crossbar_2x2_view import Crossbar2x2View
from neurolab.gui.crossbar_4x4_view import Crossbar4x4View
from neurolab.gui.widgets.validation_tests_panel import ValidationTestsPanel


class CrossbarCanvas(BaseCrossbarCanvas):
    """Widget de dibujo interactivo para el canvas del Crossbar 1x1."""

    MIN_WIDTH = 900
    MIN_HEIGHT = 600

    def __init__(self, parent=None, elements=None, on_element_clicked=None, on_zoom_changed=None):
        super().__init__(parent, elements, on_element_clicked, on_zoom_changed)
        self.spike_flash = 0.0

    def _draw_wires(self, painter: QPainter):
        sensor = self.elements.get('sensor_1')
        mem = self.elements.get('M11')
        mv = self.elements.get('M_v')
        neuron = self.elements.get('neuron_1')
        act = self.elements.get('actuator_1')

        if not (sensor and mem and neuron and act):
            return

        # --- FILA 1 (PRE) — Horizontal ---
        pen_row = QPen(get_qcolor('sensor'), CANVAS['wire_width'])
        painter.setPen(pen_row)
        # Tramo Sensor -> M11
        x_start_row = sensor.x + 45
        x_end_row = mem.x - 45
        painter.drawLine(int(x_start_row), int(sensor.y), int(x_end_row), int(mem.y))

        # Tramo Continuación de Fila
        pen_dash = QPen(get_qcolor('sensor'), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen_dash)
        painter.drawLine(int(mem.x + 45), int(mem.y), int(mem.x + 280), int(mem.y))
        # Flecha al final de la fila
        painter.setPen(pen_row)
        painter.drawLine(int(mem.x + 270), int(mem.y - 5), int(mem.x + 285), int(mem.y))
        painter.drawLine(int(mem.x + 270), int(mem.y + 5), int(mem.x + 285), int(mem.y))

        # Etiqueta FILA 1 (PRE) arriba
        painter.setFont(FONTS['subtitle'])
        painter.setPen(get_qcolor('sensor'))
        painter.drawText(int((sensor.x + mem.x) / 2 - 45), int(sensor.y - 20), "FILA 1 (PRE)")

        # --- COLUMNA 1 (POST) — Vertical ---
        pen_col = QPen(get_qcolor('neuron'), CANVAS['wire_width'])
        painter.setPen(pen_col)
        # Segmento M11 -> M_v
        painter.drawLine(int(mem.x), int(mem.y + 35), int(mv.x), int(mv.y - 30))

        # Segmento M_v -> Neurona
        painter.drawLine(int(mv.x), int(mv.y + 30), int(neuron.x), int(neuron.y - 45))
        # Flecha hacia neurona
        painter.drawLine(int(neuron.x - 5), int(neuron.y - 55), int(neuron.x), int(neuron.y - 45))
        painter.drawLine(int(neuron.x + 5), int(neuron.y - 55), int(neuron.x), int(neuron.y - 45))

        # Etiqueta COLUMNA 1 (POST) a la derecha
        painter.setFont(FONTS['subtitle'])
        painter.setPen(get_qcolor('neuron'))
        painter.drawText(int(mem.x + 75), int(mv.y - 45), "COLUMNA 1 (POST)")

        # --- Neurona -> Actuador ---
        pen_out = QPen(get_qcolor('wire'), CANVAS['wire_width'])
        painter.setPen(pen_out)
        painter.drawLine(int(neuron.x), int(neuron.y + 45), int(act.x), int(act.y - 30))
        # Flechas hacia actuador
        painter.drawLine(int(act.x - 5), int(act.y - 40), int(act.x), int(act.y - 30))
        painter.drawLine(int(act.x + 5), int(act.y - 40), int(act.x), int(act.y - 30))

        painter.setFont(FONTS['small'])
        painter.setPen(get_qcolor('neuron'))
        painter.drawText(int(neuron.x + 55), int((neuron.y + act.y) / 2), "SPIKES")

        # --- ANIMACIÓN DE PARTÍCULAS LUMINOSAS SI ESTÁ ACTIVO ---
        if self.is_animating:
            # 1. Partículas en la Fila 1 (Sensor -> M11)
            n_particles_row = 4
            for p_idx in range(n_particles_row):
                offset = (self.anim_time * 1.5 + p_idx / n_particles_row) % 1.0
                px = x_start_row + offset * (x_end_row - x_start_row)
                py = sensor.y
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(137, 180, 250, 220)))  # Azul brillante
                painter.drawEllipse(QPointF(px, py), 5, 5)
                painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
                painter.drawEllipse(QPointF(px, py), 2.5, 2.5)

            # 2. Partículas en la Columna 1 (M11 -> M_v -> Neurona)
            y_start_col = mem.y + 35
            y_end_col = neuron.y - 45
            n_particles_col = 4
            for p_idx in range(n_particles_col):
                offset = (self.anim_time * 1.2 + p_idx / n_particles_col) % 1.0
                px = mem.x
                py = y_start_col + offset * (y_end_col - y_start_col)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(166, 227, 161, 220)))  # Verde brillante
                painter.drawEllipse(QPointF(px, py), 5, 5)
                painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
                painter.drawEllipse(QPointF(px, py), 2.5, 2.5)

            # 3. Partículas de Spike (Neurona -> Actuador)
            if self.spike_flash > 0:
                y_start_out = neuron.y + 45
                y_end_out = act.y - 30
                offset = 1.0 - self.spike_flash
                px = neuron.x
                py = y_start_out + offset * (y_end_out - y_start_out)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(250, 179, 135, 240)))  # Naranja brillante
                painter.drawEllipse(QPointF(px, py), 7, 7)
                painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
                painter.drawEllipse(QPointF(px, py), 3.5, 3.5)

    def _draw_extra(self, painter: QPainter):
        painter.setFont(FONTS['subtitle'])
        painter.setPen(get_qcolor('text_dim'))
        mode_str = " — ⚡ ANIMACIÓN EN TIEMPO REAL ACTIVA" if self.is_animating else ""
        painter.drawText(QRectF(0, 15, self.width(), 30), Qt.AlignCenter,
                         f"CROSSBAR 1×1 — MATRIZ DE MEMRISTORES NO VOLÁTILES{mode_str}")

        mem = self.elements.get('M11')
        sensor = self.elements.get('sensor_1')
        if mem and sensor:
            G_uS = float(mem.params.get('G_11', mem.params.get('G', 69.4e-6))) * 1e6
            V_in = float(sensor.params.get('V_out', 1.0))
            I_uA = G_uS * V_in

            painter.setFont(FONTS['mono'])
            painter.setPen(get_qcolor('success'))
            painter.drawText(
                QRectF(self.width() - 480, self.height() - 35, 460, 25),
                Qt.AlignRight | Qt.AlignVCenter,
                f"I_out = G·V = {G_uS:.2f} μS · {V_in:.2f} V = {I_uA:.2f} μA"
            )


class CrossbarView(QWidget):
    """
    Panel principal de la vista Crossbar 1×1 (PySide6 / Qt6).
    Layout Matricial 2D con Animación en Tiempo Real continuada.
    """

    def __init__(self, parent=None, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.elements: Dict[str, VisualElement] = {}

        # Temporizador de animación en tiempo real (~30 FPS)
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(30)
        self.anim_timer.timeout.connect(self._anim_tick)
        self._sim_time: float = 0.0

        self._build_elements()
        self._build_ui()

    def _build_elements(self):
        self.elements['sensor_1'] = SensorElement('sensor_1', x=100, y=130)
        self.elements['M11'] = MemristorElement('M11', row=0, col=0, x=400, y=130)
        self.elements['M_v'] = VolatileMemristorElement('M_v', x=400, y=300)
        self.elements['neuron_1'] = NeuronElement('neuron_1', x=400, y=440)
        self.elements['actuator_1'] = ActuatorElement('actuator_1', x=400, y=545)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        self.sub_tabs = QTabWidget()
        main_layout.addWidget(self.sub_tabs)

        # ── TAB 1: Matriz & Esquema 2D ─────────────────────────────────────
        tab1 = QWidget()
        lay1 = QVBoxLayout(tab1)
        lay1.setContentsMargins(10, 10, 10, 10)
        lay1.setSpacing(8)

        # Header
        header = QHBoxLayout()
        lbl_title = QLabel("🔷 Crossbar 1×1")
        lbl_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text']};")
        lbl_subtitle = QLabel("— Click en cualquier elemento para configurarlo")
        lbl_subtitle.setStyleSheet(f"font-size: 11px; color: {COLORS['text_dim']};")

        header.addWidget(lbl_title)
        header.addWidget(lbl_subtitle)
        header.addStretch()
        lay1.addLayout(header)

        # Controles
        controls = QHBoxLayout()
        btn_sim = QPushButton("▶ Pasó Único")
        btn_sim.setToolTip("Ejecuta un único paso de simulación")
        btn_sim.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['text']}; font-weight: bold; padding: 6px 14px; border: 1px solid #45475a; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #45475a; }}
        """)
        btn_sim.clicked.connect(self._on_simulate)

        self.btn_realtime = QPushButton("⚡ Tiempo Real (ON)")
        self.btn_realtime.setToolTip("Inicia/Pausa la simulación continua y animación visual a 30 FPS")
        self.btn_realtime.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: #11111b; font-weight: bold; padding: 6px 16px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #94e2d5; }}
        """)
        self.btn_realtime.clicked.connect(self.toggle_realtime_animation)

        btn_reset = QPushButton("⏹ Reset")
        btn_reset.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: #11111b; font-weight: bold; padding: 6px 16px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #f9e2af; }}
        """)
        btn_reset.clicked.connect(self._on_reset)

        btn_export = QPushButton("📊 Exportar")
        btn_export.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['text']}; border: 1px solid #45475a;
                padding: 6px 14px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #45475a; }}
        """)
        btn_export.clicked.connect(self._on_export)

        controls.addWidget(self.btn_realtime)
        controls.addWidget(btn_sim)
        controls.addWidget(btn_reset)
        controls.addWidget(btn_export)
        controls.addStretch()

        # Controles de Zoom y Desplazamiento
        btn_style_zoom = f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['text']}; border: 1px solid #45475a;
                padding: 6px 14px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #45475a; color: #89b4fa; }}
        """

        self.btn_zoom_in = QPushButton("🔍 +")
        self.btn_zoom_out = QPushButton("🔍 -")
        self.btn_zoom_reset = QPushButton("📌 Reset Vista")
        self.lbl_zoom = QLabel("100%")

        self.btn_zoom_in.setToolTip("Acercar Zoom (o rueda del mouse)")
        self.btn_zoom_out.setToolTip("Alejar Zoom (o rueda del mouse)")
        self.btn_zoom_reset.setToolTip("Restablecer Posición y Zoom (100%)")

        self.btn_zoom_in.setStyleSheet(btn_style_zoom)
        self.btn_zoom_out.setStyleSheet(btn_style_zoom)
        self.btn_zoom_reset.setStyleSheet(btn_style_zoom)
        self.lbl_zoom.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 12px; padding: 0 4px;")

        controls.addWidget(self.btn_zoom_in)
        controls.addWidget(self.btn_zoom_out)
        controls.addWidget(self.btn_zoom_reset)
        controls.addWidget(self.lbl_zoom)

        lay1.addLayout(controls)

        # Canvas
        self.canvas = CrossbarCanvas(
            self,
            elements=self.elements,
            on_element_clicked=self._open_config,
            on_zoom_changed=self._update_zoom_label
        )
        self.btn_zoom_in.clicked.connect(self.canvas.zoom_in)
        self.btn_zoom_out.clicked.connect(self.canvas.zoom_out)
        self.btn_zoom_reset.clicked.connect(self.canvas.reset_view)

        lay1.addWidget(self.canvas, stretch=1)

        # Status Bar Inferior
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet(f"background-color: {COLORS['bg_dark']}; border-radius: 4px; padding: 4px;")
        status_layout = QHBoxLayout(self.status_frame)
        status_layout.setContentsMargins(10, 4, 10, 4)

        self.status_label = QLabel("Crossbar 1×1 cargado. 💡 Panning: Arrastra con el mouse | Zoom: Rueda del mouse.")
        self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text_dim']};")
        status_layout.addWidget(self.status_label)

        lay1.addWidget(self.status_frame)

        self.sub_tabs.addTab(tab1, "🔳 Matriz Esquema 1×1")

        # ── TAB 2: Crossbar 2×2 (Esquema Interactivo) ───────────────────────
        self.view_2x2 = Crossbar2x2View(self)
        self.sub_tabs.addTab(self.view_2x2, "🔶 Matriz Esquema 2×2")

        # ── TAB 3: Crossbar 4×4 (Esquema Interactivo) ───────────────────────
        self.view_4x4 = Crossbar4x4View(self)
        self.sub_tabs.addTab(self.view_4x4, "🔷 Matriz Esquema 4×4")

        # ── TAB 4: Pruebas de Validación (Tiempo Real) ──────────────────────
        self.tests_panel = ValidationTestsPanel(self, crossbar_view=self)
        self.sub_tabs.addTab(self.tests_panel, "🧪 Pruebas de Validación (Tiempo Real)")
        self.sub_tabs.currentChanged.connect(self._on_subtab_changed)

    def _update_zoom_label(self, zoom: float):
        self.lbl_zoom.setText(f"{int(zoom * 100)}%")

    def _on_subtab_changed(self, index: int):
        if index == 3 and hasattr(self, 'tests_panel') and self.tests_panel:
            self.tests_panel.run_current_test()


    def toggle_realtime_animation(self):
        """Alterna el estado de reproducción de la animación en tiempo real."""
        if self.canvas.is_animating:
            self.anim_timer.stop()
            self.canvas.is_animating = False
            self.btn_realtime.setText("⚡ Tiempo Real (OFF)")
            self.btn_realtime.setStyleSheet(f"""
                QPushButton {{
                    background-color: #313244; color: #a6adc8; font-weight: bold;
                    padding: 6px 16px; border-radius: 4px; border: 1px solid #45475a;
                }}
                QPushButton:hover {{ background-color: #45475a; }}
            """)
            self.status_label.setText("⏸ Simulación en tiempo real pausada.")
        else:
            self.canvas.is_animating = True
            self.anim_timer.start()
            self.btn_realtime.setText("⚡ Tiempo Real (ACTIVO)")
            self.btn_realtime.setStyleSheet(f"""
                QPushButton {{
                    background-color: #a6e3a1; color: #11111b; font-weight: bold;
                    padding: 6px 16px; border-radius: 4px;
                }}
                QPushButton:hover {{ background-color: #94e2d5; }}
            """)
            self.status_label.setText("⚡ Simulación y animación en tiempo real ejecutándose...")

    def _anim_tick(self):
        """Tick ejecutado a ~33 FPS para actualizar la física y re-dibujar la animación."""
        dt = 0.005  # 5 ms por tick físico
        self._sim_time += dt
        self.canvas.anim_time += 0.03

        # Decaimiento del destello de spike
        if self.canvas.spike_flash > 0:
            self.canvas.spike_flash = max(0.0, self.canvas.spike_flash - 0.08)

        # 1. Obtener parámetros del circuito
        sensor = self.elements['sensor_1']
        mem = self.elements['M11']
        mv = self.elements['M_v']
        neuron = self.elements['neuron_1']
        actuator = self.elements['actuator_1']

        V_in = float(sensor.params.get('V_out', 1.0))
        G_11 = float(mem.params.get('G_11', mem.params.get('G', 69.4e-6)))
        I_out = G_11 * V_in

        # 2. Integrar neurona LIF
        C_m = float(neuron.params.get('C_m', 100e-9))
        R_leak = float(neuron.params.get('R_leak', 1e6))
        R_series = float(neuron.params.get('R_series', 100e3))
        V_th = float(neuron.params.get('V_th', 2.5))
        V_reset = float(neuron.params.get('V_reset', 0.0))

        V_m = float(neuron.params.get('V_m', 0.0))

        # Dinámica dVm/dt = I_in/C_m - Vm/(R_leak*C_m)
        dVm = ((I_out - V_m / R_leak) / C_m) * dt
        V_m_next = V_m + dVm

        if V_m_next >= V_th:
            # SPIKE!
            V_m_next = V_reset
            neuron.params['spike_count'] = int(neuron.params.get('spike_count', 0)) + 1
            neuron.params['last_isi'] = float(dt)
            self.canvas.spike_flash = 1.0
            actuator.params['action'] = 'ACTIVADO (Spike!)'
        else:
            if self.canvas.spike_flash <= 0.1:
                actuator.params['action'] = 'retroceder'

        neuron.params['V_m'] = V_m_next

        spikes = neuron.params.get('spike_count', 0)
        self.status_label.setText(
            f"⚡ [TIEMPO REAL] t = {self._sim_time:.2f} s | V_in = {V_in:.2f} V | I_out = {I_out*1e6:.1f} μA | Vm = {V_m_next:.2f} V | Spikes = {spikes}"
        )

        self.canvas.update()

    def _open_config(self, element: VisualElement):
        self.status_label.setText(f"Configurando: {element.element_id}")
        dialog = ConfigDialog(self, element, on_save=self._on_config_saved)
        dialog.exec()

    def _on_config_saved(self, element: VisualElement):
        self.status_label.setText(f"✓ Configuración actualizada: {element.element_id}")

        if hasattr(element, 'on_params_changed'):
            element.on_params_changed()

        self.canvas.update()

        # Actualización instantánea en tiempo real de la pestaña de Pruebas de Validación
        if hasattr(self, 'tests_panel') and self.tests_panel:
            self.tests_panel.run_current_test()


    def _recalculate_memristor(self, element: VisualElement):
        if hasattr(element, 'on_params_changed'):
            element.on_params_changed()

    def _on_simulate(self):
        try:
            from neurolab.crossbar import CrossbarIdeal, CrossbarConfig

            mem_params = self.elements['M11'].params
            R_on = float(mem_params.get('RON', mem_params.get('R_on', 100.0)))
            R_off = float(mem_params.get('ROFF', mem_params.get('R_off', 16000.0)))
            x0 = float(mem_params.get('x0', mem_params.get('x', 0.10)))
            D = float(mem_params.get('D', 10e-9))
            mu_v = float(mem_params.get('mu_v', 1e-14))
            is_vol = bool(mem_params.get('is_volatile', mem_params.get('enable_volatile', False)))
            tau_rel = float(mem_params.get('volatile_tau_relax', mem_params.get('tau_relax', 0.5)))

            config = CrossbarConfig(
                n_rows=1, n_cols=1,
                R_on=R_on,
                R_off=R_off,
                x0=x0,
                D=D,
                mu_v=mu_v,
                enable_volatile=is_vol,
                tau_relax=tau_rel
            )
            cb = CrossbarIdeal(config)

            V_in = float(self.elements['sensor_1'].params['V_out'])
            cb.apply_voltages([V_in])
            I_out = cb.read_currents()[0]

            neuron = self.elements['neuron_1']
            C_m = float(neuron.params['C_m'])
            V_m_increment = (I_out * 1e-3) / C_m
            neuron.params['V_m'] = float(neuron.params.get('V_m', 0.0) + V_m_increment)

            self.status_label.setText(
                f"▶ V_in = {V_in:.2f} V | I_out = {I_out*1e6:.2f} μA | ΔVm = {V_m_increment*1e3:.2f} mV"
            )
            self.canvas.update()
        except Exception as e:
            self.status_label.setText(f"❌ Error al simular: {e}")

    def _on_reset(self):
        if self.canvas.is_animating:
            self.toggle_realtime_animation()

        self._sim_time = 0.0
        self.canvas.anim_time = 0.0
        self.canvas.spike_flash = 0.0

        for elem in self.elements.values():
            elem.selected = False
            elem.hover = False
            if isinstance(elem, (MemristorElement, VolatileMemristorElement)):
                elem.params['seed'] = random.randint(10000, 999999)
                if hasattr(elem, 'on_params_changed'):
                    elem.on_params_changed()

        neuron = self.elements['neuron_1']
        neuron.params['V_m'] = 0.0
        neuron.params['spike_count'] = 0
        neuron.params['last_isi'] = 0.0

        actuator = self.elements['actuator_1']
        actuator.params['action'] = 'retroceder'

        self.status_label.setText("⏹ Estado reiniciado")
        self.canvas.update()

    def _on_export(self):
        self.status_label.setText("📊 Exportación no implementada aún (Fase 6)")
