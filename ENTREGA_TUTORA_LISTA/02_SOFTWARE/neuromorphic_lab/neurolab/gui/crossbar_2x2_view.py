"""
neurolab.gui.crossbar_2x2_view
==============================
Vista interactiva del crossbar 2×2 en PySide6 (Qt6).

Layout Matricial Físicamente Completo:
    Filas (PRE, Horizontales):
        S1 (V1) ──[M11]──[M12]──► Fila 1 (PRE)
        S2 (V2) ──[M21]──[M22]──► Fila 2 (PRE)

    Columnas (POST, Totalmente Verticales):
        Columna 1 (x=400): │ M11 │ ──► │ M21 │ ──► │ M_v1 │ ──► LIF_1 ──► Act_1
        Columna 2 (x=700): │ M12 │ ──► │ M22 │ ──► │ M_v2 │ ──► LIF_2 ──► Act_2
"""

import math
import random
from typing import Dict, Any, Optional
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QMessageBox, QInputDialog
)
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QCursor
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer

from neurolab.gui.styles import COLORS, FONTS, CANVAS, get_qcolor
from neurolab.gui.crossbar_elements import (
    SensorElement, MemristorElement, VolatileMemristorElement,
    NeuronElement, ActuatorElement, VisualElement
)
from neurolab.gui.base_view import BaseCrossbarCanvas


class Crossbar2x2Canvas(BaseCrossbarCanvas):
    """Canvas del crossbar 2×2."""

    MIN_WIDTH = 1300
    MIN_HEIGHT = 820

    def __init__(self, parent=None, elements=None, on_element_clicked=None, on_zoom_changed=None):
        super().__init__(parent, elements, on_element_clicked, on_zoom_changed)
        self.spike_flash_1 = 0.0
        self.spike_flash_2 = 0.0

    def _draw_wires(self, painter: QPainter):
        S1 = self.elements.get('S1')
        S2 = self.elements.get('S2')
        M11 = self.elements.get('M11')
        M12 = self.elements.get('M12')
        M21 = self.elements.get('M21')
        M22 = self.elements.get('M22')
        M_v1 = self.elements.get('M_v1')
        M_v2 = self.elements.get('M_v2')
        LIF_1 = self.elements.get('LIF_1')
        LIF_2 = self.elements.get('LIF_2')
        Act_1 = self.elements.get('Act_1')
        Act_2 = self.elements.get('Act_2')

        if not (S1 and S2 and M11 and M12 and M21 and M22 and M_v1 and M_v2 and LIF_1 and LIF_2 and Act_1 and Act_2):
            return

        wire_w = 3

        # ── 1. FILAS (HORIZONTALES / PRE - AZUL) ────────────────────────────
        pen_row = QPen(get_qcolor('sensor'), wire_w)
        painter.setPen(pen_row)

        # Fila 1: S1 -> M11 -> M12 -> Extremo Fila 1
        painter.drawLine(int(S1.x + 45), int(S1.y), int(M11.x - 45), int(M11.y))
        painter.drawLine(int(M11.x + 45), int(M11.y), int(M12.x - 45), int(M12.y))

        pen_dash_row = QPen(get_qcolor('sensor'), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen_dash_row)
        painter.drawLine(int(M12.x + 45), int(M12.y), int(M12.x + 130), int(M12.y))
        # Flecha final Fila 1
        painter.setPen(pen_row)
        painter.drawLine(int(M12.x + 120), int(M12.y - 5), int(M12.x + 135), int(M12.y))
        painter.drawLine(int(M12.x + 120), int(M12.y + 5), int(M12.x + 135), int(M12.y))

        # Fila 2: S2 -> M21 -> M22 -> Extremo Fila 2
        painter.drawLine(int(S2.x + 45), int(S2.y), int(M21.x - 45), int(M21.y))
        painter.drawLine(int(M21.x + 45), int(M21.y), int(M22.x - 45), int(M22.y))

        painter.setPen(pen_dash_row)
        painter.drawLine(int(M22.x + 45), int(M22.y), int(M22.x + 130), int(M22.y))
        painter.setPen(pen_row)
        painter.drawLine(int(M22.x + 120), int(M22.y - 5), int(M22.x + 135), int(M22.y))
        painter.drawLine(int(M22.x + 120), int(M22.y + 5), int(M22.x + 135), int(M22.y))

        # Etiquetas de Filas (PRE)
        painter.setFont(FONTS['subtitle'])
        painter.setPen(get_qcolor('sensor'))
        painter.drawText(QRectF(S1.x - 50, S1.y - 52, 100, 20), Qt.AlignCenter, "FILA 1 (PRE, V₁)")
        painter.drawText(QRectF(S2.x - 50, S2.y - 52, 100, 20), Qt.AlignCenter, "FILA 2 (PRE, V₂)")

        # ── 2. COLUMNAS (TOTALMENTE VERTICALES / POST - VERDE) ──────────────
        pen_col = QPen(get_qcolor('neuron'), wire_w)
        painter.setPen(pen_col)

        # Columna 1 (x=400): Totalmente Recta -> M11 -> M21 -> M_v1 -> LIF_1
        painter.drawLine(int(M11.x), int(M11.y - 60), int(M11.x), int(M11.y - 30))
        painter.drawLine(int(M11.x), int(M11.y + 30), int(M21.x), int(M21.y - 30))
        painter.drawLine(int(M21.x), int(M21.y + 30), int(M_v1.x), int(M_v1.y - 30))
        painter.drawLine(int(M_v1.x), int(M_v1.y + 30), int(LIF_1.x), int(LIF_1.y - 35))

        # Flecha hacia LIF_1
        painter.drawLine(int(LIF_1.x - 5), int(LIF_1.y - 45), int(LIF_1.x), int(LIF_1.y - 35))
        painter.drawLine(int(LIF_1.x + 5), int(LIF_1.y - 45), int(LIF_1.x), int(LIF_1.y - 35))

        # Columna 2 (x=700): Totalmente Recta -> M12 -> M22 -> M_v2 -> LIF_2
        painter.drawLine(int(M12.x), int(M12.y - 60), int(M12.x), int(M12.y - 30))
        painter.drawLine(int(M12.x), int(M12.y + 30), int(M22.x), int(M22.y - 30))
        painter.drawLine(int(M22.x), int(M22.y + 30), int(M_v2.x), int(M_v2.y - 30))
        painter.drawLine(int(M_v2.x), int(M_v2.y + 30), int(LIF_2.x), int(LIF_2.y - 35))

        # Flecha hacia LIF_2
        painter.drawLine(int(LIF_2.x - 5), int(LIF_2.y - 45), int(LIF_2.x), int(LIF_2.y - 35))
        painter.drawLine(int(LIF_2.x + 5), int(LIF_2.y - 45), int(LIF_2.x), int(LIF_2.y - 35))

        # Etiquetas de Columnas (POST) arriba
        painter.setFont(FONTS['subtitle'])
        painter.setPen(get_qcolor('neuron'))
        painter.drawText(QRectF(M11.x - 65, M11.y - 82, 130, 20), Qt.AlignCenter, "COLUMNA 1 (POST, I₁)")
        painter.drawText(QRectF(M12.x - 65, M12.y - 82, 130, 20), Qt.AlignCenter, "COLUMNA 2 (POST, I₂)")

        # ── 3. SALIDAS DE NEURONAS LIF -> ACTUADORES (VERTICALES RECTAS) ──────
        pen_act = QPen(get_qcolor('wire'), wire_w)
        painter.setPen(pen_act)
        painter.drawLine(int(LIF_1.x), int(LIF_1.y + 35), int(Act_1.x), int(Act_1.y - 30))
        painter.drawLine(int(LIF_2.x), int(LIF_2.y + 35), int(Act_2.x), int(Act_2.y - 30))

        # Flechas hacia actuadores
        painter.drawLine(int(Act_1.x - 5), int(Act_1.y - 40), int(Act_1.x), int(Act_1.y - 30))
        painter.drawLine(int(Act_1.x + 5), int(Act_1.y - 40), int(Act_1.x), int(Act_1.y - 30))
        painter.drawLine(int(Act_2.x - 5), int(Act_2.y - 40), int(Act_2.x), int(Act_2.y - 30))
        painter.drawLine(int(Act_2.x + 5), int(Act_2.y - 40), int(Act_2.x), int(Act_2.y - 30))

        painter.setFont(FONTS['small'])
        painter.drawText(int(LIF_1.x + 45), int((LIF_1.y + Act_1.y) / 2), "SPIKES")
        painter.drawText(int(LIF_2.x + 45), int((LIF_2.y + Act_2.y) / 2), "SPIKES")

        # ── 4. ANIMACIÓN DE PARTÍCULAS EN TIEMPO REAL ────────────────────────
        if self.is_animating:
            # 1. Partículas Azules (Voltajes en Filas Horizontales)
            for p_idx in range(5):
                offset = (self.anim_time * 1.5 + p_idx / 5.0) % 1.0
                px = (S1.x + 45) + offset * (M12.x + 130 - (S1.x + 45))
                py = S1.y
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(137, 180, 250, 230)))
                painter.drawEllipse(QPointF(px, py), 4.5, 4.5)

            for p_idx in range(5):
                offset = (self.anim_time * 1.5 + p_idx / 5.0) % 1.0
                px = (S2.x + 45) + offset * (M22.x + 130 - (S2.x + 45))
                py = S2.y
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(137, 180, 250, 230)))
                painter.drawEllipse(QPointF(px, py), 4.5, 4.5)

            # 2. Partículas Verdes (Corrientes en Columnas Totalmente Verticales -> M_v -> LIF)
            # Columna 1
            y_start_c1 = M11.y - 60
            y_end_c1 = LIF_1.y - 35
            for p_idx in range(5):
                offset = (self.anim_time * 1.2 + p_idx / 5.0) % 1.0
                px = M11.x
                py = y_start_c1 + offset * (y_end_c1 - y_start_c1)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(166, 227, 161, 230)))
                painter.drawEllipse(QPointF(px, py), 4.5, 4.5)

            # Columna 2
            y_start_c2 = M12.y - 60
            y_end_c2 = LIF_2.y - 35
            for p_idx in range(5):
                offset = (self.anim_time * 1.2 + p_idx / 5.0) % 1.0
                px = M12.x
                py = y_start_c2 + offset * (y_end_c2 - y_start_c2)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(166, 227, 161, 230)))
                painter.drawEllipse(QPointF(px, py), 4.5, 4.5)

            # 3. Destellos de Spike LIF -> Actuador (Naranja - Vertical)
            if self.spike_flash_1 > 0:
                offset = 1.0 - self.spike_flash_1
                px = LIF_1.x
                py = (LIF_1.y + 35) + offset * (Act_1.y - 30 - (LIF_1.y + 35))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(250, 179, 135, 250)))
                painter.drawEllipse(QPointF(px, py), 6.5, 6.5)

            if self.spike_flash_2 > 0:
                offset = 1.0 - self.spike_flash_2
                px = LIF_2.x
                py = (LIF_2.y + 35) + offset * (Act_2.y - 30 - (LIF_2.y + 35))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(250, 179, 135, 250)))
                painter.drawEllipse(QPointF(px, py), 6.5, 6.5)

    def _draw_extra(self, painter: QPainter):
        G11 = float(self.elements['M11'].params.get('G_11', 69.4e-6))
        G12 = float(self.elements['M12'].params.get('G_11', 69.4e-6))
        G21 = float(self.elements['M21'].params.get('G_11', 69.4e-6))
        G22 = float(self.elements['M22'].params.get('G_22', self.elements['M22'].params.get('G_11', 69.4e-6)))

        V1 = float(self.elements['S1'].params.get('V_out', 0.8))
        V2 = float(self.elements['S2'].params.get('V_out', 0.4))

        I1 = G11 * V1 + G21 * V2
        I2 = G12 * V1 + G22 * V2

        painter.setFont(FONTS['mono'])
        painter.setPen(get_qcolor('success'))
        painter.drawText(
            QRectF(self.width() - 550, self.height() - 60, 520, 25),
            Qt.AlignRight | Qt.AlignVCenter,
            f"I₁ = G₁₁·V₁ + G₂₁·V₂ = {I1*1e6:.2f} μA"
        )
        painter.drawText(
            QRectF(self.width() - 550, self.height() - 32, 520, 25),
            Qt.AlignRight | Qt.AlignVCenter,
            f"I₂ = G₁₂·V₁ + G₂₂·V₂ = {I2*1e6:.2f} μA"
        )


class Crossbar2x2View(QWidget):
    """
    Vista interactiva del crossbar 2×2 (PySide6 / Qt6).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.elements: Dict[str, VisualElement] = {}
        self.hovered_element = None

        # Temporizador de animación (~30 FPS)
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(30)
        self.anim_timer.timeout.connect(self._anim_tick)
        self._sim_time: float = 0.0

        self._build_elements()
        self._build_ui()

    def _build_elements(self):
        # Sensores (Filas PRE)
        self.elements['S1'] = SensorElement('S1', x=100, y=180)
        self.elements['S1'].params['name'] = 'Sensor Frontal'
        self.elements['S1'].params['V_out'] = 0.8

        self.elements['S2'] = SensorElement('S2', x=100, y=360)
        self.elements['S2'].params['name'] = 'Sensor Trasero'
        self.elements['S2'].params['V_out'] = 0.4

        # Matriz de memristores 2x2 No Volátiles (Strukov)
        self.elements['M11'] = MemristorElement('M11', row=0, col=0, x=400, y=180)
        self.elements['M12'] = MemristorElement('M12', row=0, col=1, x=700, y=180)
        self.elements['M21'] = MemristorElement('M21', row=1, col=0, x=400, y=360)
        self.elements['M22'] = MemristorElement('M22', row=1, col=1, x=700, y=360)

        # Memristores Volátiles (HfO₂) en cada Columna POST
        self.elements['M_v1'] = VolatileMemristorElement('M_v1', x=400, y=480)
        self.elements['M_v2'] = VolatileMemristorElement('M_v2', x=700, y=480)

        # Neuronas LIF (Alineadas en vertical debajo de los Memristores Volátiles)
        self.elements['LIF_1'] = NeuronElement('LIF_1', x=400, y=600)
        self.elements['LIF_2'] = NeuronElement('LIF_2', x=700, y=600)

        # Actuadores (Alineados en vertical debajo de las neuronas)
        self.elements['Act_1'] = ActuatorElement('Act_1', x=400, y=715)
        self.elements['Act_1'].params['action'] = 'girar_izquierda'
        self.elements['Act_2'] = ActuatorElement('Act_2', x=700, y=715)
        self.elements['Act_2'].params['action'] = 'girar_derecha'

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Header
        header = QHBoxLayout()
        lbl_title = QLabel("🔶 Crossbar 2×2")
        lbl_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text']};")
        lbl_subtitle = QLabel("— 4 memristores Strukov · 2 memristores volátiles HfO₂ · 2 sensores · 2 neuronas LIF")
        lbl_subtitle.setStyleSheet(f"font-size: 11px; color: {COLORS['text_dim']};")

        header.addWidget(lbl_title)
        header.addWidget(lbl_subtitle)
        header.addStretch()
        main_layout.addLayout(header)

        # Toolbar Controles
        toolbar = QHBoxLayout()

        btn_sim = QPushButton("▶ Simular Paso")
        btn_sim.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: #11111b; font-weight: bold; padding: 6px 14px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #a6e3a1; }}
        """)
        btn_sim.clicked.connect(self._on_simulate)

        self.btn_realtime = QPushButton("⚡ Tiempo Real (OFF)")
        self.btn_realtime.setStyleSheet(f"""
            QPushButton {{
                background-color: #313244; color: #a6adc8; font-weight: bold;
                padding: 6px 16px; border-radius: 4px; border: 1px solid #45475a;
            }}
            QPushButton:hover {{ background-color: #45475a; }}
        """)
        self.btn_realtime.clicked.connect(self.toggle_realtime)

        btn_reset = QPushButton("⏹ Reset")
        btn_reset.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: #11111b; font-weight: bold; padding: 6px 14px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #f9e2af; }}
        """)
        btn_reset.clicked.connect(self._on_reset)

        btn_matrix = QPushButton("📊 Ver Matriz G")
        btn_matrix.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['text']}; border: 1px solid #45475a;
                padding: 6px 14px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #45475a; }}
        """)
        btn_matrix.clicked.connect(self._on_show_matrix)

        btn_voltages = QPushButton("🔧 Configurar Voltajes")
        btn_voltages.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['text']}; border: 1px solid #45475a;
                padding: 6px 14px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #45475a; }}
        """)
        btn_voltages.clicked.connect(self._on_config_voltages)

        btn_reset_mat = QPushButton("🎯 Reset Matriz")
        btn_reset_mat.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['text']}; border: 1px solid #45475a;
                padding: 6px 14px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #45475a; }}
        """)
        btn_reset_mat.clicked.connect(self._on_reset_matrix)

        toolbar.addWidget(btn_sim)
        toolbar.addWidget(self.btn_realtime)
        toolbar.addWidget(btn_reset)
        toolbar.addWidget(btn_matrix)
        toolbar.addWidget(btn_voltages)
        toolbar.addWidget(btn_reset_mat)
        toolbar.addStretch()

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

        toolbar.addWidget(self.btn_zoom_in)
        toolbar.addWidget(self.btn_zoom_out)
        toolbar.addWidget(self.btn_zoom_reset)
        toolbar.addWidget(self.lbl_zoom)

        main_layout.addLayout(toolbar)

        # Canvas
        self.canvas = Crossbar2x2Canvas(
            self,
            elements=self.elements,
            on_element_clicked=self._open_config,
            on_zoom_changed=self._update_zoom_label
        )
        self.btn_zoom_in.clicked.connect(self.canvas.zoom_in)
        self.btn_zoom_out.clicked.connect(self.canvas.zoom_out)
        self.btn_zoom_reset.clicked.connect(self.canvas.reset_view)

        main_layout.addWidget(self.canvas, stretch=1)

        # Status Bar Inferior
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet(f"background-color: {COLORS['bg_dark']}; border-radius: 4px; padding: 4px;")
        status_layout = QHBoxLayout(self.status_frame)
        status_layout.setContentsMargins(10, 4, 10, 4)

        self.status_label = QLabel("Crossbar 2×2 listo. Click para configurar. 💡 Panning: Arrastra con el mouse | Zoom: Rueda del mouse.")
        self.status_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text_dim']};")
        status_layout.addWidget(self.status_label)

        main_layout.addWidget(self.status_frame)

    def _update_zoom_label(self, zoom: float):
        self.lbl_zoom.setText(f"{int(zoom * 100)}%")

    # ------------------------------------------------------------------
    # CÁLCULOS
    # ------------------------------------------------------------------

    def _compute_currents(self):
        G11 = float(self.elements['M11'].params.get('G_11', 69.4e-6))
        G12 = float(self.elements['M12'].params.get('G_11', 69.4e-6))
        G21 = float(self.elements['M21'].params.get('G_11', 69.4e-6))
        G22 = float(self.elements['M22'].params.get('G_22', self.elements['M22'].params.get('G_11', 69.4e-6)))

        V1 = float(self.elements['S1'].params.get('V_out', 0.8))
        V2 = float(self.elements['S2'].params.get('V_out', 0.4))

        I1 = G11 * V1 + G21 * V2
        I2 = G12 * V1 + G22 * V2
        return I1, I2

    # ------------------------------------------------------------------
    # ACCIONES DE SIMULACIÓN Y CONFIGURACIÓN
    # ------------------------------------------------------------------

    def toggle_realtime(self):
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
            self.status_label.setText("⏸ Simulación pausada.")
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
            self.status_label.setText("⚡ Simulación 2×2 en tiempo real activa...")

    def _anim_tick(self):
        self._sim_time += 0.005
        self.canvas.anim_time += 0.03
        self._on_simulate()

    def _on_simulate(self):
        I1, I2 = self._compute_currents()

        LIF_1 = self.elements['LIF_1']
        LIF_2 = self.elements['LIF_2']

        C_m1 = float(LIF_1.params.get('C_m', 100e-9))
        C_m2 = float(LIF_2.params.get('C_m', 100e-9))

        V_th1 = float(LIF_1.params.get('V_th', 2.0))
        V_th2 = float(LIF_2.params.get('V_th', 2.0))

        LIF_1.params['V_m'] = min(float(LIF_1.params.get('V_m', 0.0)) + I1 * 1e-3 / C_m1, 2.5)
        LIF_2.params['V_m'] = min(float(LIF_2.params.get('V_m', 0.0)) + I2 * 1e-3 / C_m2, 2.5)

        if LIF_1.params['V_m'] >= V_th1:
            LIF_1.params['V_m'] = float(LIF_1.params.get('V_reset', 0.0))
            LIF_1.params['spike_count'] = int(LIF_1.params.get('spike_count', 0)) + 1
            self.canvas.spike_flash_1 = 1.0

        if LIF_2.params['V_m'] >= V_th2:
            LIF_2.params['V_m'] = float(LIF_2.params.get('V_reset', 0.0))
            LIF_2.params['spike_count'] = int(LIF_2.params.get('spike_count', 0)) + 1
            self.canvas.spike_flash_2 = 1.0

        if self.canvas.spike_flash_1 > 0:
            self.canvas.spike_flash_1 = max(0.0, self.canvas.spike_flash_1 - 0.1)
        if self.canvas.spike_flash_2 > 0:
            self.canvas.spike_flash_2 = max(0.0, self.canvas.spike_flash_2 - 0.1)

        self.status_label.setText(
            f"▶ I₁ = {I1*1e6:.2f} μA | I₂ = {I2*1e6:.2f} μA | "
            f"Spikes: LIF_1={LIF_1.params['spike_count']}, LIF_2={LIF_2.params['spike_count']}"
        )
        self.canvas.update()

    def _on_reset(self):
        if self.canvas.is_animating:
            self.toggle_realtime()

        for elem in self.elements.values():
            elem.selected = False
            elem.hover = False
            if isinstance(elem, (MemristorElement, VolatileMemristorElement)):
                elem.params['seed'] = random.randint(10000, 999999)
                if hasattr(elem, 'on_params_changed'):
                    elem.on_params_changed()

        for key in ['LIF_1', 'LIF_2']:
            self.elements[key].params['V_m'] = 0.0
            self.elements[key].params['spike_count'] = 0

        self.status_label.setText("⏹ Reset completo")
        self.canvas.update()

    def _on_reset_matrix(self):
        for key in ['M11', 'M12', 'M21', 'M22']:
            mem = self.elements[key]
            mem.params['x0'] = 0.10
            mem.params['seed'] = random.randint(10000, 999999)
            if hasattr(mem, 'on_params_changed'):
                mem.on_params_changed()

        self.status_label.setText("🎯 Matriz 2×2 reiniciada")
        self.canvas.update()

    def _on_show_matrix(self):
        G11 = float(self.elements['M11'].params.get('G_11', 69.4e-6)) * 1e6
        G12 = float(self.elements['M12'].params.get('G_11', 69.4e-6)) * 1e6
        G21 = float(self.elements['M21'].params.get('G_11', 69.4e-6)) * 1e6
        G22 = float(self.elements['M22'].params.get('G_22', self.elements['M22'].params.get('G_11', 69.4e-6))) * 1e6

        V1 = float(self.elements['S1'].params.get('V_out', 0.8))
        V2 = float(self.elements['S2'].params.get('V_out', 0.4))

        I1 = G11 * V1 + G21 * V2
        I2 = G12 * V1 + G22 * V2

        msg = (
            f"Matriz de conductancias G (μS):\n\n"
            f"         Col 1      Col 2\n"
            f"Fila 1   {G11:8.2f}   {G12:8.2f}\n"
            f"Fila 2   {G21:8.2f}   {G22:8.2f}\n\n"
            f"Voltajes Aplicados:\n"
            f"V₁ (S1) = {V1:.2f} V\n"
            f"V₂ (S2) = {V2:.2f} V\n\n"
            f"Operación I = G^T · V:\n"
            f"I₁ = {G11:.2f}·{V1:.2f} + {G21:.2f}·{V2:.2f} = {I1:.2f} μA\n"
            f"I₂ = {G12:.2f}·{V1:.2f} + {G22:.2f}·{V2:.2f} = {I2:.2f} μA"
        )
        QMessageBox.information(self, "Matriz G del Crossbar 2×2", msg)

    def _on_config_voltages(self):
        val1, ok1 = QInputDialog.getDouble(
            self, "Voltaje V₁", "Voltaje Sensor S1 (V):",
            value=float(self.elements['S1'].params.get('V_out', 0.8)),
            minValue=0.0, maxValue=5.0, decimals=2
        )
        if ok1:
            self.elements['S1'].params['V_out'] = val1

        val2, ok2 = QInputDialog.getDouble(
            self, "Voltaje V₂", "Voltaje Sensor S2 (V):",
            value=float(self.elements['S2'].params.get('V_out', 0.4)),
            minValue=0.0, maxValue=5.0, decimals=2
        )
        if ok2:
            self.elements['S2'].params['V_out'] = val2

        self.status_label.setText(f"✓ Voltajes actualizados: V₁ = {self.elements['S1'].params['V_out']:.2f} V, V₂ = {self.elements['S2'].params['V_out']:.2f} V")
        self.canvas.update()

    def _open_config(self, element: VisualElement):
        for elem in self.elements.values():
            elem.selected = False
        element.selected = True
        self.canvas.update()

        self.status_label.setText(f"Configurando: {element.element_id}")
        dlg = ConfigDialog(self, element, on_save=self._on_config_saved)
        dlg.exec()

    def _on_config_saved(self, element: VisualElement):
        self.status_label.setText(f"✓ Configuración actualizada: {element.element_id}")
        if hasattr(element, 'on_params_changed'):
            element.on_params_changed()
        self.canvas.update()
