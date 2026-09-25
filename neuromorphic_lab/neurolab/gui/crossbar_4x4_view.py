"""
neurolab.gui.crossbar_4x4_view
==============================
Vista interactiva del crossbar 4×4 en PySide6 (Qt6).

Layout Matricial Físicamente Completo:
    Filas (PRE, Horizontales):
        S1 (V1) ──[M11]──[M12]──[M13]──[M14]──► Fila 1 (PRE)
        S2 (V2) ──[M21]──[M22]──[M23]──[M24]──► Fila 2 (PRE)
        S3 (V3) ──[M31]──[M32]──[M33]──[M34]──► Fila 3 (PRE)
        S4 (V4) ──[M41]──[M42]──[M43]──[M44]──► Fila 4 (PRE)

    Columnas (POST, Verticales):
        Columna 1: M11 → M21 → M31 → M41 ──► LIF_1 ──► Act_1
        Columna 2: M12 → M22 → M32 → M42 ──► LIF_2 ──► Act_2
        Columna 3: M13 → M23 → M33 → M43 ──► LIF_3 ──► Act_3
        Columna 4: M14 → M24 → M34 → M44 ──► LIF_4 ──► Act_4
"""

import math
import random
from typing import Dict, Any, Optional
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QMessageBox, QInputDialog, QDialog, QGroupBox, QFormLayout,
    QComboBox, QDoubleSpinBox, QSpinBox
)
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QCursor
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer

from neurolab.gui.styles import COLORS, FONTS, CANVAS, get_qcolor
from neurolab.gui.crossbar_elements import (
    SensorElement, MemristorElement, VolatileMemristorElement,
    NeuronElement, ActuatorElement, VisualElement,
    get_G_matrix, set_G_matrix, compute_currents, matrix_to_heatmap_string
)
from neurolab.gui.config_dialogs import ConfigDialog
from neurolab.gui.base_view import BaseCrossbarCanvas
from neurolab.crossbar.addressing import (
    AddressDecoder, RowDriver, ColumnDriver,
    ReadConfig, WriteConfig
)
from neurolab.crossbar.plasticity import (
    STDPRule, RSTDPRule,
    STDPConfig, RSTDPConfig,
    Trace,
)
from neurolab.gui.read_write_config_dialog import ReadWriteConfigDialog



class BatchVolatileDialog(QDialog):
    """Diálogo modal para configurar todos los memristores volátiles (M_v1..M_v4) simultáneamente."""

    def __init__(self, parent=None, current_elements=None):
        super().__init__(parent)
        self.elements = current_elements or {}
        self.setWindowTitle("⚡ Configurar Todos los Memristores Volátiles (M_v1 .. M_v4)")
        self.resize(480, 360)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog {{ background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', sans-serif; }}
            QGroupBox {{ border: 1px solid #45475a; border-radius: 6px; margin-top: 10px; font-weight: bold; color: #89b4fa; padding-top: 10px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
            QLabel {{ color: #a6adc8; }}
            QComboBox, QDoubleSpinBox {{ background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px; }}
            QPushButton {{ background-color: #89b4fa; color: #11111b; font-weight: bold; padding: 6px 14px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: #b4befe; }}
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        grp = QGroupBox("Parámetros Globales Memristores Volátiles (M_v1 .. M_v4)")
        form = QFormLayout(grp)

        self.combo_model = QComboBox()
        self.combo_model.addItem("HfO₂ Volátil Neurona (Wang et al. 2025)", "HfO2_volatile")
        self.combo_model.addItem("Ag:SiO₂ Difusivo (Yang et al. 2017)", "diffusive_Ag_SiO2")
        self.combo_model.addItem("NbOₓ Threshold Switch (Pickett et al. 2013)", "NbOx_threshold")

        self.spin_tau = QDoubleSpinBox()
        self.spin_tau.setRange(0.001, 10.0)
        self.spin_tau.setSingleStep(0.05)
        self.spin_tau.setValue(0.30)
        self.spin_tau.setSuffix(" s")

        self.spin_ron = QDoubleSpinBox()
        self.spin_ron.setRange(0.1, 500.0)
        self.spin_ron.setValue(10.0)
        self.spin_ron.setSuffix(" kΩ")

        self.spin_roff = QDoubleSpinBox()
        self.spin_roff.setRange(1.0, 50000.0)
        self.spin_roff.setValue(500.0)
        self.spin_roff.setSuffix(" kΩ")

        self.spin_x0 = QDoubleSpinBox()
        self.spin_x0.setRange(0.001, 1.0)
        self.spin_x0.setSingleStep(0.01)
        self.spin_x0.setValue(0.05)

        form.addRow("Modelo Volátil:", self.combo_model)
        form.addRow("Tiempo Relajación (τ_relax):", self.spin_tau)
        form.addRow("Resistencia R_ON:", self.spin_ron)
        form.addRow("Resistencia R_OFF:", self.spin_roff)
        form.addRow("Estado Inicial x_0:", self.spin_x0)

        layout.addWidget(grp)

        btn_box = QHBoxLayout()
        btn_apply = QPushButton("✓ Aplicar a los 4 Memristores Volátiles")
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet("background-color: #313244; color: #cdd6f4;")

        btn_apply.clicked.connect(self._apply)
        btn_cancel.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_apply)
        layout.addLayout(btn_box)

    def _apply(self):
        model_key = self.combo_model.currentData()
        tau = self.spin_tau.value()
        ron = self.spin_ron.value() * 1e3
        roff = self.spin_roff.value() * 1e3
        x0 = self.spin_x0.value()

        for j in range(4):
            key = f'M_v{j+1}'
            if key in self.elements:
                mv = self.elements[key]
                mv.params['model'] = model_key
                mv.params['tau_relax'] = tau
                mv.params['volatile_tau_relax'] = tau
                mv.params['RON'] = ron
                mv.params['ROFF'] = roff
                mv.params['x0'] = x0
                if hasattr(mv, 'on_params_changed'):
                    mv.on_params_changed()

        self.accept()


class BatchGridDialog(QDialog):
    """Diálogo modal para configurar memristores de la matriz 4x4 por orden/subgrilla (1x1, 2x2, 1x3, 4x4, etc.)."""

    def __init__(self, parent=None, current_elements=None):
        super().__init__(parent)
        self.elements = current_elements or {}
        self.setWindowTitle("🔲 Programación por Orden / Sub-Matriz (R×C)")
        self.resize(500, 440)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog {{ background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', sans-serif; }}
            QGroupBox {{ border: 1px solid #45475a; border-radius: 6px; margin-top: 10px; font-weight: bold; color: #89b4fa; padding-top: 10px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
            QLabel {{ color: #a6adc8; }}
            QComboBox, QDoubleSpinBox, QSpinBox {{ background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px; }}
            QPushButton {{ background-color: #a6e3a1; color: #11111b; font-weight: bold; padding: 6px 14px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: #94e2d5; }}
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        grp_order = QGroupBox("1. Orden / Sub-Matriz Afectada")
        form_order = QFormLayout(grp_order)

        self.combo_preset = QComboBox()
        self.combo_preset.addItem("1×1 — Solo M11 (1 memristor)", (1, 1))
        self.combo_preset.addItem("2×2 — Submatriz M11..M22 (4 memristores)", (2, 2))
        self.combo_preset.addItem("1×3 — Submatriz M11..M13 (3 memristores)", (1, 3))
        self.combo_preset.addItem("3×3 — Submatriz M11..M33 (9 memristores)", (3, 3))
        self.combo_preset.addItem("4×4 — Toda la Matriz Crossbar (16 memristores)", (4, 4))
        self.combo_preset.addItem("Personalizado...", None)
        self.combo_preset.setCurrentIndex(1)  # Default 2x2

        self.spin_r = QSpinBox()
        self.spin_r.setRange(1, 4)
        self.spin_r.setValue(2)

        self.spin_c = QSpinBox()
        self.spin_c.setRange(1, 4)
        self.spin_c.setValue(2)

        self.combo_preset.currentIndexChanged.connect(self._on_preset_change)

        form_order.addRow("Preset por Orden:", self.combo_preset)
        form_order.addRow("Filas (R):", self.spin_r)
        form_order.addRow("Columnas (C):", self.spin_c)

        layout.addWidget(grp_order)

        grp_model = QGroupBox("2. Modelo No Volátil y Conductancia")
        form_model = QFormLayout(grp_model)

        self.combo_model = QComboBox()
        self.combo_model.addItem("Strukov (TiO₂-x, Nature 2008)", "strukov")
        self.combo_model.addItem("Prezioso 2015 (Al₂O₃/TiO₂-x, Nature 2015)", "prezioso")
        self.combo_model.addItem("Jo 2010 (Ag/a-Si, Nano Lett 2010)", "jo2010")

        self.spin_g = QDoubleSpinBox()
        self.spin_g.setRange(1.0, 2000.0)
        self.spin_g.setSingleStep(10.0)
        self.spin_g.setValue(200.0)
        self.spin_g.setSuffix(" μS")

        self.spin_x0 = QDoubleSpinBox()
        self.spin_x0.setRange(0.01, 1.0)
        self.spin_x0.setSingleStep(0.05)
        self.spin_x0.setValue(0.10)

        form_model.addRow("Modelo Memristores:", self.combo_model)
        form_model.addRow("Conductancia Objetivo G:", self.spin_g)
        form_model.addRow("Estado Inicial x_0:", self.spin_x0)

        layout.addWidget(grp_model)

        btn_box = QHBoxLayout()
        btn_apply = QPushButton("✓ Programar Sub-Matriz Seleccionada")
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet("background-color: #313244; color: #cdd6f4;")

        btn_apply.clicked.connect(self._apply)
        btn_cancel.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_apply)
        layout.addLayout(btn_box)

    def _on_preset_change(self):
        val = self.combo_preset.currentData()
        if val is not None:
            r, c = val
            self.spin_r.setValue(r)
            self.spin_c.setValue(c)

    def _apply(self):
        n_r = self.spin_r.value()
        n_c = self.spin_c.value()
        model_key = self.combo_model.currentData()
        G_uS = self.spin_g.value()
        x0_val = self.spin_x0.value()

        for i in range(n_r):
            for j in range(n_c):
                key = f'M{i+1}{j+1}'
                if key in self.elements:
                    mem = self.elements[key]
                    mem.params['model'] = model_key
                    RON = float(mem.params.get('RON', 100.0))
                    ROFF = float(mem.params.get('ROFF', 16000.0))
                    R_target = 1.0 / (G_uS * 1e-6) if G_uS > 0 else ROFF
                    if ROFF != RON:
                        x_calc = float(np.clip((R_target - ROFF) / (RON - ROFF), 0.01, 1.0))
                        mem.params['x0'] = x_calc
                        mem.params['x'] = x_calc
                    else:
                        mem.params['x0'] = x0_val
                        mem.params['x'] = x0_val

                    if hasattr(mem, 'on_params_changed'):
                        mem.on_params_changed()

        self.accept()


class Crossbar4x4Canvas(BaseCrossbarCanvas):
    """Canvas del crossbar 4×4."""

    MIN_WIDTH = 1400
    MIN_HEIGHT = 750

    def __init__(self, parent=None, elements=None, on_element_clicked=None, on_zoom_changed=None):
        super().__init__(parent, elements, on_element_clicked, on_zoom_changed)
        self.spike_flashes = [0.0, 0.0, 0.0, 0.0]

    def _draw_wires(self, painter: QPainter):
        wire_w = 3
        pen_row = QPen(get_qcolor('sensor'), wire_w)
        pen_col = QPen(get_qcolor('neuron'), wire_w)
        pen_act = QPen(get_qcolor('wire'), wire_w)

        x_memristores = [300, 480, 660, 840]
        y_filas = [150, 310, 470, 630]

        # --- FILAS (HORIZONTALES / PRE - AZUL) ---
        painter.setPen(pen_row)
        for i in range(4):
            S = self.elements.get(f'S{i+1}')
            M1 = self.elements.get(f'M{i+1}1')
            M4 = self.elements.get(f'M{i+1}4')
            if S and M1 and M4:
                painter.drawLine(int(S.x + 45), int(S.y), int(M1.x - 45), int(M1.y))
                for j in range(3):
                    Ma = self.elements.get(f'M{i+1}{j+1}')
                    Mb = self.elements.get(f'M{i+1}{j+2}')
                    if Ma and Mb:
                        painter.drawLine(int(Ma.x + 45), int(Ma.y), int(Mb.x - 45), int(Mb.y))
                pen_dash = QPen(get_qcolor('sensor'), 2, Qt.DashLine)
                painter.setPen(pen_dash)
                painter.drawLine(int(M4.x + 45), int(M4.y), int(M4.x + 130), int(M4.y))
                painter.setPen(pen_row)
                painter.drawLine(int(M4.x + 120), int(M4.y - 5), int(M4.x + 135), int(M4.y))
                painter.drawLine(int(M4.x + 120), int(M4.y + 5), int(M4.x + 135), int(M4.y))

        # Etiquetas de fila
        painter.setFont(FONTS['subtitle'])
        view = self.view
        v_rows = getattr(view, 'V_rows', np.full(4, 0.2))
        v_cols = getattr(view, 'V_cols', np.zeros(4))
        is_prog = (view and hasattr(view, 'mode') and view.mode == 'program_v2')

        for i in range(4):
            S = self.elements.get(f'S{i+1}')
            if S:
                v_r_val = float(v_rows[i])
                if is_prog:
                    tg_r, _ = getattr(view, 'target_cell', (0, 0))
                    col_str = get_qcolor('selected') if i == tg_r else get_qcolor('sensor')
                else:
                    col_str = QColor('#f38ba8') if abs(v_r_val) > 0.49 else get_qcolor('sensor')
                painter.setPen(col_str)
                painter.drawText(QRectF(S.x - 70, S.y - 52, 140, 20), Qt.AlignCenter,
                                 f"FILA {i+1} (V={v_r_val:+.2f}V)")

        # --- COLUMNAS (VERTICALES / POST - VERDE) ---
        winner_col = getattr(view, 'winner_j', None)

        for j in range(4):
            M1 = self.elements.get(f'M1{j+1}')
            M4 = self.elements.get(f'M4{j+1}')
            Mv = self.elements.get(f'M_v{j+1}')
            LIF = self.elements.get(f'LIF_{j+1}')
            if M1 and M4 and Mv and LIF:
                if is_prog and j == getattr(view, 'target_cell', (0, 0))[1]:
                    painter.setPen(QPen(QColor('#f38ba8'), wire_w))
                elif winner_col is not None:
                    if j == winner_col:
                        glow_pen = QPen(QColor(166, 227, 161, 80), 9)
                        painter.setPen(glow_pen)
                        painter.drawLine(int(M1.x), int(M1.y - 60), int(M1.x), int(LIF.y - 35))
                        painter.setPen(QPen(QColor('#a6e3a1'), 5))
                    else:
                        painter.setPen(QPen(QColor('#313244'), 1.5))
                else:
                    painter.setPen(pen_col)

                painter.drawLine(int(M1.x), int(M1.y - 60), int(M1.x), int(M1.y - 30))
                for i in range(3):
                    Ma = self.elements.get(f'M{i+1}{j+1}')
                    Mb = self.elements.get(f'M{i+2}{j+1}')
                    if Ma and Mb:
                        painter.drawLine(int(Ma.x), int(Ma.y + 30), int(Mb.x), int(Mb.y - 30))
                painter.drawLine(int(M4.x), int(M4.y + 30), int(Mv.x), int(Mv.y - 25))

                if winner_col is not None:
                    col_wire_pen = QPen(QColor('#a6e3a1'), 5) if j == winner_col else QPen(QColor('#313244'), 1.5)
                    painter.setPen(col_wire_pen)
                else:
                    painter.setPen(pen_col)

                painter.drawLine(int(Mv.x), int(Mv.y + 25), int(LIF.x), int(LIF.y - 35))
                painter.drawLine(int(LIF.x - 5), int(LIF.y - 45), int(LIF.x), int(LIF.y - 35))
                painter.drawLine(int(LIF.x + 5), int(LIF.y - 45), int(LIF.x), int(LIF.y - 35))

        # Etiquetas de columna
        painter.setFont(FONTS['subtitle'])
        for j in range(4):
            M1 = self.elements.get(f'M1{j+1}')
            if M1:
                v_c_val = float(v_cols[j])
                if is_prog:
                    _, tg_c = getattr(view, 'target_cell', (0, 0))
                    col_str = QColor('#f38ba8') if j == tg_c else get_qcolor('neuron')
                    lbl_col = f"COL {j+1} (V={v_c_val:+.2f}V)"
                elif winner_col is not None:
                    if j == winner_col:
                        col_str = QColor('#f9e2af')
                        lbl_col = f"🏆 COL {j+1} (GANADORA)"
                    else:
                        col_str = QColor('#6c7086')
                        lbl_col = f"COL {j+1} (INHIBIDA)"
                else:
                    col_str = get_qcolor('neuron')
                    lbl_col = f"COL {j+1} (V={v_c_val:+.2f}V)"

                painter.setPen(col_str)
                painter.drawText(QRectF(M1.x - 85, 88, 170, 20), Qt.AlignCenter, lbl_col)

        # --- LIF → Actuador ---
        for j in range(4):
            LIF = self.elements.get(f'LIF_{j+1}')
            Act = self.elements.get(f'Act_{j+1}')
            if LIF and Act:
                if winner_col is not None:
                    act_pen = QPen(QColor('#a6e3a1'), 4) if j == winner_col else QPen(QColor('#313244'), 1.5)
                else:
                    act_pen = pen_act
                painter.setPen(act_pen)
                painter.drawLine(int(LIF.x), int(LIF.y + 35), int(Act.x), int(Act.y - 30))
                painter.drawLine(int(Act.x - 5), int(Act.y - 40), int(Act.x), int(Act.y - 30))
                painter.drawLine(int(Act.x + 5), int(Act.y - 40), int(Act.x), int(Act.y - 30))

        # --- Animación en tiempo real ---
        if self.is_animating:
            for i in range(4):
                S = self.elements.get(f'S{i+1}')
                M4 = self.elements.get(f'M{i+1}4')
                if S and M4:
                    for p_idx in range(4):
                        offset = (self.anim_time * 1.5 + p_idx / 4.0) % 1.0
                        px = (S.x + 45) + offset * (M4.x + 130 - (S.x + 45))
                        py = S.y
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(QColor(137, 180, 250, 230)))
                        painter.drawEllipse(QPointF(px, py), 4.0, 4.0)
            for j in range(4):
                M1 = self.elements.get(f'M1{j+1}')
                LIF = self.elements.get(f'LIF_{j+1}')
                if M1 and LIF:
                    y_start = M1.y - 60
                    y_end = LIF.y - 35
                    for p_idx in range(4):
                        offset = (self.anim_time * 1.2 + p_idx / 4.0) % 1.0
                        px = M1.x
                        py = y_start + offset * (y_end - y_start)
                        painter.setPen(Qt.NoPen)
                        particle_col = QColor(166, 227, 161, 240) if (winner_col is None or j == winner_col) else QColor(108, 112, 134, 100)
                        painter.setBrush(QBrush(particle_col))
                        painter.drawEllipse(QPointF(px, py), 4.0, 4.0)

    def _draw_extra(self, painter: QPainter):
        """Decoders + drivers + título + ecuación matricial limpiamente distribuidos."""
        view = self.view
        if not view or not hasattr(view, 'row_decoder'):
            return

        painter.setFont(FONTS['small'])

        # 1. Decoders y Drivers Fila (Esquina superior izquierda)
        painter.setPen(QPen(QColor('#f9e2af'), 1.5))
        painter.setBrush(QBrush(QColor('#181825')))
        painter.drawRoundedRect(QRectF(15, 10, 210, 24), 5, 5)
        painter.setPen(QColor('#f9e2af'))
        addr_row_str = view.row_decoder.address_to_binary(view.row_decoder.last_address) \
            if view.mode == 'program_v2' else "OFF"
        painter.drawText(QRectF(15, 10, 210, 24), Qt.AlignCenter,
                         f"🎯 Decoder ROW [{addr_row_str}] → Addr {view.row_decoder.last_address}")

        painter.setPen(QPen(QColor('#f38ba8'), 1.5))
        painter.setBrush(QBrush(QColor('#181825')))
        painter.drawRoundedRect(QRectF(15, 38, 210, 24), 5, 5)
        painter.setPen(QColor('#f38ba8'))
        painter.drawText(QRectF(15, 38, 210, 24), Qt.AlignCenter,
                         f"⚡ Driver ROW (WL): {view.row_driver.V_active:+.2f}V")

        # 2. Decoders y Drivers Columna (Esquina superior derecha)
        painter.setPen(QPen(QColor('#f9e2af'), 1.5))
        painter.setBrush(QBrush(QColor('#181825')))
        painter.drawRoundedRect(QRectF(750, 10, 220, 24), 5, 5)
        painter.setPen(QColor('#f9e2af'))
        addr_col_str = view.col_decoder.address_to_binary(view.col_decoder.last_address) \
            if view.mode == 'program_v2' else "OFF"
        painter.drawText(QRectF(750, 10, 220, 24), Qt.AlignCenter,
                         f"🎯 Decoder COL [{addr_col_str}] → Addr {view.col_decoder.last_address}")

        painter.setPen(QPen(QColor('#f38ba8'), 1.5))
        painter.setBrush(QBrush(QColor('#181825')))
        painter.drawRoundedRect(QRectF(750, 38, 220, 24), 5, 5)
        painter.setPen(QColor('#f38ba8'))
        painter.drawText(QRectF(750, 38, 220, 24), Qt.AlignCenter,
                         f"⚡ Driver COL (BL): {view.col_driver.V_active:+.2f}V")

        # 3. Badge Título Central (Limpio y sin solapamientos)
        painter.setPen(QPen(QColor('#313244'), 1))
        painter.setBrush(QBrush(QColor('#181825')))
        painter.drawRoundedRect(QRectF(240, 14, 490, 28), 6, 6)

        painter.setFont(FONTS['subtitle'])
        painter.setPen(get_qcolor('text'))
        mode_str = " — ⚡ TIEMPO REAL ACTIVO" if self.is_animating else ""
        painter.drawText(QRectF(240, 14, 490, 28), Qt.AlignCenter,
                         f"CROSSBAR 4×4 — MATRIZ DE 16 MEMRISTORES STRUKOV{mode_str}")

        # 4. Bus de Inhibición Lateral Interneuronal WTA (si hay un ganador activo)
        winner_j = getattr(view, 'winner_j', None)
        if winner_j is not None:
            LIF_win = self.elements.get(f'LIF_{winner_j+1}')
            LIF_1 = self.elements.get('LIF_1')
            LIF_4 = self.elements.get('LIF_4')
            if LIF_win and LIF_1 and LIF_4:
                y_inhib = LIF_win.y
                pen_inhib = QPen(QColor('#f38ba8'), 2, Qt.DashLine)
                painter.setPen(pen_inhib)
                painter.drawLine(int(LIF_1.x - 30), int(y_inhib), int(LIF_4.x + 30), int(y_inhib))

                painter.setFont(FONTS['small'])
                painter.setPen(QColor('#f38ba8'))
                painter.drawText(QRectF(LIF_1.x - 40, y_inhib - 48, (LIF_4.x - LIF_1.x) + 80, 18),
                                 Qt.AlignCenter, "⚡ BUS DE INHIBICIÓN LATERAL WTA (R-STDP LTD ACTIVO)")

        # 5. Ecuación matricial de lectura en la parte inferior
        G = get_G_matrix(self.elements, 4, 4)
        V = np.array([float(self.elements[f'S{i+1}'].params.get('V_out', 0.5))
                      for i in range(4)])
        I = compute_currents(G, V)
        painter.setFont(FONTS['mono'])
        painter.setPen(get_qcolor('success'))
        text_i = "   │   ".join([f"I_{j+1} = {I[j]*1e6:.1f} μA" for j in range(4)])
        painter.drawText(QRectF(0, self.height() - 35, self.width(), 25),
                         Qt.AlignCenter,
                         f"⚡ Operación Matricial I = G^T · V:   {text_i}")


class Crossbar4x4View(QWidget):
    """Vista interactiva completa del Crossbar 4×4 integrada en PySide6 (Qt6)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.elements: Dict[str, VisualElement] = {}
        self._sim_time = 0.0

        # --- Decodificadores y drivers explícitos de Hardware ---
        self.row_decoder = AddressDecoder(n_lines=4)
        self.col_decoder = AddressDecoder(n_lines=4)
        self.row_driver = RowDriver(n_rows=4)
        self.col_driver = ColumnDriver(n_cols=4)

        # --- Configuraciones explícitas ---
        self.read_cfg = ReadConfig()
        self.write_cfg = WriteConfig()

        # --- Modos de Operación Crossbar Real ---
        self.mode = "read"         # "read" (Lectura no destructiva) o "program_v2" (Programación V/2)
        self.target_cell = (0, 0)  # Celda objetivo para programación V/2 (row, col)
        self.V_rows = np.full(4, 0.2)
        self.V_cols = np.zeros(4)

        # ═══════════════════════════════════════════════════════════════
        # Modos de Plasticidad (OFF / STDP / R-STDP)
        # ═══════════════════════════════════════════════════════════════
        self.plasticity_mode = "off"   # 'off' | 'stdp' | 'rstdp'

        stdp_cfg = STDPConfig(
            A_plus=0.05, A_minus=0.025,
            tau_plus=20e-3, tau_minus=20e-3,
            eta=1.0, G_min=1e-6, G_max=500e-6,
        )
        self.stdp_rule = STDPRule(stdp_cfg)
        self.stdp_rule.reset(n_rows=4, n_cols=4)

        rstdp_cfg = RSTDPConfig(
            A_plus=0.05, A_minus=0.025,
            tau_plus=20e-3, tau_minus=20e-3,
            eta=1.0, G_min=1e-6, G_max=500e-6,
            R=0.0,
        )
        self.rstdp_rule = RSTDPRule(rstdp_cfg)
        self.rstdp_rule.reset(n_rows=4, n_cols=4)

        self.spike_pre = np.zeros(4)   # Spikes de sensores (filas)
        self.spike_post = np.zeros(4)  # Spikes de LIF (columnas)
        self.reward = 0.0

        # --- Selectividad Secuencial (Decoder multiplexing por fila) ---
        self.sequence_index = 0
        self.sequence_period_ticks = 100  # 100 ticks (~3.0 s) por fila para observación clara y estable
        self._sequence_counter = 0

        # --- Recompensa Causal por Columna (R local dependiente de la actividad) ---
        self.column_rewards = np.zeros(4)
        self.column_spike_history = np.zeros(4)

        # --- Acumulador de Evidencia (Winner Accumulator por Integración Leaky de Spikes, Gold & Shadlen 2007) ---
        self.evidence_accumulator = np.zeros(4)   # Vector de evidencia por columna (cuantos)
        self.tau_evidence = 100e-3                 # Constante de integración de evidencia (100 ms)
        self.evidence_threshold = 3.0             # Umbral de evidencia acumulada para declarar ganador (3 cuantos)
        self.winner_j = None                      # Columna del ganador activo
        self.winner_lock_time = 0.0               # Instante físico hasta el que se mantiene la decisión firme (s)

        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(30)
        self.anim_timer.timeout.connect(self._anim_tick)

        self._build_elements()
        self._init_ui()


    def _build_elements(self):
        x_sensores = 80
        x_memristores = [300, 480, 660, 840]
        y_filas = [150, 310, 470, 630]

        # 4 Sensores (Filas 1 a 4)
        for i in range(4):
            key = f'S{i+1}'
            self.elements[key] = SensorElement(key, x=x_sensores, y=y_filas[i])
            self.elements[key].params['name'] = f'Sensor {i+1}'
            self.elements[key].params['V_out'] = 0.8 if i == 0 else 0.4 if i == 1 else 0.3 if i == 2 else 0.2

        # 16 Memristores Strukov (Matriz 4x4)
        for i in range(4):
            for j in range(4):
                key = f'M{i+1}{j+1}'
                self.elements[key] = MemristorElement(key, row=i, col=j, x=x_memristores[j], y=y_filas[i])

        # 4 Memristores Volátiles (HfO₂), 4 Neuronas LIF, 4 Actuadores (al final de cada columna)
        acciones = ['girar_izq', 'girar_der', 'avanzar', 'retroceder']
        y_volatiles = 740
        y_neuronas = 850
        y_actuadores = 950

        for j in range(4):
            key_mv = f'M_v{j+1}'
            self.elements[key_mv] = VolatileMemristorElement(key_mv, x=x_memristores[j], y=y_volatiles)
            self.elements[key_mv].params['volatile_tau_relax'] = 0.3

            key_lif = f'LIF_{j+1}'
            self.elements[key_lif] = NeuronElement(key_lif, x=x_memristores[j], y=y_neuronas)
            self.elements[key_lif].params['C_m'] = 100e-9
            self.elements[key_lif].params['R_leak'] = 1e6
            self.elements[key_lif].params['V_th_base'] = 2.5
            self.elements[key_lif].params['V_th'] = 2.5
            self.elements[key_lif].params['V_adapt_inc'] = 0.15
            self.elements[key_lif].params['tau_adapt'] = 0.05
            self.elements[key_lif].params['V_m'] = 0.0
            self.elements[key_lif].params['spike_count'] = 0

            key_act = f'Act_{j+1}'
            self.elements[key_act] = ActuatorElement(key_act, x=x_memristores[j], y=y_actuadores)
            self.elements[key_act].params['action'] = acciones[j]

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ═══════════════════════════════════════════════════════════════
        # FILA 1: BARRA DE LOS 4 MODOS UNIFICADOS
        # ═══════════════════════════════════════════════════════════════
        toolbar_modes_main = QHBoxLayout()

        lbl_m_title = QLabel("🔷 Modo del Crossbar:")
        lbl_m_title.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 12px;")
        toolbar_modes_main.addWidget(lbl_m_title)

        self.btn_mode_read = QPushButton("📖 1. Lectura")
        self.btn_mode_prog = QPushButton("⚡ 2. Modo V/2")
        self.btn_plast_stdp = QPushButton("🧠 3. STDP")
        self.btn_plast_rstdp = QPushButton("🎯 4. R-STDP")
        self.btn_plast_off = QPushButton("🔒 OFF")  # Alias interno para tests

        self.btn_mode_read.setToolTip("Modo 1: Lectura paralela no destructiva (G no cambia)")
        self.btn_mode_prog.setToolTip("Modo 2: Programación manual celda target V/2 con La Cruz")
        self.btn_plast_stdp.setToolTip("Modo 3: Aprendizaje no supervisado por coincidencia temporal local")
        self.btn_plast_rstdp.setToolTip("Modo 4: Aprendizaje por refuerzo modulado por señal global R")

        self.btn_mode_read.clicked.connect(self._set_mode_1_read)
        self.btn_mode_prog.clicked.connect(self._set_mode_2_v2)
        self.btn_plast_stdp.clicked.connect(self._set_mode_3_stdp)
        self.btn_plast_rstdp.clicked.connect(self._set_mode_4_rstdp)

        self.btn_pulse_ltp = QPushButton("➕ LTP (+2V)")
        self.btn_pulse_ltd = QPushButton("➖ LTD (-2V)")
        self.btn_pulse_ltp.clicked.connect(lambda: self._apply_programming_pulse(2.0))
        self.btn_pulse_ltd.clicked.connect(lambda: self._apply_programming_pulse(-2.0))

        self.btn_reward_plus = QPushButton("✅ Éxito (+1)")
        self.btn_reward_minus = QPushButton("❌ Error (−1)")
        self.btn_reward_plus.setStyleSheet(
            "QPushButton { background-color: #a6e3a1; color: #11111b; font-weight: bold; padding: 5px 10px; border-radius: 4px; }"
        )
        self.btn_reward_minus.setStyleSheet(
            "QPushButton { background-color: #f38ba8; color: #11111b; font-weight: bold; padding: 5px 10px; border-radius: 4px; }"
        )
        self.btn_reward_plus.clicked.connect(lambda: self._apply_reward(+1.0))
        self.btn_reward_minus.clicked.connect(lambda: self._apply_reward(-1.0))

        toolbar_modes_main.addWidget(self.btn_mode_read)
        toolbar_modes_main.addWidget(self.btn_mode_prog)
        toolbar_modes_main.addWidget(self.btn_plast_stdp)
        toolbar_modes_main.addWidget(self.btn_plast_rstdp)
        toolbar_modes_main.addSpacing(15)
        toolbar_modes_main.addWidget(self.btn_pulse_ltp)
        toolbar_modes_main.addWidget(self.btn_pulse_ltd)
        toolbar_modes_main.addSpacing(15)
        toolbar_modes_main.addWidget(self.btn_reward_plus)
        toolbar_modes_main.addWidget(self.btn_reward_minus)
        toolbar_modes_main.addStretch()

        main_layout.addLayout(toolbar_modes_main)

        # ═══════════════════════════════════════════════════════════════
        # FILA 2: SENSORES Y ACCIONES DE SIMULACIÓN
        # ═══════════════════════════════════════════════════════════════
        toolbar_controls_main = QHBoxLayout()

        lbl_s_title = QLabel("⚡ Sensores S1..S4:")
        lbl_s_title.setStyleSheet("color: #a6e3a1; font-weight: bold; font-size: 11px;")
        toolbar_controls_main.addWidget(lbl_s_title)

        self.spin_v1 = QDoubleSpinBox()
        self.spin_v2 = QDoubleSpinBox()
        self.spin_v3 = QDoubleSpinBox()
        self.spin_v4 = QDoubleSpinBox()
        self.sensor_spinboxes = [self.spin_v1, self.spin_v2, self.spin_v3, self.spin_v4]

        for i, sb in enumerate(self.sensor_spinboxes):
            sb.setRange(-5.0, 5.0)
            sb.setSingleStep(0.1)
            sb.setDecimals(2)
            sb.setValue(float(self.elements[f'S{i+1}'].params.get('V_out', 0.2)))
            sb.setPrefix(f"S{i+1}: ")
            sb.setSuffix("V")
            sb.setStyleSheet("""
                QDoubleSpinBox {
                    background-color: #1e1e2e; color: #a6e3a1; font-weight: bold;
                    border: 1px solid #45475a; border-radius: 4px; padding: 2px 4px; font-size: 11px;
                }
            """)
            idx = i
            sb.valueChanged.connect(lambda val, s_idx=idx: self._on_sensor_spinbox_changed(s_idx, val))
            toolbar_controls_main.addWidget(sb)

        btn_style_default = """
            QPushButton {
                background-color: #313244; color: #cdd6f4; font-weight: bold;
                padding: 5px 10px; border-radius: 4px; border: 1px solid #45475a;
            }
            QPushButton:hover { background-color: #45475a; color: #89b4fa; }
        """
        btn_style_primary = """
            QPushButton {
                background-color: #89b4fa; color: #11111b; font-weight: bold;
                padding: 5px 10px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #b4befe; }
        """
        btn_style_warning = """
            QPushButton {
                background-color: #f38ba8; color: #11111b; font-weight: bold;
                padding: 5px 10px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #f5e0dc; }
        """
        btn_style_accent = """
            QPushButton {
                background-color: #cba6f7; color: #11111b; font-weight: bold;
                padding: 5px 10px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #f5c2e7; }
        """

        self.btn_simulate = QPushButton("▶ Simular")
        self.btn_realtime = QPushButton("⚡ Tiempo Real")
        self.btn_reset = QPushButton("⏹ Reset")
        self.btn_matrix = QPushButton("📊 Matriz G")
        self.btn_grid_config = QPushButton("🔲 Sub-Matriz")
        self.btn_rw_config = QPushButton("⚙️ Config R/W")

        self.btn_simulate.setStyleSheet(btn_style_primary)
        self.btn_realtime.setStyleSheet(btn_style_default)
        self.btn_reset.setStyleSheet(btn_style_warning)
        self.btn_matrix.setStyleSheet(btn_style_default)
        self.btn_grid_config.setStyleSheet(btn_style_default)
        self.btn_rw_config.setStyleSheet(btn_style_accent)

        self.btn_simulate.clicked.connect(self._on_simulate)
        self.btn_realtime.clicked.connect(self.toggle_realtime)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_matrix.clicked.connect(self._on_show_matrix)
        self.btn_grid_config.clicked.connect(self._on_config_grid)
        self.btn_rw_config.clicked.connect(self._on_open_rw_config)

        toolbar_controls_main.addWidget(self.btn_simulate)
        toolbar_controls_main.addWidget(self.btn_realtime)
        toolbar_controls_main.addWidget(self.btn_reset)
        toolbar_controls_main.addWidget(self.btn_matrix)
        toolbar_controls_main.addWidget(self.btn_grid_config)
        toolbar_controls_main.addWidget(self.btn_rw_config)

        self.btn_zoom_in = QPushButton("🔍 +")
        self.btn_zoom_out = QPushButton("🔍 -")
        self.btn_zoom_reset = QPushButton("📌 Reset Vista")
        self.lbl_zoom = QLabel("100%")

        self.btn_zoom_in.setStyleSheet(btn_style_default)
        self.btn_zoom_out.setStyleSheet(btn_style_default)
        self.btn_zoom_reset.setStyleSheet(btn_style_default)
        self.lbl_zoom.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 11px;")

        toolbar_controls_main.addWidget(self.btn_zoom_in)
        toolbar_controls_main.addWidget(self.btn_zoom_out)
        toolbar_controls_main.addWidget(self.btn_zoom_reset)
        toolbar_controls_main.addWidget(self.lbl_zoom)
        toolbar_controls_main.addStretch()

        main_layout.addLayout(toolbar_controls_main)
        self._update_mode_button_styles()


        # Canvas Qt
        self.canvas = Crossbar4x4Canvas(
            self,
            elements=self.elements,
            on_element_clicked=self._open_config,
            on_zoom_changed=self._update_zoom_label
        )
        self.btn_zoom_in.clicked.connect(self.canvas.zoom_in)
        self.btn_zoom_out.clicked.connect(self.canvas.zoom_out)
        self.btn_zoom_reset.clicked.connect(self.canvas.reset_view)

        main_layout.addWidget(self.canvas, 1)

        # Status Label
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet("background-color: #181825; border-radius: 4px; border: 1px solid #45475a;")
        status_layout = QHBoxLayout(self.status_frame)
        status_layout.setContentsMargins(10, 4, 10, 4)

        self.status_label = QLabel("Crossbar 4×4 listo. Click para configurar. 💡 Panning: Arrastra con el mouse | Zoom: Rueda del mouse.")
        self.status_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        status_layout.addWidget(self.status_label)

        main_layout.addWidget(self.status_frame)

    def _update_zoom_label(self, zoom: float):
        self.lbl_zoom.setText(f"{int(zoom * 100)}%")

    def _set_mode_1_read(self):
        """Modo 1: Lectura (Inferencia Pura - G no cambia)."""
        self.mode = "read"
        self.plasticity_mode = "off"
        self.reward = 0.0
        if hasattr(self, 'rstdp_rule'):
            self.rstdp_rule.set_reward(0.0)
        self._update_mode_button_styles()
        self.status_label.setText("📖 Modo 1 [Lectura]: Voltajes V_i aplican inferencia I = G^T · V (Conductancias G persistentes e inalteradas).")
        self.canvas.update()

    def _set_mode_2_v2(self):
        """Modo 2: Programación V/2 (La Cruz)."""
        self.mode = "program_v2"
        self.plasticity_mode = "off"
        self.reward = 0.0
        if hasattr(self, 'rstdp_rule'):
            self.rstdp_rule.set_reward(0.0)
        self._update_mode_button_styles()
        self._sync_sensors_to_mode()
        tg = self.target_cell
        self.status_label.setText(f"⚡ Modo 2 [Programación V/2]: Target = M{tg[0]+1}{tg[1]+1} (🔴 Full V_P = 2.0V | 🟡 Half V_P/2 = 1.0V en la CRUZ).")
        self.canvas.update()

    def _set_mode_3_stdp(self):
        """Modo 3: STDP (Aprendizaje No Supervisado Hebbiano)."""
        self.mode = "read"  # Bug #1: Forzar modo física = "read"
        self.plasticity_mode = "stdp"
        self.reward = 0.0
        if hasattr(self, 'rstdp_rule'):
            self.rstdp_rule.set_reward(0.0)
        self.stdp_rule.reset(n_rows=4, n_cols=4)
        self._update_mode_button_styles()
        self.status_label.setText("🧠 Modo 3 [STDP Activo]: Aprendizaje Hebbiano no supervisado por coincidencia temporal local (Sin decodificador).")
        self.canvas.update()

    def _set_mode_4_rstdp(self):
        """Modo 4: R-STDP (Aprendizaje por Refuerzo modulado por R)."""
        self.mode = "read"  # Bug #2: Forzar modo física = "read"
        self.plasticity_mode = "rstdp"
        self.rstdp_rule.reset(n_rows=4, n_cols=4)
        self.reward = 0.0
        if hasattr(self, 'rstdp_rule'):
            self.rstdp_rule.set_reward(0.0)
        self._update_mode_button_styles()
        self.status_label.setText("🎯 Modo 4 [R-STDP Activo]: Aprendizaje por refuerzo — Presiona ✅ Éxito (+1) o ❌ Error (-1) para modular G.")
        self.canvas.update()

    def _set_mode_read(self):
        self._set_mode_1_read()

    def _set_mode_prog(self):
        self._set_mode_2_v2()

    def _set_plasticity_off(self):
        self._set_mode_1_read()

    def _set_plasticity_stdp(self):
        self._set_mode_3_stdp()

    def _set_plasticity_rstdp(self):
        self._set_mode_4_rstdp()

    def _sync_sensors_to_mode(self):
        """Sincroniza los spinboxes de los sensores con la simulación V/2 actual."""
        if self.mode == "program_v2":
            tg_r, _ = self.target_cell
            for i in range(4):
                v_val = 1.0 if i == tg_r else 0.0
                self.elements[f'S{i+1}'].params['V_out'] = v_val
            self._update_sensor_spinboxes()

    def _update_mode_button_styles(self):
        style_active = "QPushButton { background-color: #a6e3a1; color: #11111b; font-weight: bold; padding: 5px 12px; border-radius: 4px; }"
        style_inactive = "QPushButton { background-color: #313244; color: #cdd6f4; font-weight: bold; padding: 5px 12px; border-radius: 4px; border: 1px solid #45475a; }"
        style_pulse = "QPushButton { background-color: #f9e2af; color: #11111b; font-weight: bold; padding: 5px 12px; border-radius: 4px; } QPushButton:hover { background-color: #fab387; }"

        # Determinar cuál de los 4 modos está activo
        is_m1 = (self.mode == "read" and self.plasticity_mode == "off")
        is_m2 = (self.mode == "program_v2" and self.plasticity_mode == "off")
        is_m3 = (self.plasticity_mode == "stdp")
        is_m4 = (self.plasticity_mode == "rstdp")

        self.btn_mode_read.setStyleSheet(style_active if is_m1 else style_inactive)
        self.btn_mode_prog.setStyleSheet(style_active if is_m2 else style_inactive)
        self.btn_plast_stdp.setStyleSheet(style_active if is_m3 else style_inactive)
        self.btn_plast_rstdp.setStyleSheet(style_active if is_m4 else style_inactive)

        self.btn_pulse_ltp.setStyleSheet(style_pulse)
        self.btn_pulse_ltd.setStyleSheet(style_pulse)

        style_reward_plus_active = "QPushButton { background-color: #a6e3a1; color: #11111b; font-weight: bold; padding: 5px 10px; border-radius: 4px; border: 2px solid #ffffff; }"
        style_reward_plus_inactive = "QPushButton { background-color: #a6e3a1; color: #11111b; font-weight: bold; padding: 5px 10px; border-radius: 4px; }"
        style_reward_minus_active = "QPushButton { background-color: #f38ba8; color: #11111b; font-weight: bold; padding: 5px 10px; border-radius: 4px; border: 2px solid #ffffff; }"
        style_reward_minus_inactive = "QPushButton { background-color: #f38ba8; color: #11111b; font-weight: bold; padding: 5px 10px; border-radius: 4px; }"

        self.btn_reward_plus.setStyleSheet(style_reward_plus_active if (is_m4 and self.reward > 0) else style_reward_plus_inactive)
        self.btn_reward_minus.setStyleSheet(style_reward_minus_active if (is_m4 and self.reward < 0) else style_reward_minus_inactive)

        self.btn_pulse_ltp.setEnabled(True)
        self.btn_pulse_ltd.setEnabled(True)
        self.btn_reward_plus.setEnabled(True)
        self.btn_reward_minus.setEnabled(True)

    def _apply_reward(self, R: float, is_auto: bool = False):
        """
        Aplica R-STDP (Aprendizaje por Refuerzo Modulado por Recompensa Global R):
        - R = +1.0 (Éxito / Recompensa): Induce LTP (+25.0 μS) en la celda target y celdas activas.
        - R = -1.0 (Error / Penalización): Induce LTD (-25.0 μS) en la celda target y celdas activas.
        """
        if self.plasticity_mode != "rstdp":
            self.mode = "read"
            self.plasticity_mode = "rstdp"

        self.reward = float(R)
        self.rstdp_rule.set_reward(R)

        tg_r, tg_c = self.target_cell
        dG_pulse = R * 25.0e-6  # +25.0 uS para Éxito, -25.0 uS para Error

        affected_cells = []
        max_dG_uS = 0.0

        # Celdas a actualizar: celda objetivo M_ij + cualquier celda con sensor activo (V > 0.49V)
        target_keys = {f'M{tg_r+1}{tg_c+1}'}
        for i in range(4):
            v_s = float(self.elements[f'S{i+1}'].params.get('V_out', 0.2))
            if v_s > 0.49:
                for j in range(4):
                    target_keys.add(f'M{i+1}{j+1}')

        for key in target_keys:
            mem = self.elements.get(key)
            if mem:
                p = mem.params
                G_old = float(p.get('G', 69.4e-6))
                G_new = float(np.clip(G_old + dG_pulse, 1.0e-6, 500.0e-6))

                RON = float(p.get('RON', 100.0))
                ROFF = float(p.get('ROFF', 16000.0))
                R_new = 1.0 / max(1e-12, G_new)
                x_calc = (ROFF - R_new) / (ROFF - RON) if ROFF != RON else 0.1
                x_calc = float(np.clip(x_calc, 0.01, 0.99))

                p['x'] = x_calc
                p['x0'] = x_calc
                p['G'] = G_new
                p['G_11'] = G_new
                p['R'] = R_new
                p['R_11'] = R_new

                if hasattr(mem, 'on_params_changed'):
                    mem.on_params_changed()

                affected_cells.append(key)
                if abs(dG_pulse) * 1e6 > abs(max_dG_uS):
                    max_dG_uS = dG_pulse * 1e6

        tag = "✅ ÉXITO (+1)" if R > 0 else "❌ PENALIZACIÓN ERROR (−1)"
        auto_tag = " [AUTO-ENTORNO]" if is_auto else " [MANUAL]"
        cells_str = ", ".join(affected_cells[:4])
        target_mem = self.elements.get(f'M{tg_r+1}{tg_c+1}')
        g_target_uS = float(target_mem.params.get('G', 69.4e-6)) * 1e6 if target_mem else 0.0

        self.status_label.setText(
            f"🎯 R-STDP Real{auto_tag} {tag}: Recompensa R = {R:+.0f} │ ΔG = {max_dG_uS:+.1f} μS │ Celda target M{tg_r+1}{tg_c+1} G = {g_target_uS:.1f} μS │ Celdas actualizadas ({len(affected_cells)}): {cells_str}"
        )
        self._update_mode_button_styles()
        self.canvas.update()





    def _update_plasticity_button_styles(self):
        """Actualiza estilos de los botones de plasticidad."""
        style_active = """
            QPushButton {
                background-color: #a6e3a1; color: #11111b;
                font-weight: bold; padding: 6px 12px; border-radius: 4px;
            }
        """
        style_inactive = """
            QPushButton {
                background-color: #313244; color: #cdd6f4;
                font-weight: bold; padding: 6px 12px;
                border-radius: 4px; border: 1px solid #45475a;
            }
        """
        if not hasattr(self, 'btn_plast_off'):
            return

        modes = [
            (self.btn_plast_off, "off"),
            (self.btn_plast_stdp, "stdp"),
            (self.btn_plast_rstdp, "rstdp"),
        ]
        for btn, mode in modes:
            btn.setStyleSheet(style_active if self.plasticity_mode == mode else style_inactive)

        if hasattr(self, 'btn_pulse_ltp') and hasattr(self, 'btn_pulse_ltd'):
            self.btn_pulse_ltp.setEnabled(True)
            self.btn_pulse_ltd.setEnabled(True)


    def _apply_programming_pulse(self, v_pulse: float):
        """Aplica un pulso de programación V/2 (LTP o LTD) a la celda objetivo M_ij."""
        tg_r, tg_c = self.target_cell
        target_mem = self.elements.get(f'M{tg_r+1}{tg_c+1}')
        if not target_mem:
            return

        g_old = float(target_mem.params.get('G', 69.4e-6)) * 1e6

        self.V_rows = np.zeros(4)
        self.V_cols = np.zeros(4)
        self.V_rows[tg_r] = v_pulse / 2.0
        self.V_cols[tg_c] = -v_pulse / 2.0

        # Sub-pasos de evolución para un cambio claro y visible de G
        for _ in range(5):
            for i in range(4):
                for j in range(4):
                    mem = self.elements.get(f'M{i+1}{j+1}')
                    if not mem:
                        continue
                    p = mem.params

                    v_cell = float(self.V_rows[i]) - float(self.V_cols[j])
                    v_abs = abs(v_cell)
                    v_th = 0.5
                    if v_abs > v_th:
                        v_sign = 1.0 if v_cell > 0 else -1.0
                        v_overdrive = v_abs - v_th
                        x_curr = float(p.get('x', p.get('x0', 0.10)))
                        p_exp = float(p.get('window_p', 2))
                        st = 1.0 if v_cell > 0 else 0.0
                        f_win = max(0.1, float(1.0 - (x_curr - st)**(2 * float(p_exp))))
                        d2d_factor = float(p.get('d2d_factor', 1.0))
                        k_rate = 0.35
                        dxdt = v_sign * k_rate * (v_overdrive / 1.5) * f_win * d2d_factor

                        x_next = float(np.clip(x_curr + dxdt * 0.1, 0.01, 0.99))
                        p['x'] = x_next
                        p['x0'] = x_next

                        RON = float(p.get('RON', 100.0))
                        ROFF = float(p.get('ROFF', 16000.0))
                        R_val = RON * x_next + ROFF * (1.0 - x_next)
                        G_val = 1.0 / max(1.0, R_val)
                        p['R'] = R_val
                        p['G'] = G_val
                        p['G_11'] = G_val
                        p['R_11'] = R_val
                        if hasattr(mem, 'on_params_changed'):
                            mem.on_params_changed()


        g_new = float(target_mem.params.get('G', 69.4e-6)) * 1e6
        tag = "➕ LTP (+2V)" if v_pulse > 0 else "➖ LTD (-2V)"
        self.status_label.setText(
            f"⚡ Pulso {tag} aplicado a M{tg_r+1}{tg_c+1}: G cambió de {g_old:.1f} μS → {g_new:.1f} μS │ (Cruz V/2 programada en Fila {tg_r+1} y Col {tg_c+1})"
        )
        self.canvas.update()


    def _on_sensor_spinbox_changed(self, sensor_idx: int, val: float):
        s_key = f'S{sensor_idx+1}'
        if s_key in self.elements:
            self.elements[s_key].params['V_out'] = val

            v_tag = "⚡ ESCRITURA" if abs(val) > 0.49 else "📖 LECTURA"
            self.status_label.setText(f"{v_tag} en {s_key}: Voltaje ajustado a {val:.2f} V")
            if abs(val) > 0.49:
                for _ in range(10):
                    self._step_physics(0.02)
            self.canvas.update()

    def _set_all_sensors_read(self):
        for i, sb in enumerate(self.sensor_spinboxes):
            sb.setValue(0.2)
        self.status_label.setText("📖 Todos los sensores ajustados a Voltaje de Lectura No Destructivo (0.20V)")
        self.canvas.update()

    def _set_sensor_write(self, sensor_idx: int, v_write: float = 2.0):
        if 0 <= sensor_idx < 4:
            self.sensor_spinboxes[sensor_idx].setValue(v_write)
            self.status_label.setText(f"⚡ Voltaje de ESCRITURA (+{v_write:.2f}V) aplicado a S{sensor_idx+1}. ¡Memristores de la Fila {sensor_idx+1} evolucionando conductancia!")
            self.canvas.update()

    def _update_sensor_spinboxes(self):
        for i, sb in enumerate(self.sensor_spinboxes):
            v_val = float(self.V_rows[i]) if hasattr(self, 'V_rows') and i < len(self.V_rows) else float(self.elements[f'S{i+1}'].params.get('V_out', 0.2))
            sb.blockSignals(True)
            sb.setValue(v_val)
            sb.blockSignals(False)

    def _open_config(self, element: VisualElement):
        for elem in self.elements.values():
            elem.selected = False
        element.selected = True

        if isinstance(element, MemristorElement):
            self.target_cell = (element.row, element.col)
            if self.mode == "program_v2":
                self.status_label.setText(
                    f"🎯 Target V/2 seleccionado: {element.element_id} (Fila {element.row+1}, Columna {element.col+1}). Aplica pulsos ➕ LTP o ➖ LTD."
                )
                self.canvas.update()

        if isinstance(element, SensorElement):
            v_curr = float(element.params.get('V_out', 0.2))
            if v_curr <= 0.5:
                v_new = 2.0
                msg = f"⚡ Voltaje de ESCRITURA (+{v_new:.2f}V) activo en {element.element_id} (Fila {element.element_id[1]})"
            else:
                v_new = 0.2
                msg = f"📖 Voltaje de LECTURA ({v_new:.2f}V) activo en {element.element_id}"
            element.params['V_out'] = v_new
            self._update_sensor_spinboxes()
            if v_new > 0.49:
                for _ in range(10):
                    self._step_physics(0.02)
            self.status_label.setText(msg)
            self.canvas.update()
            return

        if self.canvas.is_animating:
            self.status_label.setText(f"Seleccionado: {element.element_id} │ ⚡ [TIEMPO REAL ACTIVO] (Pausa la simulación para editar parámetros en diálogo)")
            self.canvas.update()
            return

        self.canvas.update()
        self.status_label.setText(f"Configurando: {element.element_id}")
        dlg = ConfigDialog(self, element, on_save=self._on_config_saved)
        dlg.exec()

    def _on_config_saved(self, element: VisualElement):
        self.status_label.setText(f"✓ Configuración actualizada: {element.element_id}")
        if hasattr(element, 'on_params_changed'):
            element.on_params_changed()
        self.canvas.update()

    def toggle_realtime(self):
        if self.canvas.is_animating:
            self.anim_timer.stop()
            self.canvas.is_animating = False
            self.btn_realtime.setText("⚡ Tiempo Real (OFF)")
            self.btn_realtime.setStyleSheet("""
                QPushButton {
                    background-color: #313244; color: #a6adc8; font-weight: bold;
                    padding: 6px 14px; border-radius: 4px; border: 1px solid #45475a;
                }
                QPushButton:hover { background-color: #45475a; }
            """)
            self.status_label.setText("⏸ Simulación en tiempo real pausada.")
        else:
            self.canvas.is_animating = True
            self.anim_timer.start()
            self.btn_realtime.setText("⚡ Tiempo Real (ACTIVO)")
            self.btn_realtime.setStyleSheet("""
                QPushButton {
                    background-color: #a6e3a1; color: #11111b; font-weight: bold;
                    padding: 6px 14px; border-radius: 4px;
                }
                QPushButton:hover { background-color: #94e2d5; }
            """)
            self.status_label.setText("⚡ Simulación 4×4 en tiempo real activa... 💡 Haz clic en los Sensores (S1..S4) para simular detección!")

    def _step_physics(self, dt: float):
        """
        Evolución física realista del Crossbar 4×4 con V_cell = V_row - V_col:
        - Modo Lectura: V_col = 0. Si V_row <= 0.5V, G_ij permanece constante (lectura no destructiva).
        - Modo Programación V/2: V_row[tg_r] = 1V, V_col[tg_c] = -1V. Target ve 2.0V, Cruz ve 1.0V.
        """
        if self.mode == "program_v2":
            tg_r, tg_c = self.target_cell
            self.V_rows = np.zeros(4)
            self.V_cols = np.zeros(4)
            self.V_rows[tg_r] = 1.0
            self.V_cols[tg_c] = -1.0
        elif self.plasticity_mode in ("stdp", "rstdp") and self.canvas.is_animating:
            # Selectividad Secuencial (address decoder multiplexing por fila)
            self._sequence_counter += 1
            if self._sequence_counter >= self.sequence_period_ticks:
                self._sequence_counter = 0
                self.sequence_index = (self.sequence_index + 1) % 4

            self.V_rows = np.zeros(4)
            s_val = float(self.elements[f'S{self.sequence_index+1}'].params.get('V_out', 0.8))
            self.V_rows[self.sequence_index] = s_val if abs(s_val) > 0.1 else 0.8
            self.V_cols = np.zeros(4)
        else:
            self.V_rows = np.array([float(self.elements[f'S{i+1}'].params.get('V_out', 0.2)) for i in range(4)])
            self.V_cols = np.zeros(4)

        # 1. Memristores Strukov M_ij
        for i in range(4):
            for j in range(4):
                mem = self.elements.get(f'M{i+1}{j+1}')
                if not mem:
                    continue
                p = mem.params

                v_cell = float(self.V_rows[i]) - float(self.V_cols[j])
                v_th_write = 0.5
                if abs(v_cell) > v_th_write:
                    x_curr = float(p.get('x', p.get('x0', 0.10)))
                    p_exp = float(p.get('window_p', 2))
                    st = 1.0 if v_cell > 0 else 0.0
                    f_win = max(0.0, float(1.0 - (x_curr - st)**(2 * float(p_exp))))

                    v_sign = 1.0 if v_cell > 0 else -1.0
                    v_overdrive = abs(v_cell) - v_th_write
                    k_rate = 0.10

                    d2d_factor = float(p.get('d2d_factor', 1.0))

                    if bool(p.get('chk_c2c', True)):
                        c2c_sig = float(p.get('c2c_sigma', 0.05))
                        c2c_noise = float(np.clip(np.random.normal(1.0, c2c_sig), 0.5, 1.5))
                    else:
                        c2c_noise = 1.0

                    dxdt = v_sign * k_rate * (v_overdrive / 1.5) * f_win * d2d_factor * c2c_noise

                    x_next = float(np.clip(x_curr + dxdt * dt, 0.01, 0.99))
                    p['x'] = x_next
                    RON = float(p.get('RON', 100.0))
                    ROFF = float(p.get('ROFF', 16000.0))
                    R_val = RON * x_next + ROFF * (1.0 - x_next)
                    G_val = 1.0 / max(1.0, R_val)
                    p['R'] = R_val
                    p['G'] = G_val
                    p['G_11'] = G_val
                    p['R_11'] = R_val

        # 2. Memristores Volátiles M_v1 .. M_v4 (Relajación difusiva)
        G_mat = get_G_matrix(self.elements, 4, 4)
        I_cols = compute_currents(G_mat, self.V_rows)

        for j in range(4):
            mv = self.elements.get(f'M_v{j+1}')
            if not mv:
                continue
            p_v = mv.params
            tau_rel = float(p_v.get('volatile_tau_relax', 0.3))
            x_v = float(p_v.get('x', 0.05))
            i_col = I_cols[j]

            alpha_drive = 100.0
            dxdt_v = alpha_drive * i_col - (x_v / max(1e-4, tau_rel))
            x_v_next = float(np.clip(x_v + dxdt_v * dt, 0.001, 1.0))
            p_v['x'] = x_v_next

        # 3. Aplicar plasticidad (si está activa STDP o R-STDP Y el modo es "read")
        if self.plasticity_mode in ("stdp", "rstdp") and self.mode == "read":
            self.spike_pre = (self.V_rows > 0.5).astype(float)


            for j in range(4):
                LIF = self.elements.get(f'LIF_{j+1}')
                if LIF:
                    Vm = float(LIF.params.get('V_m', 0.0))
                    Vth = float(LIF.params.get('V_th', 2.5))
                    prev_Vm = float(LIF.params.get('_prev_Vm', 0.0))
                    if prev_Vm >= Vth and Vm < 1.0:
                        self.spike_post[j] = 1.0
                    else:
                        self.spike_post[j] = 0.0
                    LIF.params['_prev_Vm'] = Vm

            G_matrix = get_G_matrix(self.elements, 4, 4)

            if self.plasticity_mode == "stdp":
                dG = self.stdp_rule.apply(
                    G_matrix, self.spike_pre, self.spike_post, dt
                )
            else:  # rstdp
                dG = self.rstdp_rule.apply(
                    G_matrix, self.spike_pre, self.spike_post, dt
                )

            for i in range(4):
                for j in range(4):
                    if abs(dG[i, j]) > 1e-12:
                        mem = self.elements.get(f'M{i+1}{j+1}')
                        if mem:
                            G_old = float(mem.params.get('G', 69.4e-6))
                            G_new = np.clip(
                                G_old + dG[i, j],
                                self.stdp_rule.G_min,
                                self.stdp_rule.G_max,
                            )
                            mem.params['G'] = float(G_new)
                            mem.params['G_11'] = float(G_new)
                            if hasattr(mem, 'on_params_changed'):
                                mem.on_params_changed()


    def _anim_tick(self):
        dt = 0.005
        self._sim_time += dt
        self.canvas.anim_time += 0.03

        self._step_physics(dt)

        G = get_G_matrix(self.elements, 4, 4)
        I = compute_currents(G, self.V_rows)

        # 1. Estado de inhibición previa o candado activo
        current_winner = self.winner_j if (self.winner_j is not None and self._sim_time < self.winner_lock_time) else None

        # 2. Integración de potencial de membrana con Inhibición Lateral WTA activa
        spikes_this_tick = np.zeros(4)
        for j in range(4):
            LIF = self.elements[f'LIF_{j+1}']
            C_m = float(LIF.params.get('C_m', 100e-9))
            R_leak = float(LIF.params.get('R_leak', 1e6))
            V_th_base = float(LIF.params.get('V_th_base', 2.5))
            V_th = float(LIF.params.get('V_th', V_th_base))
            tau_adapt = float(LIF.params.get('tau_adapt', 0.10))
            V_m = float(LIF.params.get('V_m', 0.0))

            # Decaimiento del umbral adaptativo hacia V_th_base
            V_th += (V_th_base - V_th) * (dt / max(1e-4, tau_adapt))
            LIF.params['V_th'] = V_th

            # Si otra neurona es la ganadora activa, inhibición lateral clamp a 0V (I_effective = 0)
            if current_winner is not None and j != current_winner:
                V_m_next = 0.0
                LIF.params['V_m'] = 0.0
                LIF.params['_v_m_next'] = 0.0
                spikes_this_tick[j] = 0.0
            else:
                dVm = ((I[j] - V_m / R_leak) / C_m) * dt
                V_m_next = V_m + dVm
                LIF.params['_v_m_next'] = V_m_next
                if V_m_next >= V_th:
                    spikes_this_tick[j] = 1.0

        # 3. Integración Leaky del Acumulador de Evidencia (Gold & Shadlen 2007)
        decay_ev = np.exp(-dt / self.tau_evidence)
        self.evidence_accumulator = self.evidence_accumulator * decay_ev + spikes_this_tick

        # 4. Decisión de Ganador por Acumulación de Evidencia (Winner Accumulator)
        if current_winner is not None:
            winner_j = current_winner
        else:
            max_idx = int(np.argmax(self.evidence_accumulator))
            if self.evidence_accumulator[max_idx] >= self.evidence_threshold:
                self.winner_j = max_idx
                self.winner_lock_time = self._sim_time + 0.50  # 500 ms de candado de inercia temporal
                winner_j = max_idx
            else:
                if self.winner_j is not None and self.evidence_accumulator[self.winner_j] < (self.evidence_threshold * 0.5):
                    self.winner_j = None
                winner_j = self.winner_j

        # Margen de dominancia física
        sorted_ev = np.sort(self.evidence_accumulator)
        max_ev = sorted_ev[-1]
        second_ev = sorted_ev[-2]
        margin = max_ev / max(1e-6, second_ev)

        column_rewards = np.full(4, -1.0)
        spikes_str = []

        for j in range(4):
            LIF = self.elements[f'LIF_{j+1}']
            Act = self.elements[f'Act_{j+1}']
            V_th = float(LIF.params.get('V_th', 2.5))
            V_adapt_inc = float(LIF.params.get('V_adapt_inc', 0.02))

            if winner_j is not None and j == winner_j:
                LIF.params['V_m'] = 0.0
                LIF.params['V_th'] = V_th + V_adapt_inc  # Auto-frenado homeostático por disparo
                LIF.params['spike_count'] = int(LIF.params.get('spike_count', 0)) + int(spikes_this_tick[j])
                LIF.params['is_winner'] = True
                Act.params['is_winner'] = True
                Act.params['action'] = f'🏆 GANADOR (×{margin:.1f})'
                column_rewards[j] = +1.0  # Ganador recibe R = +1.0 (LTP / Recompensa)
                self.spike_post[j] = 1.0
            else:
                LIF.params['V_m'] = 0.0 if winner_j is not None else float(LIF.params.get('_v_m_next', 0.0))
                LIF.params['is_winner'] = False
                Act.params['is_winner'] = False
                Act.params['action'] = '🚫 INHIBIDO' if winner_j is not None else 'listo'
                column_rewards[j] = -1.0  # Perdedores reciben R = -1.0 (LTD / Penalización)
                self.spike_post[j] = 0.0

            spikes_str.append(f"LIF_{j+1}={LIF.params['spike_count']}")

        # 5. Aplicar vector de recompensas por columna a R-STDP
        if self.plasticity_mode == "rstdp" and self.canvas.is_animating:
            self.rstdp_rule.set_reward(column_rewards)

        if not (self.plasticity_mode == "rstdp" and self.canvas.is_animating):
            margin_info = f" │ Margen: ×{margin:.1f}" if winner_j is not None else ""
            self.status_label.setText(
                f"⚡ [4×4 REALTIME] t = {self._sim_time:.2f} s │ I_outs = [{I[0]*1e6:.1f}, {I[1]*1e6:.1f}, {I[2]*1e6:.1f}, {I[3]*1e6:.1f}] μA │ Spikes: {' '.join(spikes_str)}{margin_info}"
            )
        self.canvas.update()
        self.canvas.update()

    def _on_simulate(self):
        dt = 0.01
        self._step_physics(dt)

        G = get_G_matrix(self.elements, 4, 4)
        I = compute_currents(G, self.V_rows)

        for j in range(4):
            LIF = self.elements[f'LIF_{j+1}']
            C_m = float(LIF.params.get('C_m', 100e-9))
            V_m_inc = (I[j] * 1e-3) / C_m
            LIF.params['V_m'] = min(LIF.params.get('V_m', 0.0) + V_m_inc, 2.5)

            if LIF.params['V_m'] >= LIF.params.get('V_th', 2.5):
                LIF.params['V_m'] = 0.0
                LIF.params['spike_count'] = int(LIF.params.get('spike_count', 0)) + 1

        self.status_label.setText(
            f"▶ Simulación Paso Completa: I = [{I[0]*1e6:.1f}, {I[1]*1e6:.1f}, {I[2]*1e6:.1f}, {I[3]*1e6:.1f}] μA"
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

        for i in range(4):
            lif = self.elements[f'LIF_{i+1}']
            lif.params['V_m'] = 0.0
            lif.params['V_th'] = float(lif.params.get('V_th_base', 2.5))
            lif.params['spike_count'] = 0

        self.status_label.setText("⏹ Reset completo (Semillas estocásticas renovadas)")
        self.canvas.update()

    def _on_reset_matrix(self):
        for i in range(4):
            for j in range(4):
                mem = self.elements[f'M{i+1}{j+1}']
                mem.params['x0'] = 0.10
                mem.params['x'] = 0.10
                mem.params['seed'] = random.randint(10000, 999999)
                if hasattr(mem, 'on_params_changed'):
                    mem.on_params_changed()

        self.status_label.setText("🎯 Matriz 4×4 reiniciada (Nuevas semillas estocásticas asignadas)")
        self.canvas.update()

    def _on_show_matrix(self):
        G = get_G_matrix(self.elements, 4, 4)
        msg = matrix_to_heatmap_string(G)
        QMessageBox.information(self, "Matriz G del Crossbar 4×4", msg)

    def _on_config_volatiles(self):
        dlg = BatchVolatileDialog(self, current_elements=self.elements)
        if dlg.exec():
            self.status_label.setText("⚡ Todos los 4 memristores volátiles (M_v1..M_v4) actualizados")
            self.canvas.update()

    def _on_config_grid(self):
        dlg = BatchGridDialog(self, current_elements=self.elements)
        if dlg.exec():
            r = dlg.spin_r.value()
            c = dlg.spin_c.value()
            self.status_label.setText(f"🔲 Sub-matriz {r}×{c} programada con éxito")
            self.canvas.update()

    def _on_open_rw_config(self):
        """Abre el diálogo de configuración explícita de lectura/escritura."""
        dlg = ReadWriteConfigDialog(
            self,
            read_cfg=self.read_cfg,
            write_cfg=self.write_cfg
        )
        if dlg.exec():
            self.read_cfg = dlg.get_read_config()
            self.write_cfg = dlg.get_write_config()

            # Sincronizar con drivers
            self.row_driver.V_active = self.write_cfg.V_row_active()
            self.col_driver.V_active = self.write_cfg.V_col_active()

            self.status_label.setText(
                f"⚙️ Configuración actualizada │ {self.read_cfg.describe()} │ {self.write_cfg.describe()}"
            )
            self.canvas.update()

    def _on_config_voltages(self):
        for i in range(4):
            val, ok = QInputDialog.getDouble(
                self, f"Voltaje V_{i+1}", f"Voltaje Sensor S{i+1} (V):",
                value=float(self.elements[f'S{i+1}'].params.get('V_out', 0.5)),
                minValue=0.0, maxValue=5.0, decimals=2
            )
            if ok:
                self.elements[f'S{i+1}'].params['V_out'] = val

        self.status_label.setText("✓ Voltajes de los 4 sensores actualizados")
        self.canvas.update()

    def _on_uniform(self):
        G_val, ok = QInputDialog.getDouble(
            self, "Programación Uniforme", "Conductancia Objetivo G (μS):",
            value=200.0, minValue=62.5, maxValue=500.0, decimals=1
        )
        if ok:
            G_mat = np.full((4, 4), G_val * 1e-6)
            set_G_matrix(self.elements, G_mat)
            self.status_label.setText(f"📐 Matriz 4×4 programada a {G_val:.1f} μS")
            self.canvas.update()
