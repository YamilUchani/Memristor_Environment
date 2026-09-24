"""
neurolab.gui.base_view
=======================
Clase base común para los canvas interactivos del Crossbar.

Proporciona:
  - Estado de zoom/pan
  - Eventos de mouse (wheel, press, move, release)
  - Primitivas de dibujo comunes:
      _draw_sensor, _draw_memristor, _draw_volatile_memristor,
      _draw_neuron, _draw_actuator
  - Estado de animación
  - Helpers widget_to_canvas, set_zoom, reset_view

Las subclases DEBEN definir:
  - _draw_wires(painter)      ← conexiones específicas del layout
  - _draw_grid(painter)       ← opcional, override
  - _draw_extra(painter)      ← opcional, decoders, títulos
  - MIN_WIDTH, MIN_HEIGHT     ← tamaño mínimo del canvas
"""

from typing import Dict, Any, Optional
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont
from PySide6.QtCore import Qt, QRectF, QPointF

from neurolab.gui.styles import COLORS, FONTS, get_qcolor


class BaseCrossbarCanvas(QWidget):
    """Canvas base con dibujo, pan, zoom, hit-test e interacción."""

    # --- Subclases pueden sobrescribir ---
    MIN_WIDTH: int = 900
    MIN_HEIGHT: int = 600

    def __init__(self, parent=None, elements=None,
                 on_element_clicked=None, on_zoom_changed=None):
        super().__init__(parent)
        self.view = parent
        self.elements: Dict[str, Any] = elements or {}
        self.on_element_clicked = on_element_clicked
        self.on_zoom_changed = on_zoom_changed
        self.hovered_element: Optional[Any] = None

        # --- Estado de zoom y pan ---
        self.zoom_factor: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.is_panning: bool = False
        self.last_pan_pos: QPointF = QPointF()

        # --- Estado de animación ---
        self.is_animating: bool = False
        self.anim_time: float = 0.0

        self.setMouseTracking(True)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.setStyleSheet(f"background-color: {COLORS['bg_canvas']};")

    # ================================================================
    # CONVERSIÓN DE COORDENADAS
    # ================================================================

    def widget_to_canvas(self, wx: float, wy: float) -> tuple[float, float]:
        """Convierte coordenadas de widget a canvas (aplicando pan/zoom)."""
        cx = (wx - self.pan_x) / self.zoom_factor
        cy = (wy - self.pan_y) / self.zoom_factor
        return cx, cy

    # ================================================================
    # ZOOM Y PAN
    # ================================================================

    def set_zoom(self, new_zoom: float,
                 anchor_x: float = None, anchor_y: float = None):
        """Cambia el zoom manteniendo un punto ancla fijo."""
        new_zoom = max(0.4, min(3.0, new_zoom))
        if anchor_x is None:
            anchor_x = self.width() / 2.0
        if anchor_y is None:
            anchor_y = self.height() / 2.0

        scale_ratio = new_zoom / self.zoom_factor
        self.pan_x = anchor_x - (anchor_x - self.pan_x) * scale_ratio
        self.pan_y = anchor_y - (anchor_y - self.pan_y) * scale_ratio
        self.zoom_factor = new_zoom
        self.update()
        if self.on_zoom_changed:
            self.on_zoom_changed(self.zoom_factor)

    def zoom_in(self):
        self.set_zoom(self.zoom_factor * 1.15)

    def zoom_out(self):
        self.set_zoom(self.zoom_factor / 1.15)

    def reset_view(self):
        """Vuelve a zoom=100% y pan=(0,0)."""
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()
        if self.on_zoom_changed:
            self.on_zoom_changed(self.zoom_factor)

    # ================================================================
    # EVENTOS DE MOUSE
    # ================================================================

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        pos = event.position()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.set_zoom(self.zoom_factor * factor, pos.x(), pos.y())
        event.accept()

    def mousePressEvent(self, event):
        pos = event.position()
        cx, cy = self.widget_to_canvas(pos.x(), pos.y())

        clicked_elem = None
        for elem in self.elements.values():
            if elem.hit_test(cx, cy):
                clicked_elem = elem
                break

        # Iniciar pan: clic derecho/medio o clic izquierdo sobre fondo vacío
        if event.button() in (Qt.RightButton, Qt.MiddleButton) or \
           (event.button() == Qt.LeftButton and clicked_elem is None):
            self.is_panning = True
            self.last_pan_pos = pos
            self.setCursor(Qt.ClosedHandCursor)
            if event.button() in (Qt.RightButton, Qt.MiddleButton):
                return

        if event.button() == Qt.LeftButton:
            for elem in self.elements.values():
                elem.selected = (elem == clicked_elem)
            self.update()
            if clicked_elem and self.on_element_clicked:
                self.on_element_clicked(clicked_elem)

    def mouseMoveEvent(self, event):
        pos = event.position()

        if self.is_panning:
            dx = pos.x() - self.last_pan_pos.x()
            dy = pos.y() - self.last_pan_pos.y()
            self.pan_x += dx
            self.pan_y += dy
            self.last_pan_pos = pos
            self.update()
            return

        cx, cy = self.widget_to_canvas(pos.x(), pos.y())
        hovered = None
        for elem in self.elements.values():
            if elem.hit_test(cx, cy):
                hovered = elem
                break

        if hovered != self.hovered_element:
            if self.hovered_element:
                self.hovered_element.hover = False
            if hovered:
                hovered.hover = True
            self.hovered_element = hovered
            self.setCursor(Qt.PointingHandCursor if hovered else Qt.ArrowCursor)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.is_panning:
            self.is_panning = False
            pos = event.position()
            cx, cy = self.widget_to_canvas(pos.x(), pos.y())
            hovered = None
            for elem in self.elements.values():
                if elem.hit_test(cx, cy):
                    hovered = elem
                    break
            self.setCursor(Qt.PointingHandCursor if hovered else Qt.ArrowCursor)

    # ================================================================
    # PRIMITIVAS DE DIBUJO
    # ================================================================

    def _draw_grid(self, painter: QPainter):
        """Rejilla de fondo. Override si el layout lo requiere."""
        pen = QPen(get_qcolor('bg_grid'), 1, Qt.PenStyle.DotLine)
        painter.setPen(pen)
        step = 40
        for x in range(-2000, 4000, step):
            painter.drawLine(x, -2000, x, 4000)
        for y in range(-2000, 4000, step):
            painter.drawLine(-2000, y, 4000, y)

    def _draw_sensor(self, painter: QPainter, elem):
        """Dibuja un sensor cuadrado (S_i)."""
        x, y = elem.x, elem.y
        view = getattr(self, 'view', None)

        # Obtener voltaje actual
        s_idx = int(elem.element_id.replace('S', '')) - 1 \
            if (elem.element_id and elem.element_id.startswith('S')) else 0
        if view and hasattr(view, 'V_rows') and 0 <= s_idx < len(view.V_rows):
            v_out = float(view.V_rows[s_idx])
        else:
            v_out = float(elem.params.get('V_out', 0.20))

        v_th_write = 0.49
        if abs(v_out) > v_th_write:
            border_color = QColor('#f38ba8')
            bg_color = QColor('#31222b')
            border_width = 3
            badge_str = "⚡ ESCRITURA"
            badge_color = QColor('#f38ba8')
        else:
            border_color = get_qcolor('selected' if elem.selected
                                       else ('hover' if elem.hover else 'sensor'))
            bg_color = QColor('#1e2d3d')
            border_width = 2
            badge_str = "📖 LECTURA"
            badge_color = QColor('#89b4fa')

        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(QRectF(x - 45, y - 30, 90, 60), 8, 8)

        painter.setFont(FONTS['label'])
        painter.setPen(get_qcolor('text'))
        painter.drawText(QRectF(x - 40, y - 25, 80, 18), Qt.AlignCenter,
                         elem.element_id)

        painter.setFont(FONTS['small'])
        painter.setPen(badge_color)
        painter.drawText(QRectF(x - 40, y - 8, 80, 14), Qt.AlignCenter, badge_str)

        painter.setFont(FONTS['mono'])
        painter.setPen(QColor('#ffffff'))
        painter.drawText(QRectF(x - 40, y + 8, 80, 18), Qt.AlignCenter,
                         f"{v_out:+.2f} V")

    def _draw_memristor(self, painter: QPainter, elem):
        """Dibuja un memristor (M_ij)."""
        x, y = elem.x, elem.y
        r, c = elem.row, elem.col

        view = getattr(self, 'view', None)
        is_prog = (view and hasattr(view, 'mode') and view.mode == 'program_v2')

        v_rows = getattr(view, 'V_rows', np.full(4, 0.2))
        v_cols = getattr(view, 'V_cols', np.zeros(4))
        v_cell = float(v_rows[r]) - float(v_cols[c])

        v_th_write = 0.5
        abs_v = abs(v_cell)

        if abs_v < 1e-9:
            border_color = get_qcolor('selected' if elem.selected
                                       else ('hover' if elem.hover else 'memristor'))
            bg_color = get_qcolor('bg_panel')
            border_width = 2
            v_net_str = f"⚪ Vnet={v_cell:+.1f}V"
            badge_color = get_qcolor('text_dim')
        elif abs_v <= v_th_write:
            border_color = get_qcolor('selected' if elem.selected
                                       else ('hover' if elem.hover else 'memristor'))
            bg_color = get_qcolor('bg_panel')
            border_width = 2
            v_net_str = f"📖 L: {v_cell:+.2f}V"
            badge_color = QColor('#89b4fa')
        else:
            if is_prog:
                tg_r, tg_c = getattr(view, 'target_cell', (0, 0))
                if r == tg_r and c == tg_c:
                    border_color = QColor('#f38ba8')
                    bg_color = QColor('#31222b')
                    border_width = 3
                    v_net_str = f"🔴 Full V={v_cell:+.1f}V"
                    badge_color = QColor('#f38ba8')
                else:
                    border_color = QColor('#f9e2af')
                    bg_color = QColor('#2b291d')
                    border_width = 2
                    v_net_str = f"🟡 Half V={v_cell:+.1f}V"
                    badge_color = QColor('#f9e2af')
            else:
                border_color = QColor('#f38ba8')
                bg_color = QColor('#31222b')
                border_width = 3
                v_net_str = f"⚡ E: {v_cell:+.2f}V"
                badge_color = QColor('#f38ba8')

        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(QRectF(x - 45, y - 30, 90, 60), 6, 6)

        painter.setFont(FONTS['label'])
        painter.setPen(get_qcolor('text'))
        painter.drawText(QRectF(x - 40, y - 25, 80, 16), Qt.AlignCenter,
                         elem.element_id)

        painter.setFont(FONTS['small'])
        painter.setPen(badge_color)
        painter.drawText(QRectF(x - 44, y - 9, 88, 14), Qt.AlignCenter, v_net_str)

        painter.setFont(FONTS['mono'])
        painter.setPen(QColor('#a6e3a1') if any(s in v_net_str for s in ['⚡', '🔴', '🟡'])
                       else get_qcolor('success'))
        G_val = float(elem.params.get('G', elem.params.get('G_11', 69.4e-6))) * 1e6
        painter.drawText(QRectF(x - 40, y + 6, 80, 18), Qt.AlignCenter,
                         f"{G_val:.1f} μS")

        # Barra de traza (si STDP/R-STDP está activo)
        parent_view = getattr(self, 'parent_view', view)
        if parent_view is not None and getattr(parent_view, 'plasticity_mode', 'off') != "off":
            stdp_rule = getattr(parent_view, 'stdp_rule', None)
            if stdp_rule and stdp_rule.trace_pre and stdp_rule.trace_post:
                trace_pre_val = float(stdp_rule.trace_pre.values[r])
                trace_post_val = float(stdp_rule.trace_post.values[c])
                bar_w = 60
                bar_h = 3
                pre_w = int(trace_pre_val * bar_w / 2)
                post_w = int(trace_post_val * bar_w / 2)
                painter.setPen(Qt.NoPen)
                if pre_w > 0:
                    painter.setBrush(QBrush(QColor('#fbbf24')))
                    painter.drawRect(QRectF(x - 30, y + 26, pre_w, bar_h))
                if post_w > 0:
                    painter.setBrush(QBrush(QColor('#ec4899')))
                    painter.drawRect(QRectF(x, y + 26, post_w, bar_h))

    def _draw_volatile_memristor(self, painter: QPainter, elem):
        """Dibuja un memristor volátil HfO₂ (M_v_i)."""
        x, y = elem.x, elem.y
        color = get_qcolor('selected' if elem.selected
                           else ('hover' if elem.hover else 'memristor_v'))

        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(get_qcolor('bg_panel')))
        painter.drawRoundedRect(QRectF(x - 45, y - 30, 90, 60), 6, 6)

        painter.setFont(FONTS['label'])
        painter.setPen(get_qcolor('text'))
        painter.drawText(QRectF(x - 40, y - 25, 80, 20), Qt.AlignCenter,
                         elem.element_id)

        tau_rel = elem.params.get('volatile_tau_relax', 0.3)
        painter.setFont(FONTS['small'])
        painter.setPen(get_qcolor('text_dim'))
        painter.drawText(QRectF(x - 40, y - 5, 80, 16), Qt.AlignCenter,
                         f"τ = {tau_rel:.2f} s")
        painter.setPen(get_qcolor('memristor_v'))
        painter.drawText(QRectF(x - 40, y + 12, 80, 16), Qt.AlignCenter,
                         "Volátil HfO₂")

    def _draw_neuron(self, painter: QPainter, elem):
        """Dibuja una neurona LIF (círculo)."""
        x, y = elem.x, elem.y
        color = get_qcolor('selected' if elem.selected
                           else ('hover' if elem.hover else 'neuron'))

        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(get_qcolor('bg_panel')))
        painter.drawEllipse(QPointF(x, y), 35, 35)

        painter.setFont(FONTS['label'])
        painter.setPen(get_qcolor('text'))
        painter.drawText(QRectF(x - 30, y - 22, 60, 20), Qt.AlignCenter,
                         elem.element_id)

        painter.setFont(FONTS['small'])
        painter.setPen(get_qcolor('text_dim'))
        vm = float(elem.params.get('V_m', 0.0))
        spikes = int(elem.params.get('spike_count', 0))
        painter.drawText(QRectF(x - 30, y - 2, 60, 16), Qt.AlignCenter,
                         f"{vm:.2f}V")
        painter.drawText(QRectF(x - 30, y + 12, 60, 16), Qt.AlignCenter,
                         f"⚡ {spikes}")

    def _draw_actuator(self, painter: QPainter, elem):
        """Dibuja un actuador (rectángulo)."""
        x, y = elem.x, elem.y
        color = get_qcolor('selected' if elem.selected
                           else ('hover' if elem.hover else 'actuator'))

        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(get_qcolor('bg_panel')))
        painter.drawRoundedRect(QRectF(x - 45, y - 30, 90, 60), 8, 8)

        painter.setFont(FONTS['label'])
        painter.setPen(get_qcolor('text'))
        painter.drawText(QRectF(x - 40, y - 25, 80, 20), Qt.AlignCenter,
                         elem.element_id)

        painter.setFont(FONTS['small'])
        painter.setPen(get_qcolor('text_dim'))
        act_name = str(elem.params.get('action', 'Girar'))
        painter.drawText(QRectF(x - 40, y, 80, 20), Qt.AlignCenter, act_name)

    # ================================================================
    # HOOKS PARA SUBCLASES
    # ================================================================

    def _draw_wires(self, painter: QPainter):
        """Override obligatorio en subclases: dibuja las conexiones."""
        raise NotImplementedError

    def _draw_extra(self, painter: QPainter):
        """Override opcional: overlays, decoders, ecuaciones."""
        pass

    # ================================================================
    # paintEvent estándar
    # ================================================================

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Aplicar pan/zoom
        painter.save()
        painter.translate(self.pan_x, self.pan_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        # Capas
        self._draw_grid(painter)
        self._draw_wires(painter)

        # Elementos (por tipo)
        for elem in self.elements.values():
            cls_name = type(elem).__name__
            if cls_name == 'SensorElement':
                self._draw_sensor(painter, elem)
            elif cls_name == 'MemristorElement':
                self._draw_memristor(painter, elem)
            elif cls_name == 'VolatileMemristorElement':
                self._draw_volatile_memristor(painter, elem)
            elif cls_name == 'NeuronElement':
                self._draw_neuron(painter, elem)
            elif cls_name == 'ActuatorElement':
                self._draw_actuator(painter, elem)

        self._draw_extra(painter)
        painter.restore()
