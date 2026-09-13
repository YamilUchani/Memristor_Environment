from PySide6.QtWidgets import QWidget, QHBoxLayout, QDoubleSpinBox, QSpinBox, QPushButton, QComboBox, QAbstractSpinBox
from PySide6.QtCore import Signal, Qt, QObject, QEvent


class FocusDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox que ignora la rueda del ratón a menos que tenga el foco activo (clic previo)."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class FocusSpinBox(QSpinBox):
    """QSpinBox que ignora la rueda del ratón a menos que tenga el foco activo (clic previo)."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class FocusComboBox(QComboBox):
    """QComboBox que ignora la rueda del ratón a menos que tenga el foco activo (clic previo)."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoUnfocusedWheelEventFilter(QObject):
    """
    Filtro de eventos Qt global que evita que cualquier control numérico o desplegable
    cambie de valor al girar la rueda del ratón por encima sin haber hecho clic primero.
    """
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if isinstance(obj, (QAbstractSpinBox, QComboBox, ArrowDoubleSpinBox, ArrowSpinBox)):
                has_focus = obj.hasFocus() or (hasattr(obj, 'spin') and obj.spin.hasFocus())
                if not has_focus:
                    event.ignore()
                    return True
        return super().eventFilter(obj, event)


class ArrowDoubleSpinBox(QWidget):
    """
    Control numérico flotante con botones independientes ▲ y ▼ fuera de la barra de texto.
    Solo responde a la rueda del ratón si primero ha sido enfocado con un clic.
    """
    valueChanged = Signal(float)

    def __init__(self, value=1.0, min_val=0.0, max_val=100.0, step=0.1, decimals=2, suffix="", parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Spinbox con protección de foco contra scroll no deseado
        self.spin = FocusDoubleSpinBox()
        self.spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.spin.setRange(min_val, max_val)
        self.spin.setDecimals(decimals)
        self.spin.setValue(value)
        self.spin.setSingleStep(step)
        if suffix:
            self.spin.setSuffix(suffix)

        # Botón ARRIBA ▲ fuera de la barra
        self.btn_up = QPushButton("▲")
        self.btn_up.setFixedSize(26, 26)
        self.btn_up.setCursor(Qt.PointingHandCursor)
        self.btn_up.setFocusPolicy(Qt.NoFocus)
        self.btn_up.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #89b4fa;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #89b4fa;
                color: #11111b;
            }
            QPushButton:pressed {
                background-color: #b4befe;
            }
        """)

        # Botón ABAJO ▼ fuera de la barra
        self.btn_down = QPushButton("▼")
        self.btn_down.setFixedSize(26, 26)
        self.btn_down.setCursor(Qt.PointingHandCursor)
        self.btn_down.setFocusPolicy(Qt.NoFocus)
        self.btn_down.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #89b4fa;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #89b4fa;
                color: #11111b;
            }
            QPushButton:pressed {
                background-color: #b4befe;
            }
        """)

        layout.addWidget(self.spin, stretch=1)
        layout.addWidget(self.btn_up)
        layout.addWidget(self.btn_down)
        self.setLayout(layout)

        self.btn_up.clicked.connect(self._step_up)
        self.btn_down.clicked.connect(self._step_down)
        self.spin.valueChanged.connect(self.valueChanged.emit)

    def wheelEvent(self, event):
        if self.spin.hasFocus():
            self.spin.wheelEvent(event)
        else:
            event.ignore()

    def _step_up(self):
        new_val = min(self.max_val, self.spin.value() + self.step)
        self.spin.setValue(new_val)

    def _step_down(self):
        new_val = max(self.min_val, self.spin.value() - self.step)
        self.spin.setValue(new_val)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, val: float):
        self.spin.setValue(val)


class ArrowSpinBox(QWidget):
    """
    Control numérico entero con botones independientes ▲ y ▼ fuera de la barra de texto.
    Solo responde a la rueda del ratón si primero ha sido enfocado con un clic.
    """
    valueChanged = Signal(int)

    def __init__(self, value=1, min_val=0, max_val=100, step=1, suffix="", parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.spin = FocusSpinBox()
        self.spin.setButtonSymbols(QSpinBox.NoButtons)
        self.spin.setRange(min_val, max_val)
        self.spin.setValue(value)
        self.spin.setSingleStep(step)
        if suffix:
            self.spin.setSuffix(suffix)

        self.btn_up = QPushButton("▲")
        self.btn_up.setFixedSize(26, 26)
        self.btn_up.setCursor(Qt.PointingHandCursor)
        self.btn_up.setFocusPolicy(Qt.NoFocus)
        self.btn_up.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #89b4fa;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #89b4fa;
                color: #11111b;
            }
        """)

        self.btn_down = QPushButton("▼")
        self.btn_down.setFixedSize(26, 26)
        self.btn_down.setCursor(Qt.PointingHandCursor)
        self.btn_down.setFocusPolicy(Qt.NoFocus)
        self.btn_down.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #89b4fa;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #89b4fa;
                color: #11111b;
            }
        """)

        layout.addWidget(self.spin, stretch=1)
        layout.addWidget(self.btn_up)
        layout.addWidget(self.btn_down)
        self.setLayout(layout)

        self.btn_up.clicked.connect(self._step_up)
        self.btn_down.clicked.connect(self._step_down)
        self.spin.valueChanged.connect(self.valueChanged.emit)

    def wheelEvent(self, event):
        if self.spin.hasFocus():
            self.spin.wheelEvent(event)
        else:
            event.ignore()

    def _step_up(self):
        new_val = min(self.max_val, self.spin.value() + self.step)
        self.spin.setValue(new_val)

    def _step_down(self):
        new_val = max(self.min_val, self.spin.value() - self.step)
        self.spin.setValue(new_val)

    def value(self) -> int:
        return self.spin.value()

    def setValue(self, val: int):
        self.spin.setValue(val)
