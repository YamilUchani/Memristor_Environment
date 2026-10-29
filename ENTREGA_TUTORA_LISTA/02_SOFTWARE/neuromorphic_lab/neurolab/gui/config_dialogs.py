"""
neurolab.gui.config_dialogs
===========================
Diálogo de configuración PySide6 (Qt6) para cualquier elemento del crossbar.
"""

from typing import Dict, Any, Callable
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel,
    QLineEdit, QDoubleSpinBox, QComboBox, QCheckBox, QPushButton, QScrollArea,
    QWidget, QMessageBox
)
from PySide6.QtCore import Qt
from neurolab.gui.styles import COLORS


MEMRISTOR_PRESETS = {
    'strukov': {
        'model': 'strukov',
        'reference': 'Strukov et al., Nature 453, 2008',
        'is_volatile': False,
        'volatile_tau_relax': 0.5,
        'RON': 100.0,
        'ROFF': 16000.0,
        'x0': 0.10,
        'mu_v': 1e-14,
        'clip_x': True,
        'window_type': 'Sin Ventana',
        'window_p': 2.0,
    },
    'hfo2_volatile': {
        'model': 'hfo2_volatile',
        'reference': 'Wang et al., CMOS LIF 2025',
        'is_volatile': True,
        'volatile_tau_relax': 0.5,
        'RON': 1000.0,
        'ROFF': 1000000.0,
        'x0': 0.05,
        'mu_v': 1e-14,
        'clip_x': True,
        'window_type': 'Biolek',
        'window_p': 2.0,
    },
    'prezioso': {
        'model': 'prezioso',
        'reference': 'Prezioso et al., Nature 518, 2014',
        'is_volatile': False,
        'volatile_tau_relax': 0.5,
        'RON': 4750.0,
        'ROFF': 1000000.0,
        'x0': 0.05,
        'mu_v': 1e-12,
        'clip_x': True,
        'window_type': 'Biolek',
        'window_p': 1.0,
    }
}


class ConfigDialog(QDialog):
    """Diálogo modal de configuración de un elemento del crossbar."""

    def __init__(self, parent: QWidget, element: Any, on_save: Callable = None):
        super().__init__(parent)
        self.element = element
        self.on_save = on_save
        self.config_data = element.get_config()

        self.setWindowTitle(self.config_data['title'])
        self.resize(520, 640)
        self.setModal(True)

        self.vars = {}  # Mapeo de campos a widgets/variables

        self._apply_style()
        self._build_ui()

    def _apply_style(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['text']};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QGroupBox {{
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #89b4fa;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }}
            QLabel {{
                color: {COLORS['text_label']};
            }}
            QLineEdit, QDoubleSpinBox, QComboBox {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text']};
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
            }}
            QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid #89b4fa;
            }}
            QCheckBox {{
                color: {COLORS['text']};
            }}
            QPushButton {{
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
        """)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # Título
        lbl_title = QLabel(self.config_data['title'])
        lbl_title.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {COLORS['text']}; margin-bottom: 5px;")
        main_layout.addWidget(lbl_title)

        # Presets Rápidos si es un Memristor
        has_model = any(
            any(field.get('key') == 'model' for field in sec.get('fields', []))
            for sec in self.config_data.get('sections', [])
        )
        if has_model:
            preset_bar = QHBoxLayout()
            lbl_p = QLabel("Presets:")
            lbl_p.setStyleSheet("font-weight: bold; color: #cdd6f4; font-size: 11px;")
            preset_bar.addWidget(lbl_p)

            btn_strukov = QPushButton("⚡ Strukov 2008")
            btn_strukov.setToolTip("R_ON=100Ω, R_OFF=16kΩ, NO volátil")
            btn_strukov.setStyleSheet("""
                QPushButton { background-color: #89b4fa; color: #11111b; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 11px; }
                QPushButton:hover { background-color: #b4befe; }
            """)
            btn_strukov.clicked.connect(lambda: self.apply_memristor_preset('strukov'))
            preset_bar.addWidget(btn_strukov)

            btn_hfo2 = QPushButton("🧠 HfO₂ Volátil")
            btn_hfo2.setToolTip("R_ON=1kΩ, R_OFF=1MΩ, Volátil τ=0.5s")
            btn_hfo2.setStyleSheet("""
                QPushButton { background-color: #a6e3a1; color: #11111b; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 11px; }
                QPushButton:hover { background-color: #94e2d5; }
            """)
            btn_hfo2.clicked.connect(lambda: self.apply_memristor_preset('hfo2_volatile'))
            preset_bar.addWidget(btn_hfo2)

            btn_prezioso = QPushButton("📊 Prezioso 2014")
            btn_prezioso.setToolTip("R_ON=4.75kΩ, R_OFF=1MΩ, NO volátil")
            btn_prezioso.setStyleSheet("""
                QPushButton { background-color: #f9e2af; color: #11111b; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 11px; }
                QPushButton:hover { background-color: #fab387; }
            """)
            btn_prezioso.clicked.connect(lambda: self.apply_memristor_preset('prezioso'))
            preset_bar.addWidget(btn_prezioso)

            preset_bar.addStretch()
            main_layout.addLayout(preset_bar)

        # Áreas de desplazamiento para los campos
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        for section in self.config_data['sections']:
            self._build_section(container_layout, section)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        # Botones de Acción
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text']};
                border: 1px solid #45475a;
            }}
            QPushButton:hover {{
                background-color: #45475a;
            }}
        """)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        btn_save = QPushButton("Guardar")
        btn_save.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: #11111b;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #94e2d5;
            }}
        """)
        btn_save.clicked.connect(self._on_save)
        btn_layout.addWidget(btn_save)

        main_layout.addLayout(btn_layout)

    def apply_memristor_preset(self, preset_key: str):
        """Aplica los parámetros físicos de un preset a todos los controles del diálogo."""
        if preset_key not in MEMRISTOR_PRESETS:
            return
        data = MEMRISTOR_PRESETS[preset_key]
        for key, val in data.items():
            if key in self.vars:
                widget, ftype, field = self.vars[key]
                widget.blockSignals(True)
                if ftype == 'combo':
                    widget.setCurrentText(str(val))
                elif ftype in ('text', 'readonly'):
                    widget.setText(str(val))
                elif ftype == 'check':
                    widget.setChecked(bool(val))
                elif ftype in ('float', 'int'):
                    widget.setValue(float(val))
                elif ftype == 'sci':
                    widget.setText(f"{float(val):.2e}")
                widget.blockSignals(False)

    def _on_model_combo_changed(self, text: str):
        """Reacciona al cambio en el combo box de Modelo Matemático Base cargando sus valores por defecto."""
        text_clean = str(text).lower()
        if 'prezioso' in text_clean:
            self.apply_memristor_preset('prezioso')
        elif 'hfo' in text_clean or 'volatil' in text_clean:
            self.apply_memristor_preset('hfo2_volatile')
        elif 'strukov' in text_clean:
            self.apply_memristor_preset('strukov')

    def _build_section(self, parent_layout: QVBoxLayout, section: Dict[str, Any]):
        group = QGroupBox(section['name'])
        form = QFormLayout(group)
        form.setContentsMargins(10, 15, 10, 10)
        form.setSpacing(10)

        for field in section['fields']:
            self._build_field(form, field)

        parent_layout.addWidget(group)

    def _build_field(self, form: QFormLayout, field: Dict[str, Any]):
        key = field['key']
        label = QLabel(field['label'])
        ftype = field['type']

        if ftype == 'text':
            widget = QLineEdit(str(field['value']))
            form.addRow(label, widget)

        elif ftype == 'int':
            widget = QDoubleSpinBox()
            widget.setDecimals(0)
            if 'min' in field:
                widget.setMinimum(float(field['min']))
            if 'max' in field:
                widget.setMaximum(float(field['max']))
            if 'step' in field:
                widget.setSingleStep(float(field['step']))
            widget.setValue(float(field['value']))
            form.addRow(label, widget)

        elif ftype == 'float':
            widget = QDoubleSpinBox()
            widget.setDecimals(4)
            if 'min' in field:
                widget.setMinimum(float(field['min']))
            else:
                widget.setMinimum(-1e9)
            if 'max' in field:
                widget.setMaximum(float(field['max']))
            else:
                widget.setMaximum(1e9)
            if 'step' in field:
                widget.setSingleStep(float(field['step']))
            widget.setValue(float(field['value']))
            form.addRow(label, widget)

        elif ftype == 'sci':
            widget = QLineEdit(f"{field['value']:.2e}")
            form.addRow(label, widget)

        elif ftype == 'combo':
            widget = QComboBox()
            widget.addItems(field['values'])
            widget.setCurrentText(str(field['value']))
            form.addRow(label, widget)
            if key == 'model':
                widget.currentTextChanged.connect(self._on_model_combo_changed)

        elif ftype == 'check':
            widget = QCheckBox()
            widget.setChecked(bool(field['value']))
            if field.get('readonly'):
                widget.setEnabled(False)
            form.addRow(label, widget)

        elif ftype == 'readonly':
            widget = QLineEdit(str(field['value']))
            widget.setReadOnly(True)
            widget.setStyleSheet(f"background-color: {COLORS['bg_dark']}; color: {COLORS['text_dim']};")
            form.addRow(label, widget)

        else:
            widget = QLineEdit(str(field['value']))
            form.addRow(label, widget)

        if 'help' in field:
            lbl_help = QLabel(f"ℹ {field['help']}")
            lbl_help.setStyleSheet(f"font-size: 10px; color: {COLORS['text_dim']};")
            form.addRow("", lbl_help)

        self.vars[key] = (widget, ftype, field)

    def _on_save(self):
        new_params = {}
        for key, (widget, ftype, field) in self.vars.items():
            try:
                if ftype == 'text':
                    new_params[key] = widget.text()
                elif ftype in ('float', 'int'):
                    new_params[key] = float(widget.value())
                elif ftype == 'sci':
                    new_params[key] = float(widget.text())
                elif ftype == 'combo':
                    new_params[key] = widget.currentText()
                elif ftype == 'check':
                    new_params[key] = bool(widget.isChecked())
                elif ftype == 'readonly':
                    continue
            except Exception as e:
                QMessageBox.critical(self, "Error de Validación", f"Valor inválido en {field['label']}: {e}")
                return

        # Conversiones de unidades según el nombre de parámetro
        if 'D_nm' in new_params:
            new_params['D'] = new_params['D_nm'] * 1e-9  # nm → m
        elif 'D' in new_params and new_params['D'] > 1e-6:
            new_params['D'] = new_params['D'] * 1e-9  # nm → m

        if 'RON_kOm' in new_params:
            new_params['RON'] = new_params['RON_kOm'] * 1e3  # kΩ → Ω
        if 'ROFF_kOm' in new_params:
            new_params['ROFF'] = new_params['ROFF_kOm'] * 1e3  # kΩ → Ω

        if 'C_m_nF' in new_params:
            new_params['C_m'] = new_params['C_m_nF'] * 1e-9  # nF → F
        elif 'C_m' in new_params and new_params['C_m'] > 1e-6:
            new_params['C_m'] = new_params['C_m'] * 1e-9  # nF → F

        if 'R_series_kOm' in new_params:
            new_params['R_series'] = new_params['R_series_kOm'] * 1e3  # kΩ → Ω

        if 'R_leak_kOm' in new_params:
            new_params['R_leak'] = new_params['R_leak_kOm'] * 1e3  # kΩ → Ω
        elif 'R_leak' in new_params and 1.0 < new_params['R_leak'] <= 100000.0 and 'R_leak_kOm' not in new_params:
            new_params['R_leak'] = new_params['R_leak'] * 1e3  # kΩ → Ω

        if 'tau_ref_ms' in new_params:
            new_params['tau_ref'] = new_params['tau_ref_ms'] * 1e-3  # ms → s
        elif 'tau_ref' in new_params and new_params['tau_ref'] > 0.01:
            new_params['tau_ref'] = new_params['tau_ref'] * 1e-3  # ms → s

        if 'dt_us' in new_params:
            new_params['dt'] = new_params['dt_us'] * 1e-6  # μs → s
        elif 'dt' in new_params and new_params['dt'] > 0.01:
            new_params['dt'] = new_params['dt'] * 1e-6  # μs → s

        self.element.update_params(new_params)

        if self.on_save:
            self.on_save(self.element)

        self.accept()


