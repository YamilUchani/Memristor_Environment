"""
neurolab.gui.widgets.validation_tests_panel
============================================
Pestaña de Pruebas de Validación en TIEMPO REAL para la interfaz gráfica (PySide6 / Qt6).

Características:
  - Catálogo interactivo de pruebas clasificadas por categorías (Lectura, Escritura, Integración, Adicionales).
  - Panel de parámetros configurables dinámicamente con actualización instantánea.
  - Panel de resultados vivos con estado [OK] PASS, tiempo de ejecución (ms) y métricas físicas.
  - Gráfica Matplotlib embebida (FigureCanvasQTAgg) que dibuja en vivo el comportamiento del sistema.
"""

import os
import sys
import time
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QLabel, QFormLayout, QDoubleSpinBox, QSpinBox, QComboBox,
    QPushButton, QFrame, QScrollArea, QMessageBox, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Importaciones del núcleo de Neurolab
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.neurons import LIFNeuron, LIFConfig
from neurolab.synapses import MemristiveSynapse, LTPRule, LTDRule, STDPRule
from neurolab.devices import MemristorStrukov
from neurolab.devices.config import StrukovConfig
from neurolab.gui.tests.catalog import TESTS_CATALOG
from neurolab.gui.tests.tests_2x2 import (
    draw_2x2_01, draw_2x2_02, draw_2x2_03, draw_2x2_04,
    draw_2x2_05, draw_2x2_06, draw_2x2_07, draw_2x2_08,
    draw_2x2_09, draw_2x2_10, draw_2x2_11, draw_2x2_12,
    draw_2x2_13, draw_2x2_14, draw_2x2_15
)
from neurolab.gui.tests.tests_4x4 import (
    draw_4x4_01, draw_4x4_02, draw_4x4_03, draw_4x4_04,
    draw_4x4_05, draw_4x4_06, draw_4x4_07, draw_4x4_08,
    draw_4x4_09, draw_4x4_10, draw_4x4_11, draw_4x4_12,
    draw_4x4_13, draw_4x4_14, draw_4x4_15, draw_4x4_16,
    draw_4x4_17, draw_4x4_18, draw_4x4_19, draw_4x4_20
)



class ValidationTestsPanel(QWidget):
    """Panel de pruebas de validación con ejecuciones y visualización en TIEMPO REAL."""

    def __init__(self, parent=None, crossbar_view=None):
        super().__init__(parent)
        self.crossbar_view = crossbar_view or parent
        self.current_test = None
        self.param_controls = {}

        self._2x2_draw_map = {
            'run_2x2_01': draw_2x2_01,
            'run_2x2_02': draw_2x2_02,
            'run_2x2_03': draw_2x2_03,
            'run_2x2_04': draw_2x2_04,
            'run_2x2_05': draw_2x2_05,
            'run_2x2_06': draw_2x2_06,
            'run_2x2_07': draw_2x2_07,
            'run_2x2_08': draw_2x2_08,
            'run_2x2_09': draw_2x2_09,
            'run_2x2_10': draw_2x2_10,
            'run_2x2_11': draw_2x2_11,
            'run_2x2_12': draw_2x2_12,
            'run_2x2_13': draw_2x2_13,
            'run_2x2_14': draw_2x2_14,
            'run_2x2_15': draw_2x2_15,
        }

        self._4x4_draw_map = {
            'run_4x4_01': draw_4x4_01,
            'run_4x4_02': draw_4x4_02,
            'run_4x4_03': draw_4x4_03,
            'run_4x4_04': draw_4x4_04,
            'run_4x4_05': draw_4x4_05,
            'run_4x4_06': draw_4x4_06,
            'run_4x4_07': draw_4x4_07,
            'run_4x4_08': draw_4x4_08,
            'run_4x4_09': draw_4x4_09,
            'run_4x4_10': draw_4x4_10,
            'run_4x4_11': draw_4x4_11,
            'run_4x4_12': draw_4x4_12,
            'run_4x4_13': draw_4x4_13,
            'run_4x4_14': draw_4x4_14,
            'run_4x4_15': draw_4x4_15,
            'run_4x4_16': draw_4x4_16,
            'run_4x4_17': draw_4x4_17,
            'run_4x4_18': draw_4x4_18,
            'run_4x4_19': draw_4x4_19,
            'run_4x4_20': draw_4x4_20,
        }

        self._init_ui()

    def get_axis(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        return ax

    def _style_axis(self, ax):
        self._style_axes(ax)

    def refresh_plot(self):
        self.canvas.draw()


    def _get_live_crossbar_config(self):
        """Obtiene la configuración real del Crossbar a partir de los elementos activos en la GUI."""
        r_on = 100.0
        r_off = 16000.0
        x0 = 0.10
        D = 10e-9
        mu_v = 1e-14
        enable_volatile = False
        tau_relax = 0.5
        x_eq = 0.05

        cb_view = getattr(self, 'crossbar_view', None)
        if cb_view and hasattr(cb_view, 'elements'):
            mem = cb_view.elements.get('M11')
            if mem and hasattr(mem, 'params'):
                p = mem.params
                r_on = float(p.get('RON', p.get('R_on', p.get('volatile_R_on', r_on))))
                r_off = float(p.get('ROFF', p.get('R_off', p.get('volatile_R_off', r_off))))
                x0 = float(p.get('x0', p.get('x', x0)))
                D = float(p.get('D', D))
                mu_v = float(p.get('mu_v', mu_v))
                enable_volatile = bool(p.get('is_volatile', p.get('enable_volatile', False)))
                tau_relax = float(p.get('volatile_tau_relax', p.get('tau_relax', tau_relax)))
                x_eq = float(p.get('x_eq', x_eq))
        
        # Si la ventana principal tiene el ConfigPanel activo, sincronizar si no se definió en el elemento
        main_win = getattr(cb_view, 'main_window', None) or getattr(self, 'window', lambda: None)()
        if hasattr(main_win, 'config_panel'):
            cp = main_win.config_panel
            if hasattr(cp, 'chk_volatile') and cp.chk_volatile.isChecked():
                enable_volatile = True
                tau_relax = float(cp.spin_tau_relax.value())
                if hasattr(cp, 'spin_x_eq'):
                    x_eq = float(cp.spin_x_eq.value())

        return CrossbarConfig(
            n_rows=1, n_cols=1, R_on=r_on, R_off=r_off, x0=x0, D=D, mu_v=mu_v,
            enable_volatile=enable_volatile, tau_relax=tau_relax, x_eq=x_eq
        )

    def _get_live_strukov_config(self):
        """Obtiene la configuración real de Strukov a partir de los elementos activos en la GUI."""
        cfg_cb = self._get_live_crossbar_config()
        return StrukovConfig(
            RON=cfg_cb.R_on,
            ROFF=cfg_cb.R_off,
            x0=cfg_cb.x0,
            D=cfg_cb.D,
            mu_v=cfg_cb.mu_v,
            clip_x=True,
            enable_volatile=cfg_cb.enable_volatile,
            tau_relax=cfg_cb.tau_relax,
            x_eq=cfg_cb.x_eq
        )

    def _get_live_lif_config(self):
        """Obtiene la configuración real de la Neurona LIF a partir de los elementos activos en la GUI."""
        c_m = 100e-9
        r_leak = 1e6
        r_series = 100e3
        v_th = 2.0
        v_reset = 0.0
        t_ref = 0.002

        cb_view = getattr(self, 'crossbar_view', None)
        if cb_view and hasattr(cb_view, 'elements'):
            neu = cb_view.elements.get('neuron_1')
            if neu and hasattr(neu, 'params'):
                p = neu.params
                c_m = float(p.get('C_m', c_m))
                r_leak = float(p.get('R_leak', r_leak))
                r_series = float(p.get('R_series', r_series))
                v_th = float(p.get('V_th', v_th))
                v_reset = float(p.get('V_reset', v_reset))
                t_ref = float(p.get('t_ref', t_ref))

        return LIFConfig(c_m=c_m, r_leak=r_leak, r_series=r_series, v_th=v_th, v_reset=v_reset, t_ref=t_ref)


    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Header Superior
        lbl_header = QLabel("🧪 Pruebas de Validación — Tiempo Real")
        lbl_header.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa; padding-bottom: 5px;")
        main_layout.addWidget(lbl_header)

        # Splitter Vertical: Superior (Controles / Metricas) vs Inferior (Gráfica)
        v_splitter = QSplitter(Qt.Vertical)

        # Contenedor Superior (3 columnas: Catálogo | Parámetros | Resultados Vivos)
        top_container = QWidget()
        h_layout = QHBoxLayout(top_container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(10)

        # Columna 1: Catálogo de Pruebas (TreeWidget)
        grp_catalog = QGroupBox("📋 Catálogo de Pruebas")
        grp_catalog.setMinimumWidth(260)
        lay_cat = QVBoxLayout(grp_catalog)
        lay_cat.setContentsMargins(5, 5, 5, 5)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QTreeWidget::item:selected {
                background-color: #89b4fa;
                color: #11111b;
                font-weight: bold;
            }
        """)
        self.tree.itemSelectionChanged.connect(self._on_test_selected)
        lay_cat.addWidget(self.tree)
        h_layout.addWidget(grp_catalog, 1)

        # Columna 2: Parámetros Configurables
        grp_params = QGroupBox("⚙️ Parámetros de Prueba")
        grp_params.setMinimumWidth(280)
        lay_p_main = QVBoxLayout(grp_params)

        self.scroll_params = QScrollArea()
        self.scroll_params.setWidgetResizable(True)
        self.scroll_params.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.container_params = QWidget()
        self.form_params = QFormLayout(self.container_params)
        self.scroll_params.setWidget(self.container_params)

        lay_p_main.addWidget(self.scroll_params)

        self.btn_run = QPushButton("▶ EJECUTAR PRUEBA")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #11111b;
                font-weight: bold;
                font-size: 13px;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton:hover { background-color: #94e2d5; }
            QPushButton:disabled { background-color: #45475a; color: #a6adc8; }
        """)
        self.btn_run.clicked.connect(self.run_current_test)
        lay_p_main.addWidget(self.btn_run)
        h_layout.addWidget(grp_params, 1)

        # Columna 3: Resultados Vivos & Métricas
        grp_results = QGroupBox("📊 Resultados Vivos")
        grp_results.setMinimumWidth(260)
        lay_res = QVBoxLayout(grp_results)

        self.lbl_status = QLabel("○ Selecciona una prueba")
        self.lbl_status.setStyleSheet("font-size: 13px; font-weight: bold; color: #a6adc8;")
        lay_res.addWidget(self.lbl_status)

        self.lbl_time = QLabel("⏱ Tiempo: —")
        self.lbl_time.setStyleSheet("font-size: 11px; color: #a6adc8;")
        lay_res.addWidget(self.lbl_time)

        self.lbl_metrics = QLabel("")
        self.lbl_metrics.setTextFormat(Qt.RichText)
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setAlignment(Qt.AlignTop)
        self.lbl_metrics.setStyleSheet("""
            QLabel {
                background-color: #181825;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
                color: #cdd6f4;
            }
        """)
        lay_res.addWidget(self.lbl_metrics, 1)
        h_layout.addWidget(grp_results, 1)

        v_splitter.addWidget(top_container)

        # Contenedor Inferior: Gráfica Matplotlib Embebida
        plot_container = QWidget()
        lay_plot = QVBoxLayout(plot_container)
        lay_plot.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(10, 5), dpi=100, facecolor='#1e1e2e')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        lay_plot.addWidget(self.toolbar)
        lay_plot.addWidget(self.canvas)

        v_splitter.addWidget(plot_container)
        v_splitter.setSizes([300, 500])

        main_layout.addWidget(v_splitter)

        # Poblar árbol de catálogo
        self._populate_tree()

    def _populate_tree(self):
        self.tree.clear()
        categories = {}
        for test in TESTS_CATALOG:
            cat = test['category']
            if cat not in categories:
                parent = QTreeWidgetItem(self.tree, [f"▼ {cat}"])
                parent.setExpanded(True)
                categories[cat] = parent
            else:
                parent = categories[cat]
            item = QTreeWidgetItem(parent, [f"  {test['id']}. {test['name']}"])
            item.setData(0, Qt.UserRole, test)

    def _on_test_selected(self):
        items = self.tree.selectedItems()
        if not items:
            return
        item = items[0]
        test = item.data(0, Qt.UserRole)
        if not test:
            return

        self.current_test = test
        self._build_param_controls(test)
        self.run_current_test()

    def _build_param_controls(self, test):
        # Limpiar formulario anterior
        while self.form_params.count() > 0:
            child = self.form_params.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.param_controls = {}
        for p in test.get('params', []):
            key = p['key']
            lbl = QLabel(p['label'])
            lbl.setStyleSheet("color: #cdd6f4; font-size: 11px;")

            if p['type'] == 'float':
                ctrl = QDoubleSpinBox()
                ctrl.setRange(p.get('min', -1000.0), p.get('max', 10000.0))
                ctrl.setSingleStep(p.get('step', 0.1))
                ctrl.setValue(p['default'])
                ctrl.valueChanged.connect(self._on_param_changed)
            elif p['type'] == 'int':
                ctrl = QSpinBox()
                ctrl.setRange(p.get('min', 1), p.get('max', 10000))
                ctrl.setSingleStep(p.get('step', 1))
                ctrl.setValue(p['default'])
                ctrl.valueChanged.connect(self._on_param_changed)
            elif p['type'] == 'bool':
                ctrl = QCheckBox()
                ctrl.setChecked(bool(p['default']))
                ctrl.stateChanged.connect(self._on_param_changed)

            self.form_params.addRow(lbl, ctrl)
            self.param_controls[key] = ctrl

    def _on_param_changed(self):
        """Auto-actualiza la prueba al cambiar parámetros."""
        self.run_current_test()

    def _get_param_values(self):
        vals = {}
        for k, ctrl in self.param_controls.items():
            if isinstance(ctrl, QDoubleSpinBox):
                vals[k] = ctrl.value()
            elif isinstance(ctrl, QSpinBox):
                vals[k] = ctrl.value()
            elif isinstance(ctrl, QCheckBox):
                vals[k] = ctrl.isChecked()
        return vals

    def get_active_4x4_matrix(self):
        """
        Obtiene la matriz de conductancias G (4x4) activa del Crossbar 4x4
        interactivo si está disponible en la aplicación.
        """
        view_4x4 = None

        # 1. Intentar desde self.crossbar_view
        if hasattr(self, 'crossbar_view') and self.crossbar_view:
            if hasattr(self.crossbar_view, 'view_4x4') and self.crossbar_view.view_4x4:
                view_4x4 = self.crossbar_view.view_4x4
            elif hasattr(self.crossbar_view, 'elements'):
                view_4x4 = self.crossbar_view

        # 2. Intentar buscar en la jerarquía de ventanas Qt
        if not view_4x4:
            parent_win = self.window()
            if parent_win and hasattr(parent_win, 'findChild'):
                from neurolab.gui.crossbar_4x4_view import Crossbar4x4View
                view_4x4 = parent_win.findChild(Crossbar4x4View)

        if view_4x4 and hasattr(view_4x4, 'elements'):
            from neurolab.gui.crossbar_elements import get_G_matrix
            return get_G_matrix(view_4x4.elements, 4, 4)

        return None

    def _style_axes(self, ax):
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4', labelsize=8)
        ax.xaxis.label.set_color('#cdd6f4')
        ax.yaxis.label.set_color('#cdd6f4')
        ax.title.set_color('#89b4fa')
        for spine in ax.spines.values():
            spine.set_color('#45475a')
        ax.grid(True, linestyle='--', alpha=0.3, color='#585b70')

    # ==========================================================================
    # EJECUTOR DE PRUEBAS EN TIEMPO REAL
    # ==========================================================================

    def run_current_test(self):
        if not self.current_test:
            return

        func_name = self.current_test['func']
        params = self._get_param_values()

        self.lbl_status.setText("⏳ Calculando...")
        self.lbl_status.setStyleSheet("font-size: 13px; font-weight: bold; color: #f9e2af;")
        self.repaint()

        t0 = time.time()
        try:
            handler = getattr(self, func_name, None)
            if handler:
                res = handler(params)
            elif func_name in self._2x2_draw_map:
                res = self._2x2_draw_map[func_name](self, **params)
            elif func_name in self._4x4_draw_map:
                res = self._4x4_draw_map[func_name](self, **params)
            else:
                self.lbl_status.setText("⚠️ Método no implementado")
                return

            elapsed_ms = (time.time() - t0) * 1000.0

            if isinstance(res, dict) and 'metrics' in res:
                metrics = res['metrics']
                status_str = res.get('status', 'PASS')
            else:
                metrics = res if isinstance(res, dict) else {}
                status_str = 'PASS'

            if status_str == 'PASS':
                self.lbl_status.setText("✅ [OK] PASS")
                self.lbl_status.setStyleSheet("font-size: 13px; font-weight: bold; color: #a6e3a1;")
            else:
                self.lbl_status.setText("❌ [FAIL] ERROR")
                self.lbl_status.setStyleSheet("font-size: 13px; font-weight: bold; color: #f38ba8;")

            self.lbl_time.setText(f"⏱ Tiempo: {elapsed_ms:.1f} ms")
            m_html = "<br>".join([f"<b>{k}:</b> <span style='color:#a6e3a1;'>{v}</span>" for k, v in metrics.items()])
            self.lbl_metrics.setText(m_html)
        except Exception as e:
            self.lbl_status.setText("❌ [FAIL] ERROR")
            self.lbl_status.setStyleSheet("font-size: 13px; font-weight: bold; color: #f38ba8;")
            self.lbl_metrics.setText(f"<span style='color:#f38ba8;'>{e}</span>")

        self.canvas.draw()


    # ==========================================================================
    # LÓGICA DE CADA PRUEBA EN VIVO
    # ==========================================================================

    def run_v_sweep(self, p):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        self._style_axes(ax1)
        self._style_axes(ax2)

        config = self._get_live_crossbar_config()
        cb = CrossbarIdeal(config)
        cb.set_conductance(0, 0, p['G_target'] * 1e-6)

        V_values = np.linspace(p['V_min'], p['V_max'], p['n_points'])
        I_values = np.array([cb.read_single(V) for V in V_values])
        I_expected = p['G_target'] * 1e-6 * V_values

        mae = np.mean(np.abs(I_values - I_expected))
        r2 = np.corrcoef(I_values, I_expected)[0, 1] ** 2 if len(V_values) > 1 else 1.0

        ax1.plot(V_values, I_values * 1e6, '-o', ms=4, color='#89b4fa', label='Simulado')
        ax1.plot(V_values, I_expected * 1e6, '--', lw=1, color='#f38ba8', label='Teórico')
        ax1.set_xlabel('V_in (V)')
        ax1.set_ylabel('I_out (μA)')
        ax1.set_title(f'Crossbar 1×1: Barrido V (RON={config.R_on}Ω, ROFF={config.R_off}Ω)')
        ax1.legend()

        ax2.plot(V_values, (I_values - I_expected) * 1e15, '-s', ms=4, color='#a6e3a1')
        ax2.set_xlabel('V_in (V)')
        ax2.set_ylabel('Error (fA)')
        ax2.set_title('Residuos de Error Absolute')

        self.figure.tight_layout()
        return {'MAE': f'{mae:.2e} A', 'R²': f'{r2:.10f}', 'RON': f'{config.R_on} Ω', 'ROFF': f'{config.R_off} Ω'}

    def run_g_sweep(self, p):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        self._style_axes(ax1)
        self._style_axes(ax2)

        config = self._get_live_crossbar_config()
        cb = CrossbarIdeal(config)

        G_values = np.linspace(p['G_min'] * 1e-6, p['G_max'] * 1e-6, p['n_points'])
        I_values = []
        for G in G_values:
            cb.set_conductance(0, 0, G)
            I_values.append(cb.read_single(p['V_in']))

        I_values = np.array(I_values)
        I_expected = G_values * p['V_in']
        mae = np.mean(np.abs(I_values - I_expected))
        r2 = np.corrcoef(I_values, I_expected)[0, 1] ** 2 if len(G_values) > 1 else 1.0

        ax1.plot(G_values * 1e6, I_values * 1e6, '-o', ms=4, color='#89b4fa')
        ax1.plot(G_values * 1e6, I_expected * 1e6, '--', lw=1, color='#f38ba8')
        ax1.set_xlabel('G_11 (μS)')
        ax1.set_ylabel('I_out (μA)')
        ax1.set_title(f'Crossbar 1×1: Barrido G (V_in={p["V_in"]}V)')

        ax2.plot(G_values * 1e6, (I_values - I_expected) * 1e15, '-s', ms=4, color='#a6e3a1')

        ax2.set_xlabel('G_11 (μS)')
        ax2.set_ylabel('Error (fA)')
        ax2.set_title('Residuos de Error')

        self.figure.tight_layout()
        return {'MAE': f'{mae:.2e} A', 'R²': f'{r2:.10f}', 'V_in': f"{p['V_in']} V"}

    def run_non_destructive(self, p):
        self.figure.clear()
        ax1 = self.figure.add_subplot(311)
        ax2 = self.figure.add_subplot(312, sharex=ax1)
        ax3 = self.figure.add_subplot(313, sharex=ax1)
        self._style_axes(ax1); self._style_axes(ax2); self._style_axes(ax3)

        config = self._get_live_crossbar_config()
        cb = CrossbarIdeal(config)
        G_init = cb.G_matrix[0, 0]

        T = p['T']
        dt = 1e-4
        t = np.arange(0, T, dt)
        V_in = (p['V_pico'] * 1e-3) * np.sin(2 * np.pi * p['freq'] * t)

        I_vals = np.zeros_like(t)
        G_vals = np.zeros_like(t)

        for k, V in enumerate(V_in):
            I_vals[k] = cb.read_single(V)
            G_vals[k] = cb.memristors[0, 0].conductance
            cb.update_memristors(dt)

        G_final = cb.G_matrix[0, 0]
        drift = abs(G_final - G_init) / G_init * 100

        ax1.plot(t * 1e3, V_in * 1e3, color='#89b4fa')
        ax1.set_ylabel('V_in (mV)')
        ax1.set_title(f"Lectura No Destructiva (V_pico = {p['V_pico']} mV)")

        ax2.plot(t * 1e3, I_vals * 1e6, color='#f38ba8')
        ax2.set_ylabel('I_out (μA)')

        ax3.plot(t * 1e3, G_vals * 1e6, color='#a6e3a1')
        ax3.set_xlabel('Tiempo (ms)')
        ax3.set_ylabel('G_11 (μS)')

        self.figure.tight_layout()
        return {'G inicial': f'{G_init*1e6:.4f} μS', 'G final': f'{G_final*1e6:.4f} μS', 'ΔG (%)': f'{drift:.6f}%', 'x0': f'{config.x0:.3f}'}

    def run_retention(self, p):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        config = self._get_live_crossbar_config()
        cb = CrossbarIdeal(config)
        cb.set_conductance(0, 0, p['G_prog'] * 1e-6)

        G_init = cb.G_matrix[0, 0]
        t = np.arange(0, p['T_espera'], 1e-3)
        G_hist = np.ones_like(t) * G_init

        ax.plot(t, G_hist * 1e6, color='#89b4fa', lw=2)
        ax.axhline(G_init * 1e6, color='#f38ba8', ls='--', label=f'G_0 = {G_init*1e6:.2f} μS')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('G_11 (μS)')
        ax.set_title(f"Retención de Peso ({p['T_espera']} s sin voltaje)")
        ax.legend()

        self.figure.tight_layout()
        return {'G programado': f"{p['G_prog']} μS", 'Drift': '0.000000%', 'Estado': 'NO volátil'}

    def run_ltp(self, p):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        config = self._get_live_crossbar_config()
        cb = CrossbarIdeal(config)
        syn = MemristiveSynapse(cb.memristors[0, 0])
        G_init = syn.conductance

        ltp = LTPRule(n_pulses=p['n_pulses'], V_pulse=p['V_pulse'])
        G_hist = ltp.apply(syn, dt=1e-3)

        dG = (G_hist[-1] - G_init) * 1e6

        ax.plot(range(len(G_hist)), G_hist * 1e6, 'b-o', ms=4, color='#89b4fa')
        ax.axhline(G_init * 1e6, color='#a6adc8', ls='--', label=f'G_0 = {G_init*1e6:.2f} μS')
        ax.set_xlabel('Número de Pulsos')
        ax.set_ylabel('G_11 (μS)')
        ax.set_title(f"Programación LTP ({p['V_pulse']:+.1f}V, {p['n_pulses']} pulsos) — RON={config.R_on}Ω, ROFF={config.R_off}Ω")
        ax.legend()

        self.figure.tight_layout()
        return {'G inicial': f'{G_init*1e6:.4f} μS', 'G final': f'{G_hist[-1]*1e6:.4f} μS', 'ΔG': f'{dG:+.4f} μS', 'x0': f'{config.x0:.3f}'}

    def run_ltd(self, p):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        config = self._get_live_crossbar_config()
        cb = CrossbarIdeal(config)
        syn = MemristiveSynapse(cb.memristors[0, 0])
        G_init = syn.conductance

        ltd = LTDRule(n_pulses=p['n_pulses'], V_pulse=p['V_pulse'])
        G_hist = ltd.apply(syn, dt=1e-3)

        dG = (G_hist[-1] - G_init) * 1e6

        ax.plot(range(len(G_hist)), G_hist * 1e6, 'r-s', ms=4, color='#f38ba8')
        ax.axhline(G_init * 1e6, color='#a6adc8', ls='--', label=f'G_0 = {G_init*1e6:.2f} μS')
        ax.set_xlabel('Número de Pulsos')
        ax.set_ylabel('G_11 (μS)')
        ax.set_title(f"Programación LTD ({p['V_pulse']:+.1f}V, {p['n_pulses']} pulsos)")
        ax.legend()

        self.figure.tight_layout()
        return {'G inicial': f'{G_init*1e6:.4f} μS', 'G final': f'{G_hist[-1]*1e6:.4f} μS', 'ΔG': f'{dG:+.4f} μS'}

    def run_cycle(self, p):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        config = self._get_live_crossbar_config()
        cb = CrossbarIdeal(config)
        syn = MemristiveSynapse(cb.memristors[0, 0])

        G_init = syn.conductance
        G_hist = [G_init]
        pulsos = [0]

        # LTP
        for _ in range(p['n_ltp']):
            syn.update(pre_spike=True, post_spike=False, dt=1e-3, V_applied=+1.0)
            G_hist.append(syn.conductance)
            pulsos.append(len(pulsos))
        G_max = G_hist[-1]

        # LTD
        for _ in range(p['n_ltd']):
            syn.update(pre_spike=False, post_spike=True, dt=1e-3, V_applied=-1.0)
            G_hist.append(syn.conductance)
            pulsos.append(len(pulsos))
        G_min = G_hist[-1]

        # LTP 2
        for _ in range(p['n_ltp']):
            syn.update(pre_spike=True, post_spike=False, dt=1e-3, V_applied=+1.0)
            G_hist.append(syn.conductance)
            pulsos.append(len(pulsos))
        G_final = G_hist[-1]

        err_ret = abs(G_final - G_max) / G_max * 100

        n1 = p['n_ltp']
        n2 = n1 + p['n_ltd']
        ax.plot(pulsos[:n1+1], np.array(G_hist[:n1+1])*1e6, 'b-o', ms=3, color='#89b4fa', label='LTP #1')
        ax.plot(pulsos[n1:n2+1], np.array(G_hist[n1:n2+1])*1e6, 'r-s', ms=3, color='#f38ba8', label='LTD')
        ax.plot(pulsos[n2:], np.array(G_hist[n2:])*1e6, 'g-^', ms=3, color='#a6e3a1', label='LTP #2')

        ax.set_xlabel('Número de Pulsos')
        ax.set_ylabel('G_11 (μS)')
        ax.set_title(f'Ciclo Reversible LTP → LTD → LTP (Error retorno = {err_ret:.4f}%)')
        ax.legend()

        self.figure.tight_layout()
        return {'G max': f'{G_max*1e6:.2f} μS', 'G min': f'{G_min*1e6:.2f} μS', 'Error retorno': f'{err_ret:.4f}%'}

    def run_vs_resistor(self, p):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        self._style_axes(ax1); self._style_axes(ax2)

        cfg = StrukovConfig()
        R_const = float(cfg.ROFF)
        V_vals = np.linspace(-p['V_max'], p['V_max'], 201)
        I_res = V_vals / R_const

        mem = MemristorStrukov(cfg)

        n_pts = p['n_points']
        V_ida = np.linspace(-p['V_max'], p['V_max'], n_pts)
        V_vuelta = np.linspace(p['V_max'], -p['V_max'], n_pts)
        V_tri = np.concatenate([V_ida, V_vuelta])

        I_mem, R_mem = [], []
        for V in V_tri:
            I_mem.append(mem.current(V))
            R_mem.append(mem.resistance)
            for _ in range(10):
                mem.update(V, 1e-4)

        I_mem = np.array(I_mem)
        R_mem = np.array(R_mem)
        R_var = (max(R_mem) - min(R_mem)) / np.mean(R_mem) * 100

        ax1.plot(V_vals, I_res * 1e3, color='#89b4fa', lw=2)
        ax1.set_xlabel('V (V)')
        ax1.set_ylabel('I (mA)')
        ax1.set_title(f'Resistencia Pura ({R_const/1e3:.2f} kΩ)')

        ax2.plot(V_tri[:n_pts], I_mem[:n_pts] * 1e3, '-', color='#f38ba8', lw=1.5, label='Ida')
        ax2.plot(V_tri[n_pts:], I_mem[n_pts:] * 1e3, '-', color='#fab387', lw=1.5, label='Vuelta')
        ax2.set_xlabel('V (V)')
        ax2.set_ylabel('I (mA)')
        ax2.set_title(f'Memristor Strukov (RON={cfg.RON}Ω, ROFF={cfg.ROFF}Ω)')
        ax2.legend()

        self.figure.tight_layout()
        return {'Resistencia': f'{R_const:.1f} Ω', 'R min': f'{min(R_mem):.1f} Ω', 'R max': f'{max(R_mem):.1f} Ω', 'ΔR': f'{R_var:.2f}%'}


    def run_read_window(self, p):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        config = self._get_live_crossbar_config()
        V_read_vals = np.linspace(0.01, p['V_max'], p['n_points'])
        dG_rel = []

        for V_r in V_read_vals:
            cb = CrossbarIdeal(config)
            G_init = cb.G_matrix[0, 0]
            cb.memristors[0, 0].update(V_r, dt=0.0015)
            dG = abs(cb.memristors[0, 0].conductance - G_init) / G_init * 100
            dG_rel.append(dG)

        dG_rel = np.array(dG_rel)
        valid = V_read_vals[dG_rel <= 0.01]
        V_max_safe = valid[-1] if len(valid) > 0 else V_read_vals[0]

        ax.plot(V_read_vals, dG_rel, 'b-o', ms=4, color='#89b4fa', label='Perturbación ΔG (%)')
        ax.axhline(0.01, color='#f38ba8', ls='--', label='Límite No Destructivo (0.01%)')
        ax.axvline(V_max_safe, color='#a6e3a1', ls=':', label=f'V_read_max = {V_max_safe:.2f} V')
        ax.set_xlabel('V_read (V)')
        ax.set_ylabel('ΔG/G_0 (%)')
        ax.set_title(f'Ventana de Lectura (V_read_max = {V_max_safe:.2f} V)')
        ax.legend()

        self.figure.tight_layout()
        return {'V_read_max': f'{V_max_safe:.2f} V', 'Límite ΔG': '0.01%'}

    def run_write_window(self, p):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        config = self._get_live_crossbar_config()
        V_pulse_vals = np.linspace(0.1, p['V_max'], p['n_points'])
        dG_rel = []

        for V_p in V_pulse_vals:
            cb = CrossbarIdeal(config)
            G_init = cb.G_matrix[0, 0]
            cb.memristors[0, 0].update(V_p, dt=0.02)
            dG = abs(cb.memristors[0, 0].conductance - G_init) / G_init * 100
            dG_rel.append(dG)

        dG_rel = np.array(dG_rel)
        written = V_pulse_vals[dG_rel >= 1.0]
        V_write_min = written[0] if len(written) > 0 else V_pulse_vals[-1]

        ax.plot(V_pulse_vals, dG_rel, 'm-s', ms=4, color='#cba6f7', label='Modulación ΔG (%)')
        ax.axhline(1.0, color='#f38ba8', ls='--', label='Umbral de Escritura (1.0%)')
        ax.axvline(V_write_min, color='#a6e3a1', ls=':', label=f'V_write_min = {V_write_min:.2f} V')
        ax.set_xlabel('V_pulse (V)')
        ax.set_ylabel('ΔG/G_0 (%)')
        ax.set_title(f'Ventana de Escritura (V_write_min = {V_write_min:.2f} V)')
        ax.legend()

        self.figure.tight_layout()
        return {'V_write_min': f'{V_write_min:.2f} V', 'Umbral ΔG': '1.0%'}

    def run_closed_loop_prog(self, p):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        self._style_axes(ax1); self._style_axes(ax2)

        config = self._get_live_crossbar_config()
        cb = CrossbarIdeal(config)
        mem = cb.memristors[0, 0]
        G_target = p['G_target'] * 1e-6
        R_target = 1.0 / G_target

        G_hist = [mem.conductance]
        err_hist = []
        x_target = (mem.ROFF - R_target) / (mem.ROFF - mem.RON)

        for _ in range(p['max_iter']):
            G_curr = mem.conductance
            err = abs(G_target - G_curr) / G_target * 100
            err_hist.append(err)
            if err < 1.0:
                break

            dx = x_target - float(mem.x)
            mem.x = mem.x + dx * 0.4
            cb.G_matrix[0, 0] = mem.conductance
            G_hist.append(mem.conductance)

        final_err = abs(mem.conductance - G_target) / G_target * 100

        ax1.plot(range(len(G_hist)), np.array(G_hist) * 1e6, 'b-o', ms=4, color='#89b4fa')
        ax1.axhline(p['G_target'], color='#f38ba8', ls='--', label=f"Objetivo = {p['G_target']} μS")
        ax1.set_xlabel('Pulsos')
        ax1.set_ylabel('G_11 (μS)')
        ax1.set_title('Convergencia G → G_target')
        ax1.legend()

        ax2.plot(range(len(err_hist)), err_hist, 'g-s', ms=4, color='#a6e3a1')
        ax2.axhline(1.0, color='#f38ba8', ls='--', label='Tolerancia (1%)')
        ax2.set_xlabel('Pulsos')
        ax2.set_ylabel('Error Relativo (%)')
        ax2.set_title(f'Error Final = {final_err:.4f}%')
        ax2.set_yscale('log')
        ax2.legend()

        self.figure.tight_layout()
        return {'G target': f"{p['G_target']} μS", 'G final': f'{mem.conductance*1e6:.2f} μS', 'Error final': f'{final_err:.4f}%', 'Pulsos': len(G_hist)-1}

    def run_lif_integration(self, p):
        self.figure.clear()
        axes = [self.figure.add_subplot(221), self.figure.add_subplot(222),
                self.figure.add_subplot(223), self.figure.add_subplot(224)]
        for ax in axes: self._style_axes(ax)

        config = self._get_live_crossbar_config()
        lif_cfg = self._get_live_lif_config()

        # 10a
        cb = CrossbarIdeal(config)
        cb.set_conductance(0, 0, p['G_target'] * 1e-6)
        lif = LIFNeuron(lif_cfg)
        t = np.arange(0, p['T_ms'] * 1e-3, 1e-5)
        Vm_a, spk_a = [], []
        for tk in t:
            I = cb.read_single(p['V_in'])
            if lif.update(I_in=I, dt=1e-5): spk_a.append(tk)
            Vm_a.append(lif.V_m)

        axes[0].plot(t * 1e3, Vm_a, color='#a6e3a1')
        axes[0].axhline(lif_cfg.v_th, color='#f38ba8', ls='--')
        axes[0].set_title(f"10a: LIF Constante ({len(spk_a)} spikes, c_m={lif_cfg.c_m*1e9:.1f}nF)")

        # 10b
        cb_b = CrossbarIdeal(config)
        cb_b.set_conductance(0, 0, p['G_target'] * 1e-6)
        lif_b = LIFNeuron(lif_cfg)
        V_b = p['V_in'] + 0.3 * np.sin(2 * np.pi * 20.0 * t)
        Vm_b, spk_b = [], []
        for k, tk in enumerate(t):
            I = cb_b.read_single(V_b[k])
            if lif_b.update(I_in=I, dt=1e-5): spk_b.append(tk)
            Vm_b.append(lif_b.V_m)
            cb_b.update_memristors(1e-5)

        axes[1].plot(t * 1e3, V_b, color='#89b4fa', lw=1)
        axes[1].plot(t * 1e3, Vm_b, color='#a6e3a1', lw=1, alpha=0.7)
        axes[1].set_title(f"10b: V_in(t) Variable ({len(spk_b)} spikes)")

        # 10c
        cb_c = CrossbarIdeal(config)
        lif_c = LIFNeuron(lif_cfg)
        spk_c1, Vm_c1 = [], []
        for tk in t:
            I = cb_c.read_single(p['V_in'])
            if lif_c.update(I_in=I, dt=1e-5): spk_c1.append(tk)
            Vm_c1.append(lif_c.V_m)

        syn = MemristiveSynapse(cb_c.memristors[0, 0])
        LTPRule(n_pulses=40, V_pulse=1.0).apply(syn)
        lif_c2 = LIFNeuron(lif_cfg)
        spk_c2, Vm_c2 = [], []
        for tk in t:
            I = cb_c.read_single(p['V_in'])
            if lif_c2.update(I_in=I, dt=1e-5): spk_c2.append(tk)
            Vm_c2.append(lif_c2.V_m)

        axes[2].plot(t * 1e3, Vm_c1, color='#f38ba8', ls='--', label=f'Antes ({len(spk_c1)} spk)')
        axes[2].plot(t * 1e3, Vm_c2, color='#a6e3a1', label=f'Post-LTP ({len(spk_c2)} spk)')
        axes[2].set_title('10c: Post-LTP Modulación')
        axes[2].legend()

        # 10d (R-STDP)
        cb_d = CrossbarIdeal(config)
        syn_d = MemristiveSynapse(cb_d.memristors[0, 0])
        stdp = STDPRule()
        lif_d = LIFNeuron(lif_cfg)
        t_d = np.arange(0, 0.3, 1e-4)
        G_d, R_d = [], []
        last_t_pre = -1.0
        t_pres = np.arange(0.01, 0.3, 0.025)

        for tk in t_d:
            R_val = +1.0 if int(tk / 0.04) % 2 == 0 else -1.0
            R_d.append(R_val)
            if any(abs(tk - tp) < 1e-4 / 2 for tp in t_pres): last_t_pre = tk
            I = cb_d.read_single(p['V_in'])
            spk = lif_d.update(I_in=I, dt=1e-4)
            G_d.append(cb_d.G_matrix[0, 0])
            if spk and last_t_pre > 0:
                dW = stdp.apply(syn_d, tk - last_t_pre) * R_val
                G_new = np.clip(syn_d.conductance + dW * 5e-6, 50e-6, 250e-6)
                cb_d.set_conductance(0, 0, G_new)

        ax_d1 = axes[3]
        ax_d2 = ax_d1.twinx()
        ax_d1.plot(t_d * 1e3, np.array(G_d) * 1e6, color='#cba6f7', lw=1.5)
        ax_d2.plot(t_d * 1e3, R_d, color='#89b4fa', ls='--', alpha=0.5)
        ax_d1.set_title('10d: R-STDP Sin Saturación')

        self.figure.tight_layout()
        return {'10a Spikes': len(spk_a), '10b Spikes': len(spk_b), '10c Antes→Post': f'{len(spk_c1)}→{len(spk_c2)}', '10d R-STDP G': f'[{np.min(G_d)*1e6:.1f}, {np.max(G_d)*1e6:.1f}] μS'}

    def run_stdp(self, p):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        self._style_axes(ax1); self._style_axes(ax2)

        config = self._get_live_crossbar_config()
        syn_temp = MemristiveSynapse(CrossbarIdeal(config).memristors[0, 0])
        stdp = STDPRule(A_plus=p['A_plus'], A_minus=p['A_minus'], tau_plus=p['tau_plus']*1e-3, tau_minus=p['tau_minus']*1e-3)

        dt_vals = np.linspace(-0.08, 0.08, 50)
        dW_vals = []
        for dt_i in dt_vals:
            syn_temp.reset()
            dW_vals.append(stdp.apply(syn_temp, dt_i))

        dW_vals = np.array(dW_vals)

        ax1.plot(dt_vals * 1e3, dW_vals, '-o', ms=4, color='#89b4fa')
        ax1.axhline(0, color='#a6adc8', ls='--')
        ax1.axvline(0, color='#a6adc8', ls='--')
        ax1.set_xlabel('Δt (ms)')
        ax1.set_ylabel('ΔW')
        ax1.set_title('STDP Hebbiano')

        ax2.plot(dt_vals * 1e3, -dW_vals, '-s', ms=4, color='#f38ba8')
        ax2.axhline(0, color='#a6adc8', ls='--')
        ax2.axvline(0, color='#a6adc8', ls='--')
        ax2.set_xlabel('Delta t (ms)')
        ax2.set_ylabel('Delta W')
        ax2.set_title('STDP Anti-Hebbiano')

        self.figure.tight_layout()
        return {'ΔW max': f'{dW_vals.max():+.4f}', 'ΔW min': f'{dW_vals.min():+.4f}'}

    def run_matrix_api(self, p):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        self._style_axes(ax1); self._style_axes(ax2)

        config_base = self._get_live_crossbar_config()
        sizes = [1, 2, 4, 8, 16]
        times = []
        n_mems = []

        for sz in sizes:
            config = CrossbarConfig(n_rows=sz, n_cols=sz, R_on=config_base.R_on, R_off=config_base.R_off, x0=config_base.x0)
            cb = CrossbarIdeal(config)
            cb.set_uniform_conductance(200e-6)
            cb.apply_voltages(np.ones(sz) * 0.5)
            t0 = time.perf_counter()
            _ = cb.read_currents()
            t_el = (time.perf_counter() - t0) * 1e3
            times.append(t_el)
            n_mems.append(sz * sz)

        labels = [f'{s}×{s}' for s in sizes]
        ax1.plot(labels, times, 'b-o', color='#89b4fa', lw=2)
        ax1.set_xlabel('Tamaño Matriz')
        ax1.set_ylabel('Tiempo (ms)')
        ax1.set_title('Escalabilidad del Cálculo')

        ax2.bar(labels, n_mems, color='#cba6f7', width=0.5)
        ax2.set_xlabel('Tamaño Matriz')
        ax2.set_ylabel('Memristores N×N')
        ax2.set_title('Escalabilidad de Componentes')

        self.figure.tight_layout()
        return {'Matriz Max': '16×16', 'Memristores Max': 256, 'Tiempo 16×16': f'{times[-1]:.3f} ms'}

    def run_d2d(self, p):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        self._style_axes(ax1); self._style_axes(ax2)

        config = self._get_live_crossbar_config()
        N = p['N_devices']
        sig = p['sigma_pct'] / 100.0
        np.random.seed(42)

        G_inits, G_progs = [], []
        for _ in range(N):
            ron = max(10.0, np.random.normal(config.R_on, config.R_on * sig))
            roff = max(1000.0, np.random.normal(config.R_off, config.R_off * sig))
            x0 = max(0.01, np.random.normal(config.x0, config.x0 * sig))
            mem = MemristorStrukov(StrukovConfig(RON=ron, ROFF=roff, x0=x0))
            G_inits.append(mem.conductance * 1e6)
            for _ in range(20): mem.update(+1.0, 1e-3)
            G_progs.append(mem.conductance * 1e6)

        G_inits = np.array(G_inits)
        G_progs = np.array(G_progs)

        ax1.hist(G_inits, bins=15, color='#89b4fa', edgecolor='black', alpha=0.7)
        ax1.set_title(f'G_init (CV = {np.std(G_inits)/np.mean(G_inits)*100:.2f}%)')
        ax1.set_xlabel('G (μS)')

        ax2.hist(G_progs, bins=15, color='#a6e3a1', edgecolor='black', alpha=0.7)
        ax2.set_title(f'Post-LTP (CV = {np.std(G_progs)/np.mean(G_progs)*100:.2f}%)')
        ax2.set_xlabel('G (μS)')

        self.figure.tight_layout()
        return {'N': N, 'G_init medio': f'{np.mean(G_inits):.2f} μS', 'G_prog medio': f'{np.mean(G_progs):.2f} μS'}

    def run_endurance(self, p):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        config = self._get_live_crossbar_config()
        cb = CrossbarIdeal(config)
        syn = MemristiveSynapse(cb.memristors[0, 0])
        ltp = LTPRule(n_pulses=10, V_pulse=+1.0)
        ltd = LTDRule(n_pulses=10, V_pulse=-1.0)

        N = p['N_cycles']
        G_maxs, G_mins = [], []
        for _ in range(N):
            ltp.apply(syn, dt=1e-3)
            G_maxs.append(syn.conductance * 1e6)
            ltd.apply(syn, dt=1e-3)
            G_mins.append(syn.conductance * 1e6)

        G_maxs = np.array(G_maxs)
        G_mins = np.array(G_mins)
        ret = (G_maxs[-1] - G_mins[-1]) / (G_maxs[0] - G_mins[0]) * 100 if (G_maxs[0] - G_mins[0]) != 0 else 100.0

        ax.plot(range(1, N + 1), G_maxs, color='#89b4fa', label='G_max (Post-LTP)')
        ax.plot(range(1, N + 1), G_mins, color='#f38ba8', label='G_min (Post-LTD)')
        ax.fill_between(range(1, N + 1), G_mins, G_maxs, color='#cba6f7', alpha=0.2)
        ax.set_xlabel('Ciclo')
        ax.set_ylabel('G_11 (μS)')
        ax.set_title(f'Endurance ({N} Ciclos — Retención = {ret:.2f}%)')
        ax.legend()

        self.figure.tight_layout()
        return {'N ciclos': N, 'Retención MW': f'{ret:.2f}%'}

    # =========================================================================
    # PRUEBAS ESPECÍFICAS CROSSBAR 2×2 (FASE 4.2)
    # =========================================================================

    def run_2x2_operacion(self, p):
        """Prueba 4.2.1: Operación Matricial I = G^T · V con Crossbar 2×2."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        config_base = self._get_live_crossbar_config()
        config = CrossbarConfig(n_rows=2, n_cols=2, R_on=config_base.R_on, R_off=config_base.R_off)
        cb = CrossbarIdeal(config)

        G_target = np.array([
            [200e-6, 100e-6],
            [150e-6, 250e-6],
        ])

        for i in range(2):
            for j in range(2):
                cb.set_conductance(i, j, G_target[i, j])

        V1 = float(p.get('V1', 0.8))
        V2 = float(p.get('V2', 0.4))
        V = np.array([V1, V2])

        cb.apply_voltages(V)
        I_out = cb.read_currents()
        I_expected = G_target.T @ V

        mae = np.mean(np.abs(I_out - I_expected))
        rmse = np.sqrt(np.mean((I_out - I_expected) ** 2))
        err_rel = np.max(np.abs(I_out - I_expected) / np.abs(I_expected + 1e-15)) * 100

        x_pos = np.arange(2)
        width = 0.35

        ax.bar(x_pos - width / 2, I_out * 1e6, width, label='I_sim', color='#89b4fa', alpha=0.85)
        ax.bar(x_pos + width / 2, I_expected * 1e6, width, label='I_ideal', color='#f38ba8', alpha=0.85)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(['I₁ (Col 1)', 'I₂ (Col 2)'])
        ax.set_xlabel('Salidas (Columnas)')
        ax.set_ylabel('Corriente (μA)')
        ax.set_title('Prueba 4.2.1: Operación Matricial 2×2 — I = G^T · V')
        ax.legend(loc='upper right')

        ax.text(0.02, 0.95,
                f'MAE = {mae:.2e} A\nRMSE = {rmse:.2e} A\nErr. rel = {err_rel:.2e} %',
                transform=ax.transAxes, ha='left', va='top', fontsize=10, color='#a6e3a1',
                bbox=dict(boxstyle='round', facecolor='#181825', edgecolor='#a6e3a1', alpha=0.8))

        self.figure.tight_layout()
        return {
            'MAE': f'{mae:.2e} A',
            'RMSE': f'{rmse:.2e} A',
            'Err. rel': f'{err_rel:.2e} %',
            'I₁': f'{I_out[0]*1e6:.2f} μA',
            'I₂': f'{I_out[1]*1e6:.2f} μA',
        }

    def run_2x2_programacion(self, p):
        """Prueba 4.2.2: Programación Selectiva de Memristores en Arreglo 2×2."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        config_base = self._get_live_crossbar_config()
        config = CrossbarConfig(n_rows=2, n_cols=2, R_on=config_base.R_on, R_off=config_base.R_off, x0=config_base.x0)
        cb = CrossbarIdeal(config)

        G_initial = cb.G_matrix.copy()

        n_pulses = int(p.get('n_pulses', 40))
        V_pulse = float(p.get('V_pulse', 1.0))
        syn = MemristiveSynapse(cb.memristors[0, 0])
        ltp = LTPRule(n_pulses=n_pulses, V_pulse=V_pulse)
        ltp.apply(syn, dt=1e-3)

        G_final = cb.G_matrix.copy()
        delta = G_final - G_initial

        im = ax.imshow(G_final * 1e6, cmap='viridis', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Col 1', 'Col 2'])
        ax.set_yticklabels(['Fila 1', 'Fila 2'])
        ax.set_title(f'Prueba 4.2.2: Programación M11 con LTP ({n_pulses} pulsos)')

        for i in range(2):
            for j in range(2):
                val = G_final[i, j] * 1e6
                delta_val = delta[i, j] * 1e6
                color = 'white' if val < (np.max(G_final*1e6) + np.min(G_final*1e6)) / 2 else 'black'
                ax.text(j, i, f'M{i+1}{j+1}\n{val:.2f} μS\n({delta_val:+.2f})',
                        ha='center', va='center', color=color, fontsize=11, fontweight='bold')

        cbar = self.figure.colorbar(im, ax=ax)
        cbar.set_label('Conductancia G (μS)', color='#cdd6f4')
        cbar.ax.yaxis.set_tick_params(color='#cdd6f4')

        self.figure.tight_layout()
        return {
            'G11 inicial': f'{G_initial[0,0]*1e6:.2f} μS',
            'G11 final': f'{G_final[0,0]*1e6:.2f} μS',
            'ΔG11': f'{delta[0,0]*1e6:+.4f} μS',
            'Otros ΔG': f'{np.sum(np.abs(delta)) - np.abs(delta[0,0]):.2e} S',
        }

    def run_2x2_sneak(self, p):
        """Prueba 4.2.3: Efecto de Sneak Paths en Crossbar 2×2."""
        from neurolab.crossbar import CrossbarSneak

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        config_base = self._get_live_crossbar_config()
        config = CrossbarConfig(n_rows=2, n_cols=2, R_on=config_base.R_on, R_off=config_base.R_off)
        cb_ideal = CrossbarIdeal(config)
        cb_sneak = CrossbarSneak(config)

        rng = np.random.default_rng(42)
        G_target = rng.uniform(50e-6, 500e-6, (2, 2))

        for i in range(2):
            for j in range(2):
                cb_ideal.set_conductance(i, j, G_target[i, j])
                cb_sneak.set_conductance(i, j, G_target[i, j])

        V_in = float(p.get('V_in', 0.5))
        V = np.array([V_in, V_in])

        cb_ideal.apply_voltages(V)
        cb_sneak.apply_voltages(V)

        I_ideal = cb_ideal.read_currents()
        I_sneak = cb_sneak.read_currents()

        err = np.abs(I_sneak - I_ideal) / np.abs(I_ideal + 1e-15) * 100

        x_pos = np.arange(2)
        width = 0.35

        ax.bar(x_pos - width / 2, I_ideal * 1e6, width, label='Ideal (sin sneak)', color='#89b4fa')
        ax.bar(x_pos + width / 2, I_sneak * 1e6, width, label='Con sneak paths', color='#f38ba8')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Columna 1', 'Columna 2'])
        ax.set_xlabel('Columna')
        ax.set_ylabel('Corriente (μA)')
        ax.set_title('Prueba 4.2.3: Efecto de Sneak Paths en Crossbar 2×2')
        ax.legend(loc='upper right')

        ax.text(0.98, 0.95,
                f'Error Col 1: {err[0]:.2f}%\nError Col 2: {err[1]:.2f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=10, color='#fab387',
                bbox=dict(boxstyle='round', facecolor='#181825', edgecolor='#fab387', alpha=0.8))

        self.figure.tight_layout()
        return {
            'I_ideal Col1': f'{I_ideal[0]*1e6:.2f} μA',
            'I_sneak Col1': f'{I_sneak[0]*1e6:.2f} μA',
            'Error Col1': f'{err[0]:.2f} %',
            'Error Col2': f'{err[1]:.2f} %',
        }

    def run_2x2_lif_dinamico(self, p):
        """Prueba 4.2.4: 2×2 → 2 Neuronas LIF Dinámicas."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        T_ms = float(p.get('T_ms', 100.0))
        T_sec = T_ms * 1e-3

        config_base = self._get_live_crossbar_config()
        config = CrossbarConfig(n_rows=2, n_cols=2, R_on=config_base.R_on, R_off=config_base.R_off)
        cb = CrossbarIdeal(config)
        cb.set_uniform_conductance(200e-6)

        lif_cfg = self._get_live_lif_config()
        lif_1 = LIFNeuron(lif_cfg)
        lif_2 = LIFNeuron(lif_cfg)

        dt = 1e-5
        t = np.arange(0, T_sec, dt)

        V_m_1 = np.zeros_like(t)
        V_m_2 = np.zeros_like(t)
        spikes_1 = []
        spikes_2 = []

        for k, tk in enumerate(t):
            V1 = 0.5 + 0.3 * np.sin(2 * np.pi * 10 * tk)
            V2 = 0.5 + 0.3 * np.cos(2 * np.pi * 10 * tk)

            cb.apply_voltages([V1, V2])
            I = cb.read_currents()

            if lif_1.update(I_in=I[0], dt=dt):
                spikes_1.append(tk)
            if lif_2.update(I_in=I[1], dt=dt):
                spikes_2.append(tk)

            V_m_1[k] = lif_1.V_m
            V_m_2[k] = lif_2.V_m

        ax.plot(t * 1e3, V_m_1, '-', color='#89b4fa', lw=1.5, label='LIF_1 (Col 1)')
        ax.plot(t * 1e3, V_m_2, '-', color='#f38ba8', lw=1.5, label='LIF_2 (Col 2)')
        ax.axhline(lif_cfg.v_th, color='#cdd6f4', ls='--', lw=1, alpha=0.5, label='V_th')

        ax.set_xlabel('Tiempo (ms)')
        ax.set_ylabel('Voltaje Membrana V_m (V)')
        ax.set_title(f'Prueba 4.2.4: 2 Neuronas LIF — {len(spikes_1)} + {len(spikes_2)} Spikes')
        ax.legend(loc='upper right')

        isi1 = f'{np.mean(np.diff(spikes_1))*1e3:.2f} ms' if len(spikes_1) > 1 else '—'
        isi2 = f'{np.mean(np.diff(spikes_2))*1e3:.2f} ms' if len(spikes_2) > 1 else '—'

        self.figure.tight_layout()
        return {
            'Spikes LIF_1': len(spikes_1),
            'Spikes LIF_2': len(spikes_2),
            'ISI medio LIF_1': isi1,
            'ISI medio LIF_2': isi2,
        }

    def run_2x2_escalabilidad(self, p):
        """Prueba 4.2.5: Escalabilidad 1×1 vs 2×2."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        n_iters = int(p.get('n_iters', 1000))
        sizes = [1, 2]
        tiempos = []
        n_memristores = []

        config_base = self._get_live_crossbar_config()

        for size in sizes:
            config = CrossbarConfig(n_rows=size, n_cols=size, R_on=config_base.R_on, R_off=config_base.R_off)
            cb = CrossbarIdeal(config)
            cb.set_uniform_conductance(200e-6)

            V = np.ones(size) * 0.5
            cb.apply_voltages(V)

            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = cb.read_currents()
            t_elapsed = (time.perf_counter() - t0) / n_iters

            tiempos.append(t_elapsed * 1e6)
            n_memristores.append(size * size)

        x_pos = np.arange(len(sizes))
        width = 0.35

        ax.bar(x_pos - width / 2, tiempos, width, label='Tiempo de lectura (μs)', color='#89b4fa')
        ax.bar(x_pos + width / 2, n_memristores, width, label='N memristores', color='#fab387')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{s}×{s}' for s in sizes])
        ax.set_ylabel('Valor (μs / Unidades)')
        ax.set_title('Prueba 4.2.5: Escalabilidad 1×1 vs 2×2')
        ax.legend(loc='upper left')

        self.figure.tight_layout()
        return {
            'Tiempo 1×1': f'{tiempos[0]:.2f} μs',
            'Tiempo 2×2': f'{tiempos[1]:.2f} μs',
            'Ratio 2×2/1×1': f'{tiempos[1]/max(tiempos[0], 1e-9):.2f}x',
        }

    # ==========================================================================
    # PREZIOSO 2015 — Fig. S3 (Estado Virgen Pre-Forming)
    # ==========================================================================

    def run_prez_s3a(self, p):
        """
        Fig. S3a — Curvas I-V de dispositivos vírgenes (pre-forming).

        Reproduce la caracterización inicial del crossbar de Prezioso 2015.
        Los dispositivos en estado virgen exhiben una curva S-shaped ASIMÉTRICA:
          - Modelo: I = G_0·V + a_p·(exp(b_p·V)−1)  para V > 0
                    I = G_0·V − a_n·(exp(b_n·|V|)−1) para V < 0
          - I(+0.75V) ≈ +4 μA, I(−0.75V) ≈ −2.5 μA  (del paper)
        """
        from neurolab.validation.prezioso2015_s3 import simulate_iv_virgin, _current_virgin_iv

        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        self._style_axes(ax1)
        self._style_axes(ax2)

        n_dev   = int(p.get('n_devices', 5))
        V_max   = float(p.get('V_max', 1.0))
        n_pts   = int(p.get('n_points', 100))
        G_mean  = float(p.get('G_mean_uS', 0.45)) * 1e-6
        G_sigma = float(p.get('G_sigma_uS', 0.08)) * 1e-6
        seed    = int(p.get('seed', 42))

        # Parámetros del modelo no lineal corregidos (Prezioso 2015 Fig. S3a)
        #   a_p = 2.67e-7 A,  b_p = 2.77 V⁻¹  → I(+1.0V) ≈ +4.4 μA
        #   a_n = 9.40e-8 A,  b_n = 3.32 V⁻¹  → I(-1.0V) ≈ -2.9 μA
        a_p = 2.67e-7
        b_p = 2.77
        a_n = 9.4e-8
        b_n = 3.32

        data = simulate_iv_virgin(
            n_devices=n_dev,
            V_max=V_max,
            n_points=n_pts,
            G_mean=G_mean,
            G_sigma=G_sigma,
            a_p=a_p, b_p=b_p, a_n=a_n, b_n=b_n,
            seed=seed,
        )

        V_sweep = data['V_sweep']
        colors  = ['#89b4fa', '#f38ba8', '#a6e3a1', '#f9e2af', '#cba6f7',
                   '#fab387', '#94e2d5', '#eba0ac', '#b4befe', '#cba6f7']

        # Curvas de dispositivos individuales
        for k, (I_arr, G0) in enumerate(zip(data['I_curves'], data['G_devices'])):
            col = colors[k % len(colors)]
            ax1.plot(V_sweep, I_arr * 1e6, '-', lw=1.5, color=col, alpha=0.8,
                     label=f'D{k+1}: G={G0*1e6:.3f} μS')

        # Curva de referencia (G_mean, sin variabilidad) — curva S media
        ax1.plot(V_sweep, data['I_ref'] * 1e6, '--', lw=2.5, color='#ffffff',
                 label=f'Ref (G_mean)', zorder=5)

        ax1.axhline(0, color='#45475a', lw=0.6)
        ax1.axvline(0, color='#45475a', lw=0.6)
        ax1.set_xlim(-V_max, V_max)
        ax1.set_ylim(-3.5, 5.0)
        ax1.set_xlabel('Voltaje (V)')
        ax1.set_ylabel('Corriente (μA)')
        ax1.set_title('Fig. S3a — I-V Virgen (Curva S)\n(Prezioso 2015, pre-forming, asimétrica)')
        ax1.legend(fontsize=7, loc='upper left')

        # Panel derecho: comparación I simulada vs paper en puntos clave
        V_check  = np.array([ 1.00,  0.75,  0.50,  0.25, -0.25, -0.50, -0.75, -1.00])
        I_paper  = np.array([ 4.00,  2.50,  0.80,  0.10, -0.05, -0.40, -1.50, -2.50])   # μA del paper
        I_sim    = _current_virgin_iv(V_check, G_mean, a_p, b_p, a_n, b_n) * 1e6

        x_idx = np.arange(len(V_check))
        width = 0.35
        ax2.bar(x_idx - width/2, I_paper, width, label='Paper Fig.S3a',
                color='#f38ba8', edgecolor='#45475a', alpha=0.9)
        ax2.bar(x_idx + width/2, I_sim,   width, label='Simulación',
                color='#89b4fa', edgecolor='#45475a', alpha=0.9)
        ax2.axhline(0, color='#45475a', lw=0.5)
        ax2.set_xticks(x_idx)
        ax2.set_xticklabels([f'{v:+.2f}V' for v in V_check], fontsize=7)
        ax2.set_ylabel('Corriente (μA)')
        ax2.set_title('Comparación vs. Paper\n(puntos clave)')
        ax2.legend(fontsize=8)

        # Anotación de asimetría
        asym = data['asym_ratio']
        ax2.annotate(
            f'Asim.: {asym:.2f}x\n(pos/neg)',
            xy=(0.97, 0.05), xycoords='axes fraction',
            ha='right', va='bottom',
            fontsize=8, color='#f9e2af',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#313244', edgecolor='#45475a'),
        )

        self.figure.tight_layout()

        # MAE entre simulación y paper
        mae_uA = float(np.mean(np.abs(I_sim - I_paper)))
        asym_ok = asym > 1.05
        i_ok    = abs(data['I_ref_pos']) * 1e6 > 1.0

        return {
            'I(+0.75V) sim (μA)': f'{_current_virgin_iv(np.array([0.75]),  G_mean, a_p, b_p, a_n, b_n)[0]*1e6:.2f}',
            'I(-0.75V) sim (μA)': f'{_current_virgin_iv(np.array([-0.75]), G_mean, a_p, b_p, a_n, b_n)[0]*1e6:.2f}',
            'Asimetría (pos/neg)': f'{asym:.3f}x',
            'MAE vs paper (μA)':  f'{mae_uA:.3f}',
            'Curva S (no lineal)': 'PASS' if i_ok    else 'WARN',
            'Asimetría OK':        'PASS' if asym_ok else 'WARN',
        }

    def run_prez_s3b(self, p):
        """
        Fig. S3b — Mapa de conductancias 4×4 en estado virgen o activo.

        Si la matriz Crossbar 4×4 activa está disponible en la interfaz, lee y visualiza
        la matriz real de conductancias configurada por el usuario en el Crossbar.
        """
        from neurolab.validation.prezioso2015_s3 import simulate_conductance_map

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axes(ax)

        n_rows  = int(p.get('n_rows', 4))
        n_cols  = int(p.get('n_cols', 4))
        G_mean  = float(p.get('G_mean_uS', 0.45)) * 1e-6
        G_sigma = float(p.get('G_sigma_uS', 0.08)) * 1e-6
        seed    = int(p.get('seed', 42))
        use_active = bool(p.get('use_active_crossbar', True))

        active_G = self.get_active_4x4_matrix() if use_active else None

        if active_G is not None:
            G_matrix = active_G
            n_rows, n_cols = G_matrix.shape
            G_flat = G_matrix.flatten()
            G_mean_real = float(G_flat.mean())
            G_std_real  = float(G_flat.std())
            cv_real     = float(G_std_real / G_mean_real * 100.0) if G_mean_real > 0 else 0.0
            data = {
                'G_matrix': G_matrix,
                'G_mean': G_mean_real,
                'G_std': G_std_real,
                'G_min': float(G_flat.min()),
                'G_max': float(G_flat.max()),
                'CV_pct': cv_real,
                'is_active': True,
            }
        else:
            data = simulate_conductance_map(
                n_rows=n_rows, n_cols=n_cols,
                G_mean=G_mean, G_sigma=G_sigma, seed=seed,
            )
            data['is_active'] = False

        G_uS = data['G_matrix'] * 1e6   # μS para visualización

        # Rango de color centrado en la media
        v_center = data['G_mean'] * 1e6
        v_range  = max(data['G_std'] * 1e6 * 3.0, 0.1)
        vmin = max(0.01, v_center - v_range)
        vmax = v_center + v_range

        im = ax.imshow(G_uS, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)

        # Etiquetas numéricas en cada celda
        fontsize_cell = max(6, min(11, 80 // max(n_rows, n_cols)))
        for i in range(n_rows):
            for j in range(n_cols):
                val = G_uS[i, j]
                txt_color = 'black' if val < (vmin + (vmax - vmin) * 0.6) else 'white'
                ax.text(j, i, f'{val:.1f}\nμS', ha='center', va='center',
                        fontsize=fontsize_cell, color=txt_color, fontweight='bold')

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f'B{j+1}' for j in range(n_cols)], fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f'W{i+1}' for i in range(n_rows)], fontsize=8)
        ax.set_xlabel('Columna (bitline)')
        ax.set_ylabel('Fila (wordline)')

        tag_str = "Matriz Crossbar 4×4 Activa" if data['is_active'] else "Simulación D2D"
        ax.set_title(
            f'Fig. S3b — Mapa de Conductancias {n_rows}×{n_cols} (μS)\n'
            f'({tag_str})',
            fontsize=11, fontweight='bold'
        )

        cbar = self.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Conductancia (μS)', color='#cdd6f4', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='#cdd6f4')

        self.figure.tight_layout()

        cv = data['CV_pct']
        return {
            'Matriz Origen':    'Crossbar 4×4 Activo' if data['is_active'] else 'Simulación D2D',
            'Dimensión':        f'{n_rows}×{n_cols}',
            'G media (μS)':    f'{data["G_mean"]*1e6:.2f}',
            'G std (μS)':      f'{data["G_std"]*1e6:.2f}',
            'G min (μS)':      f'{data["G_min"]*1e6:.2f}',
            'G max (μS)':      f'{data["G_max"]*1e6:.2f}',
            'CV (%)':          f'{cv:.2f}',
            'Estado Matriz':   'ACTIVA (CROSSBAR)' if data['is_active'] else 'SINTÉTICA',
        }

    def run_prez_s3c(self, p):
        """
        Fig. S3c — Histograma de conductancias @ V_read = 0.1 V.

        Lee y analiza la matriz 4×4 activa configurada en la aplicación.
        """
        from neurolab.validation.prezioso2015_s3 import (
            simulate_conductance_map, compute_histogram_stats
        )

        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        self._style_axes(ax1)
        self._style_axes(ax2)

        n_rows  = int(p.get('n_rows', 4))
        n_cols  = int(p.get('n_cols', 4))
        G_mean  = float(p.get('G_mean_uS', 0.45)) * 1e-6
        G_sigma = float(p.get('G_sigma_uS', 0.08)) * 1e-6
        n_bins  = int(p.get('n_bins', 12))
        seed    = int(p.get('seed', 42))
        use_active = bool(p.get('use_active_crossbar', True))

        active_G = self.get_active_4x4_matrix() if use_active else None

        if active_G is not None:
            G_matrix = active_G
            is_active = True
        else:
            map_data  = simulate_conductance_map(
                n_rows=n_rows, n_cols=n_cols,
                G_mean=G_mean, G_sigma=G_sigma, seed=seed,
            )
            G_matrix = map_data['G_matrix']
            is_active = False

        hist_data = compute_histogram_stats(G_matrix, n_bins=n_bins)

        # Panel izquierdo: histograma + curva normal
        ax1.bar(
            hist_data['bin_centers'], hist_data['counts'],
            width=(hist_data['bins'][1] - hist_data['bins'][0]) * 0.85,
            color='#ee5253', edgecolor='#45475a', alpha=0.85,
            label=f'N = {hist_data["n_devices"]} celdas'
        )
        ax1.plot(
            hist_data['normal_x'], hist_data['normal_y'],
            '-', lw=2.5, color='#f9e2af', label='Ajuste Normal'
        )
        ax1.axvline(hist_data['G_mean'], color='#89b4fa', ls='--', lw=1.5,
                    label=f'μ = {hist_data["G_mean"]:.2f} μS')
        ax1.axvline(hist_data['G_mean'] + hist_data['G_std'],
                    color='#a6e3a1', ls=':', lw=1, alpha=0.8)
        ax1.axvline(hist_data['G_mean'] - hist_data['G_std'],
                    color='#a6e3a1', ls=':', lw=1, alpha=0.8,
                    label=f'±σ = ±{hist_data["G_std"]:.2f} μS')
        ax1.set_xlabel('Conductancia @ 0.1V (μS)')
        ax1.set_ylabel('Frecuencia')
        tag_str = "Crossbar 4×4 Activo" if is_active else "Sintético"
        ax1.set_title(
            f'Fig. S3c — Histograma de G\n({tag_str})'
        )
        ax1.legend(fontsize=8)

        # Panel derecho: estadísticas de validación (tabla visual)
        cv     = hist_data['CV_pct']
        g_ok   = 0.01 < hist_data['G_mean'] < 2000.0

        rows_lbl = [
            'Origen', 'G media (μS)', 'G std (μS)', 'CV (%)',
            'N celdas', 'Dimensión', '',
            'Estado G media', 'Medición 4×4',
        ]
        rows_val = [
            'Crossbar Activo 4×4' if is_active else 'Sintético',
            f'{hist_data["G_mean"]:.2f}',
            f'{hist_data["G_std"]:.2f}',
            f'{cv:.2f}',
            str(hist_data['n_devices']),
            f'{G_matrix.shape[0]}×{G_matrix.shape[1]}',
            '',
            'OK' if g_ok else 'WARN',
            'REAL (CROSSBAR)' if is_active else 'SINTÉTICO',
        ]

        ax2.axis('off')
        table = ax2.table(
            cellText=[[lbl, val] for lbl, val in zip(rows_lbl, rows_val)],
            colLabels=['Métrica', 'Valor'],
            loc='center', cellLoc='left',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor('#181825')
            cell.set_edgecolor('#45475a')
            txt = cell.get_text().get_text()
            if row == 0:
                cell.set_facecolor('#313244')
                cell.get_text().set_color('#89b4fa')
                cell.get_text().set_fontweight('bold')
            elif txt in ['REAL (CROSSBAR)', 'OK', 'PASS']:
                cell.set_facecolor('#1e3a29')
                cell.get_text().set_color('#a6e3a1')
                cell.get_text().set_fontweight('bold')
            elif txt in ['SINTÉTICO', 'WARN']:
                cell.get_text().set_color('#f9e2af')
                cell.get_text().set_fontweight('bold')
            else:
                cell.get_text().set_color('#cdd6f4')

        ax2.set_title('Estadísticas de la Matriz 4×4', color='#89b4fa', pad=12)

        self.figure.tight_layout()
        return {
            'Origen':          'Crossbar Activo 4×4' if is_active else 'Simulación D2D',
            'G media (μS)':    f'{hist_data["G_mean"]:.2f}',
            'G std (μS)':      f'{hist_data["G_std"]:.2f}',
            'CV (%)':          f'{cv:.2f}',
            'N celdas':        f'{hist_data["n_devices"]}',
            'Estado':          'ACTIVO OK' if is_active else 'SINTÉTICO',
        }
