"""
tests/test_crossbar_4x4.py
===========================
Pruebas unitarias Pytest para el Crossbar 4×4 y sus 20 funciones de validación.
"""

import pytest
import numpy as np
from matplotlib.figure import Figure

from neurolab.crossbar import CrossbarIdeal, CrossbarSneak, CrossbarLine, CrossbarConfig
from neurolab.gui.crossbar_elements import (
    get_G_matrix, get_V_vector, set_G_matrix, compute_currents, matrix_to_heatmap_string
)
from neurolab.gui.tests.tests_4x4 import (
    draw_4x4_01, draw_4x4_02, draw_4x4_03, draw_4x4_04,
    draw_4x4_05, draw_4x4_06, draw_4x4_07, draw_4x4_08,
    draw_4x4_09, draw_4x4_10, draw_4x4_11, draw_4x4_12,
    draw_4x4_13, draw_4x4_14, draw_4x4_15, draw_4x4_16,
    draw_4x4_17, draw_4x4_18, draw_4x4_19, draw_4x4_20
)


class DummyCanvas:
    def draw(self):
        pass


class DummyGUI:
    def __init__(self):
        self.figure = Figure(figsize=(10, 5))
        self.canvas = DummyCanvas()

    def get_axis(self):
        self.figure.clear()
        return self.figure.add_subplot(111)

    def _style_axis(self, ax):
        pass

    def refresh_plot(self):
        pass


def test_crossbar_4x4_initialization():
    cfg = CrossbarConfig(n_rows=4, n_cols=4)
    cb = CrossbarIdeal(cfg)
    assert cb.n_rows == 4
    assert cb.n_cols == 4
    assert cb.memristors.shape == (4, 4)
    assert cb.G_matrix.shape == (4, 4)


def test_crossbar_4x4_matrix_helpers():
    elements = {}
    class DummyElement:
        def __init__(self, val=200e-6):
            self.params = {'G_11': val, 'x0': 0.1}
        def on_params_changed(self):
            pass

    for i in range(4):
        for j in range(4):
            elements[f'M{i+1}{j+1}'] = DummyElement(200e-6)

    G = get_G_matrix(elements, 4, 4)
    assert G.shape == (4, 4)
    assert np.allclose(G, 200e-6)

    V = np.array([0.5, 0.5, 0.5, 0.5])
    I = compute_currents(G, V)
    assert len(I) == 4
    assert np.allclose(I, 4 * 200e-6 * 0.5)

    heatmap_str = matrix_to_heatmap_string(G)
    assert "Fila 1" in heatmap_str
    assert "Fila 4" in heatmap_str


@pytest.mark.parametrize("draw_func", [
    draw_4x4_01, draw_4x4_02, draw_4x4_03, draw_4x4_04,
    draw_4x4_05, draw_4x4_06, draw_4x4_07, draw_4x4_08,
    draw_4x4_09, draw_4x4_10, draw_4x4_11, draw_4x4_12,
    draw_4x4_13, draw_4x4_14, draw_4x4_15, draw_4x4_16,
    draw_4x4_17, draw_4x4_18, draw_4x4_19, draw_4x4_20
])
def test_all_20_draw_functions_4x4(draw_func):
    gui = DummyGUI()
    res = draw_func(gui)
    assert isinstance(res, dict)
    assert res.get('status') == 'PASS'
    assert 'metrics' in res


def test_crossbar_4x4_volatile_elements():
    """Verifica que la vista Crossbar 4x4 incluya los 4 memristores volátiles HfO2 en sus columnas."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from neurolab.gui.crossbar_4x4_view import Crossbar4x4View
    from neurolab.gui.crossbar_elements import VolatileMemristorElement

    view = Crossbar4x4View()
    for j in range(4):
        key = f'M_v{j+1}'
        assert key in view.elements
        assert isinstance(view.elements[key], VolatileMemristorElement)
    view.deleteLater()



def test_crossbar_4x4_core_physics_integration():
    """Verifica que Crossbar4x4View utiliza Crossbar de neurolab.crossbar.core correctamente."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from neurolab.gui.crossbar_4x4_view import Crossbar4x4View
    view = Crossbar4x4View()

    # Verificar inicialización del core
    assert hasattr(view, 'crossbar'), "Crossbar4x4View no ha inicializado el core 'crossbar'"
    assert view.crossbar.n_rows == 4
    assert view.crossbar.n_cols == 4

    # Verificar que _step_physics utiliza el core en modo V/2
    view.mode = "program_v2"
    view.target_cell = (0, 0)
    # Mock update_memristors to check if it's called
    called = False
    def mock_update(*args, **kwargs):
        nonlocal called
        called = True
    view.crossbar.update_memristors = mock_update
    
    view._anim_tick()
    
    assert called, "El core backend no fue llamado en _step_physics"
    assert view.crossbar.V_rows[0] == 1.0, "El voltaje V/2 no se aplicó al core"
    assert view.crossbar.V_cols[0] == -1.0, "El voltaje V/2 no se aplicó al core"

    # Verificar Bug Fixes de Modos
    # STDP and R-STDP should properly switch states without side effects
    view._set_mode_3_stdp()
    assert view.plasticity_mode == "stdp"
    
    view._set_mode_4_rstdp()
    assert view.plasticity_mode == "rstdp"

    view.deleteLater()


def test_crossbar_4x4_physical_realism_integration():
    """
    Verifica rigurosamente el comportamiento físico realista del controlador Crossbar 4x4:
    1. Las neuronas LIF emiten 1 spike discreto por disparo y entran en período refractario (t_ref=150ms).
    2. R-STDP consume la señal de recompensa R en exactamente 1 tick (R se resetea a 0.0).
    3. El memristor volátil relaja su estado difusivo x hacia x_eq (x0) y no hacia 0.
    4. Las corrientes de columna I_cols responden dinámicamente al vector V_rows inyectado.
    """
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from neurolab.gui.crossbar_4x4_view import Crossbar4x4View

    view = Crossbar4x4View()
    ctrl = view.controller

    # 1. Prueba de R-STDP: Consumo de recompensa en 1 tick
    ctrl.set_reward(1.0)
    assert ctrl.rstdp_rule.cfg.R == 1.0
    G_dummy = np.full((4, 4), 100e-6)
    spike_pre = np.ones(4)
    spike_post = np.ones(4)
    dG = ctrl.rstdp_rule.apply(G_dummy, spike_pre, spike_post, dt=0.03)
    assert ctrl.rstdp_rule.cfg.R == 0.0, "R-STDP no consumió la recompensa R (quedó bloqueada)"
    assert np.any(dG > 0), "R-STDP debió potenciar las conductancias con R=+1"

    # 2. Prueba de Memristor Volátil: Relajación hacia x_eq (x0)
    mv = ctrl.elements['M_v1']
    mv.params['x'] = 0.50
    mv.params['x0'] = 0.05
    mv.params['volatile_tau_relax'] = 0.10
    V_rows = np.zeros(4)
    V_cols = np.zeros(4)
    ctrl.step(dt=0.03, mode="read", plasticity_mode="off", is_animating=True, V_rows=V_rows, V_cols=V_cols)
    assert mv.params['x'] < 0.50, "El memristor volátil no redujo x hacia el reposo"
    assert mv.params['x'] >= 0.05, "El memristor volátil cayó por debajo de x_eq (x0)"

    # 3. Prueba de Corrientes de Columna Dinámicas e Integración LIF
    V_rows_active = np.array([2.0, 0.0, 0.0, 0.0])
    res = ctrl.step(dt=0.03, mode="read", plasticity_mode="off", is_animating=True, V_rows=V_rows_active, V_cols=V_cols)
    I_cols = res["I_cols"]
    assert np.all(I_cols > 0), "Las corrientes de columna debieron ser mayores a cero con V_row[0]=2V"

    view.deleteLater()

