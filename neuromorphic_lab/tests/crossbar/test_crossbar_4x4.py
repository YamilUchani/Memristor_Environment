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


def test_crossbar_4x4_7_bug_fixes():
    """Verifica los 7 bug fixes de modos, plasticidad y botones en la vista 4x4."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from neurolab.gui.crossbar_4x4_view import Crossbar4x4View

    view = Crossbar4x4View()

    # BUG #1: _set_mode_3_stdp forza mode = "read"
    view.mode = "program_v2"
    view._set_mode_3_stdp()
    assert view.mode == "read"
    assert view.plasticity_mode == "stdp"

    # BUG #2: _set_mode_4_rstdp forza mode = "read"
    view.mode = "program_v2"
    view._set_mode_4_rstdp()
    assert view.mode == "read"
    assert view.plasticity_mode == "rstdp"

    # BUG #3: _set_mode_1_read resetea reward a 0
    view.reward = 1.0
    view._set_mode_1_read()
    assert view.reward == 0.0
    assert view.plasticity_mode == "off"
    assert view.mode == "read"

    # BUG #4: _step_physics no aplica STDP en modo program_v2
    view.mode = "program_v2"
    view.plasticity_mode = "stdp"
    view._step_physics(0.01)
    assert view.mode == "program_v2"

    # BUG #5: spike_pre se obtiene desde V_rows
    view.mode = "read"
    view.plasticity_mode = "stdp"
    view.V_rows = np.array([1.0, 0.0, 0.0, 0.0])
    view._step_physics(0.01)
    assert view.spike_pre[0] == 1.0
    assert view.spike_pre[1] == 0.0

    # BUG #6 & Action buttons: _apply_reward aplica modulación de recompensa
    view._set_mode_4_rstdp()
    view.spike_pre = np.zeros(4)
    view.spike_post = np.zeros(4)
    view.rstdp_rule.reset(4, 4)
    view._apply_reward(1.0)
    assert "R-STDP" in view.status_label.text()


    # Los 4 botones de acción permanecen siempre habilitados y responsivos
    assert view.btn_pulse_ltp.isEnabled()
    assert view.btn_pulse_ltd.isEnabled()
    assert view.btn_reward_plus.isEnabled()
    assert view.btn_reward_minus.isEnabled()
    view.deleteLater()



