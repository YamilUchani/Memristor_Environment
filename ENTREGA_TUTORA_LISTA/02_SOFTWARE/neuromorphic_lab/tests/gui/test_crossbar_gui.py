"""
test_crossbar_gui.py
====================
Tests unitarios para la interfaz gráfica del Crossbar 1x1 (PySide6).
"""

import sys
import pytest
from PySide6.QtWidgets import QApplication

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


from neurolab.gui.crossbar_view import CrossbarView, CrossbarCanvas
from neurolab.gui.crossbar_elements import (
    SensorElement, MemristorElement, VolatileMemristorElement, NeuronElement, ActuatorElement
)
from neurolab.gui.config_dialogs import ConfigDialog


def test_crossbar_view_initialization(qapp):
    """Verifica la creación del panel CrossbarView."""
    view = CrossbarView()
    assert 'sensor_1' in view.elements
    assert 'M11' in view.elements
    assert 'M_v' in view.elements
    assert 'neuron_1' in view.elements
    assert 'actuator_1' in view.elements

    assert isinstance(view.elements['sensor_1'], SensorElement)
    assert isinstance(view.elements['M11'], MemristorElement)
    assert isinstance(view.elements['M_v'], VolatileMemristorElement)
    assert isinstance(view.elements['neuron_1'], NeuronElement)
    assert isinstance(view.elements['actuator_1'], ActuatorElement)
    view.deleteLater()


def test_crossbar_elements_hit_test(qapp):
    """Verifica la detección de click en los elementos visuales."""
    view = CrossbarView()
    s = view.elements['sensor_1']
    m = view.elements['M11']

    assert s.hit_test(s.x, s.y) is True
    assert s.hit_test(0, 0) is False

    assert m.hit_test(m.x, m.y) is True
    assert m.hit_test(0, 0) is False
    view.deleteLater()


def test_crossbar_simulation(qapp):
    """Verifica la ejecución del botón ▶ Simular en la vista GUI."""
    view = CrossbarView()

    # Voltaje inicial de sensor = 1.0 V
    sensor = view.elements['sensor_1']
    sensor.params['V_out'] = 1.0

    # Simular
    view._on_simulate()

    # Vm de neurona debe haber subido
    neuron = view.elements['neuron_1']
    assert neuron.params['V_m'] > 0
    assert "V_in" in view.status_label.text()
    view.deleteLater()


def test_crossbar_reset(qapp):
    """Verifica la ejecución del botón ⏹ Reset."""
    view = CrossbarView()
    neuron = view.elements['neuron_1']
    neuron.params['V_m'] = 1.5

    view._on_reset()
    assert neuron.params['V_m'] == 0.0
    assert "reiniciado" in view.status_label.text()
    view.deleteLater()


def test_crossbar_pan_zoom(qapp):
    """Verifica las funcionalidades de zoom, pan y conversión de coordenadas del canvas."""
    view = CrossbarView()
    canvas = view.canvas

    assert canvas.zoom_factor == 1.0
    assert canvas.pan_x == 0.0
    assert canvas.pan_y == 0.0

    # Probamos zoom in
    canvas.zoom_in()
    assert canvas.zoom_factor > 1.0

    # Probamos zoom out
    canvas.zoom_out()
    assert abs(canvas.zoom_factor - 1.0) < 0.01

    # Probamos reset
    canvas.set_zoom(2.0)
    canvas.pan_x = 50.0
    canvas.pan_y = 100.0
    assert canvas.zoom_factor == 2.0

    cx, cy = canvas.widget_to_canvas(150.0, 300.0)
    assert cx == (150.0 - 50.0) / 2.0  # 50.0
    assert cy == (300.0 - 100.0) / 2.0 # 100.0

    canvas.reset_view()
    assert canvas.zoom_factor == 1.0
    assert canvas.pan_x == 0.0
    assert canvas.pan_y == 0.0
    view.deleteLater()


