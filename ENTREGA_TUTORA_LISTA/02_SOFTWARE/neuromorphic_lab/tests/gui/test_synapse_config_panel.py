import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from neurolab.gui.widgets.synapse_config_panel import SynapseConfigPanel
from neurolab.gui.widgets.synapse_plot_canvas import SynapseMplCanvas


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_ltp_ltd_voltage_control_updates_simulation_live(qapp):
    panel = SynapseConfigPanel()
    canvas = SynapseMplCanvas()
    emissions = []
    panel.simulation_requested.connect(
        lambda exp_type, sim_data: emissions.append((exp_type, sim_data))
    )
    panel.simulation_requested.connect(canvas.plot_experiment)

    panel.memristor_subtabs.setCurrentIndex(1)
    panel.combo_exp.setCurrentIndex(8)
    _, initial = emissions[-1]
    assert "R² residual:" in panel.metrics_label.text()
    assert "r² de Pearson" in panel.metrics_label.text()
    assert np.isfinite(initial["residual_r2"])
    assert np.isfinite(initial["pearson_r2"])
    assert "R² =" in canvas.figure.axes[0].get_title()
    assert "r² =" in canvas.figure.axes[0].get_title()
    initial_curve = initial["G_sim"].copy()
    initial_plot_curve = np.asarray(canvas.figure.axes[0].lines[2].get_ydata()).copy()

    panel.spin_vamp.setValue(3.0)
    exp_type, updated = emissions[-1]
    updated_plot_curve = np.asarray(canvas.figure.axes[0].lines[2].get_ydata())

    assert exp_type == "jo2010_ltp_ltd"
    assert updated["V_pulse"] == 3.0
    assert not np.array_equal(initial_curve, updated["G_sim"])
    assert not np.array_equal(initial_plot_curve, updated_plot_curve)
    assert "±3.00 V" in panel.metrics_label.text()
    assert not panel.spin_aplus.isEnabled()
    assert not panel.spin_tauplus.isEnabled()
    assert panel.spin_vamp.isEnabled()
    panel.close()
    canvas.close()


def test_stdp_controls_update_stdp_curve_live(qapp):
    panel = SynapseConfigPanel()
    emissions = []
    panel.simulation_requested.connect(
        lambda exp_type, sim_data: emissions.append((exp_type, sim_data))
    )
    panel.combo_exp.setCurrentIndex(2)
    initial_curve = emissions[-1][1]["dW_hebbian"].copy()

    panel.spin_aplus.setValue(0.5)
    exp_type, updated = emissions[-1]

    assert exp_type == "stdp_curves"
    assert not np.array_equal(initial_curve, updated["dW_hebbian"])
    assert panel.spin_aplus.isEnabled()
    assert not panel.spin_vamp.isEnabled()
    panel.close()


def test_ltp_ltd_protocol_controls_update_real_simulation(qapp):
    panel = SynapseConfigPanel()
    canvas = SynapseMplCanvas()
    panel.simulation_requested.connect(canvas.plot_experiment)
    panel.memristor_subtabs.setCurrentIndex(1)
    panel.combo_exp.setCurrentIndex(8)
    initial_raw_curve = np.asarray(canvas.figure.axes[1].lines[0].get_ydata()).copy()
    assert canvas.figure.axes[0].lines[2].get_xdata()[-1] == 200

    panel.spin_pulse_width.setValue(2.0)
    wider_pulse_curve = np.asarray(canvas.figure.axes[1].lines[0].get_ydata()).copy()
    assert not np.array_equal(initial_raw_curve, wider_pulse_curve)
    assert "ancho 2 ms" in panel.metrics_label.text()

    panel.spin_n_ltp.setValue(80)
    exp_type_data = canvas.figure.axes[1].lines[0].get_xdata()
    assert exp_type_data[-1] == 80
    assert canvas.figure.axes[0].lines[2].get_xdata()[-1] == 180
    assert "cobertura:" in panel.metrics_label.text()
    assert panel.spin_n_ltp.value() == 80

    panel.spin_interpulse.setValue(2.0)
    assert "intervalo 2 ms" in panel.metrics_label.text()
    panel.close()
    canvas.close()
