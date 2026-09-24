"""
test_crossbar_1x1.py
====================
Tests unitarios para crossbar 1x1.
"""

import pytest
import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def test_1x1_initialization():
    """Verifica la creación del crossbar 1x1."""
    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    assert cb.n_rows == 1
    assert cb.n_cols == 1
    assert cb.G_matrix.shape == (1, 1)
    assert cb.memristors.shape == (1, 1)


def test_1x1_ideal_operation():
    """I = G · V debe ser exacto."""
    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    G_target = 200e-6
    cb.set_conductance(0, 0, G_target)

    V_in = 0.5
    I_out = cb.read_single(V_in)
    I_expected = G_target * V_in

    assert I_out == pytest.approx(I_expected, rel=1e-10)


def test_1x1_sweep_voltage():
    """Barrido de voltaje."""
    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    G_target = 200e-6
    cb.set_conductance(0, 0, G_target)

    for V_in in [-0.1, -0.05, 0.0, 0.05, 0.1]:
        I_out = cb.read_single(V_in)
        I_expected = G_target * V_in
        assert I_out == pytest.approx(I_expected, rel=1e-10)


def test_1x1_sweep_conductance():
    """Barrido de conductancia."""
    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    V_in = 0.5
    for G_target in [62.5e-6, 100e-6, 200e-6, 500e-6, 1000e-6]:
        cb.set_conductance(0, 0, G_target)
        I_out = cb.read_single(V_in)
        I_expected = G_target * V_in
        assert I_out == pytest.approx(I_expected, rel=1e-10)


def test_1x1_voltage_array():
    """Uso de apply_voltages + read_currents."""
    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    G_target = 200e-6
    cb.set_conductance(0, 0, G_target)

    V = np.array([0.5])
    cb.apply_voltages(V)
    I = cb.read_currents()
    I_expected = G_target * 0.5

    assert I.shape == (1,)
    assert I[0] == pytest.approx(I_expected, rel=1e-10)


def test_1x1_read_single_requires_1x1():
    """read_single solo funciona en 1x1."""
    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)

    with pytest.raises(ValueError):
        cb.read_single(0.5)


def test_1x1_integration_with_memristor():
    """Verifica que el memristor interno es de Fase 1."""
    from neurolab.devices import MemristorStrukov

    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    assert isinstance(cb.memristors[0, 0], MemristorStrukov)


def test_1x1_integration_with_synapse():
    """Verifica que se puede crear una sinapsis (Fase 3)."""
    from neurolab.synapses import MemristiveSynapse

    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    syn = MemristiveSynapse(cb.memristors[0, 0])
    assert syn.conductance == pytest.approx(
        cb.memristors[0, 0].conductance, rel=1e-10)


def test_1x1_integration_with_lif():
    """Verifica que la corriente puede alimentar una LIF (Fase 2)."""
    from neurolab.neurons import LIFNeuron

    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)
    cb.set_conductance(0, 0, 200e-6)

    lif = LIFNeuron()

    V_in = 0.5
    I_out = cb.read_single(V_in)
    lif.step(current_input=I_out, dt=1e-4)

    assert lif.v_membrane > lif.config.v_rest

