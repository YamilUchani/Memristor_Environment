"""
test_synapses.py
================
Tests unitarios y de integración para el módulo neurolab.synapses.
"""

import pytest
import numpy as np
from neurolab.devices import MemristorStrukov
from neurolab.configs import StrukovConfig
from neurolab.synapses import (
    Synapse,
    MemristiveSynapse,
    LTPRule,
    LTDRule,
    STDPRule,
    AntiSTDPRule,
    PlasticityConfig
)


def test_memristive_synapse_initialization():
    cfg = StrukovConfig(RON=100.0, ROFF=16_000.0, x0=0.10)
    mem = MemristorStrukov(cfg)
    syn = MemristiveSynapse(mem)

    assert syn.G_min == pytest.approx(1.0 / 16000.0)
    assert syn.G_max == pytest.approx(1.0 / 100.0)
    assert 0.0 <= syn.weight <= 1.0
    assert syn.conductance == pytest.approx(mem.conductance)


def test_weight_conductance_mapping_clipping():
    cfg = StrukovConfig(RON=100.0, ROFF=16_000.0, x0=0.50)
    mem = MemristorStrukov(cfg)
    syn = MemristiveSynapse(mem)

    # Modificar peso directamente
    syn.weight = 0.80
    assert syn.weight == pytest.approx(0.80)
    assert syn.conductance > syn.G_min

    # Clipping de límites
    syn.weight = 1.5
    assert syn.weight == pytest.approx(1.0)

    syn.weight = -0.5
    assert syn.weight == pytest.approx(0.0)


def test_ltp_ltd_rules():
    cfg = StrukovConfig(RON=100.0, ROFF=16_000.0, x0=0.10)
    mem = MemristorStrukov(cfg)
    syn = MemristiveSynapse(mem)

    G_initial = syn.conductance

    # Aplicar LTP (pulsos positivos)
    ltp = LTPRule(n_pulses=10, V_pulse=+1.0)
    G_ltp = ltp.apply(syn, dt=1e-3)

    assert len(G_ltp) == 11
    assert G_ltp[-1] > G_initial  # Monótonamente creciente

    # Aplicar LTD (pulsos negativos)
    ltd = LTDRule(n_pulses=10, V_pulse=-1.0)
    G_ltd = ltd.apply(syn, dt=1e-3)

    assert G_ltd[-1] < G_ltp[-1]  # Monótonamente decreciente


def test_pulse_width_and_interval_are_integrated_as_real_time():
    class RecordingSynapse(Synapse):
        def __init__(self):
            super().__init__()
            self.updates = []

        def update(self, pre_spike, post_spike, dt, V_applied=None):
            self.updates.append((pre_spike, post_spike, dt, V_applied))
            self.conductance += dt * 1e-3 * abs(V_applied or 0.0)

    synapse = RecordingSynapse()
    history = LTPRule(
        n_pulses=2,
        V_pulse=1.0,
        pulse_width=2e-3,
        saturation=False,
        t_off=3e-3,
    ).apply(synapse, dt=1e-3)

    assert len(history) == 3
    assert len(synapse.updates) == 10
    assert sum(update[2] for update in synapse.updates) == pytest.approx(10e-3)
    assert sum(update[3] == 0.0 for update in synapse.updates) == 6
    assert synapse.updates[0][2] == pytest.approx(1e-3)
    assert synapse.updates[2][2] == pytest.approx(1e-3)


def test_stdp_hebbian_and_anti_hebbian():
    hebbian = STDPRule(A_plus=0.05, A_minus=-0.025, tau_plus=17e-3, tau_minus=34e-3)
    anti_hebbian = AntiSTDPRule(A_plus=0.05, A_minus=-0.025, tau_plus=17e-3, tau_minus=34e-3)

    # Pre antes que post (dt > 0) -> LTP Hebbiano
    dW_hebb_pos = hebbian.delta_w(0.010)
    assert dW_hebb_pos > 0.0

    # Post antes que pre (dt < 0) -> LTD Hebbiano
    dW_hebb_neg = hebbian.delta_w(-0.010)
    assert dW_hebb_neg < 0.0

    # Anti-Hebbiano invertido
    dW_anti_pos = anti_hebbian.delta_w(0.010)
    assert dW_anti_pos < 0.0

    dW_anti_neg = anti_hebbian.delta_w(-0.010)
    assert dW_anti_neg > 0.0


def test_stdp_changes_memristive_conductance_for_both_spike_orders():
    cfg = StrukovConfig(RON=100.0, ROFF=16_000.0, x0=0.50)
    synapse = MemristiveSynapse(MemristorStrukov(cfg))
    stdp = STDPRule(
        A_plus=0.05,
        A_minus=-0.025,
        tau_plus=17e-3,
        tau_minus=34e-3,
    )
    initial_conductance = synapse.conductance

    # Project convention: dt = t_post - t_pre, so pre-before-post is LTP.
    stdp.apply(synapse, dt=10e-3)
    potentiated_conductance = synapse.conductance
    assert potentiated_conductance > initial_conductance

    synapse.conductance = initial_conductance
    stdp.apply(synapse, dt=-10e-3)
    assert synapse.conductance < initial_conductance


def test_plasticity_config():
    config = PlasticityConfig()
    assert config.n_pulses_ltp == 40
    assert config.v_pulse_ltp == 1.0
    assert config.tau_plus == 17e-3
