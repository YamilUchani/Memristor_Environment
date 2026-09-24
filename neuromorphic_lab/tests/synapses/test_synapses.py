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


def test_plasticity_config():
    config = PlasticityConfig()
    assert config.n_pulses_ltp == 40
    assert config.v_pulse_ltp == 1.0
    assert config.tau_plus == 17e-3
