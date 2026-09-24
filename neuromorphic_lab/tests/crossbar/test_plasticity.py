"""
neuromorphic_lab/tests/test_plasticity.py
=========================================
Pruebas unitarias para las reglas de plasticidad (Trace, STDP, R-STDP).
"""

import pytest
import numpy as np
from neurolab.crossbar.plasticity import (
    Trace, STDPConfig, STDPRule, RSTDPConfig, RSTDPRule
)
from neurolab.crossbar import Crossbar
from neurolab.crossbar.configs import CrossbarConfig


def test_trace_decay_and_spike():
    trace = Trace(n_channels=4, tau=20e-3)
    assert np.all(trace.values == 0.0)
    
    # Step with spike on channel 1
    spikes = np.array([0.0, 1.0, 0.0, 0.0])
    trace.step(spikes, dt=1e-3)
    assert pytest.approx(trace.values[1], rel=1e-3) == 1.0
    
    # Step without spikes: exponential decay
    trace.step(np.zeros(4), dt=1e-3)
    expected_decay = np.exp(-1e-3 / 20e-3)
    assert pytest.approx(trace.values[1], rel=1e-3) == expected_decay
    assert trace.values[0] == 0.0
    
    # Reset
    trace.reset()
    assert np.all(trace.values == 0.0)


def test_stdp_rule_ltp_ltd():
    cfg = STDPConfig(A_plus=0.05, A_minus=0.025, tau_plus=20e-3, tau_minus=20e-3, eta=1.0)
    stdp = STDPRule(cfg)
    stdp.reset(4, 4)
    
    # Pre spike on row 1 (index 0) at t=0
    dG_t0 = stdp.apply(np.zeros((4,4)), np.array([1., 0., 0., 0.]), np.array([0., 0., 0., 0.]), dt=1e-3)
    assert np.all(dG_t0 == 0.0)
    assert pytest.approx(stdp.trace_pre.values[0], rel=1e-3) == 1.0
    
    # Post spike on col 2 (index 1) at t=1ms (pre before post -> LTP)
    dG_t1 = stdp.apply(np.zeros((4,4)), np.array([0., 0., 0., 0.]), np.array([0., 1., 0., 0.]), dt=1e-3)
    assert dG_t1[0, 1] > 0.0  # LTP on cell (0,1)
    assert dG_t1[1, 0] == 0.0
    
    # Post spike on col 3 (index 2) at t=2ms, then pre spike on row 2 (index 1) at t=3ms (post before pre -> LTD)
    stdp.reset(4, 4)
    stdp.apply(np.zeros((4,4)), np.array([0., 0., 0., 0.]), np.array([0., 0., 1., 0.]), dt=1e-3)
    dG_ltd = stdp.apply(np.zeros((4,4)), np.array([0., 1., 0., 0.]), np.array([0., 0., 0., 0.]), dt=1e-3)
    assert dG_ltd[1, 2] < 0.0  # LTD on cell (1,2)


def test_rstdp_rule_reward_gating():
    cfg = RSTDPConfig(A_plus=0.05, A_minus=0.025, R=0.0)
    rstdp = RSTDPRule(cfg)
    rstdp.reset(4, 4)
    
    # Pre spike
    rstdp.apply(np.zeros((4,4)), np.array([1., 0., 0., 0.]), np.array([0., 0., 0., 0.]), dt=1e-3)
    
    # R = 0.0 -> no dG applied
    rstdp.set_reward(0.0)
    dG_r0 = rstdp.apply(np.zeros((4,4)), np.array([0., 0., 0., 0.]), np.array([0., 1., 0., 0.]), dt=1e-3)
    assert np.all(dG_r0 == 0.0)
    
    # R = +1.0 -> positive reinforcement
    rstdp.reset(4, 4)
    rstdp.apply(np.zeros((4,4)), np.array([1., 0., 0., 0.]), np.array([0., 0., 0., 0.]), dt=1e-3)
    rstdp.set_reward(+1.0)
    dG_rplus = rstdp.apply(np.zeros((4,4)), np.array([0., 0., 0., 0.]), np.array([0., 1., 0., 0.]), dt=1e-3)
    assert dG_rplus[0, 1] > 0.0
    
    # R = -1.0 -> negative reinforcement (inverted sign)
    rstdp.reset(4, 4)
    rstdp.apply(np.zeros((4,4)), np.array([1., 0., 0., 0.]), np.array([0., 0., 0., 0.]), dt=1e-3)
    rstdp.set_reward(-1.0)
    dG_rminus = rstdp.apply(np.zeros((4,4)), np.array([0., 0., 0., 0.]), np.array([0., 1., 0., 0.]), dt=1e-3)
    assert dG_rminus[0, 1] < 0.0
