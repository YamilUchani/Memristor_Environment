"""
Prueba 4.2.4: 2×2 → 2 Neuronas LIF Dinámicas.
"""

import numpy as np
import pytest
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.neurons import LIFNeuron, LIFConfig


def test_2x2_lif_dinamico():
    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)
    cb.set_uniform_conductance(200e-6)

    lif_cfg = LIFConfig(c_m=100e-9, r_leak=1e6, v_th=2.0, v_reset=0.0)
    lif_1 = LIFNeuron(lif_cfg)
    lif_2 = LIFNeuron(lif_cfg)

    dt = 1e-4
    t = np.arange(0, 0.05, dt)
    spikes_1 = 0
    spikes_2 = 0

    for tk in t:
        V1 = 1.0
        V2 = 1.0
        cb.apply_voltages([V1, V2])
        I = cb.read_currents()

        if lif_1.update(I_in=I[0], dt=dt):
            spikes_1 += 1
        if lif_2.update(I_in=I[1], dt=dt):
            spikes_2 += 1

    assert spikes_1 > 0
    assert spikes_2 > 0
    assert spikes_1 == spikes_2  # Por simetría en la entrada
