"""
Prueba 4.2.3: Efecto de Sneak Paths en Crossbar 2×2.
"""

import numpy as np
import pytest
from neurolab.crossbar import CrossbarIdeal, CrossbarSneak, CrossbarConfig


def test_2x2_sneak_paths():
    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb_ideal = CrossbarIdeal(config)
    cb_sneak = CrossbarSneak(config)

    cb_ideal.set_uniform_conductance(200e-6)
    cb_sneak.set_uniform_conductance(200e-6)

    V = np.array([0.5, 0.5])
    cb_ideal.apply_voltages(V)
    cb_sneak.apply_voltages(V)

    I_ideal = cb_ideal.read_currents()
    I_sneak = cb_sneak.read_currents()

    # La corriente con sneak path debe ser mayor debido a las corrientes parásitas
    assert np.all(I_sneak >= I_ideal)
    assert len(I_sneak) == 2
