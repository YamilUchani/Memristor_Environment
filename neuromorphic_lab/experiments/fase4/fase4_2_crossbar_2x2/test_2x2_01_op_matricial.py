"""
Prueba 4.2.1: Operación Matricial I = G^T · V con Crossbar 2×2.
"""

import numpy as np
import pytest
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def test_2x2_op_matricial():
    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)

    G_target = np.array([
        [200e-6, 100e-6],
        [150e-6, 250e-6],
    ])

    for i in range(2):
        for j in range(2):
            cb.set_conductance(i, j, G_target[i, j])

    V = np.array([0.8, 0.4])
    cb.apply_voltages(V)
    I_out = cb.read_currents()
    I_expected = G_target.T @ V

    mae = np.mean(np.abs(I_out - I_expected))
    assert mae < 1e-12, f"Error MAE demasiado alto: {mae}"
    assert len(I_out) == 2
    assert cb.get_conductance(0, 0) == 200e-6
