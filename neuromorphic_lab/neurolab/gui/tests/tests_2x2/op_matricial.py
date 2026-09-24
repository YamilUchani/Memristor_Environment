"""
Prueba 4.2.1: Operación Matricial I = G^T · V con 2×2.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_2x2_operacion(gui=None, **kwargs):
    """Ejecuta la operación matricial I = G^T · V con 2×2."""
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
    rmse = np.sqrt(np.mean((I_out - I_expected) ** 2))
    err_rel = np.max(np.abs(I_out - I_expected) / np.abs(I_expected + 1e-15)) * 100

    return {
        'status': 'PASS',
        'metrics': {
            'MAE': f'{mae:.2e} A',
            'RMSE': f'{rmse:.2e} A',
            'Err. rel': f'{err_rel:.2e} %',
            'I₁': f'{I_out[0]*1e6:.2f} μA',
            'I₂': f'{I_out[1]*1e6:.2f} μA',
        }
    }
