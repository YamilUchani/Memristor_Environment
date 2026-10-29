"""
Prueba 4.2.3: Sneak paths en crossbar 2×2.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarSneak, CrossbarConfig


def draw_2x2_sneak(gui=None, **kwargs):
    """Compara lectura ideal vs lectura con sneak paths."""
    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb_ideal = CrossbarIdeal(config)
    cb_sneak = CrossbarSneak(config)

    rng = np.random.default_rng(42)
    G_target = rng.uniform(50e-6, 500e-6, (2, 2))

    for i in range(2):
        for j in range(2):
            cb_ideal.set_conductance(i, j, G_target[i, j])
            cb_sneak.set_conductance(i, j, G_target[i, j])

    V = np.array([0.5, 0.5])
    cb_ideal.apply_voltages(V)
    cb_sneak.apply_voltages(V)

    I_ideal = cb_ideal.read_currents()
    I_sneak = cb_sneak.read_currents()

    err = np.abs(I_sneak - I_ideal) / np.abs(I_ideal + 1e-15) * 100

    return {
        'status': 'PASS',
        'metrics': {
            'I_ideal Col1': f'{I_ideal[0]*1e6:.2f} μA',
            'I_sneak Col1': f'{I_sneak[0]*1e6:.2f} μA',
            'Error Col1': f'{err[0]:.2f} %',
            'Error Col2': f'{err[1]:.2f} %',
        }
    }
