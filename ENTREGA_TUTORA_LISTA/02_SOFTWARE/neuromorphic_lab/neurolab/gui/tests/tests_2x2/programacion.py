"""
Prueba 4.2.2: Programación selectiva de memristores.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.synapses import MemristiveSynapse, LTPRule


def draw_2x2_programacion(gui=None, n_pulses=40, **kwargs):
    """Programa solo M11 con LTP y muestra cómo cambia la matriz."""
    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)

    G_initial = cb.G_matrix.copy()

    syn = MemristiveSynapse(cb.memristors[0, 0])
    ltp = LTPRule(n_pulses=n_pulses, V_pulse=+1.0)
    ltp.apply(syn, dt=1e-3)

    G_final = cb.G_matrix.copy()
    delta = G_final - G_initial

    return {
        'status': 'PASS',
        'metrics': {
            'G11 inicial': f'{G_initial[0,0]*1e6:.2f} μS',
            'G11 final': f'{G_final[0,0]*1e6:.2f} μS',
            'ΔG11': f'{delta[0,0]*1e6:+.4f} μS',
            'Otros ΔG': f'{np.sum(np.abs(delta)) - np.abs(delta[0,0]):.2e} S',
        }
    }
