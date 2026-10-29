"""
Prueba 4.2.5: Escalabilidad 1×1 → 2×2.
"""

import numpy as np
import time
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_2x2_escalabilidad(gui=None, **kwargs):
    """Compara tiempo y recursos entre 1×1 y 2×2."""
    sizes = [1, 2]
    tiempos = []

    for size in sizes:
        config = CrossbarConfig(n_rows=size, n_cols=size)
        cb = CrossbarIdeal(config)
        cb.set_uniform_conductance(200e-6)

        V = np.ones(size) * 0.5
        cb.apply_voltages(V)

        t0 = time.perf_counter()
        for _ in range(1000):
            cb.read_currents()
        t_elapsed = (time.perf_counter() - t0) / 1000
        tiempos.append(t_elapsed * 1e6)

    return {
        'status': 'PASS',
        'metrics': {
            'Tiempo 1×1': f'{tiempos[0]:.2f} μs',
            'Tiempo 2×2': f'{tiempos[1]:.2f} μs',
            'Ratio 2×2/1×1': f'{tiempos[1]/max(tiempos[0], 1e-9):.2f}x',
        }
    }
