"""
Prueba 2x2_13: Escalabilidad 1×1 → 2×2.
"""

import numpy as np
import time
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_2x2_13(gui, **kwargs):
    """
    Compara tiempos de cómputo entre 1×1 y 2×2.
    """
    sizes = [1, 2]
    tiempos_lectura = []
    tiempos_update = []
    n_memristores = []

    for size in sizes:
        config = CrossbarConfig(n_rows=size, n_cols=size)
        cb = CrossbarIdeal(config)
        cb.set_uniform_conductance(200e-6)

        V = np.ones(size) * 0.5
        cb.apply_voltages(V)

        t0 = time.perf_counter()
        for _ in range(1000):
            cb.read_currents()
        t_read = (time.perf_counter() - t0) / 1000

        t0 = time.perf_counter()
        for _ in range(100):
            cb.update_memristors(1e-4)
        t_upd = (time.perf_counter() - t0) / 100

        tiempos_lectura.append(t_read * 1e6)
        tiempos_update.append(t_upd * 1e6)
        n_memristores.append(size * size)

    fig = gui.figure
    fig.clear()

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    x_pos = np.arange(len(sizes))
    width = 0.35

    ax1.bar(x_pos - width / 2, tiempos_lectura, width,
            label='Lectura', color='#4a9eff', edgecolor='white')
    ax1.bar(x_pos + width / 2, tiempos_update, width,
            label='Update', color='#ffb84d', edgecolor='white')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{s}×{s}\n({n} mem)' for s, n in
                          zip(sizes, n_memristores)])
    ax1.set_ylabel('Tiempo (μs)', color='white')
    ax1.set_title('Tiempos por operación',
                  color='white', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', facecolor='#252526',
               edgecolor='#555', labelcolor='white')
    ax1.grid(True, alpha=0.2, axis='y')

    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    ax2.plot(sizes, tiempos_lectura, 'b-o', ms=10, lw=2,
             label='Lectura')
    ax2.plot(sizes, tiempos_update, 'r-s', ms=10, lw=2,
             label='Update')

    ax2.set_xlabel('Tamaño N (N×N)', color='white')
    ax2.set_ylabel('Tiempo (μs)', color='white')
    ax2.set_title('Escalabilidad 1×1 → 2×2',
                  color='white', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', facecolor='#252526',
               edgecolor='#555', labelcolor='white')
    ax2.grid(True, alpha=0.2)
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([f'{s}×{s}' for s in sizes])

    fig.tight_layout()
    gui.canvas.draw()

    ratio = tiempos_lectura[1] / max(tiempos_lectura[0], 1e-9)

    return {
        'status': 'PASS',
        'metrics': {
            'Lectura 1×1': f'{tiempos_lectura[0]:.2f} μs',
            'Lectura 2×2': f'{tiempos_lectura[1]:.2f} μs',
            'Ratio': f'{ratio:.2f}x',
            'Update 1×1': f'{tiempos_update[0]:.2f} μs',
            'Update 2×2': f'{tiempos_update[1]:.2f} μs',
        }
    }
