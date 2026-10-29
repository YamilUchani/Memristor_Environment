"""
neurolab.gui.tests.tests_4x4.test_13_escalabilidad
===================================================
Prueba 4x4_13: Escalabilidad del Arreglo (1×1 vs 2×2 vs 4×4).
"""

import time
import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_4x4_13(gui, **kwargs):
    """Evalúa la velocidad de lectura y actualización en arreglos 1x1, 2x2 y 4x4."""
    sizes = [(1, 1), (2, 2), (4, 4)]
    names = ['1×1 (1 cell)', '2×2 (4 cells)', '4×4 (16 cells)']
    n_runs = 1000

    t_read = []
    t_update = []

    for nr, nc in sizes:
        cfg = CrossbarConfig(n_rows=nr, n_cols=nc)
        cb = CrossbarIdeal(cfg)
        V = np.ones(nr) * 0.5
        cb.apply_voltages(V)

        # Benchmark Lectura
        t0 = time.perf_counter()
        for _ in range(n_runs):
            cb.read_currents()
        t1 = time.perf_counter()
        t_read.append((t1 - t0) / n_runs * 1e6)  # en us

        # Benchmark Update
        t0 = time.perf_counter()
        for _ in range(n_runs):
            cb.update_memristors(1e-3)
        t1 = time.perf_counter()
        t_update.append((t1 - t0) / n_runs * 1e6)  # en us

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)
    x_pos = np.arange(len(sizes))
    width = 0.35

    ax1.bar(x_pos - width/2, t_read, width, label='Lectura I = G^T·V', color='#2563eb', edgecolor='white')
    ax1.bar(x_pos + width/2, t_update, width, label='Actualización dx/dt', color='#d97706', edgecolor='white')

    max_t = max(max(t_read), max(t_update))
    for k in range(3):
        ax1.text(k - width/2, t_read[k] + (max_t * 0.02), f'{t_read[k]:.1f}μs', ha='center', color=text_color, fontsize=8)
        ax1.text(k + width/2, t_update[k] + (max_t * 0.02), f'{t_update[k]:.1f}μs', ha='center', color=text_color, fontsize=8)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names)
    ax1.set_ylabel('Tiempo por Paso (μs)', color=text_color)
    ax1.set_title('Tiempo de Ejecución según Tamaño de Matriz', color=title_color, fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, max_t * 1.25)

    # PANEL 2: Rendimiento Ops/s
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    ops_sec = [1e6 / t_r for t_r in t_read]
    ax2.plot(x_pos, ops_sec, 'o-', color='#047857', linewidth=2.5, markersize=8)

    for k in range(3):
        ax2.text(k, ops_sec[k] * 1.05, f'{ops_sec[k]/1e3:.1f}k ops/s', ha='center',
                 color=text_color, fontsize=9, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names)
    ax2.set_ylabel('Operaciones de Lectura / Segundo', color=text_color)
    ax2.set_title('Rendimiento Vectorial O(1)\nOperaciones Matriciales', color=title_color, fontsize=10, fontweight='bold', pad=25)

    fig.suptitle('Prueba 4×4_13: Escalabilidad y Rendimiento de Cómputo N×N',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'Lectura 1x1': f'{t_read[0]:.2f} μs',
            'Lectura 2x2': f'{t_read[1]:.2f} μs',
            'Lectura 4x4': f'{t_read[2]:.2f} μs',
            'Rendimiento 4x4': f'{ops_sec[2]/1e3:.1f} kOps/s',
        }
    }
