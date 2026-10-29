"""
Prueba 2x2_12: Interferencia entre celdas (cross-talk).
"""

import numpy as np
import matplotlib.pyplot as plt
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.synapses import MemristiveSynapse, LTPRule


def draw_2x2_12(gui, **kwargs):
    """
    Mide la interferencia: programar M11 y ver el efecto en las otras.
    """
    config = CrossbarConfig(n_rows=2, n_cols=2, x0=0.10)
    cb = CrossbarIdeal(config)

    G_initial = cb.G_matrix.copy()

    syn = MemristiveSynapse(cb.memristors[0, 0])
    ltp = LTPRule(n_pulses=40, V_pulse=+1.0)
    ltp.apply(syn, dt=1e-3)

    G_final = cb.G_matrix.copy()
    delta = (G_final - G_initial) * 1e6

    cross_talk = np.abs(delta).sum() - np.abs(delta[0, 0])

    fig = gui.figure
    fig.clear()

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    im = ax1.imshow(delta, cmap='RdBu_r',
                     aspect='auto', vmin=-0.1, vmax=2.5)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Col 1', 'Col 2'])
    ax1.set_yticklabels(['Fila 1', 'Fila 2'])
    ax1.set_title('ΔG tras programar M11 (μS)',
                  color='white', fontsize=12, fontweight='bold')

    for i in range(2):
        for j in range(2):
            val = delta[i, j]
            color = 'white' if abs(val) > 1.0 else 'black'
            ax1.text(j, i, f'{val:+.4f}',
                     ha='center', va='center',
                     color=color, fontsize=13, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('ΔG (μS)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    labels = ['M11\n(programado)', 'M12', 'M21', 'M22']
    values = [delta[0, 0], delta[0, 1], delta[1, 0], delta[1, 1]]
    colors = ['#4a9eff' if abs(v) > 0.5 else '#ff6b6b' for v in values]

    bars = ax2.bar(labels, values, color=colors, edgecolor='white')

    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 h + 0.05 if h > 0 else h - 0.05,
                 f'{val:+.4f}',
                 ha='center', va='bottom' if h > 0 else 'top',
                 color='white', fontsize=10, fontweight='bold')

    ax2.axhline(0, color='white', ls='--', lw=1, alpha=0.5)
    ax2.set_ylabel('ΔG (μS)', color='white')
    ax2.set_title('Análisis de Cross-talk',
                  color='white', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='y')

    rel_pct = (cross_talk / max(abs(delta[0, 0]), 1e-15)) * 100
    ax2.text(0.98, 0.98,
             f'Cross-talk: {cross_talk:.2e} μS\n'
             f'Relación: {rel_pct:.4f}%',
             transform=ax2.transAxes,
             ha='right', va='top', fontsize=10, color='#10ac84',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                       edgecolor='#10ac84', alpha=0.9))

    fig.tight_layout()
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'ΔG M11': f'{delta[0,0]:+.4f} μS',
            'ΔG M12': f'{delta[0,1]:+.6f} μS',
            'ΔG M21': f'{delta[1,0]:+.6f} μS',
            'ΔG M22': f'{delta[1,1]:+.6f} μS',
            'Cross-talk': f'{cross_talk:.2e} μS',
        }
    }
