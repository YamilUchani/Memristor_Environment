"""
Prueba 2x2_08: Efecto de sneak paths en 2×2.
"""

import numpy as np
import matplotlib.pyplot as plt
from neurolab.crossbar import CrossbarIdeal, CrossbarSneak, CrossbarConfig


def draw_2x2_08(gui, seed=42, **kwargs):
    """
    Compara lectura ideal vs con sneak paths.
    """
    config = CrossbarConfig(n_rows=2, n_cols=2)

    cb_ideal = CrossbarIdeal(config)
    cb_sneak = CrossbarSneak(config)

    rng = np.random.default_rng(seed)
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

    err_abs = np.abs(I_sneak - I_ideal)
    err_rel = err_abs / np.abs(I_ideal + 1e-15) * 100

    fig = gui.figure
    fig.clear()

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    x_pos = np.arange(2)
    width = 0.35

    ax1.bar(x_pos - width / 2, I_ideal * 1e6, width,
            label='Ideal', color='#4a9eff', edgecolor='white')
    ax1.bar(x_pos + width / 2, I_sneak * 1e6, width,
            label='Con sneak', color='#ff6b6b', edgecolor='white')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['Columna 1', 'Columna 2'])
    ax1.set_xlabel('Columna', color='white')
    ax1.set_ylabel('I (μA)', color='white')
    ax1.set_title('Comparación ideal vs sneak',
                  color='white', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', facecolor='#252526',
               edgecolor='#555', labelcolor='white')
    ax1.grid(True, alpha=0.2, axis='y')

    for k in range(2):
        ax1.text(k - width / 2, I_ideal[k] * 1e6 + 1,
                 f'{I_ideal[k]*1e6:.1f}', ha='center', va='bottom',
                 color='white', fontsize=9, fontweight='bold')
        ax1.text(k + width / 2, I_sneak[k] * 1e6 + 1,
                 f'{I_sneak[k]*1e6:.1f}', ha='center', va='bottom',
                 color='white', fontsize=9, fontweight='bold')

    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    im = ax2.imshow(G_target * 1e6, cmap='viridis',
                     aspect='auto', vmin=50, vmax=500)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Col 1', 'Col 2'])
    ax2.set_yticklabels(['Fila 1', 'Fila 2'])
    ax2.set_title('Matriz G (μS)',
                  color='white', fontsize=12, fontweight='bold')

    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{G_target[i,j]*1e6:.1f}',
                     ha='center', va='center',
                     color='white', fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('G (μS)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax1.text(0.02, 0.98,
             f'Error Col 1: {err_rel[0]:.2f}%\n'
             f'Error Col 2: {err_rel[1]:.2f}%',
             transform=ax1.transAxes,
             ha='left', va='top', fontsize=10, color='#ff9f43',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                       edgecolor='#ff9f43', alpha=0.9))

    fig.tight_layout()
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'I_ideal C1': f'{I_ideal[0]*1e6:.2f} μA',
            'I_sneak C1': f'{I_sneak[0]*1e6:.2f} μA',
            'Error C1': f'{err_rel[0]:.2f} %',
            'Error C2': f'{err_rel[1]:.2f} %',
        }
    }
