"""
Prueba 2x2_14: Programación uniforme de todas las celdas.
"""

import numpy as np
import matplotlib.pyplot as plt
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_2x2_14(gui, G_target=300.0, **kwargs):
    """
    Programa todas las celdas al mismo valor y verifica uniformidad.
    """
    G_target_val = float(G_target) * 1e-6

    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)

    cb.set_uniform_conductance(G_target_val)

    G_matrix = cb.G_matrix.copy()
    error = np.max(np.abs(G_matrix - G_target_val))
    error_rel = (error / G_target_val) * 100

    fig = gui.figure
    fig.clear()

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    im = ax1.imshow(G_matrix * 1e6, cmap='viridis', aspect='auto')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Col 1', 'Col 2'])
    ax1.set_yticklabels(['Fila 1', 'Fila 2'])
    ax1.set_title('Matriz G Programada (μS)', color='white', fontsize=12, fontweight='bold')

    for i in range(2):
        for j in range(2):
            val = G_matrix[i, j] * 1e6
            ax1.text(j, i, f'M{i+1}{j+1}\n{val:.2f} μS',
                     ha='center', va='center', color='white', fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Conductancia G (μS)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    labels = ['M11', 'M12', 'M21', 'M22']
    values = G_matrix.flatten() * 1e6
    bars = ax2.bar(labels, values, color='#4a9eff', edgecolor='white')
    ax2.axhline(float(G_target), color='#ff6b6b', ls='--', lw=1.5, label=f'Target = {G_target:.1f} μS')

    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 2,
                 f'{val:.2f}', ha='center', va='bottom',
                 color='white', fontsize=10, fontweight='bold')

    ax2.set_ylabel('G (μS)', color='white')
    ax2.set_title('Uniformidad de Celdas 2×2', color='white', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', facecolor='#252526', edgecolor='#555', labelcolor='white')
    ax2.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    gui.canvas.draw()

    return {
        'status': 'PASS' if error_rel < 0.01 else 'FAIL',
        'metrics': {
            'G target': f'{G_target:.2f} μS',
            'G medio': f'{np.mean(values):.2f} μS',
            'Desviación Std': f'{np.std(values):.2e} μS',
            'Error máx': f'{error*1e6:.2e} μS ({error_rel:.4f}%)',
        }
    }
