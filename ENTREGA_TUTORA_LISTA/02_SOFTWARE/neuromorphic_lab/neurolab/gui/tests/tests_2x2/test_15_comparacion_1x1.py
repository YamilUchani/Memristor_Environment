"""
Prueba 2x2_15: Comparación directa 1×1 vs 2×2.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_2x2_15(gui, **kwargs):
    """
    Comparación directa de arquitectura y operación 1×1 vs 2×2.
    """
    config_1x1 = CrossbarConfig(n_rows=1, n_cols=1)
    cb_1x1 = CrossbarIdeal(config_1x1)
    cb_1x1.set_conductance(0, 0, 200e-6)
    cb_1x1.apply_voltages([0.8])
    I_1x1 = cb_1x1.read_currents()[0]

    config_2x2 = CrossbarConfig(n_rows=2, n_cols=2)
    cb_2x2 = CrossbarIdeal(config_2x2)
    G_2x2 = np.array([
        [200e-6, 100e-6],
        [150e-6, 250e-6]
    ])
    for i in range(2):
        for j in range(2):
            cb_2x2.set_conductance(i, j, G_2x2[i, j])
    cb_2x2.apply_voltages([0.8, 0.4])
    I_2x2 = cb_2x2.read_currents()

    fig = gui.figure
    fig.clear()

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)
    ax1.bar(['I_out (1×1)'], [I_1x1 * 1e6], width=0.4, color='#4a9eff', edgecolor='white')
    ax1.set_ylabel('Corriente (μA)', color='white')
    ax1.set_title('Crossbar 1×1 (Escalar: I = G·V)', color='white', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.text(0, I_1x1 * 1e6 + 2, f'{I_1x1*1e6:.2f} μA', ha='center', va='bottom', color='white', fontweight='bold')

    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)
    x_pos = np.arange(2)
    width = 0.35
    bars = ax2.bar(x_pos, I_2x2 * 1e6, width, color='#ff6b6b', edgecolor='white')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['I₁ (Col 1)', 'I₂ (Col 2)'])
    ax2.set_ylabel('Corriente (μA)', color='white')
    ax2.set_title('Crossbar 2×2 (Matricial: I = G^T·V)', color='white', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='y')
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 2, f'{h:.2f} μA', ha='center', va='bottom', color='white', fontweight='bold')

    fig.tight_layout()
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            '1×1 Entradas/Salidas': '1 PRE / 1 POST',
            '2×2 Entradas/Salidas': '2 PRE / 2 POST',
            '1×1 Operación': 'I = G·V',
            '2×2 Operación': 'I = G^T·V',
            'Capacidad MAC': '1 MAC vs 4 MACs',
        }
    }
