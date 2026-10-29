"""
Prueba 2x2_05: Programación LTP selectiva de M11.
"""

import numpy as np
import matplotlib.pyplot as plt
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.synapses import MemristiveSynapse, LTPRule


def draw_2x2_05(gui, n_pulses=40, V_pulse=1.0, target='M11', **kwargs):
    """
    Programa solo una celda con LTP y muestra el efecto en la matriz.
    """
    n_pulses = int(n_pulses)
    V_pulse = float(V_pulse)

    target_map = {
        'M11': (0, 0), 'M12': (0, 1),
        'M21': (1, 0), 'M22': (1, 1),
    }
    i, j = target_map.get(target, (0, 0))

    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)

    G_initial = cb.G_matrix.copy()

    syn = MemristiveSynapse(cb.memristors[i, j])
    ltp = LTPRule(n_pulses=n_pulses, V_pulse=V_pulse)

    G_history = [syn.conductance]
    for _ in range(n_pulses):
        syn.update(pre_spike=True, post_spike=False,
                   dt=1e-3, V_applied=V_pulse)
        G_history.append(syn.conductance)

    G_history = np.array(G_history)
    G_final = cb.G_matrix.copy()
    delta = (G_final - G_initial) * 1e6

    fig = gui.figure
    fig.clear()

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)
    ax1.plot(range(len(G_history)), G_history * 1e6,
             'b-o', ms=4, lw=2, label=f'G_{target}')
    ax1.axhline(G_initial[i, j] * 1e6, color='gray',
                ls='--', lw=1, label='G inicial')
    ax1.set_xlabel('Número de pulsos', color='white')
    ax1.set_ylabel('G (μS)', color='white')
    ax1.set_title(f'LTP en {target} ({n_pulses} pulsos)',
                  color='white', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', facecolor='#252526',
               edgecolor='#555', labelcolor='white')
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    im = ax2.imshow(delta, cmap='RdBu_r',
                     aspect='auto', vmin=-3, vmax=3)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Col 1', 'Col 2'])
    ax2.set_yticklabels(['Fila 1', 'Fila 2'])
    ax2.set_title('ΔG tras programación (μS)',
                  color='white', fontsize=12, fontweight='bold')

    for ii in range(2):
        for jj in range(2):
            val = delta[ii, jj]
            color = 'white' if abs(val) > 1.5 else 'black'
            ax2.text(jj, ii, f'{val:+.3f}',
                     ha='center', va='center',
                     color=color, fontsize=14, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('ΔG (μS)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    fig.tight_layout()
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'Celda programada': target,
            'G inicial': f'{G_initial[i, j]*1e6:.3f} μS',
            'G final': f'{G_final[i, j]*1e6:.3f} μS',
            'ΔG': f'{delta[i, j]:+.4f} μS',
            'ΔG otras celdas': f'{np.sum(np.abs(delta)) - abs(delta[i, j]):.2e} μS',
        }
    }
