"""
Prueba 2x2_07: Ciclo LTP → LTD → LTP en una celda del 2×2.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.synapses import MemristiveSynapse


def draw_2x2_07(gui, n_ltp=50, n_ltd=50, **kwargs):
    """
    Aplica ciclo completo a M11 y verifica reversibilidad.
    """
    n_ltp = int(n_ltp)
    n_ltd = int(n_ltd)

    config = CrossbarConfig(n_rows=2, n_cols=2, x0=0.10)
    cb = CrossbarIdeal(config)

    cb.set_uniform_conductance(200e-6)

    syn = MemristiveSynapse(cb.memristors[0, 0])

    G_inicial = syn.conductance
    G_history = [G_inicial]
    pulsos = [0]

    for _ in range(n_ltp):
        syn.update(pre_spike=True, post_spike=False,
                   dt=1e-3, V_applied=+1.0)
        G_history.append(syn.conductance)
        pulsos.append(len(pulsos))

    G_max = G_history[-1]

    for _ in range(n_ltd):
        syn.update(pre_spike=False, post_spike=True,
                   dt=1e-3, V_applied=-1.0)
        G_history.append(syn.conductance)
        pulsos.append(len(pulsos))

    G_min = G_history[-1]

    for _ in range(n_ltp):
        syn.update(pre_spike=True, post_spike=False,
                   dt=1e-3, V_applied=+1.0)
        G_history.append(syn.conductance)
        pulsos.append(len(pulsos))

    G_final = G_history[-1]
    G_history = np.array(G_history)

    error_retorno = abs(G_final - G_max) / G_max * 100

    ax = gui.get_axis()
    ax.clear()
    gui._style_axis(ax)

    ax.plot(pulsos[:n_ltp+1], G_history[:n_ltp+1] * 1e6,
            'b-o', ms=3, lw=1.8, label=f'LTP #1 (+1V, {n_ltp})')
    ax.plot(pulsos[n_ltp:n_ltp+n_ltd+1],
            G_history[n_ltp:n_ltp+n_ltd+1] * 1e6,
            'r-s', ms=3, lw=1.8, label=f'LTD (−1V, {n_ltd})')
    ax.plot(pulsos[n_ltp+n_ltd:], G_history[n_ltp+n_ltd:] * 1e6,
            'g-^', ms=3, lw=1.8, label=f'LTP #2 (+1V, {n_ltp})')

    ax.axvline(n_ltp, color='gray', ls='--', alpha=0.5)
    ax.axvline(n_ltp + n_ltd, color='gray', ls='--', alpha=0.5)
    ax.axhline(G_max * 1e6, color='orange', ls=':', lw=1, alpha=0.6)

    ax.set_xlabel('Número de pulsos consecutivos', color='white', fontsize=11)
    ax.set_ylabel('G₁₁ (μS)', color='white', fontsize=11)
    ax.set_title('Prueba 2×2_07: Ciclo LTP → LTD → LTP en M11',
                 color='white', fontsize=13, fontweight='bold')
    ax.legend(loc='center right', facecolor='#252526',
              edgecolor='#555', labelcolor='white')
    ax.grid(True, alpha=0.2)

    ax.text(0.02, 0.98,
            f'G inicial: {G_inicial*1e6:.3f} μS\n'
            f'G max: {G_max*1e6:.3f} μS\n'
            f'G min: {G_min*1e6:.3f} μS\n'
            f'G final: {G_final*1e6:.3f} μS\n'
            f'Error retorno: {error_retorno:.4f}%',
            transform=ax.transAxes,
            ha='left', va='top', fontsize=10, color='#10ac84',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                      edgecolor='#10ac84', alpha=0.9))

    gui.refresh_plot()

    return {
        'status': 'PASS' if error_retorno < 1.0 else 'FAIL',
        'metrics': {
            'G inicial': f'{G_inicial*1e6:.3f} μS',
            'G max': f'{G_max*1e6:.3f} μS',
            'G min': f'{G_min*1e6:.3f} μS',
            'Error retorno': f'{error_retorno:.4f} %',
        }
    }
