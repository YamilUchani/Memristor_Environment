"""
Prueba 2x2_11: STDP con normalización porcentual.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.synapses import MemristiveSynapse, STDPRule


def draw_2x2_11(gui, max_percent=0.5, **kwargs):
    """
    Aplica STDP con normalización porcentual.
    """
    max_percent = float(max_percent)

    dt_values = np.linspace(-80e-3, 80e-3, 41)
    dW_values = []
    dG_values = []

    for dt in dt_values:
        config = CrossbarConfig(n_rows=2, n_cols=2, x0=0.10)
        cb = CrossbarIdeal(config)
        cb.set_uniform_conductance(200e-6)

        syn = MemristiveSynapse(cb.memristors[0, 0])
        stdp = STDPRule(max_percent_change=max_percent)

        G_before = syn.conductance
        dW = stdp.apply(syn, dt)
        G_after = syn.conductance

        dW_values.append(dW)
        dG_values.append((G_after - G_before) * 1e6)  # en μS

    dW_values = np.array(dW_values)
    dG_values = np.array(dG_values)

    # ==== DIBUJAR ====
    fig = gui.figure
    fig.clear()

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)
    ax1.plot(dt_values * 1e3, dW_values, '-o', ms=5, lw=2, color='#89b4fa')
    ax1.axhline(0, color='white', ls='--', lw=1, alpha=0.5)
    ax1.axvline(0, color='white', ls='--', lw=1, alpha=0.5)
    ax1.fill_between(dt_values * 1e3, 0, dW_values,
                      where=(dW_values > 0), color='green', alpha=0.2,
                      label='LTP')
    ax1.fill_between(dt_values * 1e3, 0, dW_values,
                      where=(dW_values < 0), color='red', alpha=0.2,
                      label='LTD')
    ax1.set_xlabel('Δt = t_post − t_pre (ms)', color='white')
    ax1.set_ylabel('ΔW (normalizado)', color='white')
    ax1.set_title('Ventana STDP (ΔW)',
                  color='white', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', facecolor='#252526',
               edgecolor='#555', labelcolor='white')
    ax1.grid(True, alpha=0.2)

    # Subplot 2: ΔG en μS
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)
    ax2.plot(dt_values * 1e3, dG_values, '-o', ms=5, lw=2, color='#f5c2e7')
    ax2.axhline(0, color='white', ls='--', lw=1, alpha=0.5)
    ax2.axvline(0, color='white', ls='--', lw=1, alpha=0.5)
    ax2.fill_between(dt_values * 1e3, 0, dG_values,
                      where=(dG_values > 0), color='green', alpha=0.2)
    ax2.fill_between(dt_values * 1e3, 0, dG_values,
                      where=(dG_values < 0), color='red', alpha=0.2)
    ax2.set_xlabel('Δt = t_post − t_pre (ms)', color='white')
    ax2.set_ylabel('ΔG (μS)', color='white')
    ax2.set_title(f'Cambio físico (máx {max_percent}% por evento)',
                  color='white', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2)

    ax2.text(0.02, 0.98,
             f'max_percent = {max_percent}%\n'
             f'ΔG max = {dG_values.max():+.4f} μS\n'
             f'ΔG min = {dG_values.min():+.4f} μS',
             transform=ax2.transAxes,
             ha='left', va='top', fontsize=10, color='#10ac84',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                       edgecolor='#10ac84', alpha=0.9))

    fig.tight_layout()
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'ΔW max': f'{dW_values.max():+.4f}',
            'Δt (ΔW max)': f'{dt_values[np.argmax(dW_values)]*1e3:+.1f} ms',
            'ΔG max': f'{dG_values.max():+.4f} μS',
            'ΔG min': f'{dG_values.min():+.4f} μS',
            'max_percent': f'{max_percent}%',
        }
    }

