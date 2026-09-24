"""
neurolab.gui.tests.tests_4x4.test_11_stdp
==========================================
Prueba 4x4_11: Ventana STDP con Escalado Físico en 4×4.
"""

import numpy as np
from neurolab.synapses.stdp import STDPRule
from neurolab.synapses import PlasticityConfig


def draw_4x4_11(gui, **kwargs):
    """Ventana STDP adaptada a la escala de conductancia de la matriz 4x4."""
    max_percent = float(kwargs.get('max_percent', 0.5))

    stdp = STDPRule(max_percent_change=max_percent)

    dt_ms = np.linspace(-50, 50, 101)
    dG_uS = np.zeros_like(dt_ms)
    G_initial_uS = 200.0  # 200 μS en celda 4x4

    for k, delta_t in enumerate(dt_ms):
        t_sec = delta_t * 1e-3
        dW_norm = stdp.delta_w(t_sec)
        max_dG = G_initial_uS * (max_percent / 100.0)
        if delta_t > 0:
            fraction = dW_norm / stdp.A_plus if stdp.A_plus != 0 else 0.0
        elif delta_t < 0:
            fraction = dW_norm / abs(stdp.A_minus) if stdp.A_minus != 0 else 0.0
        else:
            fraction = 0.0
        dG_uS[k] = max_dG * fraction

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    ax1.plot(dt_ms[dt_ms > 0], dG_uS[dt_ms > 0], color='#2563eb', linewidth=2.5, label='LTP (Post - Pre > 0)')
    ax1.plot(dt_ms[dt_ms < 0], dG_uS[dt_ms < 0], color='#dc2626', linewidth=2.5, label='LTD (Post - Pre < 0)')
    ax1.axhline(0, color='#4b5563', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='#4b5563', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Diferencia de Tiempo Δt = t_post − t_pre (ms)', color=text_color)
    ax1.set_ylabel('Cambio de Conductancia ΔG (μS)', color=text_color)
    ax1.set_title('Ventana STDP Asimétrica (G_0 = 200 μS)', color=title_color, fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right')

    # PANEL 2: Diagrama de Matriz 4x4 STDP
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    # Matriz de cambios dG aleatorios STDP
    rng = np.random.default_rng(123)
    dt_matrix_ms = rng.uniform(-40, 40, (4, 4))
    dG_matrix_uS = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            dt_ij = dt_matrix_ms[i, j]
            dW_ij = stdp.delta_w(dt_ij * 1e-3)
            max_dG_ij = G_initial_uS * (max_percent / 100.0)
            if dt_ij > 0:
                frac_ij = dW_ij / stdp.A_plus if stdp.A_plus != 0 else 0.0
            elif dt_ij < 0:
                frac_ij = dW_ij / abs(stdp.A_minus) if stdp.A_minus != 0 else 0.0
            else:
                frac_ij = 0.0
            dG_matrix_uS[i, j] = max_dG_ij * frac_ij

    im = ax2.imshow(dG_matrix_uS, cmap='coolwarm', aspect='auto', vmin=-1.2, vmax=1.2)

    for i in range(4):
        for j in range(4):
            val = dG_matrix_uS[i, j]
            ax2.text(j, i, f'{val:+.2f}', ha='center', va='center',
                     color='white' if abs(val) > 0.6 else 'black', fontsize=9, fontweight='bold')

    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels([f'Col {j+1}' for j in range(4)])
    ax2.set_yticklabels([f'Fila {i+1}' for i in range(4)])
    ax2.set_title('Matriz ΔG Resultante por Parejas de Spikes (μS)', color=title_color, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('ΔG (μS)', color=text_color)

    fig.suptitle('Prueba 4×4_11: Ventana STDP con Escalado Porcentual Físico Proporcional',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    max_ltp = np.max(dG_uS)
    max_ltd = np.min(dG_uS)

    return {
        'status': 'PASS',
        'metrics': {
            'Max ΔG LTP': f'+{max_ltp:.3f} μS',
            'Max ΔG LTD': f'{max_ltd:.3f} μS',
            'Escala porcentual': f'{max_percent:.1f} %',
        }
    }
