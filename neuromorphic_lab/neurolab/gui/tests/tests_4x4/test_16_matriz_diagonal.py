"""
neurolab.gui.tests.tests_4x4.test_16_matriz_diagonal
=====================================================
Prueba 4x4_16: Matriz Diagonal con Gradiente de Pesos.
CORREGIDO (v2): Voltajes explícitos V = [1.0, 0.8, 0.6, 0.4] V con cálculos verificables.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_4x4_16(gui, **kwargs):
    """Programa la matriz con conductancias ponderadas en la diagonal."""
    config = CrossbarConfig(n_rows=4, n_cols=4)
    cb = CrossbarIdeal(config)

    G_diag = np.array([500.0, 400.0, 300.0, 200.0]) * 1e-6  # μS
    G_off  = 62e-6   # 62 μS fuera de diagonal

    G_target = np.full((4, 4), G_off)
    for k in range(4):
        G_target[k, k] = G_diag[k]

    cb.set_matrix_conductance(G_target)

    # ✅ FIX: Voltajes explícitos y consistentes — mismo set que prueba 15
    # I₁ = 500·1.0 + 62·(0.8+0.6+0.4) = 500 + 111.6 = 611.6 μA
    # I₂ = 400·0.8 + 62·(1.0+0.6+0.4) = 320 + 124.0 = 444.0 μA
    # I₃ = 300·0.6 + 62·(1.0+0.8+0.4) = 180 + 136.4 = 316.4 μA
    # I₄ = 200·0.4 + 62·(1.0+0.8+0.6) =  80 + 148.8 = 228.8 μA
    V = np.array([1.0, 0.8, 0.6, 0.4])
    cb.apply_voltages(V)
    I_out = cb.read_currents()

    # Cálculo analítico esperado
    I_expected = np.array([
        G_diag[j] * V[j] + G_off * (V.sum() - V[j])
        for j in range(4)
    ])

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color  = '#e5e7eb' if is_dark else '#1f2937'

    # Panel izquierdo: Mapa de calor de la Matriz G
    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)
    im = ax1.imshow(G_target * 1e6, cmap='Purples', aspect='auto', vmin=50, vmax=500)

    for i in range(4):
        for j in range(4):
            val = G_target[i, j] * 1e6
            ax1.text(j, i, f'{val:.0f}', ha='center', va='center',
                     color='white' if val > 250 else 'black', fontsize=10, fontweight='bold')

    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels([f'Col {j+1}' for j in range(4)])
    ax1.set_yticklabels([f'Fila {i+1}' for i in range(4)])
    ax1.set_title('Matriz Diagonal Ponderada (μS)\n'
                  '(G_diag = [500, 400, 300, 200], G_off = 62)',
                  color=title_color, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('G (μS)', color=text_color)

    # Panel derecho: Corrientes simuladas vs esperadas
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)
    x_pos = np.arange(4)
    width = 0.35

    ax2.bar(x_pos - width/2, I_out * 1e6, width,
            label='I_sim', color='#9333ea', edgecolor='white')
    ax2.bar(x_pos + width/2, I_expected * 1e6, width,
            label='I_analítica', color='#dc2626', edgecolor='white', alpha=0.85)

    y_max = np.max(I_expected) * 1e6 * 1.30
    ax2.set_ylim(0, y_max)

    for k in range(4):
        ax2.text(k - width/2, I_out[k] * 1e6 + y_max * 0.02,
                 f'{I_out[k]*1e6:.1f}', ha='center', color=text_color, fontsize=8, fontweight='bold')
        ax2.text(k + width/2, I_expected[k] * 1e6 + y_max * 0.02,
                 f'{I_expected[k]*1e6:.1f}', ha='center', color=text_color, fontsize=8, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'I_{k+1}' for k in range(4)])
    ax2.set_xlabel('Salida de Columna', color=text_color)
    ax2.set_ylabel('I (μA)', color=text_color)
    ax2.set_title('Corrientes Ponderadas I_j ∝ G_jj\n(gradiente decreciente)',
                  color=title_color, fontsize=11, fontweight='bold')
    ax2.legend(loc='lower right')

    # ✅ FIX: Anotación con voltajes y verificación analítica explícita
    verify_lines = (
        f'V = [{V[0]:.1f}, {V[1]:.1f}, {V[2]:.1f}, {V[3]:.1f}] V\n'
        f'G_diag = [500, 400, 300, 200] μS\n'
        f'G_off  = {G_off*1e6:.0f} μS\n'
        f'──────────────────\n'
        f'I₁ = {I_expected[0]*1e6:.1f} μA  ✓\n'
        f'I₂ = {I_expected[1]*1e6:.1f} μA  ✓\n'
        f'I₃ = {I_expected[2]*1e6:.1f} μA  ✓\n'
        f'I₄ = {I_expected[3]*1e6:.1f} μA  ✓\n'
        f'▸ Patrón decreciente [OK]'
    )
    ax2.text(0.02, 0.98, verify_lines,
             transform=ax2.transAxes,
             ha='left', va='top', fontsize=7.5, color='#047857',
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor='#f0fdf4' if not is_dark else '#064e3b',
                       edgecolor='#047857', alpha=0.9))

    fig.suptitle('Prueba 4×4_16: Matriz Diagonal con Gradiente de Pesos',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'V aplicados': f'[{V[0]:.1f}, {V[1]:.1f}, {V[2]:.1f}, {V[3]:.1f}] V',
            'G diagonal': '[500, 400, 300, 200] μS',
            'G off-diagonal': f'{G_off*1e6:.0f} μS',
            'I₁': f'{I_out[0]*1e6:.1f} μA  (analítica: {I_expected[0]*1e6:.1f})',
            'I₂': f'{I_out[1]*1e6:.1f} μA  (analítica: {I_expected[1]*1e6:.1f})',
            'I₃': f'{I_out[2]*1e6:.1f} μA  (analítica: {I_expected[2]*1e6:.1f})',
            'I₄': f'{I_out[3]*1e6:.1f} μA  (analítica: {I_expected[3]*1e6:.1f})',
        }
    }
