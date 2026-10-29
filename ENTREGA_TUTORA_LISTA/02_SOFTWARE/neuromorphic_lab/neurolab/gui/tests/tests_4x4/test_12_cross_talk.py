"""
neurolab.gui.tests.tests_4x4.test_12_cross_talk
================================================
Prueba 4x4_12: Cross-talk e Interferencia Parásita entre Líneas (Evaluación Cuantitativa).
"""

import numpy as np
from neurolab.crossbar import Crossbar, CrossbarConfig


def draw_4x4_12(gui, **kwargs):
    """Verifica cuantitativamente el aislamiento y acotación del cross-talk en programación real."""
    cb = Crossbar(CrossbarConfig(n_rows=4, n_cols=4, x0=0.30))
    cb.set_uniform_conductance(69.4e-6)
    G_before = cb.G_matrix.copy()

    # Programación selectiva de la celda (0,0) mediante esquema 1T1R / V/2
    cb.program_1T1R(0, 0, V_program=2.0, dt=1e-3)

    G_after = cb.G_matrix
    delta_target = abs(G_after[0, 0] - G_before[0, 0])
    delta_offdiag = np.max(np.abs(G_after - G_before - np.diag(np.diag(G_after - G_before))))

    # Criterio real: el ratio de perturbación parásita (cross-talk) en celdas no objetivo debe ser < 10%
    cross_talk_ratio = float(delta_offdiag / max(delta_target, 1e-12))
    assert cross_talk_ratio < 0.1, f"Cross-talk excesivo: {cross_talk_ratio}"

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    dG_matrix_uS = (G_after - G_before) * 1e6
    vmax = max(np.max(np.abs(dG_matrix_uS)), 1.0)
    im = ax1.imshow(dG_matrix_uS, cmap='Blues', aspect='auto', vmin=0, vmax=vmax)

    for i in range(4):
        for j in range(4):
            val = dG_matrix_uS[i, j]
            ax1.text(j, i, f'{val:+.2f}', ha='center', va='center',
                     color='white' if val > vmax * 0.5 else 'black', fontsize=9, fontweight='bold')

    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels([f'Col {j+1}' for j in range(4)])
    ax1.set_yticklabels([f'Fila {i+1}' for i in range(4)])
    ax1.set_title('Perturbación Neta ΔG (μS)', color=title_color, fontsize=11, fontweight='bold')

    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    categories = ['ΔG Target (M11)', 'Max Cross-talk']
    values = [delta_target * 1e6, delta_offdiag * 1e6]
    ax2.bar(categories, values, color=['#2563eb', '#dc2626'], width=0.45)
    ax2.set_ylabel('ΔG (μS)', color=text_color)
    ax2.set_title(f'Ratio Cross-talk: {cross_talk_ratio*100:.2f}% (< 10%)', color=title_color, fontsize=11, fontweight='bold')

    fig.suptitle('Prueba 4×4_12: Evaluación Cuantitativa de Cross-talk',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'cross_talk_ratio': cross_talk_ratio,
        'metrics': {
            'ΔG Target': f'+{delta_target*1e6:.2f} μS',
            'Max Cross-talk': f'{delta_offdiag*1e6:.2e} μS',
            'Cross-talk Ratio': f'{cross_talk_ratio*100:.2f} % (< 10%)',
        }
    }
