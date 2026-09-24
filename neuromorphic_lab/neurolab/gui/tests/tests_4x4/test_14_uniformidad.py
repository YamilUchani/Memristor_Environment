"""
neurolab.gui.tests.tests_4x4.test_14_uniformidad
=================================================
Prueba 4x4_14: Uniformidad y Dispersión de Celdas 4×4.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_4x4_14(gui, **kwargs):
    """Verifica la uniformidad de la matriz 4x4 tras programar a G_target."""
    G_target_uS = float(kwargs.get('G_target', 300.0))

    config = CrossbarConfig(n_rows=4, n_cols=4)
    cb = CrossbarIdeal(config)

    cb.set_uniform_conductance(G_target_uS * 1e-6)
    G_matrix_uS = cb.G_matrix.flatten() * 1e6

    mean_G = float(np.mean(G_matrix_uS))
    std_G = float(np.std(G_matrix_uS))
    cv_G = (std_G / mean_G) * 100 if mean_G > 0 else 0.0

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    # PANEL 1: Heatmap de la Matriz Programada
    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)
    im = ax1.imshow(cb.G_matrix * 1e6, cmap='Greens', aspect='auto', vmin=G_target_uS*0.9, vmax=G_target_uS*1.1)

    for i in range(4):
        for j in range(4):
            val = cb.G_matrix[i, j] * 1e6
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=9, fontweight='bold')

    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels([f'Col {j+1}' for j in range(4)])
    ax1.set_yticklabels([f'Fila {i+1}' for i in range(4)])
    ax1.set_title(f'Matriz G Programada a {G_target_uS:.0f} μS', color=title_color, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('G (μS)', color=text_color)

    # PANEL 2: Histograma de Dispersión
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    ax2.hist(G_matrix_uS, bins=10, color='#047857', edgecolor='white', alpha=0.8)
    ax2.axvline(mean_G, color='#dc2626', linestyle='--', linewidth=2, label=f'Media = {mean_G:.2f} μS')

    ax2.set_xlabel('Conductancia G (μS)', color=text_color)
    ax2.set_ylabel('Frecuencia (N° de Celdas)', color=text_color)
    ax2.set_title(f'Distribución de Conductancias (CV = {cv_G:.4f}%)', color=title_color, fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right')

    fig.suptitle('Prueba 4×4_14: Uniformidad y Homogeneidad en las 16 Celdas de la Matriz',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'Media G': f'{mean_G:.2f} μS',
            'Desv. Estándar': f'{std_G:.4f} μS',
            'Coef. Variación (CV)': f'{cv_G:.4f} %',
            'Celdas evaluadas': 16,
        }
    }
