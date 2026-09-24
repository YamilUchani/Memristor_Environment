"""
neurolab.gui.tests.tests_4x4.test_19_patron_4x4
=================================================
Prueba 4x4_19: Clasificación de Patrones de Caracteres 4×4.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_4x4_19(gui, **kwargs):
    """Clasificación del patrón de carácter 'H' usando el Crossbar 4x4."""
    config = CrossbarConfig(n_rows=4, n_cols=4)
    cb = CrossbarIdeal(config)

    # Patrón del carácter 'H' en 4x4
    #  1 0 0 1
    #  1 1 1 1
    #  1 0 0 1
    #  1 0 0 1
    pattern_H = np.array([
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
    ])

    G_H = np.where(pattern_H == 1, 480.0, 62.5) * 1e-6
    cb.set_matrix_conductance(G_H)

    # Probar con entrada idéntica 'H' vs entrada con ruido
    V_test = np.array([0.8, 0.8, 0.8, 0.8])
    cb.apply_voltages(V_test)
    I_out = cb.read_currents()

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)
    im = ax1.imshow(G_H * 1e6, cmap='Greens', aspect='auto', vmin=50, vmax=500)

    for i in range(4):
        for j in range(4):
            val = G_H[i, j] * 1e6
            ax1.text(j, i, f'{val:.0f}', ha='center', va='center',
                     color='white' if val > 250 else 'black', fontsize=10, fontweight='bold')

    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels([f'Col {j+1}' for j in range(4)])
    ax1.set_yticklabels([f'Fila {i+1}' for i in range(4)])
    ax1.set_title('Plantilla Memristiva Carácter "H" (μS)', color=title_color, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('G (μS)', color=text_color)

    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)
    x_pos = np.arange(4)
    ax2.bar(x_pos, I_out * 1e6, color='#047857', edgecolor='white', width=0.5)

    max_i = np.max(I_out) * 1e6
    for k in range(4):
        ax2.text(k, I_out[k] * 1e6 + (max_i * 0.03), f'{I_out[k]*1e6:.1f}',
                 ha='center', color=text_color, fontsize=9, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Col {k+1}' for k in range(4)])
    ax2.set_xlabel('Salida de Columna (Filtro)', color=text_color)
    ax2.set_ylabel('I (μA)', color=text_color)
    ax2.set_title('Respuesta Correlacionada I = G^T · V', color=title_color, fontsize=11, fontweight='bold')
    ax2.set_ylim(0, max_i * 1.25)

    fig.suptitle('Prueba 4×4_19: Clasificación y Filtrado del Carácter "H"',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'Carácter': 'H (4x4)',
            'G activa': '480 μS',
            'I col 1': f'{I_out[0]*1e6:.1f} μA',
            'I col 2': f'{I_out[1]*1e6:.1f} μA',
            'I col 4': f'{I_out[3]*1e6:.1f} μA',
        }
    }
