"""
neurolab.gui.tests.tests_4x4.test_08_sneak
===========================================
Prueba 4x4_08: Sneak Paths en 4×4.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarSneak, CrossbarConfig


def draw_4x4_08(gui, **kwargs):
    """Compara lectura ideal vs con sneak paths en 4×4."""
    config = CrossbarConfig(n_rows=4, n_cols=4)
    cb_ideal = CrossbarIdeal(config)
    cb_sneak = CrossbarSneak(config)

    rng = np.random.default_rng(42)
    G_target = rng.uniform(50e-6, 500e-6, (4, 4))

    for i in range(4):
        for j in range(4):
            cb_ideal.set_conductance(i, j, G_target[i, j])
            cb_sneak.set_conductance(i, j, G_target[i, j])

    V = np.ones(4) * 0.5

    cb_ideal.apply_voltages(V)
    cb_sneak.apply_voltages(V)

    I_ideal = cb_ideal.read_currents()
    I_sneak = cb_sneak.read_currents()

    err_rel = np.abs(I_sneak - I_ideal) / np.abs(I_ideal) * 100

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)
    x_pos = np.arange(4)
    width = 0.35

    ax1.bar(x_pos - width/2, I_ideal * 1e6, width,
            label='Ideal (sin sneak)', color='#2563eb', edgecolor='white')
    ax1.bar(x_pos + width/2, I_sneak * 1e6, width,
            label='Sneak (con parásitas)', color='#dc2626', edgecolor='white')

    max_i = max(np.max(I_ideal), np.max(I_sneak)) * 1e6
    for k in range(4):
        ax1.text(k - width/2, I_ideal[k] * 1e6 + (max_i * 0.02),
                 f'{I_ideal[k]*1e6:.0f}', ha='center',
                 color=text_color, fontsize=8)
        ax1.text(k + width/2, I_sneak[k] * 1e6 + (max_i * 0.02),
                 f'{I_sneak[k]*1e6:.0f}', ha='center',
                 color=text_color, fontsize=8)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Col {k+1}' for k in range(4)])
    ax1.set_xlabel('Columna de Salida', color=text_color)
    ax1.set_ylabel('Corriente I (μA)', color=text_color)
    ax1.set_title('Corrientes Ideal vs Sneak (4×4)', color=title_color, fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, max_i * 1.25)

    # Error por columna
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)
    ax2.bar([f'Col {j+1}' for j in range(4)], err_rel, color='#d97706', edgecolor='white', width=0.5)

    max_err = np.max(err_rel)
    for k in range(4):
        ax2.text(k, err_rel[k] + (max_err * 0.03), f'{err_rel[k]:.1f}%',
                 ha='center', color=text_color, fontsize=9, fontweight='bold')

    ax2.set_ylabel('Error relativo (%)', color=text_color)
    ax2.set_title(f'Error Parásito por Columna (Media = {np.mean(err_rel):.1f}%)',
                  color=title_color, fontsize=11, fontweight='bold')
    ax2.set_ylim(0, max_err * 1.25)

    fig.suptitle('Prueba 4×4_08: Evaluación de Sneak Paths y Corrientes Parásitas',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'Error medio': f'{np.mean(err_rel):.2f} %',
            'Error máx': f'{np.max(err_rel):.2f} %',
        }
    }
