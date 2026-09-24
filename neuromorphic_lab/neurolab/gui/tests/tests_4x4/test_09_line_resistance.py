"""
neurolab.gui.tests.tests_4x4.test_09_line_resistance
======================================================
Prueba 4x4_09: Resistencias de Línea R_línea en 4×4.
"""

import numpy as np
from neurolab.crossbar import CrossbarLine, CrossbarConfig


def draw_4x4_09(gui, **kwargs):
    """Evalúa la degradación de corrientes por resistencia de línea R_line en 4×4."""
    R_line_param = float(kwargs.get('line_R', 50.0))

    R_line_values = np.linspace(0, 200, 21)
    n_pts = len(R_line_values)

    I_cols = np.zeros((n_pts, 4))
    V_app = np.ones(4) * 0.5

    for k, R_line in enumerate(R_line_values):
        config = CrossbarConfig(n_rows=4, n_cols=4, line_resistance=R_line)
        cb = CrossbarLine(config)
        cb.set_uniform_conductance(300e-6)
        cb.apply_voltages(V_app)
        I_cols[k] = cb.read_currents()

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'
    colors = ['#2563eb', '#d97706', '#9333ea', '#047857']

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    for j in range(4):
        ax1.plot(R_line_values, I_cols[:, j] * 1e6, label=f'Col {j+1}', color=colors[j], linewidth=2)

    ax1.set_xlabel('Resistencia de Línea R_línea (Ω)', color=text_color)
    ax1.set_ylabel('Corriente de Salida I (μA)', color=text_color)
    ax1.set_title('Atenuación de Corriente vs R_línea', color=title_color, fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right')

    # PANEL 2: Caída porcentual a R_line_param
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    idx_target = np.argmin(np.abs(R_line_values - R_line_param))
    I_ideal_col = I_cols[0]
    I_real_col = I_cols[idx_target]
    drop_pct = (I_ideal_col - I_real_col) / I_ideal_col * 100

    x_pos = np.arange(4)
    ax2.bar(x_pos, drop_pct, color=colors, edgecolor='white', width=0.5)

    for j in range(4):
        ax2.text(j, drop_pct[j] + 0.3, f'{drop_pct[j]:.1f}%', ha='center',
                 color=text_color, fontsize=9, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Col {j+1}' for j in range(4)])
    ax2.set_ylabel('Caída Nodal (%)', color=text_color)
    ax2.set_title(f'Caída Porcentual a R_línea = {R_line_values[idx_target]:.0f} Ω', color=title_color, fontsize=11, fontweight='bold')

    fig.suptitle('Prueba 4×4_09: Atenuación por Resistencias de Línea (IR Drop)',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'R_line evaluada': f'{R_line_values[idx_target]:.1f} Ω',
            'Caída Col 1': f'{drop_pct[0]:.2f} %',
            'Caída Col 4': f'{drop_pct[3]:.2f} %',
        }
    }
