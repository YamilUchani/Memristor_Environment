"""
neurolab.gui.tests.tests_4x4.test_02_barrido_v
===============================================
Prueba 4x4_02: Barrido de Voltaje en 4 Columnas.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_4x4_02(gui, **kwargs):
    """Barrido de voltaje V1 de -0.5V a +0.5V observando las 4 salidas."""
    V_min = float(kwargs.get('V_min', -0.5))
    V_max = float(kwargs.get('V_max', 0.5))
    n_points = int(kwargs.get('n_points', 41))

    config = CrossbarConfig(n_rows=4, n_cols=4)
    cb = CrossbarIdeal(config)

    # Matriz G heterogénea
    G_target = np.array([
        [200, 100, 150, 250],
        [120, 180, 140, 160],
        [180, 130, 220, 140],
        [140, 170, 190, 200],
    ]) * 1e-6

    for i in range(4):
        for j in range(4):
            cb.set_conductance(i, j, G_target[i, j])

    V_sweep = np.linspace(V_min, V_max, n_points)
    I_outs = np.zeros((n_points, 4))

    for k, V1 in enumerate(V_sweep):
        V_app = np.array([V1, 0.2, 0.1, 0.0])
        cb.apply_voltages(V_app)
        I_outs[k] = cb.read_currents()

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    colors = ['#2563eb', '#d97706', '#9333ea', '#047857']

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    r2_list = []
    for j in range(4):
        I_uA = I_outs[:, j] * 1e6
        ax1.plot(V_sweep, I_uA, label=f'I_{j+1} (Col {j+1})', color=colors[j], linewidth=2)
        r2 = float(np.corrcoef(V_sweep, I_uA)[0, 1] ** 2)
        r2_list.append(r2)

    ax1.set_xlabel('V₁ Aplicado (V)', color=text_color)
    ax1.set_ylabel('Corriente de Salida I (μA)', color=text_color)
    ax1.set_title('Barrido V₁: I = G^T · V', color=title_color, fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left')

    # Panel 2: Pendientes y Linealidad
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    slopes = [np.polyfit(V_sweep, I_outs[:, j] * 1e6, 1)[0] for j in range(4)]
    x_pos = np.arange(4)
    ax2.bar(x_pos, slopes, color=colors, edgecolor='white', width=0.5)

    for j in range(4):
        ax2.text(j, slopes[j] + 5, f'{slopes[j]:.1f} μS', ha='center',
                 color=text_color, fontsize=9, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Col {j+1}\n(G₁{j+1}={G_target[0,j]*1e6:.0f}μS)' for j in range(4)])
    ax2.set_ylabel('Pendiente dI/dV₁ (μS)', color=text_color)
    ax2.set_title('Pendientes dI_j / dV₁ (Conductancia Fila 1)', color=title_color, fontsize=11, fontweight='bold')

    fig.suptitle('Prueba 4×4_02: Barrido de Voltaje y Linealidad en 4 Columnas',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'R² Col 1': f'{r2_list[0]:.10f}',
            'R² Col 2': f'{r2_list[1]:.10f}',
            'R² Col 3': f'{r2_list[2]:.10f}',
            'R² Col 4': f'{r2_list[3]:.10f}',
            'Pendiente G11': f'{slopes[0]:.1f} μS',
        }
    }
