"""
neurolab.gui.tests.tests_4x4.test_03_barrido_g
===============================================
Prueba 4x4_03: Barrido de G₁₁ — Ideal vs Real (IR Drop R_línea=50Ω).
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarLine, CrossbarConfig


def draw_4x4_03(gui, **kwargs):
    """Barrido de G11 en 4x4 mostrando caídas por carga real (IR Drop)."""
    G_min = float(kwargs.get('G_min', 62.5))
    G_max = float(kwargs.get('G_max', 500.0))
    n_points = int(kwargs.get('n_points', 41))
    R_line = float(kwargs.get('R_line', 50.0))

    config_ideal = CrossbarConfig(n_rows=4, n_cols=4, line_resistance=0.0)
    config_real = CrossbarConfig(n_rows=4, n_cols=4, line_resistance=R_line)

    cb_ideal = CrossbarIdeal(config_ideal)
    cb_real = CrossbarLine(config_real)

    G11_values = np.linspace(G_min, G_max, n_points) * 1e-6
    V1, V2, V3, V4 = 0.8, 0.4, 0.3, 0.2

    # Configuración de matriz
    G_base = np.array([
        [100, 100, 100, 100],
        [150, 150, 150, 150],
        [120, 120, 120, 120],
        [180, 180, 180, 180],
    ]) * 1e-6

    I_ideal = np.zeros((n_points, 4))
    I_real = np.zeros((n_points, 4))
    V1_eff_real = np.zeros(n_points)

    for k, G11 in enumerate(G11_values):
        G_mat = G_base.copy()
        G_mat[0, 0] = G11

        cb_ideal.set_matrix_conductance(G_mat)
        cb_real.set_matrix_conductance(G_mat)

        V_app = np.array([V1, V2, V3, V4])
        cb_ideal.apply_voltages(V_app)
        cb_real.apply_voltages(V_app)

        I_ideal[k] = cb_ideal.read_currents()
        I_real[k] = cb_real.read_currents()

        # Voltaje efectivo Fila 1
        G_row1 = np.sum(G_mat[0, :])
        V1_eff_real[k] = V1 / (1.0 + R_line * G_row1)

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    # PANEL 1: Corrientes Ideales (Col 1 crece, Col 2 fija)
    ax1 = fig.add_subplot(2, 2, 1)
    gui._style_axis(ax1)
    ax1.plot(G11_values * 1e6, I_ideal[:, 0] * 1e6, label='I₁ (Col 1)', color='#2563eb', linewidth=2)
    ax1.plot(G11_values * 1e6, I_ideal[:, 1] * 1e6, label='I₂ (Col 2)', color='#dc2626', linewidth=2)
    ax1.set_xlabel('G₁₁ (μS)', color=text_color)
    ax1.set_ylabel('I (μA)', color=text_color)
    ax1.set_title('Caso IDEAL (R_línea = 0 Ω): V₁ fijo = 0.8V', color='#2563eb', fontsize=10, fontweight='bold')
    ax1.legend(loc='upper left')

    # PANEL 2: Caída de Voltaje Nodal V1(eff)
    ax2 = fig.add_subplot(2, 2, 2)
    gui._style_axis(ax2)
    ax2.plot(G11_values * 1e6, np.ones(n_points) * V1 * 1e3, '--', label='V₁ ideal (800 mV)', color='#4b5563', linewidth=1.5)
    ax2.plot(G11_values * 1e6, V1_eff_real * 1e3, label=f'V₁ real (R_línea={R_line:.0f}Ω)', color='#d97706', linewidth=2)
    ax2.set_xlabel('G₁₁ (μS)', color=text_color)
    ax2.set_ylabel('V₁ efectivo (mV)', color=text_color)
    ax2.set_title('EFECTO DE CARGA: Caída Nodal V₁(G₁₁)', color='#d97706', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right')

    # PANEL 3: Reducción de I2 por Carga
    ax3 = fig.add_subplot(2, 2, 3)
    gui._style_axis(ax3)
    ax3.plot(G11_values * 1e6, I_ideal[:, 1] * 1e6, '--', label='I₂ Ideal (plana)', color='#ef4444', linewidth=1.5, alpha=0.7)
    ax3.plot(G11_values * 1e6, I_real[:, 1] * 1e6, label=f'I₂ Real (R_línea={R_line:.0f}Ω)', color='#9333ea', linewidth=2)
    ax3.set_xlabel('G₁₁ (μS)', color=text_color)
    ax3.set_ylabel('I₂ (μA)', color=text_color)
    ax3.set_title('EFECTO REAL: Reducción de I₂ por caída de V₁', color='#9333ea', fontsize=10, fontweight='bold')
    ax3.legend(loc='upper right')

    # PANEL 4: Explicación Física
    ax4 = fig.add_subplot(2, 2, 4)
    gui._style_axis(ax4)
    ax4.set_xticks([])
    ax4.set_yticks([])
    for spine in ax4.spines.values():
        spine.set_color('#4b5563' if is_dark else '#cbd5e1')
        spine.set_linewidth(1.0)

    texto_card = (
        f"1. FUENTE IDEAL (R_línea = 0 Ω):\n"
        f"   • V₁ = 0.80 V (constante)\n"
        f"   • I₂ = Σ G_i2 · V_i = {I_ideal[0,1]*1e6:.1f} μA\n\n"
        f"2. CIRCUITO REAL (R_línea = {R_line:.0f} Ω):\n"
        f"   • V₁,efectivo = V₁ / (1 + R_línea·∑G)\n"
        f"   • Al crecer G₁₁ (62.5 → 500 μS):\n"
        f"     - V₁ cae: {V1_eff_real[0]*1e3:.1f} → {V1_eff_real[-1]*1e3:.1f} mV\n"
        f"     - I₂ REDUCE: {I_real[0,1]*1e6:.1f} → {I_real[-1,1]*1e6:.1f} μA\n\n"
        f"CONCLUSIÓN:\n"
        f"¡I₂ REDUCE en 4×4 porque V₁ cae por carga!"
    )

    ax4.text(0.05, 0.50, texto_card,
             transform=ax4.transAxes, ha='left', va='center', fontsize=6.6,
             color='#047857', family='monospace', linespacing=1.0)
    ax4.set_title('Explicación Física del Efecto de Carga en 4×4', color='#047857', fontsize=9.5, fontweight='bold', pad=8)

    fig.suptitle(f'Prueba 4×4_03: Barrido G₁₁ — Ideal vs Real (IR Drop R_línea={R_line:.0f}Ω)',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'R_line': f'{R_line:.1f} Ω',
            'V₁ max drop': f'{(V1_eff_real[0] - V1_eff_real[-1])*1e3:.2f} mV',
            'I₂ max drop': f'{(I_real[0,1] - I_real[-1,1])*1e6:.2f} μA',
        }
    }
