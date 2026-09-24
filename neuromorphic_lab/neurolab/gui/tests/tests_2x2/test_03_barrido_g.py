"""
Prueba 2x2_03: Barrido de G₁₁ — Fuentes Ideales vs Caída de Voltaje Real (IR Drop).

Demuestra:
1. Caso Ideal (R_line = 0 Ω): V₁ se mantiene constante (0.8 V) -> I₂ es constante.
2. Caso Real (R_line = 50 Ω): Al aumentar G₁₁, la mayor corriente produce una caída
   de voltaje V₁,efectivo(G₁₁), reduciendo I₂ para compensar (Efecto de Carga / IR drop).
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_2x2_03(gui, G_min=62.5, G_max=500.0, V1=0.8, V2=0.4,
                 R_line=50.0, n_points=41, **kwargs):
    """
    Barre G₁₁ y muestra tanto el modelo ideal como el efecto de carga de voltaje real.
    Soporta fondo blanco (publicación) y oscuro.
    """
    G_min = float(G_min) * 1e-6
    G_max = float(G_max) * 1e-6
    V1 = float(V1)
    V2 = float(V2)
    R_line = float(R_line)
    n_points = int(n_points)

    G12_fija = 100e-6
    G21_fija = 150e-6
    G22_fija = 250e-6

    G11_values = np.linspace(G_min, G_max, n_points)

    # --- 1. MODELO IDEAL (R_line = 0) ---
    config_ideal = CrossbarConfig(n_rows=2, n_cols=2, line_resistance=0.0)
    cb_ideal = CrossbarIdeal(config_ideal)

    I1_ideal = np.zeros(n_points)
    I2_ideal = np.zeros(n_points)
    V1_eff_ideal = np.full(n_points, V1)

    for k, G11 in enumerate(G11_values):
        cb_ideal.set_conductance(0, 0, G11)
        cb_ideal.set_conductance(0, 1, G12_fija)
        cb_ideal.set_conductance(1, 0, G21_fija)
        cb_ideal.set_conductance(1, 1, G22_fija)
        cb_ideal.apply_voltages([V1, V2])
        I = cb_ideal.read_currents()
        I1_ideal[k] = I[0]
        I2_ideal[k] = I[1]

    # --- 2. MODELO REAL CON RESISTENCIA DE LÍNEA / FUENTE (R_line > 0) ---
    V1_eff_real = np.zeros(n_points)
    I1_real = np.zeros(n_points)
    I2_real = np.zeros(n_points)

    for k, G11 in enumerate(G11_values):
        G_row1 = G11 + G12_fija
        V1_eff = V1 / (1.0 + R_line * G_row1)
        V2_eff = V2 / (1.0 + R_line * (G21_fija + G22_fija))

        V1_eff_real[k] = V1_eff
        I1_real[k] = G11 * V1_eff + G21_fija * V2_eff
        I2_real[k] = G12_fija * V1_eff + G22_fija * V2_eff

    # Caídas cuantitativas
    dV1_max = (V1_eff_real[0] - V1_eff_real[-1]) * 1e3   # en mV
    dI2_max = (I2_real[0] - I2_real[-1]) * 1e6            # en μA

    # ==== DIBUJAR ====
    fig = gui.figure
    fig.clear()

    # Detectar fondo claro/oscuro
    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    # Paleta adaptativa
    if is_dark:
        title_color = '#cdd6f4'
        t1_color = '#89b4fa'
        t2_color = '#fab387'
        t3_color = '#cba6f7'
        card_text_color = '#a6e3a1'
        box_bg = '#11111b'
        c_i1 = '#89b4fa'
        c_i2 = '#f38ba8'
        c_vreal = '#fab387'
        c_ireal = '#cba6f7'
        c_ideal = '#a6adc8'
        leg_bg = '#181825'
        leg_edge = '#45475a'
    else:
        title_color = '#111827'
        t1_color = '#1d4ed8'
        t2_color = '#d97706'
        t3_color = '#7e22ce'
        card_text_color = '#047857'
        box_bg = '#ffffff'
        c_i1 = '#2563eb'
        c_i2 = '#dc2626'
        c_vreal = '#d97706'
        c_ireal = '#9333ea'
        c_ideal = '#6b7280'
        leg_bg = '#ffffff'
        leg_edge = '#cbd5e1'

    # ==================================================================
    # PANEL 1: Corrientes Totales Ideales (R_line = 0 Ω)
    # ==================================================================
    ax1 = fig.add_subplot(2, 2, 1)
    gui._style_axis(ax1)
    ax1.plot(G11_values * 1e6, I1_ideal * 1e6, '-', lw=2.2, color=c_i1, label='I₁ (Col 1)')
    ax1.plot(G11_values * 1e6, I2_ideal * 1e6, '-', lw=2.2, color=c_i2, label='I₂ (Col 2)')
    ax1.set_xlabel('G₁₁ (μS)', fontsize=9)
    ax1.set_ylabel('I (μA)', fontsize=9)
    ax1.set_title('Caso IDEAL (R_línea = 0 Ω): V₁ fijo = 0.8V', color=t1_color, fontsize=10, fontweight='bold')
    ax1.legend(loc='upper left', facecolor=leg_bg, edgecolor=leg_edge, fontsize=8)
    ax1.set_ylim(80, 480)

    ax1.text(0.96, 0.05,
             'Fuente ideal → V₁ constante\nI₂ plana en 180 μA',
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=8,
             color=c_i1, family='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=box_bg, edgecolor=c_i1, alpha=0.9))

    # ==================================================================
    # PANEL 2: Caída de Voltaje Efectivo V₁(G₁₁) en Caso Real
    # ==================================================================
    ax2 = fig.add_subplot(2, 2, 2)
    gui._style_axis(ax2)
    ax2.plot(G11_values * 1e6, V1_eff_ideal * 1e3, '--', lw=1.8, color=c_ideal, label='V₁ ideal (800 mV)')
    ax2.plot(G11_values * 1e6, V1_eff_real * 1e3, '-', lw=2.2, color=c_vreal, label=f'V₁ real (R_línea={R_line:.0f}Ω)')
    ax2.set_xlabel('G₁₁ (μS)', fontsize=9)
    ax2.set_ylabel('V₁ efectivo (mV)', fontsize=9)
    ax2.set_title('EFECTO DE CARGA: Caída Nodal V₁(G₁₁)', color=t2_color, fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', facecolor=leg_bg, edgecolor=leg_edge, fontsize=8)
    ax2.set_ylim(760, 810)

    ax2.text(0.05, 0.05,
             f'Al aumentar G₁₁:\n'
             f'V₁ cae {dV1_max:.2f} mV por IR drop\n'
             f'V₁: {V1_eff_real[0]*1e3:.1f} → {V1_eff_real[-1]*1e3:.1f} mV',
             transform=ax2.transAxes, ha='left', va='bottom', fontsize=8,
             color=c_vreal, family='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=box_bg, edgecolor=c_vreal, alpha=0.9))

    # ==================================================================
    # PANEL 3: Caída Compensatoria de I₂ en Caso Real
    # ==================================================================
    ax3 = fig.add_subplot(2, 2, 3)
    gui._style_axis(ax3)
    ax3.plot(G11_values * 1e6, I2_ideal * 1e6, '--', lw=1.8, color=c_i2, alpha=0.6, label='I₂ Ideal (plana)')
    ax3.plot(G11_values * 1e6, I2_real * 1e6, '-', lw=2.2, color=c_ireal, label=f'I₂ Real (R_línea={R_line:.0f}Ω)')
    ax3.set_xlabel('G₁₁ (μS)', fontsize=9)
    ax3.set_ylabel('I₂ (μA)', fontsize=9)
    ax3.set_title('EFECTO REAL: Reducción de I₂ por caída de V₁', color=t3_color, fontsize=10, fontweight='bold')
    ax3.legend(loc='upper right', facecolor=leg_bg, edgecolor=leg_edge, fontsize=8)
    ax3.set_ylim(170, 183)

    ax3.text(0.05, 0.05,
             f'Como V₁ disminuye:\n'
             f'I₂ cae {dI2_max:.2f} μA para compensar\n'
             f'I₂: {I2_real[0]*1e6:.2f} → {I2_real[-1]*1e6:.2f} μA',
             transform=ax3.transAxes, ha='left', va='bottom', fontsize=8,
             color=c_ireal, family='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=box_bg, edgecolor=c_ireal, alpha=0.9))

    # ==================================================================
    # PANEL 4: Cuadro de Explicación Física (Encuadrado en Subplot)
    # ==================================================================
    ax4 = fig.add_subplot(2, 2, 4)
    gui._style_axis(ax4)
    ax4.set_facecolor(box_bg)
    ax4.set_xticks([])
    ax4.set_yticks([])

    for spine in ax4.spines.values():
        spine.set_color('#4b5563' if is_dark else '#cbd5e1')
        spine.set_linewidth(1.0)

    texto_card = (
        f"1. FUENTE IDEAL (R_línea = 0 Ω):\n"
        f"   • V₁ = 0.80 V (constante)\n"
        f"   • I₂ = G₁₂·V₁ + G₂₂·V₂ = 180.0 μA\n\n"
        f"2. CIRCUITO REAL (R_línea = {R_line:.0f} Ω):\n"
        f"   • V₁,efectivo = V₁ / (1 + R_línea·∑G)\n"
        f"   • Al crecer G₁₁ (62.5 → 500 μS):\n"
        f"     - V₁ cae: {V1_eff_real[0]*1e3:.1f} → {V1_eff_real[-1]*1e3:.1f} mV\n"
        f"     - I₂ REDUCE: {I2_real[0]*1e6:.1f} → {I2_real[-1]*1e6:.1f} μA\n\n"
        f"CONCLUSIÓN:\n"
        f"¡I₂ REDUCE porque V₁ cae por carga!"
    )

    ax4.text(0.05, 0.50, texto_card,
             transform=ax4.transAxes, ha='left', va='center', fontsize=6.6,
             color=card_text_color, family='monospace', linespacing=1.0)

    ax4.set_title('Explicación Física del Efecto de Carga', color=card_text_color, fontsize=9.5, fontweight='bold', pad=8)

    # Título Global
    fig.suptitle(f'Prueba 2×2_03: Barrido G₁₁ — Ideal vs Real (IR Drop R_línea={R_line:.0f}Ω)',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'R_line': f'{R_line:.1f} Ω',
            'V₁ max drop': f'{dV1_max:.2f} mV',
            'I₂ max drop': f'{dI2_max:.2f} μA',
            'V₁ eff (min)': f'{V1_eff_real[-1]*1e3:.1f} mV',
            'I₂ real (min)': f'{I2_real[-1]*1e6:.2f} μA',
        }
    }




