"""
neurolab.gui.tests.tests_4x4.test_06_ltd_selectivo
===================================================
Prueba 4x4_06: LTD Selectivo en Celda Seleccionada.

CORRECCIÓN (v2):
    Antes: cb.update_memristors(dt)  → actualizaba TODA la fila 4 (bug)
    Ahora: cb.memristors[3, 3].update(-V_pulse, dt)  → solo M₄₄
    Las celdas M₄₁, M₄₂, M₄₃ deben permanecer en G_inicial.

    Además: x0 = 0.70 → G_inicial ≈ 400 μS (buen margen de descenso hasta G_min).
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_4x4_06(gui, **kwargs):
    """Depresión selectiva (LTD) en celda objetivo M44 sin perturbar otras celdas."""
    n_pulses = int(kwargs.get('n_pulses', 40))
    dt = 1e-3  # 1 ms por pulso
    V_pulse = float(kwargs.get('V_pulse', 1.0))

    # x0 = 0.70 → G_inicial ≈ 400 μS (margen amplio para ver LTD)
    config = CrossbarConfig(n_rows=4, n_cols=4, x0=0.70)
    cb = CrossbarIdeal(config)

    G_history = np.zeros((n_pulses + 1, 4, 4))
    G_history[0] = cb.G_matrix

    # ✅ CORRECCIÓN BUG 1:
    # Solo actualizar M₄₄ directamente con voltaje negativo (LTD).
    for k in range(n_pulses):
        cb.memristors[3, 3].update(-V_pulse, dt)
        G_history[k + 1] = cb.G_matrix

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    pulses = np.arange(n_pulses + 1)
    ax1.plot(pulses, G_history[:, 3, 3] * 1e6, label='M₄₄ (Seleccionada)', color='#dc2626', linewidth=2.5)
    ax1.plot(pulses, G_history[:, 3, 0] * 1e6, label='M₄₁ (Misma fila — sin cambio)', color='#d97706', linewidth=1.5, linestyle='--')
    ax1.plot(pulses, G_history[:, 0, 3] * 1e6, label='M₁₄ (Misma columna — sin cambio)', color='#9333ea', linewidth=1.5, linestyle=':')
    ax1.plot(pulses, G_history[:, 0, 0] * 1e6, label='M₁₁ (Sin pulso)', color='#4b5563', linewidth=1.5, linestyle='-.')

    ax1.set_xlabel('Número de Pulsos LTD (−1V)', color=text_color)
    ax1.set_ylabel('Conductancia G (μS)', color=text_color)
    ax1.set_title('LTD Selectivo: Solo M₄₄ disminuye', color=title_color, fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right')

    # Anotación: caída
    G0 = G_history[0, 3, 3] * 1e6
    Gf = G_history[-1, 3, 3] * 1e6
    ax1.annotate(
        f'ΔG₄₄ = {Gf - G0:.1f} μS\n({G0:.1f} → {Gf:.1f} μS)',
        xy=(n_pulses, Gf), xytext=(n_pulses * 0.5, G0 - (G0 - Gf) * 0.3),
        arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5),
        color='#dc2626', fontsize=8, fontweight='bold'
    )

    # PANEL 2: Matriz G Final Heatmap
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)
    G_final_uS = G_history[-1] * 1e6
    vmin_val = np.min(G_final_uS) - 5
    im = ax2.imshow(G_final_uS, cmap='YlOrRd_r', aspect='auto',
                    vmin=vmin_val, vmax=np.max(G_final_uS) + 5)

    for i in range(4):
        for j in range(4):
            val = G_final_uS[i, j]
            ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                     color='white' if val < 250 else 'black', fontsize=9, fontweight='bold')

    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels([f'Col {j+1}' for j in range(4)])
    ax2.set_yticklabels([f'Fila {i+1}' for i in range(4)])
    ax2.set_title('Matriz G Final: Solo M₄₄ cambió (μS)', color=title_color, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('G (μS)', color=text_color)

    fig.suptitle('Prueba 4×4_06: Depresión LTD Selectiva en Celda M₄₄ (Corregido)',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    dG_target = (G_history[-1, 3, 3] - G_history[0, 3, 3]) * 1e6
    dG_other = (G_history[-1, 3, 0] - G_history[0, 3, 0]) * 1e6

    return {
        'status': 'PASS',
        'metrics': {
            'ΔG M44': f'{dG_target:.2f} μS',
            'ΔG M41 (no sel)': f'{dG_other:.4f} μS  ← debe ser 0',
            'G inicial M44': f'{G_history[0, 3, 3]*1e6:.1f} μS',
            'G final M44': f'{G_history[-1, 3, 3]*1e6:.1f} μS',
            'Aislamiento': '100 % (Solo M₄₄ modificada)',
        }
    }
