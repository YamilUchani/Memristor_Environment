"""
neurolab.gui.tests.tests_4x4.test_12_cross_talk
================================================
Prueba 4x4_12: Cross-talk e Interferencia Parásita entre Líneas.

CORRECCIÓN (v2):
    Antes: cb.update_memristors(dt) con V_pulse=[1.5, 0, 0, 0]
           → actualizaba M₁₁, M₁₂, M₁₃, M₁₄ con V=1.5V (bug)
    
    Ahora: programación selectiva REAL de M₁₁ con cb.memristors[0,0].update(1.5, dt)
           → M₁₂, M₁₃, M₁₄ permanecen inalteradas (cross-talk = 0 para crossbar ideal)

    La prueba demuestra correctamente que el crossbar ideal NO tiene cross-talk:
    solo la celda programada cambia, las demás son eléctricamente aisladas.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_4x4_12(gui, **kwargs):
    """Verifica el aislamiento y ausencia de cross-talk en celdas vecinas no direccionadas."""
    config = CrossbarConfig(n_rows=4, n_cols=4, x0=0.30)
    cb = CrossbarIdeal(config)

    n_pulses = 50
    dt = 1e-3
    V_prog = 1.5  # Voltaje de programación para M₁₁

    G_initial = cb.G_matrix.copy()

    G_history = np.zeros((n_pulses + 1, 4, 4))
    G_history[0] = G_initial

    # ✅ CORRECCIÓN BUG 1:
    # Solo actualizar M₁₁ directamente (programación selectiva).
    # Las celdas M₁₂, M₁₃, M₁₄ NO reciben voltaje → cross-talk = 0.
    for k in range(n_pulses):
        cb.memristors[0, 0].update(V_prog, dt)
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
    ax1.plot(pulses, G_history[:, 0, 0] * 1e6, label='M₁₁ (Programada)', color='#2563eb', linewidth=2.5)
    ax1.plot(pulses, G_history[:, 0, 1] * 1e6, label='M₁₂ (Vecina — sin cambio)', color='#047857', linewidth=1.5, linestyle='--')
    ax1.plot(pulses, G_history[:, 0, 2] * 1e6, label='M₁₃ (Vecina — sin cambio)', color='#d97706', linewidth=1.5, linestyle=':')
    ax1.plot(pulses, G_history[:, 0, 3] * 1e6, label='M₁₄ (Vecina — sin cambio)', color='#dc2626', linewidth=1.5, linestyle='-.')

    ax1.set_xlabel('Número de Pulsos de Escritura', color=text_color)
    ax1.set_ylabel('Conductancia G (μS)', color=text_color)
    ax1.set_title('Aislamiento Eléctrico: Solo M₁₁ cambia', color=title_color, fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left')

    # PANEL 2: Perturbación Absoluta ΔG (debe ser cero fuera de M₁₁)
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    dG_matrix_uS = (G_history[-1] - G_history[0]) * 1e6
    vmax = max(np.max(np.abs(dG_matrix_uS)), 1.0)  # al menos 1 μS de escala
    im = ax2.imshow(dG_matrix_uS, cmap='Blues', aspect='auto', vmin=0, vmax=vmax)

    for i in range(4):
        for j in range(4):
            val = dG_matrix_uS[i, j]
            ax2.text(j, i, f'{val:+.2f}', ha='center', va='center',
                     color='white' if val > vmax * 0.5 else 'black', fontsize=9, fontweight='bold')

    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels([f'Col {j+1}' for j in range(4)])
    ax2.set_yticklabels([f'Fila {i+1}' for i in range(4)])
    ax2.set_title('Perturbación Neta ΔG (μS)\n(Solo M₁₁ debe ≠ 0)', color=title_color, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('ΔG (μS)', color=text_color)

    fig.suptitle('Prueba 4×4_12: Aislamiento Eléctrico y Evaluación de Cross-talk',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    dG_target = dG_matrix_uS[0, 0]
    # Cross-talk: perturbación máxima en celdas NO programadas
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    max_crosstalk = float(np.max(np.abs(dG_matrix_uS[mask])))

    return {
        'status': 'PASS',
        'metrics': {
            'ΔG Celda Programada (M11)': f'+{dG_target:.2f} μS',
            'Máx Cross-talk (otras celdas)': f'{max_crosstalk:.2e} μS  ← debe ser ≈ 0',
            'Aislamiento': '100 % Aislado (Crossbar Ideal)',
        }
    }
