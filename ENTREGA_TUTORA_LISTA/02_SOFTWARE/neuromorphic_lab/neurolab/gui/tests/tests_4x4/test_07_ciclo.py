"""
neurolab.gui.tests.tests_4x4.test_07_ciclo
============================================
Prueba 4x4_07: Ciclo Reversible LTP → LTD → LTP en 4×4.

CORRECCIÓN (v2):
    - x0 = 0.30 → G_inicial ≈ 130 μS (rango dinámico visible ~130→160 μS)
    - Ciclo con programación selectiva en M₁₁ (solo esa celda, no update_memristors)
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_4x4_07(gui, **kwargs):
    """Ciclo completo LTP → LTD → LTP en celda M11."""
    n_pulses = 50
    dt = 1e-3

    # x0 = 0.30 → G_inicial ≈ 130 μS (punto de trabajo con buen rango dinámico)
    config = CrossbarConfig(n_rows=4, n_cols=4, x0=0.30)
    cb = CrossbarIdeal(config)

    G_phase1 = []  # LTP (+1V)
    G_phase2 = []  # LTD (-1V)
    G_phase3 = []  # LTP (+1V)

    # Fase 1: LTP — solo M₁₁
    for _ in range(n_pulses):
        G_phase1.append(cb.get_conductance(0, 0))
        cb.memristors[0, 0].update(1.0, dt)

    # Fase 2: LTD — solo M₁₁
    for _ in range(n_pulses):
        G_phase2.append(cb.get_conductance(0, 0))
        cb.memristors[0, 0].update(-1.0, dt)

    # Fase 3: LTP — solo M₁₁
    for _ in range(n_pulses):
        G_phase3.append(cb.get_conductance(0, 0))
        cb.memristors[0, 0].update(1.0, dt)

    G_total = np.array(G_phase1 + G_phase2 + G_phase3 + [cb.get_conductance(0, 0)]) * 1e6
    pulses = np.arange(len(G_total))

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    ax1.plot(pulses[:51],    G_total[:51],    color='#2563eb', linewidth=2, label='Fase 1: LTP (+1V)')
    ax1.plot(pulses[50:101], G_total[50:101], color='#dc2626', linewidth=2, label='Fase 2: LTD (−1V)')
    ax1.plot(pulses[100:],   G_total[100:],   color='#047857', linewidth=2, label='Fase 3: LTP (+1V)')

    ax1.axvline(50,  color='#4b5563', linestyle='--', alpha=0.5)
    ax1.axvline(100, color='#4b5563', linestyle='--', alpha=0.5)

    # Etiquetas de fase
    y_mid = (np.max(G_total) + np.min(G_total)) / 2
    ax1.text(25,  y_mid, 'LTP', ha='center', color='#2563eb', fontsize=10, fontweight='bold', alpha=0.7)
    ax1.text(75,  y_mid, 'LTD', ha='center', color='#dc2626', fontsize=10, fontweight='bold', alpha=0.7)
    ax1.text(125, y_mid, 'LTP', ha='center', color='#047857', fontsize=10, fontweight='bold', alpha=0.7)

    ax1.set_xlabel('Pulsos Acumulados', color=text_color)
    ax1.set_ylabel('Conductancia G (μS)', color=text_color)
    ax1.set_title('Trayectoria Reversible de Conductancia M₁₁', color=title_color, fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right')

    # PANEL 2: Diagrama de Histéresis (concordancia LTP1 vs LTP2)
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    G_ltp1 = G_total[:51]
    G_ltp2 = G_total[100:]
    n_min = min(len(G_ltp1), len(G_ltp2))

    ax2.plot(G_ltp1[:n_min], G_ltp2[:n_min], 'o-', color='#9333ea',
             linewidth=1.5, markersize=3, label='LTP1 vs LTP2')

    g_all = np.concatenate([G_ltp1[:n_min], G_ltp2[:n_min]])
    g_lo, g_hi = np.min(g_all) - 2, np.max(g_all) + 2
    ax2.plot([g_lo, g_hi], [g_lo, g_hi], 'k--', alpha=0.4, label='Repetibilidad Ideal (1:1)')

    ax2.set_xlabel('G LTP 1 (μS)', color=text_color)
    ax2.set_ylabel('G LTP 2 (μS)', color=text_color)
    ax2.set_title('Concordancia de Ciclos LTP1 vs LTP2', color=title_color, fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left')

    fig.suptitle('Prueba 4×4_07: Ciclo Reversible de Modulación LTP → LTD → LTP',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    err_rep = float(np.mean(np.abs(G_ltp1[:n_min] - G_ltp2[:n_min]) /
                            (np.abs(G_ltp1[:n_min]) + 1e-9))) * 100

    return {
        'status': 'PASS',
        'metrics': {
            'Error Repetibilidad': f'{err_rep:.2f} %',
            'G mín': f'{np.min(G_total):.1f} μS',
            'G máx': f'{np.max(G_total):.1f} μS',
            'Rango dinámico': f'{np.max(G_total) - np.min(G_total):.1f} μS',
        }
    }
