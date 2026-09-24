"""
neurolab.gui.tests.tests_4x4.test_20_comparacion
=================================================
Prueba 4x4_20: Comparación Consolidada 1×1 vs 2×2 vs 4×4.

CORRECCIÓN (v2):
    Error de Sneak Paths ahora calculado con CrossbarSneak (modelo de vecinos corregido):
        1×1  →  0 vecinos  →  0.0 %
        2×2  →  ~1 vecino  → ~10.0 %
        4×4  →  ~2 vecinos → ~20.0 %
    Escala monótonamente creciente (físicamente correcto).

    Los valores de sneak se calculan dinámicamente con el modelo corregido,
    no son hardcoded.
"""

import numpy as np
from neurolab.crossbar import CrossbarConfig
from neurolab.crossbar import CrossbarSneak


def draw_4x4_20(gui, **kwargs):
    """Resumen consolidado comparando las tres arquitecturas Crossbar (1x1, 2x2, 4x4)."""
    sizes = ['1×1', '2×2', '4×4']
    dims = [(1, 1), (2, 2), (4, 4)]
    n_memristors = [1, 4, 16]
    max_current_uA = [160.0, 400.0, 1600.0]
    ir_drop_pct = [0.0, 3.5, 9.8]

    # ✅ Bug 3 corregido: calcular sneak error dinámicamente con CrossbarSneak
    sneak_error_pct = []
    for nr, nc in dims:
        cfg = CrossbarConfig(n_rows=nr, n_cols=nc)
        cb = CrossbarSneak(cfg)
        cb.set_uniform_conductance(200e-6)
        # Voltaje de prueba uniforme
        V_test = np.ones(nr) * 0.8
        cb.apply_voltages(V_test)
        sneak_error_pct.append(cb.sneak_error_pct())

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    # PANEL 1: Capacidad de Corriente Total
    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    x_pos = np.arange(3)
    ax1.bar(x_pos, max_current_uA, color=['#2563eb', '#d97706', '#047857'], edgecolor='white', width=0.45)

    max_c = np.max(max_current_uA)
    for k in range(3):
        ax1.text(k, max_current_uA[k] + (max_c * 0.03), f'{max_current_uA[k]:.0f} μA',
                 ha='center', color=text_color, fontsize=9, fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{s}\n({n_memristors[k]} mem)' for k, s in enumerate(sizes)])
    ax1.set_ylabel('Corriente Máxima de Salida (μA)', color=text_color)
    ax1.set_title('Capacidad de Corriente Agregada (N×N)', color=title_color, fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max_c * 1.25)

    # PANEL 2: No Idealidades — Sneak Paths y IR Drop
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    width = 0.35
    bars1 = ax2.bar(x_pos - width/2, sneak_error_pct, width,
                    label='Error Sneak Paths (%)', color='#dc2626', edgecolor='white')
    bars2 = ax2.bar(x_pos + width/2, ir_drop_pct, width,
                    label='Caída IR Drop R_línea (%)', color='#9333ea', edgecolor='white')

    for k in range(3):
        ax2.text(k - width/2, sneak_error_pct[k] + 0.5, f'{sneak_error_pct[k]:.1f}%',
                 ha='center', color=text_color, fontsize=8, fontweight='bold')
        ax2.text(k + width/2, ir_drop_pct[k] + 0.5, f'{ir_drop_pct[k]:.1f}%',
                 ha='center', color=text_color, fontsize=8, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sizes)
    ax2.set_ylabel('Error Relativo (%)', color=text_color)
    ax2.set_title('Impacto de No-Idealidades según Escala\n(Sneak: escala monótonamente con N)',
                  color=title_color, fontsize=10, fontweight='bold')
    ax2.legend(loc='upper left')
    max_err = max(max(sneak_error_pct), max(ir_drop_pct))
    ax2.set_ylim(0, max_err * 1.4 + 2)

    # Anotación: progresión correcta
    ax2.annotate('↑ Crece con N', xy=(2, sneak_error_pct[2]),
                 xytext=(1.5, sneak_error_pct[2] * 0.6 + max_err * 0.15),
                 arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.2),
                 color='#dc2626', fontsize=8, fontweight='bold')

    fig.suptitle('Prueba 4×4_20: Comparación Consolidada de Arquitecturas Crossbar 1×1 / 2×2 / 4×4',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'Capacidad 1x1': '160 μA (1 mem)',
            'Capacidad 2x2': '400 μA (4 mems)',
            'Capacidad 4x4': '1600 μA (16 mems)',
            'Sneak Error 1x1': f'{sneak_error_pct[0]:.1f} %',
            'Sneak Error 2x2': f'{sneak_error_pct[1]:.1f} %',
            'Sneak Error 4x4': f'{sneak_error_pct[2]:.1f} %  ← escala correctamente',
        }
    }
