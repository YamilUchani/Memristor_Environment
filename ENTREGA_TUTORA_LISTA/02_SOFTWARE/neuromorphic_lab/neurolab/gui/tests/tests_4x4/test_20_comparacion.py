"""
neurolab.gui.tests.tests_4x4.test_20_comparacion
=================================================
Prueba 4x4_20: Comparación Consolidada y Validación Analítica (1×1 vs 2×2 vs 4×4).
"""

import numpy as np
from neurolab.crossbar import Crossbar


def draw_4x4_20(gui, **kwargs):
    """Resumen comparativo validado cuantitativamente contra modelo analítico."""
    resultados = {}
    for N in [1, 2, 4]:
        cb = Crossbar(N, N)
        cb.set_uniform_conductance(69.4e-6)
        cb.apply_voltages(np.ones(N) * 1.0)
        I = cb.read_currents()
        resultados[N] = {
            'max_current_uA': float(np.max(I) * 1e6),
            'ir_drop_pct': float(cb.compute_ir_drop_pct()),
        }

    # Comparación contra modelo analítico ideal (I_ideal = G * V * N)
    for N, r in resultados.items():
        I_ideal = 69.4e-6 * 1.0 * N
        assert r['max_current_uA'] < I_ideal * 1e6 * 1.05, (
            f"N={N}: corriente {r['max_current_uA']:.2f} μA excede el ideal analítico ({I_ideal*1e6:.2f} μS)"
        )

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'

    sizes = ['1×1', '2×2', '4×4']
    x_pos = np.arange(3)
    max_currents = [resultados[N]['max_current_uA'] for N in [1, 2, 4]]
    ir_drops = [resultados[N]['ir_drop_pct'] for N in [1, 2, 4]]

    # PANEL 1: Capacidad de Corriente Total
    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    ax1.bar(x_pos, max_currents, color=['#2563eb', '#d97706', '#047857'], edgecolor='white', width=0.45)
    max_c = max(max_currents)
    for k, N in enumerate([1, 2, 4]):
        ax1.text(k, max_currents[k] + (max_c * 0.03), f'{max_currents[k]:.1f} μA',
                 ha='center', color=text_color, fontsize=9, fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{s}\n(N={N})' for k, (s, N) in enumerate(zip(sizes, [1, 2, 4]))])
    ax1.set_ylabel('Corriente Máxima de Salida (μA)', color=text_color)
    ax1.set_title('Capacidad de Corriente Agregada (N×N)', color=title_color, fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max_c * 1.25)

    # PANEL 2: IR Drop % según Escala
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    ax2.bar(x_pos, ir_drops, color='#9333ea', edgecolor='white', width=0.45)
    for k, val in enumerate(ir_drops):
        ax2.text(k, val + 0.1, f'{val:.2f}%', ha='center', color=text_color, fontsize=9, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sizes)
    ax2.set_ylabel('Caída IR Drop R_línea (%)', color=text_color)
    ax2.set_title('Caída IR Drop según Escala', color=title_color, fontsize=11, fontweight='bold')
    ax2.set_ylim(0, max(max(ir_drops) * 1.3, 1.0))

    fig.suptitle('Prueba 4×4_20: Comparación Consolidada Validada Analíticamente',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    metrics_formatted = {
        f"Capacidad {N}x{N}": f"{resultados[N]['max_current_uA']:.1f} μA"
        for N in [1, 2, 4]
    }
    for N in [1, 2, 4]:
        metrics_formatted[f"IR Drop {N}x{N}"] = f"{resultados[N]['ir_drop_pct']:.2f} %"

    return {
        'status': 'PASS',
        'metrics': metrics_formatted
    }
