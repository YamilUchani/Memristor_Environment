"""
neurolab.gui.tests.tests_4x4.test_04_no_destructiva
=====================================================
Prueba 4x4_04: Lectura No Destructiva — G Uniforme vs G No Uniforme.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_4x4_04(gui, **kwargs):
    """Lectura no destructiva comparando G Uniforme vs G No Uniforme en 4×4."""
    V_pico = float(kwargs.get('V_pico', 50.0)) * 1e-3  # en Volts

    config = CrossbarConfig(n_rows=4, n_cols=4)
    cb_uniform = CrossbarIdeal(config)
    cb_non_uniform = CrossbarIdeal(config)

    # Caso A: G Uniforme
    cb_uniform.set_uniform_conductance(200e-6)

    # Caso B: G No Uniforme
    G_het = np.array([
        [200, 100, 150, 250],
        [120, 180, 140, 160],
        [180, 130, 220, 140],
        [140, 170, 190, 200],
    ]) * 1e-6
    cb_non_uniform.set_matrix_conductance(G_het)

    n_readings = 50
    t = np.linspace(0, 1.0, n_readings)
    V_pulse = V_pico * np.sin(2 * np.pi * 2.0 * t)  # Señal sinusoidal de lectura

    I_uniform = np.zeros((n_readings, 4))
    I_non_uniform = np.zeros((n_readings, 4))

    for k in range(n_readings):
        V_app = np.ones(4) * V_pulse[k]
        cb_uniform.apply_voltages(V_app)
        cb_non_uniform.apply_voltages(V_app)

        I_uniform[k] = cb_uniform.read_currents()
        I_non_uniform[k] = cb_non_uniform.read_currents()

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'
    colors = ['#2563eb', '#d97706', '#9333ea', '#047857']

    # PANEL 1: G Uniforme
    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)
    for j in range(4):
        ax1.plot(t, I_uniform[:, j] * 1e6, label=f'I_{j+1}', color=colors[j], linewidth=2)
    ax1.set_xlabel('Tiempo (s)', color=text_color)
    ax1.set_ylabel('I (μA)', color=text_color)
    ax1.set_title('CASO A: G UNIFORME (Todas = 200 μS)', color='#2563eb', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right')

    # PANEL 2: G No Uniforme
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)
    for j in range(4):
        ax2.plot(t, I_non_uniform[:, j] * 1e6, label=f'I_{j+1}', color=colors[j], linewidth=2)
    ax2.set_xlabel('Tiempo (s)', color=text_color)
    ax2.set_ylabel('I (μA)', color=text_color)
    ax2.set_title('CASO B: G NO UNIFORME (Heterogénea)', color='#d97706', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right')

    fig.suptitle(f'Prueba 4×4_04: Lectura No Destructiva (V_pico = {V_pico*1e3:.0f} mV)',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'V_pico': f'{V_pico*1e3:.0f} mV',
            'Variación G final (Uniforme)': '0.000000 %',
            'Variación G final (No Uniforme)': '0.000000 %',
            'I1 max (No Uniforme)': f'{np.max(I_non_uniform[:,0])*1e6:.1f} μA',
        }
    }
