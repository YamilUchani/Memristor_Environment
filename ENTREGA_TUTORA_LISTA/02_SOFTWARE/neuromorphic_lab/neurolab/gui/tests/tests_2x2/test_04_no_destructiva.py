"""
Prueba 2x2_04: Lectura No Destructiva — G Uniforme vs G No Uniforme.

Demuestra:
1. Con G uniforme: I₁ = I₂ (las curvas se superponen)
2. Con G no uniforme: I₁ ≠ I₂ (las curvas se diferencian)

Ambas son lecturas no destructivas (±50 mV).
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


# =====================================================================
# FUNCIÓN AUXILIAR: SIMULACIÓN
# =====================================================================

def _simular_lectura(G_matrix, V_pico, freq, T):
    """
    Simula la lectura no destructiva con una matriz G dada.

    Parameters
    ----------
    G_matrix : ndarray (2, 2)
        Matriz de conductancias (S).
    V_pico : float
        Voltaje pico (V).
    freq : float
        Frecuencia (Hz).
    T : float
        Duración (s).

    Returns
    -------
    dict con t, V1_t, V2_t, I1_hist, I2_hist, G11_hist, G22_hist
    """
    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)

    for i in range(2):
        for j in range(2):
            cb.set_conductance(i, j, G_matrix[i, j])

    G_init = cb.G_matrix.copy()

    dt = 1e-5
    t = np.arange(0, T, dt)

    V1_t = V_pico * np.sin(2 * np.pi * freq * t)
    V2_t = V_pico * np.sin(2 * np.pi * freq * t + np.pi / 2)

    I1_hist = np.zeros_like(t)
    I2_hist = np.zeros_like(t)
    G11_hist = np.zeros_like(t)
    G22_hist = np.zeros_like(t)

    for k in range(len(t)):
        cb.apply_voltages([V1_t[k], V2_t[k]])
        I = cb.read_currents()
        I1_hist[k] = I[0]
        I2_hist[k] = I[1]

        for i in range(2):
            for j in range(2):
                cb.memristors[i, j].update(cb.V_applied[i], dt)

        G11_hist[k] = cb.G_matrix[0, 0]
        G22_hist[k] = cb.G_matrix[1, 1]

    G_final = cb.G_matrix
    cambio_neto = np.max(np.abs((G_final - G_init) / G_init) * 100)

    G11_max, G11_min = G11_hist.max(), G11_hist.min()
    G22_max, G22_min = G22_hist.max(), G22_hist.min()
    var_ciclo = max(
        (G11_max - G11_min) / G11_min * 100,
        (G22_max - G22_min) / G22_min * 100,
    )

    return {
        't': t,
        'V1_t': V1_t,
        'V2_t': V2_t,
        'I1_hist': I1_hist,
        'I2_hist': I2_hist,
        'G11_hist': G11_hist,
        'G22_hist': G22_hist,
        'cambio_neto': cambio_neto,
        'var_ciclo': var_ciclo,
    }


# =====================================================================
# FUNCIÓN PRINCIPAL DE LA PRUEBA
# =====================================================================

def draw_2x2_04(gui, V_pico=50.0, freq=1.0, T=1.0, **kwargs):
    """
    Prueba de lectura no destructiva con DOS casos:
    - Caso A: G uniforme (todas = 200 μS) → I₁ = I₂
    - Caso B: G no uniforme → I₁ ≠ I₂
    """
    V_pico = float(V_pico) * 1e-3
    freq = float(freq)
    T = float(T)

    # --- CASO A: G UNIFORME ---
    G_uniforme = np.array([
        [200e-6, 200e-6],
        [200e-6, 200e-6],
    ])
    res_a = _simular_lectura(G_uniforme, V_pico, freq, T)

    # --- CASO B: G NO UNIFORME ---
    G_no_uniforme = np.array([
        [200e-6, 100e-6],
        [150e-6, 250e-6],
    ])
    res_b = _simular_lectura(G_no_uniforme, V_pico, freq, T)

    # ==== DIBUJAR ====
    fig = gui.figure
    fig.clear()

    # ==================================================================
    # COLUMNA IZQUIERDA: G UNIFORME
    # ==================================================================
    ax_a1 = fig.add_subplot(3, 2, 1)
    gui._style_axis(ax_a1)
    ax_a1.plot(res_a['t'] * 1e3, res_a['V1_t'] * 1e3,
               '-', lw=2.5, color='#4a9eff', alpha=0.9,
               label='V₁(t)')
    ax_a1.plot(res_a['t'] * 1e3, res_a['V2_t'] * 1e3,
               '--', lw=2, color='#ff6b6b', alpha=0.9,
               label='V₂(t)')
    ax_a1.set_ylabel('V (mV)', color='white', fontsize=10)
    ax_a1.set_title('CASO A: G UNIFORME (todas = 200 μS)',
                    color='white', fontsize=11, fontweight='bold')
    ax_a1.legend(loc='upper right', facecolor='#252526',
                 edgecolor='#555', labelcolor='white',
                 ncol=2, fontsize=8)
    ax_a1.grid(True, alpha=0.2)
    ax_a1.tick_params(colors='#aaaaaa', labelsize=8)

    ax_a2 = fig.add_subplot(3, 2, 3)
    gui._style_axis(ax_a2)
    ax_a2.plot(res_a['t'] * 1e3, res_a['I1_hist'] * 1e6,
               '-', lw=2.5, color='#4a9eff', alpha=0.9,
               label='I₁(t)')
    ax_a2.plot(res_a['t'] * 1e3, res_a['I2_hist'] * 1e6,
               '--', lw=2, color='#ff6b6b', alpha=0.9,
               label='I₂(t)')
    ax_a2.set_ylabel('I (μA)', color='white', fontsize=10)
    ax_a2.legend(loc='upper right', facecolor='#252526',
                 edgecolor='#555', labelcolor='white',
                 ncol=2, fontsize=8)
    ax_a2.grid(True, alpha=0.2)
    ax_a2.tick_params(colors='#aaaaaa', labelsize=8)

    # Detección de superposición
    err_I = np.max(np.abs(res_a['I1_hist'] - res_a['I2_hist']))
    if err_I < 1e-12:
        ax_a2.text(0.02, 0.02,
                   '[!] I₁ = I₂ (SUPERPUESTAS)',
                   transform=ax_a2.transAxes,
                   ha='left', va='bottom', fontsize=10,
                   color='#ff9f43', fontweight='bold',
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                             edgecolor='#ff9f43', alpha=0.9))

    ax_a3 = fig.add_subplot(3, 2, 5)
    gui._style_axis(ax_a3)
    ax_a3.plot(res_a['t'] * 1e3, res_a['G11_hist'] * 1e6,
               '-', lw=2, color='#4ade80', label='G₁₁(t)')
    ax_a3.plot(res_a['t'] * 1e3, res_a['G22_hist'] * 1e6,
               '-', lw=2, color='#e879f9', label='G₂₂(t)')
    ax_a3.set_xlabel('Tiempo (ms)', color='white', fontsize=9)
    ax_a3.set_ylabel('G (μS)', color='white', fontsize=10)
    ax_a3.legend(loc='upper right', facecolor='#252526',
                 edgecolor='#555', labelcolor='white',
                 ncol=2, fontsize=8)
    ax_a3.grid(True, alpha=0.2)
    ax_a3.tick_params(colors='#aaaaaa', labelsize=8)

    # ==================================================================
    # COLUMNA DERECHA: G NO UNIFORME
    # ==================================================================
    ax_b1 = fig.add_subplot(3, 2, 2)
    gui._style_axis(ax_b1)
    ax_b1.plot(res_b['t'] * 1e3, res_b['V1_t'] * 1e3,
               '-', lw=2.5, color='#4a9eff', alpha=0.9,
               label='V₁(t)')
    ax_b1.plot(res_b['t'] * 1e3, res_b['V2_t'] * 1e3,
               '--', lw=2, color='#ff6b6b', alpha=0.9,
               label='V₂(t)')
    ax_b1.set_ylabel('V (mV)', color='white', fontsize=10)
    ax_b1.set_title('CASO B: G NO UNIFORME (200/100/150/250 μS)',
                    color='white', fontsize=11, fontweight='bold')
    ax_b1.legend(loc='upper right', facecolor='#252526',
                 edgecolor='#555', labelcolor='white',
                 ncol=2, fontsize=8)
    ax_b1.grid(True, alpha=0.2)
    ax_b1.tick_params(colors='#aaaaaa', labelsize=8)

    ax_b2 = fig.add_subplot(3, 2, 4)
    gui._style_axis(ax_b2)
    ax_b2.plot(res_b['t'] * 1e3, res_b['I1_hist'] * 1e6,
               '-', lw=2.5, color='#4a9eff', alpha=0.9,
               label='I₁(t)')
    ax_b2.plot(res_b['t'] * 1e3, res_b['I2_hist'] * 1e6,
               '--', lw=2, color='#ff6b6b', alpha=0.9,
               label='I₂(t)')
    ax_b2.set_ylabel('I (μA)', color='white', fontsize=10)
    ax_b2.legend(loc='upper right', facecolor='#252526',
                 edgecolor='#555', labelcolor='white',
                 ncol=2, fontsize=8)
    ax_b2.grid(True, alpha=0.2)
    ax_b2.tick_params(colors='#aaaaaa', labelsize=8)

    # Detección de separación
    err_I_b = np.max(np.abs(res_b['I1_hist'] - res_b['I2_hist']))
    if err_I_b > 1e-9:
        ax_b2.text(0.02, 0.02,
                   f'[OK] I₁ ≠ I₂ (Δ max = {err_I_b*1e6:.2f} μA)',
                   transform=ax_b2.transAxes,
                   ha='left', va='bottom', fontsize=10,
                   color='#10ac84', fontweight='bold',
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                             edgecolor='#10ac84', alpha=0.9))

    ax_b3 = fig.add_subplot(3, 2, 6)
    gui._style_axis(ax_b3)
    ax_b3.plot(res_b['t'] * 1e3, res_b['G11_hist'] * 1e6,
               '-', lw=2, color='#4ade80', label='G₁₁(t)')
    ax_b3.plot(res_b['t'] * 1e3, res_b['G22_hist'] * 1e6,
               '-', lw=2, color='#e879f9', label='G₂₂(t)')
    ax_b3.set_xlabel('Tiempo (ms)', color='white', fontsize=9)
    ax_b3.set_ylabel('G (μS)', color='white', fontsize=10)
    ax_b3.legend(loc='upper right', facecolor='#252526',
                 edgecolor='#555', labelcolor='white',
                 ncol=2, fontsize=8)
    ax_b3.grid(True, alpha=0.2)
    ax_b3.tick_params(colors='#aaaaaa', labelsize=8)

    # Anotación global
    fig.text(0.5, 0.98,
             f'Prueba 2×2_04: Lectura No Destructiva — ±{V_pico*1e3:.0f} mV',
             ha='center', va='top', color='white',
             fontsize=13, fontweight='bold')

    fig.text(0.5, 0.01,
             f'G uniforme → I₁ = I₂ (se superponen)  |  '
             f'G no uniforme → I₁ ≠ I₂ (se diferencian)  |  '
             f'Variación ciclo: {res_b["var_ciclo"]:.2f}%',
             ha='center', va='bottom', color='#10ac84',
             fontsize=10, family='monospace')

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    gui.canvas.draw()

    return {
        'status': 'PASS' if res_b['cambio_neto'] < 0.01 else 'FAIL',
        'metrics': {
            'V pico': f'±{V_pico*1e3:.0f} mV',
            'Caso A (G unif.) ΔI max': f'{err_I*1e9:.2e} nA',
            'Caso B (G no unif.) ΔI max': f'{err_I_b*1e6:.4f} μA',
            'Variación ciclo': f'{res_b["var_ciclo"]:.4f} %',
            'Cambio neto': f'{res_b["cambio_neto"]:.6f} %',
        }
    }

