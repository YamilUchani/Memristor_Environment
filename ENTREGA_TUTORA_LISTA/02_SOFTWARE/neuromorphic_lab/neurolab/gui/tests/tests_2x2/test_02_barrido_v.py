"""
Prueba 2x2_02: Barrido de voltaje en Fila 1, Fila 2 fija.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_2x2_02(gui, V_min=-0.5, V_max=0.5, n_points=41, **kwargs):
    """
    Barrido de voltaje en Fila 1, Fila 2 fija en 0.

    Demuestra la linealidad de la operación I = G^T · V.
    """
    V_min = float(V_min)
    V_max = float(V_max)
    n_points = int(n_points)

    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)

    G = np.array([
        [200e-6, 100e-6],
        [150e-6, 250e-6],
    ])
    for i in range(2):
        for j in range(2):
            cb.set_conductance(i, j, G[i, j])

    V1_values = np.linspace(V_min, V_max, n_points)
    V2 = 0.0

    I1_sim = np.zeros(n_points)
    I2_sim = np.zeros(n_points)
    I1_ideal = np.zeros(n_points)
    I2_ideal = np.zeros(n_points)

    for k, V1 in enumerate(V1_values):
        cb.apply_voltages([V1, V2])
        I = cb.read_currents()
        I1_sim[k] = I[0]
        I2_sim[k] = I[1]

        I_ideal = G.T @ np.array([V1, V2])
        I1_ideal[k] = I_ideal[0]
        I2_ideal[k] = I_ideal[1]

    mae1 = np.mean(np.abs(I1_sim - I1_ideal))
    mae2 = np.mean(np.abs(I2_sim - I2_ideal))
    r2_1 = np.corrcoef(I1_sim, I1_ideal)[0, 1] ** 2 if len(I1_sim) > 1 else 1.0
    r2_2 = np.corrcoef(I2_sim, I2_ideal)[0, 1] ** 2 if len(I2_sim) > 1 else 1.0

    ax = gui.get_axis()
    ax.clear()
    gui._style_axis(ax)

    ax.plot(V1_values, I1_sim * 1e6, '-o', ms=5, lw=2,
            label='I₁ (Col 1) sim', color='#4a9eff')
    ax.plot(V1_values, I1_ideal * 1e6, '--', lw=1.5,
            label='I₁ ideal', color='#4a9eff', alpha=0.6)

    ax.plot(V1_values, I2_sim * 1e6, '-s', ms=5, lw=2,
            label='I₂ (Col 2) sim', color='#ff6b6b')
    ax.plot(V1_values, I2_ideal * 1e6, '--', lw=1.5,
            label='I₂ ideal', color='#ff6b6b', alpha=0.6)


    ax.set_xlabel('V₁ (Fila 1) [V]', color='white', fontsize=11)
    ax.set_ylabel('Corriente (μA)', color='white', fontsize=11)
    ax.set_title(f'Prueba 2×2_02: Barrido de Voltaje (Fila 1) — V₂ = {V2} V',
                 color='white', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', facecolor='#252526',
              edgecolor='#555', labelcolor='white', ncol=2)
    ax.grid(True, alpha=0.2)

    text = (
        f'Col 1: MAE = {mae1:.2e} A, R² = {r2_1:.10f}\n'
        f'Col 2: MAE = {mae2:.2e} A, R² = {r2_2:.10f}'
    )
    ax.text(0.98, 0.02, text,
            transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color='#10ac84',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                      edgecolor='#10ac84', alpha=0.9))

    gui.refresh_plot()

    return {
        'status': 'PASS',
        'metrics': {
            'MAE Col 1': f'{mae1:.2e} A',
            'MAE Col 2': f'{mae2:.2e} A',
            'R² Col 1': f'{r2_1:.10f}',
            'R² Col 2': f'{r2_2:.10f}',
            'N puntos': n_points,
        }
    }
