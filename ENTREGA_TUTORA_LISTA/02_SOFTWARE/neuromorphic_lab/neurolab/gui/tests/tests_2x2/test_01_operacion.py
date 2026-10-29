"""
Prueba 2x2_01: Operación Matricial Básica I = G^T · V.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_2x2_01(gui, **kwargs):
    """
    Verifica la operación fundamental del crossbar 2×2.

    Dibuja:
    - Matriz G
    - Voltajes de entrada
    - Corrientes de salida (sim vs ideal)
    """
    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)

    G_target = np.array([
        [200e-6, 100e-6],
        [150e-6, 250e-6],
    ])

    for i in range(2):
        for j in range(2):
            cb.set_conductance(i, j, G_target[i, j])

    V = np.array([0.8, 0.4])

    cb.apply_voltages(V)
    I_out = cb.read_currents()
    I_expected = G_target.T @ V

    mae = np.mean(np.abs(I_out - I_expected))
    rmse = np.sqrt(np.mean((I_out - I_expected) ** 2))
    err_rel = np.max(np.abs(I_out - I_expected) /
                     np.abs(I_expected + 1e-15)) * 100
    r2 = np.corrcoef(I_out, I_expected)[0, 1] ** 2 if len(I_out) > 1 else 1.0

    ax = gui.get_axis()
    ax.clear()
    gui._style_axis(ax)

    x_pos = np.arange(2)
    width = 0.35

    bars1 = ax.bar(x_pos - width / 2, I_out * 1e6, width,
                    label='I_sim', color='#4a9eff',
                    edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x_pos + width / 2, I_expected * 1e6, width,
                    label='I_ideal = G^T·V', color='#ff6b6b',
                    edgecolor='white', linewidth=1.5)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                f'{h:.2f}', ha='center', va='bottom',
                color='white', fontsize=10, fontweight='bold')

    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                f'{h:.2f}', ha='center', va='bottom',
                color='white', fontsize=10, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['I₁ = G₁₁·V₁ + G₂₁·V₂\n(Columna 1)',
                         'I₂ = G₁₂·V₁ + G₂₂·V₂\n(Columna 2)'])
    ax.set_xlabel('Salida', color='white', fontsize=11)
    ax.set_ylabel('Corriente (μA)', color='white', fontsize=11)
    ax.set_title('Prueba 2×2_01: Operación Matricial I = G^T · V',
                 color='white', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', facecolor='#252526',
              edgecolor='#555', labelcolor='white')
    ax.grid(True, alpha=0.2, axis='y')

    metrics_text = (
        f'MAE = {mae:.2e} A\n'
        f'RMSE = {rmse:.2e} A\n'
        f'Err. máx = {err_rel:.2e} %\n'
        f'R² = {r2:.10f}'
    )
    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            ha='left', va='top', fontsize=10, color='#10ac84',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                      edgecolor='#10ac84', alpha=0.9))

    gui.refresh_plot()

    return {
        'status': 'PASS',
        'metrics': {
            'MAE': f'{mae:.2e} A',
            'RMSE': f'{rmse:.2e} A',
            'Err. máx': f'{err_rel:.2e} %',
            'R²': f'{r2:.10f}',
        }
    }
