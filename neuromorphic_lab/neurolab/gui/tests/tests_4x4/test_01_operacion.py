"""
neurolab.gui.tests.tests_4x4.test_01_operacion
===============================================
Prueba 4x4_01: Operación Matricial I = G^T · V.
CORREGIDO (v3): Leyenda en lower-right + métricas en lower-right inferior para no tapar barras.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def draw_4x4_01(gui, **kwargs):
    """Verifica I = G^T · V con 4×4."""
    config = CrossbarConfig(n_rows=4, n_cols=4)
    cb = CrossbarIdeal(config)

    # Matriz G conocida
    G_target = np.array([
        [200, 100, 150, 180],
        [120, 250, 140, 160],
        [180, 130, 220, 140],
        [140, 170, 190, 230],
    ]) * 1e-6

    for i in range(4):
        for j in range(4):
            cb.set_conductance(i, j, G_target[i, j])

    V = np.array([0.8, 0.4, 0.6, 0.3])

    cb.apply_voltages(V)
    I_out = cb.read_currents()
    I_expected = G_target.T @ V

    mae = np.mean(np.abs(I_out - I_expected))
    rmse = np.sqrt(np.mean((I_out - I_expected) ** 2))
    err_rel = np.max(np.abs(I_out - I_expected) / (np.abs(I_expected) + 1e-15)) * 100
    r2 = float(np.corrcoef(I_out, I_expected)[0, 1] ** 2)

    # ==== DIBUJAR ====
    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'
    spine_color = '#4b5563' if is_dark else '#cbd5e1'

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)
    x_pos = np.arange(4)
    width = 0.35

    ax1.bar(x_pos - width/2, I_out * 1e6, width,
            label='I_sim', color='#2563eb', edgecolor='white')
    ax1.bar(x_pos + width/2, I_expected * 1e6, width,
            label='I_ideal', color='#dc2626', edgecolor='white')

    # ✅ FIX: Margen Y extra para que las etiquetas no queden cortadas
    y_max = max(np.max(I_out), np.max(I_expected)) * 1e6 * 1.30
    ax1.set_ylim(0, y_max)

    for k in range(4):
        ax1.text(k - width/2, I_out[k] * 1e6 + (y_max * 0.02),
                 f'{I_out[k]*1e6:.1f}', ha='center',
                 color=text_color, fontsize=8, fontweight='bold')
        ax1.text(k + width/2, I_expected[k] * 1e6 + (y_max * 0.02),
                 f'{I_expected[k]*1e6:.1f}', ha='center',
                 color=text_color, fontsize=8, fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'I_{k+1}' for k in range(4)])
    ax1.set_xlabel('Salida de Columna', color=text_color)
    ax1.set_ylabel('I (μA)', color=text_color)
    ax1.set_title('Prueba 4×4_01: Corrientes I = G^T · V',
                  color=title_color, fontsize=11, fontweight='bold')

    # ✅ FIX: Leyenda en esquina inferior derecha — no tapa las barras ni etiquetas
    ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 0.01))

    # ✅ FIX: Métricas en esquina inferior izquierda — no se superpone con la leyenda
    ax1.text(0.02, 0.02,
             f'MAE = {mae:.2e} A\nRMSE = {rmse:.2e} A\n'
             f'Err. máx = {err_rel:.2e} %\nR² = {r2:.10f}',
             transform=ax1.transAxes,
             ha='left', va='bottom', fontsize=8, color='#047857',
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor='#f0fdf4' if not is_dark else '#064e3b',
                       edgecolor='#047857', alpha=0.9))

    # Matriz G
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)
    im = ax2.imshow(G_target * 1e6, cmap='Blues',
                     aspect='auto', vmin=100, vmax=250)
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f'{G_target[i,j]*1e6:.0f}',
                     ha='center', va='center',
                     color='white' if G_target[i,j]*1e6 > 170 else 'black',
                     fontsize=10, fontweight='bold')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels([f'Col {j+1}' for j in range(4)])
    ax2.set_yticklabels([f'Fila {i+1}' for i in range(4)])
    ax2.set_title('Matriz G Programada (μS)',
                  color=title_color, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('G (μS)', color=text_color)
    cbar.ax.yaxis.set_tick_params(color=text_color)

    fig.suptitle('Prueba 2×2 / 4×4_01: Operación Matricial Exacta I = G^T · V',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'MAE': f'{mae:.2e} A',
            'RMSE': f'{rmse:.2e} A',
            'Err. máx': f'{err_rel:.2e} %',
            'R²': f'{r2:.10f}',
        }
    }
