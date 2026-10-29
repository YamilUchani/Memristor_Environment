"""
Prueba 2x2_09: Efecto de resistencias de línea en 2×2.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarLine, CrossbarConfig


def draw_2x2_09(gui, line_R=50.0, **kwargs):
    """
    Compara crossbar ideal vs crossbar con R_line.
    """
    line_R = float(line_R)

    config_ideal = CrossbarConfig(n_rows=2, n_cols=2, line_resistance=0.0)
    config_line = CrossbarConfig(n_rows=2, n_cols=2, line_resistance=line_R)

    cb_ideal = CrossbarIdeal(config_ideal)
    cb_line = CrossbarLine(config_line)

    rng = np.random.default_rng(42)
    G_target = rng.uniform(50e-6, 500e-6, (2, 2))

    for i in range(2):
        for j in range(2):
            cb_ideal.set_conductance(i, j, G_target[i, j])
            cb_line.set_conductance(i, j, G_target[i, j])

    V = np.array([0.5, 0.5])

    cb_ideal.apply_voltages(V)
    cb_line.apply_voltages(V)

    I_ideal = cb_ideal.read_currents()
    I_line = cb_line.read_currents()

    err_rel = np.abs(I_line - I_ideal) / np.abs(I_ideal + 1e-15) * 100

    fig = gui.figure
    fig.clear()

    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    x_pos = np.arange(2)
    width = 0.35

    ax1.bar(x_pos - width / 2, I_ideal * 1e6, width,
            label='Ideal (R_line = 0)', color='#4a9eff', edgecolor='white')
    ax1.bar(x_pos + width / 2, I_line * 1e6, width,
            label=f'R_line = {line_R:.0f} Ω', color='#ff9f43',
            edgecolor='white')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['Columna 1', 'Columna 2'])
    ax1.set_xlabel('Columna', color='white')
    ax1.set_ylabel('I (μA)', color='white')
    ax1.set_title(f'Ideal vs R_line = {line_R:.0f} Ω',
                  color='white', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', facecolor='#252526',
               edgecolor='#555', labelcolor='white')
    ax1.grid(True, alpha=0.2, axis='y')

    for k in range(2):
        ax1.text(k - width / 2, I_ideal[k] * 1e6 + 1,
                 f'{I_ideal[k]*1e6:.1f}', ha='center', va='bottom',
                 color='white', fontsize=9, fontweight='bold')
        ax1.text(k + width / 2, I_line[k] * 1e6 + 1,
                 f'{I_line[k]*1e6:.1f}', ha='center', va='bottom',
                 color='white', fontsize=9, fontweight='bold')

    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    R_values = np.linspace(0, 200, 21)
    errs = []
    for R in R_values:
        cfg = CrossbarConfig(n_rows=2, n_cols=2, line_resistance=R)
        cb_l = CrossbarLine(cfg)
        for i in range(2):
            for j in range(2):
                cb_l.set_conductance(i, j, G_target[i, j])
        cb_l.apply_voltages(V)
        I_l = cb_l.read_currents()
        e = np.mean(np.abs(I_l - I_ideal) / np.abs(I_ideal + 1e-15)) * 100
        errs.append(e)

    ax2.plot(R_values, errs, 'r-o', ms=5, lw=2)
    ax2.axvline(line_R, color='orange', ls='--', lw=2,
                label=f'R_line actual = {line_R:.0f} Ω')

    ax2.set_xlabel('R_line (Ω)', color='white')
    ax2.set_ylabel('Error medio (%)', color='white')
    ax2.set_title('Error vs R_line',
                  color='white', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', facecolor='#252526',
               edgecolor='#555', labelcolor='white')
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'R_line': f'{line_R:.0f} Ω',
            'I_ideal C1': f'{I_ideal[0]*1e6:.2f} μA',
            'I_line C1': f'{I_line[0]*1e6:.2f} μA',
            'Error C1': f'{err_rel[0]:.2f} %',
            'Error medio': f'{np.mean(err_rel):.2f} %',
        }
    }
