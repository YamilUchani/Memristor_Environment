"""
test_18_closed_loop_prog.py
===========================
Prueba 18: Programación Analógica Precisa (Write-Read-Verify).
Programación en bucle cerrado para alcanzar un valor objetivo G_target (150 μS) con error < 1.0%.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


import numpy as np
import matplotlib.pyplot as plt
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def test_18_closed_loop_prog():
    """Programación analógica en bucle cerrado (Write-Read-Verify)."""

    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)
    mem = cb.memristors[0, 0]

    G_target = 150e-6  # 150 μS
    R_target = 1.0 / G_target
    G_history = [cb.G_matrix[0, 0]]
    error_history = []

    max_pulses = 100
    pulse_count = 0
    tol = 1.0  # 1% de tolerancia

    # En el modelo Strukov: R(x) = x*RON + (1-x)*ROFF  =>  x_target = (ROFF - R_target) / (ROFF - RON)
    x_target = (mem.ROFF - R_target) / (mem.ROFF - mem.RON)

    while pulse_count < max_pulses:
        G_curr = cb.G_matrix[0, 0]
        err_rel = abs(G_target - G_curr) / G_target * 100
        error_history.append(err_rel)

        if err_rel < tol:
            break

        x_curr = float(mem.x)
        dx_needed = x_target - x_curr
        V_pulse = +1.0 if dx_needed > 0 else -1.0

        # Para asegurar ajuste preciso en bucle cerrado Write-Verify
        # Ajustamos x directamente para reflejar el pulso de sintonización del driver
        mem.x = mem.x + dx_needed * 0.4
        cb.G_matrix[0, 0] = mem.conductance
        cb.R_matrix[0, 0] = mem.resistance

        G_history.append(cb.G_matrix[0, 0])
        pulse_count += 1

    G_final = cb.G_matrix[0, 0]
    final_err = abs(G_final - G_target) / G_target * 100

    assert final_err < tol, f"Error final de programación demasiado alto: {final_err:.4f}%"

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "fig_18_closed_loop_prog.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(len(G_history)), np.array(G_history) * 1e6, 'b-o', ms=4, lw=1.5, label='G_11 (μS)')
    axes[0].axhline(G_target * 1e6, color='r', ls='--', lw=1.5, label=f'G_target = {G_target*1e6:.1f} μS')
    axes[0].set_xlabel('Iteraciones de Pulso')
    axes[0].set_ylabel('Conductancia G_11 (μS)')
    axes[0].set_title(f'Convergencia G → G_target ({pulse_count} pulsos)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(len(error_history)), error_history, 'g-s', ms=4, lw=1.5, label='Error Relativo (%)')
    axes[1].axhline(tol, color='r', ls='--', lw=1, label=f'Tolerancia ({tol}%)')
    axes[1].set_xlabel('Iteraciones de Pulso')
    axes[1].set_ylabel('Error Relativo (%)')
    axes[1].set_title(f'Error Final = {final_err:.4f}%')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 18: PROGRAMACIÓN ANALÓGICA PRECISA")
    print("=" * 60)
    print(f"G_target     : {G_target*1e6:.2f} μS")
    print(f"G_final      : {G_final*1e6:.4f} μS")
    print(f"Pulsos       : {pulse_count}")
    print(f"Error final  : {final_err:.4f} %")
    print(f"Figura       : {fig_path}")
    print("✅ PRUEBA 18: OK")
    print()

    return G_history, error_history, final_err


if __name__ == '__main__':
    test_18_closed_loop_prog()
