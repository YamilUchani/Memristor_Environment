"""
test_12_matrix_api.py
=====================
Prueba 12: Interfaz matricial escalable.
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
import time
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def test_12_matrix_api():
    """El mismo código funciona para 1×1, 2×2, 4×4, 8×8."""

    sizes = [1, 2, 4, 8, 16]
    resultados = []

    for size in sizes:
        config = CrossbarConfig(n_rows=size, n_cols=size)
        cb = CrossbarIdeal(config)

        # Programar uniformemente
        cb.set_uniform_conductance(200e-6)

        # Aplicar voltaje uniforme
        V = np.ones(size) * 0.5
        cb.apply_voltages(V)

        # Leer corrientes
        t0 = time.perf_counter()
        I = cb.read_currents()
        t_elapsed = time.perf_counter() - t0

        # Verificación: En una matriz N×N, la corriente de cada columna es la suma de N filas: I_j = Σ_i G_ij · V_i
        I_expected = np.ones(size) * (size * 200e-6 * 0.5)
        err = np.max(np.abs(I - I_expected))


        resultados.append({
            'size': size,
            'n_memristors': size * size,
            'error': err,
            'time_ms': t_elapsed * 1e3,
        })

        assert err < 1e-15, f"Error en {size}×{size}: {err}"

    import matplotlib.pyplot as plt

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "fig_12_matrix_api.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sz_labels = [f"{r['size']}×{r['size']}" for r in resultados]
    t_exec = [r['time_ms'] for r in resultados]
    n_mems = [r['n_memristors'] for r in resultados]

    axes[0].plot(sz_labels, t_exec, 'b-o', lw=2, ms=6)
    axes[0].set_xlabel('Tamaño de matriz')
    axes[0].set_ylabel('Tiempo de ejecución (ms)')
    axes[0].set_title('Escalabilidad del Crossbar: Tiempo de Cálculo')
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(sz_labels, n_mems, color='#bd93f9', width=0.5)
    axes[1].set_xlabel('Tamaño de matriz')
    axes[1].set_ylabel('Número de Memristores')
    axes[1].set_title('Escalabilidad de Componentes N×N')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 12: INTERFAZ MATRICIAL")
    print("=" * 60)
    for r in resultados:
        print(f"  {r['size']:>2}×{r['size']:<2}: "
              f"{r['n_memristors']:>4} memristores, "
              f"error = {r['error']:.2e}, "
              f"t = {r['time_ms']:.3f} ms")
    print(f"Figura    : {fig_path}")
    print("✅ PRUEBA 12: OK")
    print()

    return resultados


if __name__ == '__main__':
    test_12_matrix_api()

