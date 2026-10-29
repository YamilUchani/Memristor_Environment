"""
test_14_reset.py
================
Prueba 14: Reset completo.
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
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def test_14_reset():
    """Programar, resetear, verificar."""

    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)

    # Estado inicial
    G_inicial = cb.G_matrix[0, 0]

    # Programar a un valor
    cb.set_conductance(0, 0, 500e-6)
    G_programado = cb.G_matrix[0, 0]

    # Reset
    cb.reset()
    G_reset = cb.G_matrix[0, 0]

    # Verificar
    error_reset = abs(G_reset - G_inicial) / G_inicial * 100
    assert error_reset < 0.01, f"Error de reset: {error_reset}%"

    print("=" * 60)
    print("PRUEBA 14: RESET")
    print("=" * 60)
    print(f"G inicial    : {G_inicial*1e6:.6f} μS")
    print(f"G programado : {G_programado*1e6:.6f} μS")
    print(f"G reset      : {G_reset*1e6:.6f} μS")
    print(f"Error reset  : {error_reset:.6f} %")
    print("✅ PRUEBA 14: OK")
    print()

    return G_inicial, G_programado, G_reset


if __name__ == '__main__':
    test_14_reset()


