"""
test_13_uniform.py
==================
Prueba 13: Programación uniforme.
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


def test_13_uniform():
    """Programar todos los memristores al mismo valor."""

    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    G_target = 300e-6
    cb.set_uniform_conductance(G_target)

    G_matrix = cb.G_matrix
    error = np.max(np.abs(G_matrix - G_target))

    assert error < 1e-9, f"Error de uniformidad: {error}"

    print("=" * 60)
    print("PRUEBA 13: UNIFORMIDAD")
    print("=" * 60)
    print(f"G_target     : {G_target*1e6:.4f} μS")
    print(f"G_matrix     : {G_matrix*1e6} μS")
    print(f"Error        : {error:.2e} S")
    print("✅ PRUEBA 13: OK")
    print()

    return G_matrix


if __name__ == '__main__':
    test_13_uniform()


