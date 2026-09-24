"""
test_02_ideal.py
================
Prueba 2: Operación I = G·V exacta.
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


def test_02_ideal_operation():
    """Verifica I = G·V para un punto específico."""

    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    # Fijar G a un valor conocido
    G_target = 200e-6  # 200 μS
    cb.set_conductance(0, 0, G_target)

    # Aplicar voltaje
    V_in = 0.5  # V
    cb.apply_voltages([V_in])
    I_out = cb.read_single(V_in)

    # Cálculo esperado
    I_expected = G_target * V_in

    # Error
    err_abs = abs(I_out - I_expected)
    err_rel = err_abs / I_expected * 100

    assert err_abs < 1e-15, f"Error absoluto demasiado grande: {err_abs}"

    print("=" * 60)
    print("PRUEBA 2: OPERACIÓN I = G·V")
    print("=" * 60)
    print(f"G_11     = {G_target*1e6:.4f} μS")
    print(f"V_in     = {V_in} V")
    print(f"I_out    = {I_out*1e6:.6f} μA")
    print(f"I_expected = {I_expected*1e6:.6f} μA")
    print(f"Error abs = {err_abs:.2e} A")
    print(f"Error rel = {err_rel:.2e} %")
    print("✅ PRUEBA 2: OK")
    print()

    return I_out, I_expected


if __name__ == '__main__':
    test_02_ideal_operation()


