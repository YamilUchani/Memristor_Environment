"""
test_01_init.py
===============
Prueba 1: Inicialización del crossbar 1×1.
"""

import os
import sys

# Permitir ejecucion directa desde subdirectorios
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.devices import MemristorStrukov


def test_01_initialization():
    """Verifica la estructura del crossbar 1×1."""

    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    # --- Verificaciones estructurales ---
    assert cb.n_rows == 1, "n_rows debe ser 1"
    assert cb.n_cols == 1, "n_cols debe ser 1"
    assert cb.G_matrix.shape == (1, 1), "G_matrix debe ser 1×1"
    assert cb.R_matrix.shape == (1, 1), "R_matrix debe ser 1×1"
    assert cb.memristors.shape == (1, 1), "memristors debe ser 1×1"

    # --- Verificar que el memristor es de Fase 1 ---
    assert isinstance(cb.memristors[0, 0], MemristorStrukov), \
        "El memristor debe ser Strukov (Fase 1)"

    # --- Verificar parámetros de Strukov ---
    mem = cb.memristors[0, 0]
    assert mem.RON == 100.0, "RON debe ser 100 Ω"
    assert mem.ROFF == 16_000.0, "ROFF debe ser 16 kΩ"

    # --- Verificar estado inicial ---
    G = cb.G_matrix[0, 0]
    R = cb.R_matrix[0, 0]
    assert 60e-6 < G < 80e-6, f"G inicial debe estar en ~69 μS, es {G*1e6:.2f}"
    assert 14_000 < R < 15_000, f"R inicial debe estar en ~14.4 kΩ, es {R:.2f}"

    print("=" * 60)
    print("PRUEBA 1: INICIALIZACIÓN")
    print("=" * 60)
    print(f"n_rows       = {cb.n_rows}")
    print(f"n_cols       = {cb.n_cols}")
    print(f"G_matrix.shape = {cb.G_matrix.shape}")
    print(f"Tipo M11     = {type(cb.memristors[0, 0]).__name__}")
    print(f"RON          = {mem.RON} Ω")
    print(f"ROFF         = {mem.ROFF} Ω")
    print(f"G inicial    = {G*1e6:.4f} μS")
    print(f"R inicial    = {R:.4f} Ω")
    print("✅ PRUEBA 1: OK")
    print()

    return cb


if __name__ == '__main__':
    test_01_initialization()


