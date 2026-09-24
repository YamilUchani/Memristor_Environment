"""
validacion_crossbar_1x1.py
==========================
Validación del crossbar 1x1.

Verifica que I_out = G · V_in con un solo memristor Strukov.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def test_1x1_single_voltage():
    """Test 1: un solo voltaje, una sola lectura."""
    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    # Conductancia conocida
    G_target = 200e-6  # 200 μS
    cb.set_conductance(0, 0, G_target)

    # Aplicar voltaje
    V_in = 0.5  # V
    cb.apply_voltages([V_in])
    I_out = cb.read_single(V_in)

    # Verificar
    I_expected = G_target * V_in
    err = abs(I_out - I_expected) / I_expected * 100

    print("=" * 60)
    print("TEST 1: Lectura puntual I = G · V")
    print("=" * 60)
    print(f"V_in = {V_in} V")
    print(f"G_11 = {G_target * 1e6:.2f} μS")
    print(f"I_out = {I_out * 1e6:.4f} μA")
    print(f"I_expected = {I_expected * 1e6:.4f} μA")
    print(f"Error = {err:.6e} %")
    print()

    return I_out, I_expected, err


def test_1x1_sweep_voltage():
    """Test 2: barrido de voltaje, múltiples lecturas."""
    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    # Conductancia fija
    G_target = 200e-6
    cb.set_conductance(0, 0, G_target)

    # Barrido de voltaje (lectura no destructiva)
    V_values = np.linspace(-0.1, 0.1, 21)  # ±100 mV
    I_values = []
    I_expected_values = []

    for V_in in V_values:
        I_out = cb.read_single(V_in)
        I_values.append(I_out)
        I_expected_values.append(G_target * V_in)

    I_values = np.array(I_values)
    I_expected_values = np.array(I_expected_values)

    # Métricas
    mae = np.mean(np.abs(I_values - I_expected_values))
    rmse = np.sqrt(np.mean((I_values - I_expected_values) ** 2))
    denom = np.abs(I_expected_values)
    mask = denom > 1e-12
    err_max = np.max(np.abs(I_values[mask] - I_expected_values[mask]) / denom[mask]) * 100 if np.any(mask) else 0.0
    r2 = float(np.corrcoef(I_values, I_expected_values)[0, 1] ** 2)

    print("=" * 60)
    print("TEST 2: Barrido de voltaje (lectura no destructiva)")
    print("=" * 60)
    print(f"MAE:   {mae * 1e9:.4f} nA")
    print(f"RMSE:  {rmse * 1e9:.4f} nA")
    print(f"Err. máx: {err_max:.6e} %")
    print(f"R²:    {r2:.10f}")
    print()

    return V_values, I_values, I_expected_values


def test_1x1_sweep_conductance():
    """Test 3: barrido de conductancia, V fijo."""
    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)

    # Barrido de conductancia
    G_values = np.linspace(62.5e-6, 500e-6, 21)  # 62.5 a 500 μS
    V_in = 0.5  # V
    I_values = []
    I_expected_values = []

    for G_target in G_values:
        cb.set_conductance(0, 0, G_target)
        I_out = cb.read_single(V_in)
        I_values.append(I_out)
        I_expected_values.append(G_target * V_in)

    I_values = np.array(I_values)
    I_expected_values = np.array(I_expected_values)

    mae = np.mean(np.abs(I_values - I_expected_values))
    r2 = float(np.corrcoef(I_values, I_expected_values)[0, 1] ** 2)

    print("=" * 60)
    print("TEST 3: Barrido de conductancia (V = 0.5 V)")
    print("=" * 60)
    print(f"MAE:   {mae * 1e9:.4f} nA")
    print(f"R²:    {r2:.10f}")
    print()

    return G_values, I_values, I_expected_values


def test_1x1_dynamic():
    """Test 4: lectura dinámica (V variable, x evoluciona)."""
    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)

    # Señal senoidal de baja amplitud (no destructiva)
    T = 1.0  # s
    dt = 1e-4
    t = np.arange(0, T, dt)
    V_in = 0.1 * np.sin(2 * np.pi * 1.0 * t)  # ±100 mV

    I_values = np.zeros_like(t)
    G_values = np.zeros_like(t)

    for k, V in enumerate(V_in):
        I_values[k] = cb.read_single(V)
        G_values[k] = cb.memristors[0, 0].conductance
        cb.apply_voltages([V])
        cb.update_memristors(dt)

    print("=" * 60)
    print("TEST 4: Lectura dinámica (señal senoidal ±100 mV)")
    print("=" * 60)
    print(f"G inicial: {G_values[0] * 1e6:.4f} μS")
    print(f"G final:   {G_values[-1] * 1e6:.4f} μS")
    print(f"ΔG:        {(G_values[-1] - G_values[0]) * 1e6:.6f} μS")
    print(f"I_max:     {np.max(np.abs(I_values)) * 1e6:.4f} μA")
    print()

    return t, V_in, I_values, G_values


def test_1x1_ltp_ltd():
    """Test 5: integración con STDP (Fase 3)."""
    from neurolab.synapses import MemristiveSynapse, LTPRule, LTDRule

    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)

    syn = MemristiveSynapse(cb.memristors[0, 0])

    # LTP: 20 pulsos +1V
    ltp = LTPRule(n_pulses=20, V_pulse=+1.0)
    G_ltp = ltp.apply(syn, dt=1e-3)

    # Reset
    syn.reset()

    # LTD: 20 pulsos -1V
    ltd = LTDRule(n_pulses=20, V_pulse=-1.0)
    G_ltd = ltd.apply(syn, dt=1e-3)

    print("=" * 60)
    print("TEST 5: Integración con STDP (Fase 3)")
    print("=" * 60)
    print(f"LTP: G_i = {G_ltp[0]*1e6:.2f} μS → G_f = {G_ltp[-1]*1e6:.2f} μS")
    print(f"LTD: G_i = {G_ltd[0]*1e6:.2f} μS → G_f = {G_ltd[-1]*1e6:.2f} μS")
    print()

    return G_ltp, G_ltd


def main():
    base_dir = os.path.dirname(__file__)

    # --- Tests ---
    test_1x1_single_voltage()
    V_sweep, I_sweep, I_exp = test_1x1_sweep_voltage()
    G_sweep, I_G, I_G_exp = test_1x1_sweep_conductance()
    t_dyn, V_dyn, I_dyn, G_dyn = test_1x1_dynamic()
    G_ltp, G_ltd = test_1x1_ltp_ltd()

    # --- Figura 1: Barrido V ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(V_sweep, I_sweep * 1e6, 'b-o', ms=4, lw=1.5,
                 label='I_sim')
    axes[0].plot(V_sweep, I_exp * 1e6, 'r--', lw=1,
                 label='I_ideal = G·V')
    axes[0].set_xlabel('V_in (V)')
    axes[0].set_ylabel('I_out (μA)')
    axes[0].set_title('Crossbar 1×1: barrido de voltaje')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(V_sweep, (I_sweep - I_exp) * 1e12, 'g-s', ms=4, lw=1.5)
    axes[1].set_xlabel('V_in (V)')
    axes[1].set_ylabel('Error (pA)')
    axes[1].set_title('Error absoluto I_sim − I_ideal')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "validacion_crossbar_1x1_V.png"),
                dpi=300)
    plt.close()

    # --- Figura 2: Barrido G ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(G_sweep * 1e6, I_G * 1e6, 'b-o', ms=4, lw=1.5,
                 label='I_sim')
    axes[0].plot(G_sweep * 1e6, I_G_exp * 1e6, 'r--', lw=1,
                 label='I_ideal = G·V')
    axes[0].set_xlabel('G_11 (μS)')
    axes[0].set_ylabel('I_out (μA)')
    axes[0].set_title('Crossbar 1×1: barrido de conductancia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(G_sweep * 1e6, (I_G - I_G_exp) * 1e12, 'g-s', ms=4, lw=1.5)
    axes[1].set_xlabel('G_11 (μS)')
    axes[1].set_ylabel('Error (pA)')
    axes[1].set_title('Error absoluto I_sim − I_ideal')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "validacion_crossbar_1x1_G.png"),
                dpi=300)
    plt.close()

    # --- Figura 3: Lectura dinámica ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t_dyn * 1e3, V_dyn * 1e3, 'b-', lw=1)
    axes[0].set_ylabel('V_in (mV)')
    axes[0].set_title('Voltaje aplicado V_in(t) — lectura no destructiva')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_dyn * 1e3, I_dyn * 1e6, 'r-', lw=1)
    axes[1].set_ylabel('I_out (μA)')
    axes[1].set_title('Corriente I_out(t) = G_11(t) · V_in(t)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_dyn * 1e3, G_dyn * 1e6, 'g-', lw=1)
    axes[2].set_xlabel('Tiempo (ms)')
    axes[2].set_ylabel('G_11 (μS)')
    axes[2].set_title('Conductancia G_11(t) — varía muy poco (no destructiva)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "validacion_crossbar_1x1_din.png"),
                dpi=300)
    plt.close()

    # --- Figura 4: Integración con STDP ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(len(G_ltp)), G_ltp * 1e6, 'b-o', ms=4, lw=1.5,
                 label='LTP (+1V)')
    axes[0].set_xlabel('Número de pulsos')
    axes[0].set_ylabel('G_11 (μS)')
    axes[0].set_title('Crossbar 1×1 con LTP (Fase 3)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(len(G_ltd)), G_ltd * 1e6, 'r-s', ms=4, lw=1.5,
                 label='LTD (−1V)')
    axes[1].set_xlabel('Número de pulsos')
    axes[1].set_ylabel('G_11 (μS)')
    axes[1].set_title('Crossbar 1×1 con LTD (Fase 3)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "validacion_crossbar_1x1_stdp.png"),
                dpi=300)
    plt.close()

    # --- CSV ---
    pd.DataFrame({
        'V_in_V': V_sweep,
        'I_sim_A': I_sweep,
        'I_ideal_A': I_exp,
        'error_A': I_sweep - I_exp,
    }).to_csv(os.path.join(base_dir, "validacion_crossbar_1x1_V.csv"),
              index=False)

    pd.DataFrame({
        'G_11_S': G_sweep,
        'I_sim_A': I_G,
        'I_ideal_A': I_G_exp,
        'error_A': I_G - I_G_exp,
    }).to_csv(os.path.join(base_dir, "validacion_crossbar_1x1_G.csv"),
              index=False)


if __name__ == '__main__':
    main()
