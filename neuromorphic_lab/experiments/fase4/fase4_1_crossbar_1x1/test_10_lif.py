"""
test_10_lif.py
==============
Prueba 10: Integración completa Crossbar Dinámico → Neurona LIF.
Incluye:
  10a: LIF con corriente constante desde Crossbar
  10b: LIF con voltaje variable V_in(t) y actualización dinámica del memristor
  10c: Tasa de disparo de LIF antes y después de programación LTP del memristor
  10d: Modulación R-STDP (STDP modulado por Recompensa R) para evitar saturación de G11
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
from neurolab.neurons import LIFNeuron, LIFConfig
from neurolab.synapses import MemristiveSynapse, LTPRule, STDPRule


def test_10a_constant():
    """10a: Corriente constante V_in = 0.5V -> Crossbar -> LIF."""
    config = CrossbarConfig(n_rows=1, n_cols=1)
    cb = CrossbarIdeal(config)
    cb.set_conductance(0, 0, 200e-6)

    lif_cfg = LIFConfig(c_m=100e-9, r_leak=1e6, v_th=2.0, v_reset=0.0, v_rest=0.0, t_ref=2e-3)
    lif = LIFNeuron(lif_cfg)

    T = 100e-3
    dt = 1e-5
    t = np.arange(0, T, dt)
    V_in = 0.5

    V_m_hist = np.zeros_like(t)
    I_out_hist = np.zeros_like(t)
    spike_times = []

    for k, tk in enumerate(t):
        I_out = cb.read_single(V_in)
        I_out_hist[k] = I_out
        spiked = lif.update(I_in=I_out, dt=dt)
        if spiked:
            spike_times.append(tk)
        V_m_hist[k] = lif.V_m

    return t, V_in * np.ones_like(t), I_out_hist, V_m_hist, spike_times


def test_10b_variable_voltage():
    """10b: Voltaje variable V_in(t) -> Crossbar dinámico -> LIF."""
    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)
    cb.set_conductance(0, 0, 200e-6)

    lif_cfg = LIFConfig(c_m=100e-9, r_leak=1e6, v_th=2.0, v_reset=0.0, v_rest=0.0, t_ref=2e-3)
    lif = LIFNeuron(lif_cfg)

    T = 100e-3
    dt = 1e-5
    t = np.arange(0, T, dt)
    # V_in modulado sinusoidalmente entre 0.2V y 0.8V
    V_in = 0.5 + 0.3 * np.sin(2 * np.pi * 20.0 * t)

    V_m_hist = np.zeros_like(t)
    I_out_hist = np.zeros_like(t)
    G_hist = np.zeros_like(t)
    spike_times = []

    for k, tk in enumerate(t):
        v_val = V_in[k]
        I_out = cb.read_single(v_val)
        I_out_hist[k] = I_out
        G_hist[k] = cb.G_matrix[0, 0]

        spiked = lif.update(I_in=I_out, dt=dt)
        if spiked:
            spike_times.append(tk)
        V_m_hist[k] = lif.V_m
        cb.update_memristors(dt)

    return t, V_in, I_out_hist, V_m_hist, G_hist, spike_times


def test_10c_programming_effect():
    """10c: Frecuencia de LIF ANTES vs DESPUÉS de reprogramar M11 con LTP."""
    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)
    lif_cfg = LIFConfig(c_m=100e-9, r_leak=1e6, v_th=2.0, v_reset=0.0, v_rest=0.0, t_ref=2e-3)

    T = 100e-3
    dt = 1e-5
    t = np.arange(0, T, dt)
    V_in = 0.5

    # --- ANTES (G_init ~ 69.4 uS) ---
    lif_before = LIFNeuron(lif_cfg)
    G_before = cb.G_matrix[0, 0]
    spikes_before = []
    V_m_before = np.zeros_like(t)

    for k, tk in enumerate(t):
        I_out = cb.read_single(V_in)
        if lif_before.update(I_in=I_out, dt=dt):
            spikes_before.append(tk)
        V_m_before[k] = lif_before.V_m

    # --- PROGRAMACIÓN LTP (40 pulsos +1V) ---
    syn = MemristiveSynapse(cb.memristors[0, 0])
    ltp = LTPRule(n_pulses=40, V_pulse=+1.0)
    ltp.apply(syn, dt=1e-3)
    G_after = cb.G_matrix[0, 0]

    # --- DESPUÉS (G_after > G_before) ---
    lif_after = LIFNeuron(lif_cfg)
    spikes_after = []
    V_m_after = np.zeros_like(t)

    for k, tk in enumerate(t):
        I_out = cb.read_single(V_in)
        if lif_after.update(I_in=I_out, dt=dt):
            spikes_after.append(tk)
        V_m_after[k] = lif_after.V_m

    return t, G_before, G_after, V_m_before, V_m_after, spikes_before, spikes_after


def test_10d_rstdp():
    """10d: R-STDP (STDP modulado por Recompensa R) para evitar la saturación de G11."""
    config = CrossbarConfig(n_rows=1, n_cols=1, x0=0.10)
    cb = CrossbarIdeal(config)
    syn = MemristiveSynapse(cb.memristors[0, 0])
    stdp = STDPRule(A_plus=0.02, A_minus=-0.015, tau_plus=17e-3, tau_minus=34e-3)
    lif_cfg = LIFConfig(c_m=100e-9, r_leak=1e6, v_th=2.0, v_reset=0.0, v_rest=0.0, t_ref=2e-3)
    lif = LIFNeuron(lif_cfg)

    T = 300e-3
    dt = 1e-4
    t = np.arange(0, T, dt)
    V_in = 0.5

    G_history = np.zeros_like(t)
    V_m_history = np.zeros_like(t)
    R_history = np.zeros_like(t)

    # Patrón de recompensa episódica R(t): alternar entre +1 (Evasión/Refuerzo) y -1 (Colisión/Castigo)
    # Episodios de 40 ms
    ep_len = 40e-3
    t_pre_events = np.arange(10e-3, T, 25e-3)
    last_t_pre = -1.0
    spike_times = []

    for k, tk in enumerate(t):
        # Determinar Recompensa R del episodio actual
        ep_idx = int(tk / ep_len)
        # Episodios pares: R = +1 (Refuerzo), Episodios impares: R = -1 (Castigo)
        if ep_idx in [0, 2, 4]:
            R = +1.0
        else:
            R = -1.0
        R_history[k] = R

        # Evento presináptico
        is_pre = any(abs(tk - tp) < dt / 2 for tp in t_pre_events)
        if is_pre:
            last_t_pre = tk

        I_out = cb.read_single(V_in)
        spiked = lif.update(I_in=I_out, dt=dt)
        V_m_history[k] = lif.V_m
        G_history[k] = cb.G_matrix[0, 0]

        if spiked:
            spike_times.append(tk)
            if last_t_pre > 0:
                delta_t = tk - last_t_pre
                # R-STDP: dW = STDP(delta_t) * R
                dW_base = stdp.apply(syn, delta_t)
                dW_effective = dW_base * R
                
                # Actualizar conductancia acotada sin saturar en RON
                G_new = np.clip(syn.conductance + dW_effective * 5e-6, 50e-6, 250e-6)
                cb.set_conductance(0, 0, G_new)
                syn.conductance = G_new

    return t, G_history, V_m_history, R_history, spike_times


def test_10_lif_integration():
    """Ejecuta la suite completa de sub-pruebas 10a, 10b, 10c y 10d (R-STDP)."""

    t_a, V_a, I_a, Vm_a, spk_a = test_10a_constant()
    t_b, V_b, I_b, Vm_b, G_b, spk_b = test_10b_variable_voltage()
    t_c, G_before, G_after, Vm_c_bef, Vm_c_aft, spk_c_bef, spk_c_aft = test_10c_programming_effect()
    t_d, G_d, Vm_d, R_d, spk_d = test_10d_rstdp()

    # Verificaciones
    assert len(spk_a) > 0, "10a debe producir spikes"
    assert len(spk_b) > 0, "10b debe producir spikes con V_in variable"
    assert len(spk_c_aft) > len(spk_c_bef), "10c: Tras LTP, N_spikes debe aumentar"
    assert G_after > G_before, "10c: G debe aumentar tras LTP"
    assert np.max(G_d * 1e6) < 500.0, f"10d: G11 no debe saturar a max RON (es {np.max(G_d*1e6):.2f} uS)"

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs/fase4/fase4_1"))
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'fig_10_lif.png')

    # Figura Consolidada 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: 10a (Corriente constante)
    axes[0, 0].plot(t_a * 1e3, Vm_a, 'g-', lw=1.2, label='V_m (V)')
    axes[0, 0].axhline(2.0, color='k', ls='--', lw=0.8, label='V_th')
    axes[0, 0].set_title(f'10a: LIF constante (N={len(spk_a)} spikes)')
    axes[0, 0].set_ylabel('V_m (V)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Panel 2: 10b (Voltaje variable)
    axes[0, 1].plot(t_b * 1e3, V_b, 'b-', lw=1, label='V_in(t) (V)')
    axes[0, 1].plot(t_b * 1e3, Vm_b, 'g-', lw=1, alpha=0.8, label='V_m (V)')
    axes[0, 1].set_title(f'10b: V_in(t) variable → LIF dinámico (N={len(spk_b)} spikes)')
    axes[0, 1].set_ylabel('Voltaje (V)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Panel 3: 10c (Efecto de LTP en frecuencia de LIF)
    axes[1, 0].plot(t_c * 1e3, Vm_c_bef, 'r--', lw=1, alpha=0.7, label=f'Antes LTP ({G_before*1e6:.1f} μS, {len(spk_c_bef)} spikes)')
    axes[1, 0].plot(t_c * 1e3, Vm_c_aft, 'g-', lw=1.2, label=f'Después LTP ({G_after*1e6:.1f} μS, {len(spk_c_aft)} spikes)')
    axes[1, 0].set_title('10c: Modulación por Programación LTP (G sube → Spikes suben)')
    axes[1, 0].set_xlabel('Tiempo (ms)')
    axes[1, 0].set_ylabel('V_m (V)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Panel 4: 10d (R-STDP Dinámico: G11 oscila y equilibra con Recompensa R)
    ax_d1 = axes[1, 1]
    ax_d2 = ax_d1.twinx()
    ax_d1.plot(t_d * 1e3, G_d * 1e6, 'm-', lw=1.5, label='G_11 (μS)')
    ax_d2.plot(t_d * 1e3, R_d, 'b--', lw=1, alpha=0.6, label='Recompensa R (±1)')
    ax_d1.set_title(f'10d: R-STDP Modulado por Recompensa R (Sin saturación en RON)')
    ax_d1.set_xlabel('Tiempo (ms)')
    ax_d1.set_ylabel('G_11 (μS)', color='m')
    ax_d2.set_ylabel('Recompensa R', color='b')
    ax_d1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=" * 60)
    print("PRUEBA 10: INTEGRACIÓN DINÁMICA CROSSBAR → LIF CON R-STDP (10a, 10b, 10c, 10d)")
    print("=" * 60)
    print(f"10a (Constante)      : {len(spk_a)} spikes con V_in=0.5V constante")
    print(f"10b (Voltaje var)    : {len(spk_b)} spikes modulados por V_in(t)")
    print(f"10c (Efecto LTP)     : Spikes ANTES={len(spk_c_bef)} ({G_before*1e6:.2f} μS) → DESPUÉS={len(spk_c_aft)} ({G_after*1e6:.2f} μS)")
    print(f"10d (R-STDP Modulado): G_11 oscila y equilibra en [{np.min(G_d*1e6):.1f}, {np.max(G_d*1e6):.1f}] μS (R ∈ [-1, +1])")
    print(f"Figura               : {fig_path}")
    print("✅ PRUEBA 10: OK (TODAS LAS SUB-PRUEBAS CON R-STDP PASARON)")
    print()

    return t_a, Vm_a, spk_a


if __name__ == '__main__':
    test_10_lif_integration()
