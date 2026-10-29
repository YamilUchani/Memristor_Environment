"""
validaciones/validacion_lif_rc.py
=================================
Script de Validación Analítica para la Neurona LIF (Pestaña 2 de neurolab).
Compara la solución analítica exacta V_m(t) calculada por neurolab.core.lif_validation
contra la integración numérica Euler explícita de LIFNeuron.

Salidas (generadas en esta misma carpeta):
  - lif_rc_validation.png
  - lif_rc_validation.csv
"""

import sys
import json
from pathlib import Path

# Garantizar que el directorio raíz de neuromorphic_lab esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from neurolab.neurons.config import LIFConfig
from neurolab.neurons.lif import LIFNeuron
from neurolab.core.lif_validation import compute_lif_validation_metrics


def build_config_from_json_or_default() -> LIFConfig:
    """Carga la configuración de la Pestaña 2 desde configs/lif_config.json o usa valores por defecto."""
    config_path = ROOT_DIR / "configs" / "lif_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            c_val = data.get("c_m_value", 100.0)
            c_unit = data.get("c_m_unit", "nF")
            c_mult = 1e-9 if c_unit == "nF" else (1e-12 if c_unit == "pF" else 1e-6)
            
            rs_val = data.get("r_series_value", 100.0)
            rs_unit = data.get("r_series_unit", "kΩ")
            rs_mult = 1e3 if rs_unit == "kΩ" else (1.0 if rs_unit == "Ω" else 1e6)
            
            r_val = data.get("r_leak_value", 1.0)
            r_unit = data.get("r_leak_unit", "MΩ")
            r_mult = 1e6 if r_unit == "MΩ" else (1e3 if r_unit == "kΩ" else 1.0)
            
            return LIFConfig(
                c_m=c_val * c_mult,
                r_series=rs_val * rs_mult,
                r_leak=r_val * r_mult,
                v_rest=data.get("v_rest", 0.0),
                v_th=data.get("v_th", 2.5),
                v_reset=data.get("v_reset", 0.0),
                t_ref=data.get("t_ref", 2.0) * 1e-3
            )
        except Exception:
            pass

    # Preset hardware por defecto de la Pestaña 2
    return LIFConfig(
        c_m=100e-9,      # 100 nF
        r_series=100e3,  # 100 kΩ
        r_leak=1e6,      # 1 MΩ
        v_rest=0.0,
        v_th=2.5,        # 2.5 V umbral subumbral/superumbral
        v_reset=0.0,
        t_ref=2e-3
    )


def main():
    print("=" * 80)
    print(" VALIDACION ANALITICA CIRCUITO INTEGRADOR NEURONAL LIF (PESTAÑA 2)")
    print("=" * 80)

    output_dir = Path(__file__).resolve().parent
    cfg = build_config_from_json_or_default()
    
    print(f" Configuracion cargada: C_m={cfg.c_m*1e9:.1f}nF, R_S={cfg.r_series/1e3:.1f}kOhm, R_leak={cfg.r_leak/1e6:.1f}MOhm")

    # 1. Validación Subumbral Analítica (V_in = 2.0V < V_th = 2.5V)
    cfg_sub = LIFConfig(
        c_m=cfg.c_m, r_series=cfg.r_series, r_leak=cfg.r_leak,
        v_rest=cfg.v_rest, v_th=2.5, v_reset=cfg.v_reset, t_ref=cfg.t_ref
    )
    neuron_sub = LIFNeuron(cfg_sub)
    steps = 1000
    t = np.linspace(0, 0.1, steps)  # 100 ms
    dt = t[1] - t[0]
    v_signal_sub = np.full(steps, 2.0)
    v_sim_sub = np.zeros(steps)

    for k in range(steps):
        v_sim_sub[k] = neuron_sub.v_membrane
        neuron_sub.step(voltage_input=v_signal_sub[k], dt=dt)

    metrics = compute_lif_validation_metrics(t, v_sim_sub, v_signal_sub, cfg_sub, is_voltage_input=True)

    print(f" R² (Determinación): {metrics['r2']:.6f}  (REQUERIDO > 0.99)")
    print(f" MAE (Absoluto Medio): {metrics['mae']*1e3:.4f} mV")
    print(f" RMSE (Cuadrático):   {metrics['rmse']*1e3:.4f} mV")
    print(f" Err. Rel. Máximo:    {metrics['rel_err_max']:.3f} %")
    print("=" * 80)

    # 2. Simulación Superumbral (Tren de Pulsos de Pestaña 2: V_IN = 2.0 V @ 200 Hz, V_th = 1.0 V)
    cfg_spike = LIFConfig(
        c_m=cfg.c_m, r_series=cfg.r_series, r_leak=cfg.r_leak,
        v_rest=cfg.v_rest, v_th=1.0, v_reset=cfg.v_reset, t_ref=cfg.t_ref
    )
    neuron_spike = LIFNeuron(cfg_spike)
    steps_spike = 500000  # 0.5 s @ dt = 1e-6 s
    t_spike = np.linspace(0, 0.5, steps_spike)
    dt_spike = t_spike[1] - t_spike[0]
    v_signal_pulse = np.zeros(steps_spike)
    v_sim_spike = np.zeros(steps_spike)
    f0 = 200.0
    v0 = 2.0

    for k in range(steps_spike):
        phase = (k * dt_spike * f0) % 1.0
        v_signal_pulse[k] = v0 if phase < 0.2 else 0.0
        v_sim_spike[k] = neuron_spike.v_membrane
        neuron_spike.step(voltage_input=v_signal_pulse[k], dt=dt_spike)

    # 3. Generar Gráfico de Validación
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), facecolor='#ffffff')
    
    for ax in axes:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#000000', labelcolor='#000000')
        ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
            spine.set_linewidth(1.1)

    # Panel Superior: Comparación Analítica Subumbral
    ax1 = axes[0]
    ax1.plot(t * 1e3, metrics['v_analytical'], color='#2563eb', linewidth=2.2, label='[Analítico] Solución Exacta V_m(t)')
    ax1.plot(t[::15] * 1e3, v_sim_sub[::15], 'o', color='#dc2626', markersize=4, label='[Sim] Euler Numérico (Neuromorphic Lab)')
    ax1.set_title(f"Validación Analítica Carga RC Subumbral (R² = {metrics['r2']:.4f})", color='#0f172a', fontsize=11, fontweight='bold')
    ax1.set_xlabel("Tiempo (ms)", color='#000000', fontweight='bold')
    ax1.set_ylabel("Potencial V_m (V)", color='#000000', fontweight='bold')
    ax1.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a', loc='lower right')

    # Panel Inferior: Respuesta Superumbral con Spikes (Configuración GUI: 2.0V @ 200 Hz)
    ax2 = axes[1]
    step_ds = max(1, steps_spike // 4000)
    ax2.plot(t_spike[::step_ds] * 1e3, v_sim_spike[::step_ds], color='#059669', linewidth=1.5, label='Potencial Membrana V_m(t)')
    ax2.axhline(cfg_spike.v_th, color='#dc2626', linestyle='--', linewidth=1.4, label=f'Umbral V_th = {cfg_spike.v_th}V')
    
    if len(neuron_spike.spike_times) > 0:
        spike_times_ms = np.array(neuron_spike.spike_times) * 1e3
        ax2.plot(spike_times_ms, np.full_like(spike_times_ms, cfg_spike.v_th), '^', color='#d97706', markersize=7, label=f'Spikes Disparados ({len(neuron_spike.spike_times)})')

    ax2.set_title(f"Respuesta Superumbral en Pestaña 2 ({len(neuron_spike.spike_times)} Spikes a 200 Hz)", color='#0f172a', fontsize=11, fontweight='bold')
    ax2.set_xlabel("Tiempo (ms)", color='#000000', fontweight='bold')
    ax2.set_ylabel("Potencial V_m (V)", color='#000000', fontweight='bold')
    ax2.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a', loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / "figuras" / "lif_rc_validation.png", dpi=300)

    # 4. Guardar archivo CSV de resultados
    csv_path = output_dir / "datos" / "lif_rc_validation.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t_ms,V_sim_sub_V,V_analytical_sub_V,Err_abs_mV,V_sim_pulse_V\n")
        for k in range(steps):
            err_mv = abs(v_sim_sub[k] - metrics['v_analytical'][k]) * 1e3
            f.write(f"{t[k]*1e3:.4f},{v_sim_sub[k]:.6f},{metrics['v_analytical'][k]:.6f},{err_mv:.6f},{v_sim_spike[k]:.6f}\n")

    print("\n[OK] Archivos generados exitosamente en validaciones/:")
    print("   - lif_rc_validation.png")
    print("   - lif_rc_validation.csv")


if __name__ == "__main__":
    main()

