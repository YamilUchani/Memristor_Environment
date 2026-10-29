"""
Validación F1: el ISI decreciente está gobernado por x(t) del memristor HfO2.

Añade un 4º panel que demuestra:
    R_S(t) == R_ON * x(t) + R_OFF * (1 - x(t))

Si coinciden al 100 %, la resistencia usada por la neurona viene del estado
dinámico del memristor y no de una función externa.

Salidas:
  - hfo2_lif_isi_decreciente_v2.png (4 paneles)
  - hfo2_isi_decreciente_v2.csv
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from neurolab.core.memristor import Memristor
from neurolab.core.config import DeviceIdentity, ElectricalConfig, StrukovConfig
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.realism.volatile import VolatileDecayModifier
from neurolab.neurons.lif import LIFNeuron
from neurolab.neurons.config import LIFConfig


def run():
    print("=" * 82)
    print(" VALIDACION F1: R_S(t) VIENE DE x(t) DEL MEMRISTOR (HfO2 VOLATIL)")
    print("=" * 82)

    # ── Memristor volátil HfO2 ────────────────────────────────────────────
    R_ON  = 10_000.0     # 10 kΩ
    R_OFF = 500_000.0    # 500 kΩ
    elec = ElectricalConfig(r_on=R_ON, r_off=R_OFF, initial_state=0.05)

    mem = Memristor(
        math_model=StrukovMathModel(),
        electrical=elec,
        identity=DeviceIdentity(
            device_name="HfO2 Volatile",
            device_family="HfO2_oxide",
            model_name="strukov",
        ),
        model_config=StrukovConfig(D=10e-9, mu_v=1e-14),
        modifiers=[VolatileDecayModifier(tau_relax=1.0, x0_override=0.05)],
        clip_x=True,
    )

    # ── LIF ───────────────────────────────────────────────────────────────
    lif_cfg = LIFConfig(
        c_m=50e-9, r_series=1.0, r_leak=1e6,
        v_rest=0.0, v_th=1.0, v_reset=0.0, t_ref=1e-3,
    )
    neuron = LIFNeuron(lif_cfg)

    # ── Protocolo 2 ciclos ────────────────────────────────────────────────
    duration = 8.0
    dt       = 2e-5
    steps    = int(duration / dt)
    t        = np.arange(steps) * dt

    stim1_s, stim1_e = 0.0, 2.5
    sil1_s,  sil1_e  = 2.5, 4.5
    stim2_s, stim2_e = 4.5, 7.0
    sil2_s,  sil2_e  = 7.0, 8.0
    v_source = 1.5

    v_in = np.zeros(steps)
    stim_mask = (((t >= stim1_s) & (t < stim1_e)) |
                 ((t >= stim2_s) & (t < stim2_e)))
    v_in[stim_mask] = v_source

    v_m_arr       = np.zeros(steps)
    x_arr         = np.zeros(steps)
    r_s_arr       = np.zeros(steps)   # R leída del objeto Memristor
    r_theo_arr    = np.zeros(steps)   # R recalculada a mano: R_on*x + R_off*(1-x)

    for k in range(steps):
        v_m_arr[k] = neuron.v_membrane

        # --- Avanzar el memristor con el voltaje real de la rama ---
        v_drop = max(v_in[k] - neuron.v_membrane, 0.0)
        mem.step(voltage=v_drop, dt=dt)

        # --- Leer x, R del objeto y recalcular R a mano ---
        x_now  = mem.x
        r_now  = mem.resistance
        r_theo = R_ON * x_now + R_OFF * (1.0 - x_now)

        x_arr[k]      = x_now
        r_s_arr[k]    = r_now
        r_theo_arr[k] = r_theo

        # --- Avanzar la neurona con Memristor R_S en SERIE ---
        r_series_eff = r_now
        r_leak_eff   = lif_cfg.r_leak

        # --- Integración manual de la neurona LIF con R_S dinámica ---
        neuron.t += dt
        neuron.has_spiked = False
        if neuron.refractory_time_left > 0.0:
            neuron.refractory_time_left -= dt
            neuron.v_membrane += (-(neuron.v_membrane - lif_cfg.v_rest) / r_leak_eff / lif_cfg.c_m) * dt
        else:
            i_in   = max(v_in[k] - neuron.v_membrane, 0.0) / r_series_eff
            i_leak = (neuron.v_membrane - lif_cfg.v_rest) / r_leak_eff
            neuron.v_membrane += (i_in - i_leak) / lif_cfg.c_m * dt
            if neuron.v_membrane >= lif_cfg.v_th:
                neuron.has_spiked = True
                neuron.spike_times.append(neuron.t)
                neuron.v_membrane = lif_cfg.v_reset
                if lif_cfg.t_ref > 0:
                    neuron.refractory_time_left = lif_cfg.t_ref

    spike_times = np.array(neuron.spike_times)
    isis_all    = np.diff(spike_times) * 1e3 if len(spike_times) > 1 else np.array([])

    # --- Verificación cuantitativa de la igualdad R == R_on*x + R_off*(1-x) ---
    diff_rel = np.abs(r_s_arr - r_theo_arr) / np.maximum(np.abs(r_theo_arr), 1.0)
    max_err  = float(np.max(diff_rel)) * 100.0
    mean_err = float(np.mean(diff_rel)) * 100.0

    print(f" Muestras analizadas:         {steps:,}")
    print(f" Error maximo  |R_obj - R_teo|/R_teo: {max_err:.6e} %")
    print(f" Error medio   |R_obj - R_teo|/R_teo: {mean_err:.6e} %")
    print(f" Rango R_S:  {r_s_arr.min()/1e3:.1f} - {r_s_arr.max()/1e3:.1f} kOhm")
    print(f" Rango x:    {x_arr.min():.4f} - {x_arr.max():.4f}")
    print(f" Spikes totales: {len(spike_times)}")
    print("-" * 82)
    print(" VALORES CLAVE DE x(t):")
    print(f"   x al inicio del estimulo 1 (t=0.00s):   {x_arr[0]:.4f}")
    print(f"   x al final del estimulo 1  (t=2.49s):   {x_arr[int(2.49/dt)]:.4f}")
    print(f"   x al final del silencio 1  (t=4.49s):   {x_arr[int(4.49/dt)]:.4f}")
    print(f"   x al inicio del estimulo 2 (t=4.50s):   {x_arr[int(4.50/dt)]:.4f}")
    print(f"   x al final del estimulo 2  (t=6.99s):   {x_arr[int(6.99/dt)]:.4f}")
    print(f"   x al final del silencio 2  (t=7.99s):   {x_arr[int(7.99/dt)]:.4f}")
    print("=" * 82)

    # --- Figura 4 paneles ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             facecolor='#ffffff')
    for ax in axes:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#000000', labelcolor='#000000')
        ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_color('#1e293b')
            sp.set_linewidth(1.1)
        for s, e, c in [(stim1_s, stim1_e, '#ef4444'),
                        (sil1_s, sil1_e, '#3b82f6'),
                        (stim2_s, stim2_e, '#ef4444'),
                        (sil2_s, sil2_e, '#3b82f6')]:
            ax.axvspan(s, e, alpha=0.07, color=c, zorder=0)

    step_ds = max(1, steps // 6000)

    # Panel 1: V_m
    ax1 = axes[0]
    ax1.plot(t[::step_ds], v_m_arr[::step_ds], color='#2563eb', lw=0.9)
    ax1.axhline(lif_cfg.v_th, color='#dc2626', linestyle='--', lw=1.2,
                label=f'$V_{{th}} = {lif_cfg.v_th}$ V')
    if len(spike_times) > 0:
        ax1.vlines(spike_times, 0, lif_cfg.v_th, colors='#f59e0b',
                   lw=0.4, alpha=0.5)
    ax1.set_ylabel("$V_c$ (V)", fontweight='bold')
    ax1.set_title("(1) Potencial de membrana $V_c(t)$", fontsize=11,
                  fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    # Panel 2: x(t)
    ax2 = axes[1]
    ax2.plot(t[::step_ds], x_arr[::step_ds], color='#7c3aed', lw=2.0)
    ax2.set_ylabel("$x(t)$", fontweight='bold')
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("(2) Estado interno del memristor $x(t)$",
                  fontsize=11, fontweight='bold')

    # Panel 3: R_S(t) leída vs R recalculada a mano
    ax3 = axes[2]
    ax3.plot(t[::step_ds], r_s_arr[::step_ds] / 1e3, color='#059669',
             lw=1.6, label='$R_S(t)$ leida del objeto Memristor')
    ax3.plot(t[::step_ds], r_theo_arr[::step_ds] / 1e3, color='#f97316',
             lw=2.4, linestyle='--',
             label='$R_{ON}\\,x + R_{OFF}\\,(1-x)$ recalculada a mano')
    ax3.set_ylabel("$R_S$ (kΩ)", fontweight='bold')
    ax3.set_title(
        f"(3) R_S(t) vs R(x(t)) — Error maximo = {max_err:.2e} %",
        fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)

    # Panel 4: ISI
    ax4 = axes[3]
    if len(isis_all) > 0:
        ax4.semilogy(spike_times[1:], isis_all, 'o-',
                     color='#e11d48', lw=1.0, markersize=2)
    ax4.set_xlabel("Tiempo (s)", fontweight='bold')
    ax4.set_ylabel("ISI (ms) [log]", fontweight='bold')
    ax4.set_title("(4) Intervalo Inter-Spike — decreciente porque $R_S$ baja",
                  fontsize=11, fontweight='bold')

    plt.tight_layout()
    out_png = Path(__file__).resolve().parent / "hfo2_lif_isi_decreciente_v2.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')

    # --- CSV de consistencia ---
    out_csv = Path(__file__).resolve().parent / "hfo2_isi_decreciente_v2.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("t_s,V_m_V,x,R_obj_ohm,R_theo_ohm,diff_rel\n")
        for k in range(0, steps, max(1, steps // 5000)):
            f.write(f"{t[k]:.6f},{v_m_arr[k]:.6f},{x_arr[k]:.6f},"
                    f"{r_s_arr[k]:.4f},{r_theo_arr[k]:.4f},{diff_rel[k]:.3e}\n")

    print(f"\n[OK] Archivos generados:")
    print(f"   - {out_png.name}")
    print(f"   - {out_csv.name}")


if __name__ == "__main__":
    run()
