"""
validaciones/validacion_hfo2_isi.py
====================================
Script de Validación: Adaptación Neuronal e ISI Decreciente.
Circuito Híbrido: Memristor VOLÁTIL HfO₂ (VolatileDecayModifier) + Neurona LIF.

Protocolo 2 ciclos (sensor repetible):
  Ciclo 1 — Estímulo DC (0–2.5 s)  →  Silencio (2.5–4.5 s)
  Ciclo 2 — Estímulo DC (4.5–7 s)  →  Silencio (7–8 s)

Física:
  • Durante estímulo: x(t) sube (Strukov), R_S baja, corriente aumenta → ISI se acorta.
  • Durante silencio: x decae con tau=0.3 s → R_S recupera → sensor listo para nuevo ciclo.

Salidas:
  - hfo2_lif_isi_decreciente.png   (300 DPI, Tema Blanco Académico)
  - hfo2_lif_isi_decreciente.pdf   (Vectorial Tesis)
  - hfo2_lif_isi_decreciente.csv
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
from neurolab.circuits.hybrid import MemristorLIFCircuit


def run_hfo2_isi_validation():
    print("=" * 80)
    print(" VALIDACION: Adaptacion HfO2 Volatil + LIF (2 Ciclos Estimulo/Silencio)")
    print("=" * 80)

    # ── 1. Memristor HfO2 VOLATIL ────────────────────────────────────────────
    #   R_ON=10 kOhm, R_OFF=500 kOhm  =>  a x=0.05 arranca en ~476 kOhm
    #   Durante estimulo DC: x sube rapidamente, R_S baja hasta ~10 kOhm
    #   Durante silencio:    x decae con tau=0.3 s  =>  R_S recupera en ~1.5 s
    elec = ElectricalConfig(
        r_on=10_000.0,
        r_off=500_000.0,
        initial_state=0.05
    )
    mem = Memristor(
        math_model=StrukovMathModel(),
        electrical=elec,
        identity=DeviceIdentity(
            device_name="HfO2 Sensor Volatil",
            device_family="HfO2_oxide",
            model_name="strukov"
        ),
        model_config=StrukovConfig(D=10e-9, mu_v=1e-14),
        modifiers=[VolatileDecayModifier(tau_relax=1.0, x0_override=0.05)],
        clip_x=True
    )

    # ── 2. Neurona LIF ────────────────────────────────────────────────────────
    #   Con estimulo DC de 1.5 V y R_S inicial ~476 kOhm:
    #     I_in ~ 3.16 uA  =>  V_ss = I * R_leak = 3.16 V > V_th  =>  el LIF dispara
    #     ISI_inicial ~ 19 ms  (primer spike)
    #   Con R_S saturado en R_ON = 10 kOhm:
    #     I_in ~ 148 uA  =>  ISI ~ 0.5 ms  (disparo muy rapido)
    lif_cfg = LIFConfig(
        c_m=50e-9,           # 50 nF
        r_series=1.0,        # 1 Ohm (acoplamiento dominado por el memristor)
        r_leak=1_000_000.0,  # 1 MOhm  =>  tau_m = 50 ms
        v_rest=0.0,
        v_th=1.0,            # Umbral 1.0 V
        v_reset=0.0,
        t_ref=0.001          # 1 ms refractario
    )
    neuron = LIFNeuron(lif_cfg)
    circuit = MemristorLIFCircuit(memristor=mem, neuron=neuron)

    # ── 3. Protocolo 2 Ciclos (8 s total) ─────────────────────────────────────
    duration = 8.0
    dt       = 2e-5          # 20 us (500,000 pasos)
    steps    = int(duration / dt)
    t        = np.arange(steps) * dt

    # Fases (segundos)
    stim1_s, stim1_e = 0.0,  2.5
    sil1_s,  sil1_e  = 2.5,  4.5
    stim2_s, stim2_e = 4.5,  7.0
    sil2_s,  sil2_e  = 7.0,  8.0

    v_source = 1.5  # Estimulo DC

    v_signal  = np.zeros(steps)
    v_c_hist  = np.zeros(steps)
    r_s_hist  = np.zeros(steps)

    stim_mask = (
        ((t >= stim1_s) & (t < stim1_e)) |
        ((t >= stim2_s) & (t < stim2_e))
    )
    v_signal[stim_mask] = v_source

    for k in range(steps):
        v_c_hist[k] = neuron.v_membrane
        r_s_hist[k] = mem.resistance
        circuit.step(v_signal[k], dt)

    spike_times = np.array(neuron.spike_times)
    n_spikes    = len(spike_times)
    isis_all    = np.diff(spike_times) * 1e3   # ms (entre todos los spikes)

    # Spikes por ciclo
    spk_c1 = spike_times[(spike_times >= stim1_s) & (spike_times < stim1_e)]
    spk_c2 = spike_times[(spike_times >= stim2_s) & (spike_times < stim2_e)]
    isis_c1 = np.diff(spk_c1) * 1e3 if len(spk_c1) > 1 else np.array([])
    isis_c2 = np.diff(spk_c2) * 1e3 if len(spk_c2) > 1 else np.array([])

    print(f"  Total pasos:  {steps:,}  |  dt = {dt*1e6:.0f} us")
    print(f"  Total spikes: {n_spikes}")
    print(f"  Ciclo 1 spikes: {len(spk_c1)}", end="")
    if len(isis_c1) > 0:
        print(f"  |  ISI: {isis_c1[0]:.2f} -> {isis_c1[-1]:.3f} ms  "
              f"(Reduccion: {(isis_c1[0]-isis_c1[-1])/isis_c1[0]*100:.1f}%)")
    else:
        print()
    print(f"  Ciclo 2 spikes: {len(spk_c2)}", end="")
    if len(isis_c2) > 0:
        print(f"  |  ISI: {isis_c2[0]:.2f} -> {isis_c2[-1]:.3f} ms  "
              f"(Reduccion: {(isis_c2[0]-isis_c2[-1])/isis_c2[0]*100:.1f}%)")
    else:
        print()

    # R_S al inicio de cada ciclo
    idx_c2_start = np.searchsorted(t, stim2_s)
    print(f"  R_S inicio Ciclo 1: {r_s_hist[0]/1e3:.1f} kOhm")
    print(f"  R_S inicio Ciclo 2: {r_s_hist[idx_c2_start]/1e3:.1f} kOhm  "
          f"(recuperacion: {r_s_hist[idx_c2_start]/r_s_hist[0]*100:.0f}%)")
    print("=" * 80)

    # ── 4. Exportar CSV ────────────────────────────────────────────────────────
    output_dir = Path(__file__).resolve().parent
    csv_path   = output_dir / "datos" / "hfo2_lif_isi_decreciente.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Metrica,Ciclo1,Ciclo2,Unidad\n")
        f.write(f"Total_Spikes,{len(spk_c1)},{len(spk_c2)},count\n")
        if len(isis_c1) > 0 and len(isis_c2) > 0:
            f.write(f"Primer_ISI_ms,{isis_c1[0]:.4f},{isis_c2[0]:.4f},ms\n")
            f.write(f"Ultimo_ISI_ms,{isis_c1[-1]:.4f},{isis_c2[-1]:.4f},ms\n")
            red1 = (isis_c1[0]-isis_c1[-1])/isis_c1[0]*100
            red2 = (isis_c2[0]-isis_c2[-1])/isis_c2[0]*100
            f.write(f"Reduccion_ISI_pct,{red1:.2f},{red2:.2f},%\n")
        f.write(f"R_S_inicio_ciclo_kOhm,{r_s_hist[0]/1e3:.2f},{r_s_hist[idx_c2_start]/1e3:.2f},kOhm\n")

    # ── 5. Figura 3 paneles (Tema Blanco Academico) ───────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             facecolor='#ffffff')
    ax1, ax2, ax3 = axes

    # Paleta de fases
    clr_stim = '#ef4444'
    clr_sil  = '#3b82f6'
    alpha_bg = 0.08

    for ax in axes:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#000000', labelcolor='#000000')
        ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
            spine.set_linewidth(1.1)
        # Sombras de fase
        for s, e, clr in [(stim1_s, stim1_e, clr_stim), (sil1_s,  sil1_e,  clr_sil),
                           (stim2_s, stim2_e, clr_stim), (sil2_s,  sil2_e,  clr_sil)]:
            ax.axvspan(s, e, alpha=alpha_bg, color=clr, zorder=0)
        # Lineas de transicion
        for xv in [stim1_e, stim2_s, stim2_e]:
            ax.axvline(xv, color='#94a3b8', linewidth=0.9, linestyle=':', zorder=1)

    # Anotaciones de ciclo en la parte superior de ax1
    for label, xmid in [("Ciclo 1 — Estimulo", 1.25),
                         ("  Silencio", 3.5),
                         ("Ciclo 2 — Estimulo", 5.75),
                         (" Silencio", 7.5)]:
        ax1.text(xmid, 1.06, label, ha='center', va='bottom', fontsize=8,
                 color='#334155', transform=ax1.get_xaxis_transform())

    # ── Panel 1: V_c(t) ──
    step_ds = max(1, steps // 6000)
    ax1.plot(t[::step_ds], v_c_hist[::step_ds],
             color='#2563eb', linewidth=0.8, label='$V_c(t)$ Membrana')
    ax1.axhline(lif_cfg.v_th, color='#dc2626', linestyle='--', linewidth=1.2,
                label=f'Umbral $V_{{th}}$ = {lif_cfg.v_th} V')
    if n_spikes > 0:
        # Mostrar spikes como marcadores pequeños
        ax1.vlines(spike_times, 0, lif_cfg.v_th, colors='#f59e0b',
                   linewidth=0.4, alpha=0.6, zorder=4)
    ax1.set_title("(1) Potencial de Membrana $V_c(t)$ — Disparo Acelerado por Memristor HfO2 Volatil",
                  color='#0f172a', fontsize=11, fontweight='bold')
    ax1.set_ylabel("$V_c$ (V)", color='#000000', fontweight='bold')
    ax1.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a',
               loc='upper right', fontsize=9)

    # ── Panel 2: R_S(t) ──
    ax2.plot(t[::step_ds], r_s_hist[::step_ds] / 1e3,
             color='#059669', linewidth=2.0, label='$R_S(t)$ HfO2 Volatil')
    ax2.set_title("(2) Resistencia Sinaptica $R_S(t)$ — Baja en Estimulo, Recupera en Silencio (tau = 0.3 s)",
                  color='#0f172a', fontsize=11, fontweight='bold')
    ax2.set_ylabel("$R_S$ (kOhm)", color='#000000', fontweight='bold')
    ax2.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a',
               loc='upper right', fontsize=9)

    # ── Panel 3: ISI(t) en escala log ──
    if len(isis_all) > 0:
        # Tiempos de ISI: spike_times[1:]
        sp_isi_t = spike_times[1:]
        ax3.semilogy(sp_isi_t, isis_all, 'o-',
                     color='#7c3aed', linewidth=1.2, markersize=2.5,
                     markerfacecolor='#7c3aed', label='ISI $\\Delta t$ (ms)')
        # Anotaciones de primer/ultimo ISI en ciclo 1
        if len(isis_c1) > 1:
            ax3.annotate(f"{isis_c1[0]:.1f} ms",
                         xy=(spk_c1[1], isis_c1[0]),
                         xytext=(spk_c1[1]+0.1, isis_c1[0]*2.0),
                         fontsize=8, color='#1d4ed8',
                         arrowprops=dict(arrowstyle='->', color='#1d4ed8'))
            ax3.annotate(f"{isis_c1[-1]:.2f} ms",
                         xy=(spk_c1[-1], isis_c1[-1]),
                         xytext=(spk_c1[-1]-0.5, isis_c1[-1]*0.2),
                         fontsize=8, color='#b91c1c',
                         arrowprops=dict(arrowstyle='->', color='#b91c1c'))
        ax3.set_title("(3) Adaptacion ISI — Decreciente en Estimulo, Sensor Repetible en 2 Ciclos",
                      color='#0f172a', fontsize=11, fontweight='bold')
    else:
        ax3.set_title("(3) ISI — Sin Spikes (revisar parametros)", color='#0f172a', fontsize=11)

    ax3.set_xlabel("Tiempo (s)", color='#000000', fontweight='bold')
    ax3.set_ylabel("ISI $\\Delta t$ (ms) [escala log]", color='#000000', fontweight='bold')
    ax3.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a',
               loc='upper right', fontsize=9)

    # Leyenda global de fases
    p_stim = mpatches.Patch(color=clr_stim, alpha=0.35, label='Fase Estimulo (1.5 V DC)')
    p_sil  = mpatches.Patch(color=clr_sil,  alpha=0.35, label='Fase Silencio — Recuperacion Volatil (tau=0.3 s)')
    fig.legend(handles=[p_stim, p_sil], loc='lower center',
               ncol=2, facecolor='#f8fafc', edgecolor='#cbd5e1',
               labelcolor='#0f172a', fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_png = output_dir / "figuras" / "hfo2_lif_isi_decreciente.png"
    out_pdf = output_dir / "figuras" / "hfo2_lif_isi_decreciente.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print("\n[OK] Archivos generados en validaciones/:")
    print(f"   - {out_png.name}  (300 DPI)")
    print(f"   - {out_pdf.name}  (Vectorial Tesis)")
    print(f"   - {csv_path.name}")


if __name__ == "__main__":
    run_hfo2_isi_validation()
