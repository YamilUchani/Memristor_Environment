"""
Validación F7: estabilidad del integrador Euler con dt extremos.

Barrido dt desde 1e-7 hasta 1e-2 s con una señal senoidal de 1 Hz y 1 V.
Se compara cada corrida contra la referencia (dt = 1e-7) y se reporta:
  - MAE de la corriente
  - Error relativo máximo
  - Estado de estabilidad (OK / degradado / roto)

Salidas:
  - validacion_dt_extremos.png
  - validacion_dt_extremos.csv
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from neurolab.core.memristor import Memristor
from neurolab.core.config import DeviceIdentity, ElectricalConfig, StrukovConfig
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.realism.window import BiolekWindowModifier


def build():
    return Memristor(
        math_model=StrukovMathModel(),
        electrical=ElectricalConfig(r_on=100.0, r_off=16_000.0,
                                    initial_state=0.10),
        identity=DeviceIdentity(device_name="dt_extremos",
                                device_family="oxide_memristor",
                                model_name="strukov"),
        model_config=StrukovConfig(D=10e-9, mu_v=1e-14),
        modifiers=[BiolekWindowModifier(p=3)],
        clip_x=True,
    )


def simular(dt, T=1.0, v0=1.0, f0=1.0):
    dev   = build()
    steps = int(T / dt)
    t     = np.linspace(0.0, T, steps)
    v     = v0 * np.sin(2.0 * np.pi * f0 * t)
    i_out = np.zeros(steps)
    x_out = np.zeros(steps)
    for k in range(steps):
        i_out[k] = dev.step(voltage=v[k], dt=dt)
        x_out[k] = dev.x
    return t, i_out, x_out


def run():
    print("=" * 78)
    print(" VALIDACION F7: ESTABILIDAD CON dt EXTREMOS")
    print("=" * 78)

    DT_REF = 1e-7
    print(f" Calculando referencia con dt = {DT_REF:.0e} s...")
    t_ref, i_ref, x_ref = simular(DT_REF)
    peak_i = float(np.max(np.abs(i_ref)))
    peak_x = float(np.max(np.abs(x_ref)))

    DT_LIST = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7]

    resultados = []
    print(f"\n{'dt (s)':<10}{'MAE I (A)':<14}{'Err max I (%)':<16}"
          f"{'Err max x (%)':<16}{'Estado':<12}")
    print("-" * 78)

    for dt in DT_LIST:
        t_sim, i_sim, x_sim = simular(dt)
        i_int = np.interp(t_ref, t_sim, i_sim)
        x_int = np.interp(t_ref, t_sim, x_sim)

        mae_i = float(np.mean(np.abs(i_ref - i_int)))
        rel_i = float(np.max(np.abs(i_ref - i_int)) / (peak_i + 1e-30) * 100.0)
        rel_x = float(np.max(np.abs(x_ref - x_int)) / (peak_x + 1e-30) * 100.0)

        if rel_i < 0.1:
            estado = "OK"
        elif rel_i < 1.0:
            estado = "degradado"
        else:
            estado = "roto"

        resultados.append({
            "dt": dt, "mae_i": mae_i, "rel_i": rel_i,
            "rel_x": rel_x, "estado": estado,
        })
        print(f"{dt:<10.0e}{mae_i:<14.4e}{rel_i:<16.4f}{rel_x:<16.4f}"
              f"{estado:<12}")

    print("-" * 78)

    # --- Figura ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#ffffff')
    for ax in axes:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#000000', labelcolor='#000000')
        ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.6,
                which='both')
        for sp in ax.spines.values():
            sp.set_color('#1e293b')
            sp.set_linewidth(1.1)

    dts   = np.array([r["dt"]    for r in resultados])
    rel_i = np.array([r["rel_i"] for r in resultados])
    rel_x = np.array([r["rel_x"] for r in resultados])

    # Panel 1: log-log del error vs dt
    ax1 = axes[0]
    ax1.loglog(dts, np.maximum(rel_i, 1e-10), 'o-', color='#dc2626',
               lw=1.8, label='Error max. I (%)')
    ax1.loglog(dts, np.maximum(rel_x, 1e-10), 's-', color='#059669',
               lw=1.8, label='Error max. x (%)')
    # Referencia O(dt)
    x_ref_line = np.array([dts.min(), dts.max()])
    y_ref_line = rel_i[-2] * (x_ref_line / dts[-2]) ** 1.0
    ax1.loglog(x_ref_line, y_ref_line, ':', color='#f59e0b',
               lw=1.8, label='O(dt) teorico')
    ax1.set_xlabel("dt (s)", fontweight='bold')
    ax1.set_ylabel("Error relativo maximo (%)", fontweight='bold')
    ax1.set_title("Convergencia en rango completo de dt",
                  fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)

    # Panel 2: barras de estado
    ax2 = axes[1]
    colores_map = {'OK': '#059669', 'degradado': '#f59e0b', 'roto': '#dc2626'}
    bar_colors = [colores_map[r["estado"]] for r in resultados]
    x_pos = np.arange(len(resultados))
    ax2.bar(x_pos, np.maximum(rel_i, 1e-10), color=bar_colors,
            edgecolor='#1e293b', lw=1.0)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{r['dt']:.0e}".replace("e-0", "e-").replace("e+0", "e+") for r in resultados],
                        rotation=45, fontsize=8)
    ax2.set_ylabel("Error max. I (%)", fontweight='bold')
    ax2.set_title("Clasificacion por estabilidad",
                  fontsize=11, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.6,
             which='both', axis='y')

    ax2.legend(handles=[
        Patch(facecolor='#059669', label='OK (err < 0.1%)'),
        Patch(facecolor='#f59e0b', label='degradado (0.1-1%)'),
        Patch(facecolor='#dc2626', label='roto (> 1%)'),
    ], loc='upper left', fontsize=9)

    plt.tight_layout()
    out_png = Path(__file__).resolve().parent / "validacion_dt_extremos.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')

    # --- CSV ---
    out_csv = Path(__file__).resolve().parent / "validacion_dt_extremos.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("dt_s,MAE_I_A,ErrRelMax_I_pct,ErrRelMax_x_pct,Estado\n")
        for r in resultados:
            f.write(f"{r['dt']:.0e},{r['mae_i']:.6e},{r['rel_i']:.4f},"
                    f"{r['rel_x']:.4f},{r['estado']}\n")

    print(f"\n[OK] Archivos generados:")
    print(f"   - {out_png.name}")
    print(f"   - {out_csv.name}")


if __name__ == "__main__":
    run()
