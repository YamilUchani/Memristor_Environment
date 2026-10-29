"""
Validación F5: R(x) sigue exactamente R_on*x + R_off*(1-x).

Verifica con barrido directo sobre x que la propiedad `resistance`
del objeto Memristor coincide con la fórmula analítica en todo el rango.

Salidas:
  - validacion_r_x_consistencia.png
  - validacion_r_x_consistencia.csv
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from neurolab.core.memristor import Memristor
from neurolab.core.config import DeviceIdentity, ElectricalConfig, StrukovConfig
from neurolab.devices.models.strukov import StrukovMathModel


def run():
    print("=" * 76)
    print(" VALIDACION F5: R(x) == R_ON*x + R_OFF*(1-x)")
    print("=" * 76)

    # --- Tres dispositivos con rangos distintos ---
    casos = [
        ("Strukov TiO2",   100.0,      16_000.0),
        ("Prezioso Al2O3", 4_750.0,  1_000_000.0),
        ("HfO2 Neurona",   1_000.0,  1_000_000.0),
    ]

    x_scan = np.linspace(0.0, 1.0, 201)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#ffffff')
    for ax in axes:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#000000', labelcolor='#000000')
        ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_color('#1e293b')
            sp.set_linewidth(1.1)

    resultados = []
    print(f"\n{'Dispositivo':<20}{'Err max (%)':<14}{'R(0) Ohm':<14}{'R(1) Ohm':<14}")
    print("-" * 76)

    for nombre, r_on, r_off in casos:
        mem = Memristor(
            math_model=StrukovMathModel(),
            electrical=ElectricalConfig(r_on=r_on, r_off=r_off, initial_state=0.5),
            identity=DeviceIdentity(device_name=nombre,
                                    device_family="oxide_memristor",
                                    model_name="strukov"),
            model_config=StrukovConfig(D=10e-9, mu_v=1e-14),
            modifiers=[],
            clip_x=True,
        )

        r_obj = np.zeros_like(x_scan)
        r_teo = np.zeros_like(x_scan)
        for i, xv in enumerate(x_scan):
            mem.x = xv
            r_obj[i] = mem.resistance
            r_teo[i] = r_on * xv + r_off * (1.0 - xv)

        err = np.abs(r_obj - r_teo) / r_teo * 100.0
        err_max = float(np.max(err))

        resultados.append({
            "nombre": nombre,
            "r_on": r_on, "r_off": r_off,
            "x": x_scan, "r_obj": r_obj, "r_teo": r_teo,
            "err": err, "err_max": err_max,
        })

        print(f"{nombre:<20}{err_max:<14.3e}{r_obj[0]:<14.1f}{r_obj[-1]:<14.1f}")

    print("-" * 76)

    # --- Panel 1: R(x) ---
    ax1 = axes[0]
    colores = ['#dc2626', '#2563eb', '#059669']
    for c, res in zip(colores, resultados):
        ax1.plot(res["x"], res["r_obj"] / 1e3, color=c, lw=1.6,
                 label=f"{res['nombre']} (R_obj)")
        ax1.plot(res["x"], res["r_teo"] / 1e3, color=c, lw=2.4,
                 linestyle='--', alpha=0.7,
                 label=f"{res['nombre']} (R_teo)")
    ax1.set_xlabel("Estado interno $x$", fontweight='bold')
    ax1.set_ylabel("Resistencia (kΩ)", fontweight='bold')
    ax1.set_title("R(x) — objeto Memristor vs formula analitica",
                  fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)

    # --- Panel 2: error relativo ---
    ax2 = axes[1]
    for c, res in zip(colores, resultados):
        err_plot = np.maximum(res["err"], 1e-16)
        ax2.semilogy(res["x"], err_plot,
                     color=c, lw=1.8, label=f"{res['nombre']}")
    ax2.set_xlabel("Estado interno $x$", fontweight='bold')
    ax2.set_ylabel("Error relativo $|R_{obj}-R_{teo}|/R_{teo}$ (%)",
                   fontweight='bold')
    ax2.set_title("Error relativo entre $R$ del objeto y $R$ analitica",
                  fontsize=11, fontweight='bold')
    ax2.axhline(1e-10, color='#f59e0b', linestyle=':', lw=1.2,
                label="Umbral 1e-10 %")
    ax2.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    out_png = Path(__file__).resolve().parent / "validacion_r_x_consistencia.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')

    # --- CSV ---
    out_csv = Path(__file__).resolve().parent / "validacion_r_x_consistencia.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("dispositivo,x,R_obj_ohm,R_teo_ohm,err_rel\n")
        for res in resultados:
            for xv, ro, rt, er in zip(res["x"], res["r_obj"],
                                       res["r_teo"], res["err"]):
                f.write(f"{res['nombre']},{xv:.4f},{ro:.4f},{rt:.4f},{er:.6e}\n")

    print(f"\n[OK] Archivos generados:")
    print(f"   - {out_png.name}")
    print(f"   - {out_csv.name}")


if __name__ == "__main__":
    run()
