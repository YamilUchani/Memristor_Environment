"""
validaciones/validacion_convergencia.py
========================================
Validación de convergencia temporal del modelo de Strukov.

Objetivo:
  Demostrar que la solución numérica (Euler explícito) converge a la
  solución de referencia conforme dt -> 0, y que el orden de convergencia
  es O(dt) — consistente con un integrador de primer orden.

Metodología:
  1. Simular el mismo experimento con dt decreciente: 1e-2 -> 1e-6 s.
  2. Tomar como referencia la corrida con dt más fino (1e-6 s).
  3. Interpolar cada corrida al grid común más grueso.
  4. Calcular MAE, RMSE y error relativo vs la referencia.
  5. Graficar error vs dt en escala log-log.
  6. Estimar el orden de convergencia por regresión lineal.

Salidas (generadas en esta misma carpeta):
  - validacion_convergencia.png
  - validacion_convergencia.csv
"""

import sys
from pathlib import Path

# Garantizar que el directorio raíz de neuromorphic_lab esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from neurolab.core.config import DeviceIdentity, ElectricalConfig, StrukovConfig
from neurolab.core.memristor import Memristor
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.realism.window import BiolekWindowModifier


# =====================================================================
# 1. CONFIGURACIÓN DEL EXPERIMENTO
# =====================================================================
V0 = 1.0
F0 = 1.0
T_TOTAL = 1.0

def build_device():
    return Memristor(
        math_model=StrukovMathModel(),
        electrical=ElectricalConfig(
            r_on=100.0,
            r_off=16_000.0,
            initial_state=0.10,
        ),
        identity=DeviceIdentity(
            device_name="Convergencia",
            device_family="oxide_memristor",
            model_name="strukov",
        ),
        model_config=StrukovConfig(
            D=10e-9,
            mu_v=1e-14,
        ),
        modifiers=[BiolekWindowModifier(p=3)],
        clip_x=True,
    )


# =====================================================================
# 2. SIMULACIÓN CON UN dt DADO
# =====================================================================
def simular(dt):
    """Ejecuta la simulación con un dt dado y devuelve t, I, x."""
    dev = build_device()
    steps = int(T_TOTAL / dt)
    t = np.linspace(0.0, T_TOTAL, steps)

    v = V0 * np.sin(2.0 * np.pi * F0 * t)
    i_out = np.zeros(steps)
    x_arr = np.zeros(steps)

    for k in range(steps):
        i_out[k] = dev.step(voltage=v[k], dt=dt)
        x_arr[k] = dev.x

    return t, i_out, x_arr


def main():
    # dt más fino = referencia "verdadera"
    DT_REF = 1e-6
    print(f"Calculando referencia con dt = {DT_REF:.0e} s...")
    t_ref, i_ref, x_ref = simular(DT_REF)

    DT_LIST = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    print(f"Evaluando convergencia con {len(DT_LIST)} valores de dt...\n")

    resultados = []

    for dt in DT_LIST:
        t_sim, i_sim, x_sim = simular(dt)
        i_interp = np.interp(t_ref, t_sim, i_sim)
        x_interp = np.interp(t_ref, t_sim, x_sim)

        mae_i = float(np.mean(np.abs(i_ref - i_interp)))
        rmse_i = float(np.sqrt(np.mean((i_ref - i_interp) ** 2)))
        mae_x = float(np.mean(np.abs(x_ref - x_interp)))
        rmse_x = float(np.sqrt(np.mean((x_ref - x_interp) ** 2)))

        peak_i = float(np.max(np.abs(i_ref)))
        peak_x = float(np.max(np.abs(x_ref)))
        rel_i = float(np.max(np.abs(i_ref - i_interp)) / peak_i * 100) if peak_i > 0 else 0.0
        rel_x = float(np.max(np.abs(x_ref - x_interp)) / peak_x * 100) if peak_x > 0 else 0.0

        resultados.append({
            "dt": dt,
            "steps": int(T_TOTAL / dt),
            "mae_i": mae_i,
            "rmse_i": rmse_i,
            "mae_x": mae_x,
            "rmse_x": rmse_x,
            "rel_i": rel_i,
            "rel_x": rel_x,
        })

        print(f"  dt = {dt:.0e} s | steps = {int(T_TOTAL/dt):>8,} | "
              f"MAE_I = {mae_i:.4e} A | MAE_x = {mae_x:.4e}")

    dts = np.array([r["dt"] for r in resultados])
    maes_i = np.array([r["mae_i"] for r in resultados])
    maes_x = np.array([r["mae_x"] for r in resultados])

    mask = (maes_i > 0) & (dts > 0)
    log_dt = np.log10(dts[mask])
    log_mae_i = np.log10(maes_i[mask])
    log_mae_x = np.log10(maes_x[mask])

    orden_i, _ = np.polyfit(log_dt, log_mae_i, 1)
    orden_x, _ = np.polyfit(log_dt, log_mae_x, 1)

    print()
    print("=" * 92)
    print(" VALIDACION DE CONVERGENCIA TEMPORAL")
    print("=" * 92)
    print(f" Referencia: dt = {DT_REF:.0e} s\n")
    print(f"{'dt (s)':<10}{'Steps':<10}{'MAE_I (A)':<14}{'MAE_x':<14}{'RMSE_I (A)':<14}{'Err.rel.I (%)':<14}")
    print("-" * 92)
    for r in resultados:
        print(f"{r['dt']:<10.0e}{r['steps']:<10,}{r['mae_i']:<14.4e}{r['mae_x']:<14.4e}{r['rmse_i']:<14.4e}{r['rel_i']:<14.3f}")
    print("=" * 92)
    print(f" Orden de convergencia estimado (I):   p = {orden_i:.3f}")
    print(f" Orden de convergencia estimado (x):   p = {orden_x:.3f}")
    print(f" Teórico (Euler explícito, 1er orden): p = 1.000")
    print("=" * 92)

    output_dir = Path(__file__).resolve().parent

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#ffffff")
    ax = axes[0]
    ax.set_facecolor("#ffffff")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(DT_LIST)))
    for r, c in zip(resultados, colors):
        t_sim, i_sim, _ = simular(r["dt"])
        ax.plot(t_sim * 1e3, i_sim * 1e3, color=c, linewidth=1.2, label=f"dt = {r['dt']:.0e} s")
    ax.plot(t_ref * 1e3, i_ref * 1e3, color="#dc2626", linewidth=2.0, linestyle="--", label=f"[Ref] Referencia dt = {DT_REF:.0e} s")
    ax.set_xlabel("Tiempo (ms)", color="#000000", fontweight='bold')
    ax.set_ylabel("Corriente I(t) (mA)", color="#000000", fontweight='bold')
    ax.set_title("Corriente Simulada para Distintos dt", color="#0f172a", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor="#f8fafc", edgecolor="#cbd5e1", labelcolor="#0f172a", loc="upper right")
    ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.7)
    ax.tick_params(colors="#000000")
    for s in ax.spines.values(): s.set_color("#1e293b")

    ax = axes[1]
    ax.set_facecolor("#ffffff")
    ax.loglog(dts, maes_i, "o-", color="#dc2626", linewidth=2, markersize=7, label="[Sim] MAE — Corriente I")
    ax.loglog(dts, maes_x, "s-", color="#059669", linewidth=2, markersize=7, label="[Sim] MAE — Estado x")
    dt_ref_line = np.array([dts.min(), dts.max()])
    slope_line = maes_i[0] * (dt_ref_line / dts[0]) ** 1.0
    ax.loglog(dt_ref_line, slope_line, ":", color="#d97706", linewidth=1.8, label="[Teórico] Pendiente O(dt)")
    ax.set_xlabel("dt (s)", color="#000000", fontweight='bold')
    ax.set_ylabel("Error Absoluto vs Referencia", color="#000000", fontweight='bold')
    ax.set_title(f"Convergencia Temporal (Orden Estimado p ≈ {orden_i:.2f})", color="#0f172a", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, facecolor="#f8fafc", edgecolor="#cbd5e1", labelcolor="#0f172a")
    ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.7, which="both")
    ax.tick_params(colors="#000000")
    for s in ax.spines.values(): s.set_color("#1e293b")

    plt.tight_layout()
    plt.savefig(output_dir / "figuras" / "validacion_convergencia.png", dpi=300, bbox_inches="tight", facecolor="#ffffff")

    with open(output_dir / "datos" / "validacion_convergencia.csv", "w") as f:
        f.write("dt_s,steps,MAE_I_A,MAE_x,RMSE_I_A,ErrRel_I_pct\n")
        for r in resultados:
            f.write(f"{r['dt']:.0e},{r['steps']},{r['mae_i']:.6e},{r['mae_x']:.6e},{r['rmse_i']:.6e},{r['rel_i']:.4f}\n")

    print("\n[OK] Archivos generados en validaciones/:")
    print("   - validacion_convergencia.png")
    print("   - validacion_convergencia.csv")

if __name__ == "__main__":
    main()
