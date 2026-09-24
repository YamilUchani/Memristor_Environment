"""
validaciones/validacion_prezioso_limpio.py
===========================================
Validación Prezioso Fig 1b (Tema Blanco Académico).
Calcula la métrica R² y MAE/RMSE comparando la curva I(V) simulada con el CSV experimental.
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

from neurolab.core.config import DeviceIdentity, ElectricalConfig, PreziosoConfig
from neurolab.core.memristor import Memristor
from neurolab.devices.models.prezioso import PreziosoMathModel
from neurolab.io.validation_loader import ValidationDataLoader
from neurolab.core.validation_metrics import compute_all_metrics


def build_device():
    return Memristor(
        math_model=PreziosoMathModel(),
        electrical=ElectricalConfig(r_on=4750.0, r_off=1_000_000.0, initial_state=0.05),
        identity=DeviceIdentity(device_name="Prezioso 2014", device_family="oxide_memristor", model_name="prezioso"),
        model_config=PreziosoConfig(V_p=0.60, V_n=0.85, A_p=200.0, A_n=600.0, alpha_p=1.0, alpha_n=1.0, x_p=0.2, x_n=0.2),
        modifiers=[], clip_x=True,
    )


def main():
    print("=" * 76)
    print(" VALIDACION PREZIOSO et al. (2014) - FIGURA 1b")
    print("=" * 76)

    # 1. Cargar datos CSV de validación Prezioso
    loader = ValidationDataLoader()
    val_data = loader.load_all(dataset="prezioso")
    if val_data is None:
        print("[ERROR] No se pudo cargar el archivo CSV VvsC_Prezioso.csv.")
        return

    # 2. Simulación física
    dev = build_device()
    dt = 1e-5
    duration = 3.01
    steps = int(duration / dt)
    t = np.linspace(0.0, duration, steps)
    f0 = 3.001
    v_signal = 1.0 * np.sin(2.0 * np.pi * f0 * t)

    i_out = np.zeros(steps)
    x_state = np.zeros(steps)
    r_hist = np.zeros(steps)
    g_hist = np.zeros(steps)

    for k in range(steps):
        v = v_signal[k]
        i_out[k] = dev.step(voltage=v, dt=dt)
        x_state[k] = dev.x
        r_hist[k] = dev.resistance
        g_hist[k] = dev.conductance

    # 3. Métricas cuantitativas
    metrics = compute_all_metrics(
        t_sim=t, i_sim_mA=i_out * 1e3, x_sim=x_state,
        r_sim_kohm=r_hist / 1e3, g_sim_us=g_hist * 1e6,
        val_data=val_data, v_sim=v_signal
    )

    print(f"{'Magnitud':<25} {'MAE':<12} {'RMSE':<12} {'R²':<10}")
    print("-" * 76)
    for m in metrics:
        print(f"{m.name:<25} {m.mae:<12.4g} {m.rmse:<12.4g} {m.r2:<10.4f}")
    print("=" * 76)

    # 4. Graficar en Fondo Blanco Académico
    output_dir = Path(__file__).resolve().parent

    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#000000', labelcolor='#000000')
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
        spine.set_linewidth(1.1)

    v_ref = val_data.get("v_interp")
    i_ref_mA = val_data.get("i_val_mA")
    mask_set = v_ref >= 0.0
    mask_reset = v_ref < 0.0

    ax.scatter(v_ref[mask_set], i_ref_mA[mask_set], color='#d97706', s=20, alpha=0.85, label='[CSV] SET Experimental (Prezioso 2014)', zorder=3)
    ax.scatter(v_ref[mask_reset], i_ref_mA[mask_reset], color='#0284c7', s=20, alpha=0.85, label='[CSV] RESET Experimental (Prezioso 2014)', zorder=3)

    step_ds = max(1, steps // 4000)
    ax.plot(v_signal[::step_ds], i_out[::step_ds] * 1e3, color='#dc2626', linewidth=2.0, label='[Sim] Neuromorphic Lab', zorder=4)

    ax.set_title("Validación Curva I-V (Prezioso et al., 2014 vs Neuromorphic Lab)", color='#0f172a', fontsize=11, fontweight='bold')
    ax.set_xlabel("Voltaje V_drop (V)", color='#000000', fontweight='bold')
    ax.set_ylabel("Corriente I (mA)", color='#000000', fontweight='bold')
    ax.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a', loc='upper left')
    ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / "figuras" / "validacion_prezioso_limpio.png", dpi=300)

    # Guardar métricas en CSV
    with open(output_dir / "datos" / "validacion_prezioso.csv", "w", encoding="utf-8") as f:
        f.write("Magnitud,MAE,RMSE,ErrRelMax_pct,R2\n")
        for m in metrics:
            f.write(f"{m.name},{m.mae},{m.rmse},{m.max_rel_error_pct},{m.r2}\n")

    print("\n[OK] Archivos generados en validaciones/:")
    print("   - validacion_prezioso_limpio.png (Fondo Blanco Académico)")
    print("   - validacion_prezioso.csv")


if __name__ == "__main__":
    main()
