"""
validaciones/validacion_prezioso_v2.py
======================================
Validación Prezioso et al. (2014) — Figura 1b con asimetría R_ON.
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


R_ON_SET   = 4_750.0
R_ON_RESET = 1_750.0
R_OFF      = 1_000_000.0


def build_device():
    return Memristor(
        math_model=PreziosoMathModel(),
        electrical=ElectricalConfig(r_on=R_ON_SET, r_off=R_OFF, initial_state=0.05),
        identity=DeviceIdentity(device_name="Prezioso 2014", device_family="oxide_memristor", model_name="prezioso"),
        model_config=PreziosoConfig(V_p=0.60, V_n=0.85, A_p=200.0, A_n=600.0, alpha_p=1.0, alpha_n=1.0, x_p=0.2, x_n=0.2),
        modifiers=[], clip_x=True,
    )


def simular(dt=1e-5, T=0.4, freq=5.0, v_max=2.0):
    dev = build_device()
    steps = int(T / dt)
    t = np.linspace(0.0, T, steps)
    v_signal = v_max * np.sin(2.0 * np.pi * freq * t)

    i_out = np.zeros(steps)
    x_state = np.zeros(steps)

    for k in range(steps):
        v = v_signal[k]
        i_out[k] = dev.step(voltage=v, dt=dt)
        x_state[k] = dev.x

    return t, v_signal, i_out, x_state


def main():
    print("=" * 78)
    print(" VALIDACION PREZIOSO et al. 2014 -- FIGURA 1b (OPCION C)")
    print("=" * 78)

    t, v_signal, i_out, x_state = simular()
    output_dir = Path(__file__).resolve().parent

    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#000000', labelcolor='#000000')
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
        spine.set_linewidth(1.1)

    ax.plot(v_signal[::5], i_out[::5] * 1e3, color='#dc2626', linewidth=2.0, label='[Sim] Prezioso 2014 (Neuromorphic Lab)')
    ax.set_title("Curva I-V Prezioso (2014) — Asimetría de Conmutación", color='#0f172a', fontsize=11, fontweight='bold')
    ax.set_xlabel("Voltaje V_drop (V)", color='#000000', fontweight='bold')
    ax.set_ylabel("Corriente I (mA)", color='#000000', fontweight='bold')
    ax.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a', loc='upper left')
    ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / "figuras" / "validacion_prezioso_v2.png", dpi=300)

    # Guardar datos en CSV
    csv_path = output_dir / "datos" / "validacion_prezioso_v2.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t_s,V_signal_V,I_out_A,x_state\n")
        for k in range(0, len(t), 5):
            f.write(f"{t[k]:.6f},{v_signal[k]:.6f},{i_out[k]:.6e},{x_state[k]:.6f}\n")

    print("\n[OK] Archivos generados en validaciones/:")
    print("   - validacion_prezioso_v2.png")
    print("   - validacion_prezioso_v2.csv")


if __name__ == "__main__":
    main()
