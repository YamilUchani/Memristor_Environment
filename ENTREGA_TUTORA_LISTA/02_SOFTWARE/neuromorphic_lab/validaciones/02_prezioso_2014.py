"""
validaciones/validacion_prezioso_limpio.py
===========================================
Validación Prezioso Fig 1b (Tema Blanco Académico).
Calcula la métrica R² y MAE/RMSE comparando la curva I(V) simulada con el CSV experimental.
"""

import sys
import csv
import json
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

    # This is an illustrative reconstruction; the source figure does not
    # specify enough sweep timing to reproduce the published trajectory.
    dev = build_device()
    dt = 1e-5
    duration = 3.0
    steps = int(duration / dt)
    phase = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
    v_ref = np.asarray(val_data["v_interp"], dtype=float)
    v_offset = 0.5 * (float(np.min(v_ref)) + float(np.max(v_ref)))
    v_amplitude = 0.5 * float(np.ptp(v_ref)) * 1.01
    v_signal = v_offset + v_amplitude * np.sin(phase)

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

    v_min, v_max = float(np.min(v_signal)), float(np.max(v_signal))
    coverage = float(np.mean((v_ref >= v_min) & (v_ref <= v_max)) * 100.0)
    i_ref_mA = np.asarray(val_data["i_val_mA"], dtype=float)
    step_ds = max(1, steps // 4000)
    print(
        "Reconstrucción visual aproximada de la histéresis I-V de la Fig. 1(b). "
        "La fuente no especifica el protocolo de barrido completo; el ciclo "
        "sinusoidal simulado es una excitación ilustrativa, no el protocolo "
        "publicado. No se calculan métricas de ajuste ni se asignan ramas "
        "SET/RESET al CSV."
    )
    print(
        f"Rango de referencia: [{np.min(v_ref):.6g}, {np.max(v_ref):.6g}] V; "
        f"rango de excitación: [{v_min:.6g}, {v_max:.6g}] V; "
        f"cobertura de voltaje: {coverage:.2f}%."
    )

    # 4. Graficar en Fondo Blanco Académico
    output_dir = Path(__file__).resolve().parent
    (output_dir / "figuras").mkdir(parents=True, exist_ok=True)
    thesis_figures_dir = ROOT_DIR.parent / "PLANTILLA_PDF_FINAL" / "figuras"
    thesis_figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#000000', labelcolor='#000000')
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
        spine.set_linewidth(1.1)

    v_ref = val_data.get("v_interp")
    mask_set = v_ref >= 0.0
    mask_reset = v_ref < 0.0

    ax.scatter(v_ref[mask_set], i_ref_mA[mask_set], color='#d97706', s=20, alpha=0.85, label='Referencia digitalizada, V ≥ 0', zorder=3)
    ax.scatter(v_ref[mask_reset], i_ref_mA[mask_reset], color='#0284c7', s=20, alpha=0.85, label='Referencia digitalizada, V < 0', zorder=3)

    ax.plot(v_signal[::step_ds], i_out[::step_ds] * 1e3, color='#dc2626', linewidth=2.0, label='Reconstrucción ilustrativa del modelo', zorder=4)

    ax.set_title("Prezioso et al. (2014), Fig. 1(b): reconstrucción aproximada", color='#0f172a', fontsize=11, fontweight='bold')
    ax.set_xlabel("Voltaje V_drop (V)", color='#000000', fontweight='bold')
    ax.set_ylabel("Corriente I (mA)", color='#000000', fontweight='bold')
    ax.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a', loc='upper left')
    ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.7)
    ax.text(
        0.02, 0.02,
        "Protocolo temporal no especificado; no es una reproducción exacta",
        transform=ax.transAxes, fontsize=8, color='#7f1d1d',
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#fecaca"},
    )

    plt.tight_layout()
    plt.savefig(output_dir / "figuras" / "validacion_prezioso_limpio.png", dpi=300)
    plt.savefig(thesis_figures_dir / "fig_02_prezioso.png", dpi=300)

    # Preserve the simulated trajectory and report coverage, not unpaired fit metrics.
    data_dir = output_dir / "datos"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "validacion_prezioso_simulacion.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["assumed_cycle_phase", "voltage_V", "current_mA", "state_x"])
        writer.writerows(zip(np.linspace(0.0, 1.0, steps, endpoint=False), v_signal, i_out * 1e3, x_state))

    with (data_dir / "validacion_prezioso.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value", "units"])
        writer.writerow(["reference_voltage_min", float(np.min(v_ref)), "V"])
        writer.writerow(["reference_voltage_max", float(np.max(v_ref)), "V"])
        writer.writerow(["simulation_voltage_min", v_min, "V"])
        writer.writerow(["simulation_voltage_max", v_max, "V"])
        writer.writerow(["reference_voltage_coverage", coverage, "%"])
        writer.writerow(["fit_metrics", "not calculated", ""])

    with (data_dir / "validacion_prezioso_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "reference": "Prezioso et al. (2014), Fig. 1(b), digitized I-V hysteresis",
                "reference_sweep_protocol": "not completely specified in the available source",
                "model_parameters": {
                    "r_on_ohm": 4750.0,
                    "r_off_ohm": 1_000_000.0,
                    "initial_state_x": 0.05,
                    "V_p_V": 0.60,
                    "V_n_V": 0.85,
                    "A_p": 200.0,
                    "A_n": 600.0,
                },
                "illustrative_excitation": {
                    "waveform": "one sinusoidal cycle spanning the digitized voltage range",
                    "cycle_duration_s": duration,
                    "integration_step_s": dt,
                    "values_are_assumptions_not_paper_conditions": True,
                },
                "voltage_coverage_percent": coverage,
                "fit_metrics": "not calculated; reference time and SET/RESET branch labels are unavailable",
                "conclusion": "approximate visual reconstruction based on Fig. 1(b), not exact reproduction",
            },
            f,
            indent=2,
        )

    print("\n[OK] Archivos generados en validaciones/:")
    print("   - figuras/validacion_prezioso_limpio.png (reconstrucción aproximada)")
    print("   - PLANTILLA_PDF_FINAL/figuras/fig_02_prezioso.png")
    print("   - datos/validacion_prezioso.csv (rango y cobertura)")
    print("   - datos/validacion_prezioso_simulacion.csv (trayectoria completa)")


if __name__ == "__main__":
    main()
