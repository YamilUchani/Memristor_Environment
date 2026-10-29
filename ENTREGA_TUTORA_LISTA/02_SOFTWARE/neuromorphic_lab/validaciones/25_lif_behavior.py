"""Generate a qualitative LIF behavior figure from the Neuromorphic Lab model."""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
REPO_DIR = LAB_DIR.parent
sys.path.insert(0, str(LAB_DIR))

from neurolab.neurons.config import LIFConfig
from neurolab.neurons.lif import LIFNeuron


CONFIG_PATH = LAB_DIR / "configs" / "lif_config.json"
DATA_DIR = SCRIPT_DIR / "datos"
FIGURE_DIR = SCRIPT_DIR / "figuras"
THESIS_FIGURE = REPO_DIR / "PLANTILLA_PDF_FINAL" / "figuras" / "fig_13_lif_behavior.pdf"


def main():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config_data = json.load(config_file)

    signal = config_data["signal_panel"]
    if signal["waveform"] != "Tren de Pulsos (Unipolar)":
        raise ValueError(f"Unsupported LIF waveform: {signal['waveform']}")

    config = LIFConfig(
        c_m=float(config_data["c_m_value"]) * 1e-9,
        r_series=float(config_data["r_series_value"]) * 1e3,
        r_leak=float(config_data["r_leak_value"]) * 1e6,
        v_rest=float(config_data["v_rest"]),
        v_th=float(config_data["v_th"]),
        v_reset=float(config_data["v_reset"]),
        t_ref=float(config_data["t_ref"]) * 1e-3,
    )

    duration = float(signal["duration"])
    dt = float(signal["dt_ms"]) * 1e-3
    frequency = float(signal["f0"])
    amplitude = float(signal["v0"])
    sample_count = int(round(duration / dt))
    time = np.arange(sample_count, dtype=float) * dt
    phase = (time * frequency) % 1.0
    input_voltage = np.where(phase < 0.2, amplitude, 0.0)

    neuron = LIFNeuron(config)
    membrane_voltage = np.empty(sample_count, dtype=float)
    spike_event = np.zeros(sample_count, dtype=np.uint8)
    for index, (instant, voltage) in enumerate(zip(time, input_voltage)):
        membrane_voltage[index] = neuron.v_membrane
        if neuron.step(voltage_input=float(voltage), dt=dt, t=float(instant)):
            membrane_voltage[index] = config.v_th
            spike_event[index] = 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    thesis_figure_dir = THESIS_FIGURE.parent
    thesis_figure_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = np.unique(
        np.concatenate((np.arange(0, sample_count, 10), np.flatnonzero(spike_event)))
    )
    csv_path = DATA_DIR / "lif_behavior_qualitative.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as data_file:
        writer = csv.writer(data_file)
        writer.writerow(["time_s", "input_voltage_V", "membrane_voltage_V", "spike_event"])
        for index in sample_indices:
            writer.writerow(
                [
                    f"{time[index]:.9f}",
                    f"{input_voltage[index]:.6f}",
                    f"{membrane_voltage[index]:.6f}",
                    int(spike_event[index]),
                ]
            )

    spike_times = time[np.flatnonzero(spike_event)]
    figure, axes = plt.subplots(
        3, 1, figsize=(8.5, 6.8), sharex=True, gridspec_kw={"height_ratios": [1, 1.5, 0.55]}
    )
    figure.patch.set_facecolor("white")

    axes[0].plot(time[::10], input_voltage[::10], color="#2563eb", linewidth=0.9)
    axes[0].set_ylabel("$V_{in}$ (V)")
    axes[0].set_title("Estímulo pulsado e integración/disparo en la neurona LIF")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time[::10], membrane_voltage[::10], color="#7c3aed", linewidth=1.0)
    axes[1].axhline(
        config.v_th, color="#dc2626", linestyle="--", linewidth=1.0, label="$V_{th}$"
    )
    axes[1].scatter(
        spike_times,
        np.full_like(spike_times, config.v_th),
        color="#dc2626",
        marker="^",
        s=22,
        zorder=3,
        label="Disparos",
    )
    axes[1].set_ylabel("$V_m$ (V)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", frameon=True)

    axes[2].eventplot(
        spike_times, lineoffsets=0.5, linelengths=0.75, colors="#dc2626", linewidths=1.2
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_ylabel("Eventos")
    axes[2].set_xlabel("Tiempo (s)")
    axes[2].grid(True, axis="x", alpha=0.25)
    axes[2].set_xlim(0, duration)

    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "lif_behavior_qualitative.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "lif_behavior_qualitative.png", dpi=300, bbox_inches="tight")
    figure.savefig(THESIS_FIGURE, bbox_inches="tight")
    plt.close(figure)

    print(f"Config: C_m={config.c_m * 1e9:g} nF, R_series={config.r_series / 1e3:g} kOhm, "
          f"R_leak={config.r_leak / 1e6:g} MOhm, V_th={config.v_th:g} V, "
          f"t_ref={config.t_ref * 1e3:g} ms")
    print(f"Input: {amplitude:g} V, {frequency:g} Hz, 20% duty, duration={duration:g} s, dt={dt:g} s")
    print(f"Spike events: {len(spike_times)}")
    print(f"CSV: {csv_path}")
    print(f"Figure: {THESIS_FIGURE}")


if __name__ == "__main__":
    main()
