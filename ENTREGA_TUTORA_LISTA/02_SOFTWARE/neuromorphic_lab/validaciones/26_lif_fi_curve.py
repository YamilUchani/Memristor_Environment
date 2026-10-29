"""Generate an LIF frequency-current curve from the configured laboratory model."""

import csv
import json
from pathlib import Path
import sys

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


def main():
    config_path = LAB_DIR / "configs" / "lif_config.json"
    with config_path.open(encoding="utf-8") as config_file:
        raw = json.load(config_file)

    config = LIFConfig(
        c_m=float(raw["c_m_value"]) * 1e-9,
        r_series=float(raw["r_series_value"]) * 1e3,
        r_leak=float(raw["r_leak_value"]) * 1e6,
        v_rest=float(raw["v_rest"]),
        v_th=float(raw["v_th"]),
        v_reset=float(raw["v_reset"]),
        t_ref=float(raw["t_ref"]) * 1e-3,
    )

    currents_uA = np.array(
        [0.0, 0.5, 0.9, 1.0, 1.01, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
    )
    dt = 1e-4
    duration = 10.0
    transient = 1.0
    steps = int(round(duration / dt))
    rows = []

    for current_uA in currents_uA:
        neuron = LIFNeuron(config)
        for _ in range(steps):
            neuron.step(current_input=float(current_uA * 1e-6), dt=dt)
        observed_spikes = sum(spike_time >= transient for spike_time in neuron.spike_times)
        observed_hz = observed_spikes / (duration - transient)

        steady_voltage = config.v_rest + current_uA * 1e-6 * config.r_leak
        if steady_voltage <= config.v_th:
            predicted_hz = 0.0
        else:
            tau = config.c_m * config.r_leak
            crossing_time = tau * np.log(
                (steady_voltage - config.v_reset)
                / (steady_voltage - config.v_th)
            )
            predicted_hz = 1.0 / (crossing_time + config.t_ref)

        rows.append((current_uA, observed_hz, predicted_hz, observed_spikes))

    data_dir = SCRIPT_DIR / "datos"
    figure_dir = SCRIPT_DIR / "figuras"
    thesis_figure_dir = REPO_DIR / "PLANTILLA_PDF_FINAL" / "figuras"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    thesis_figure_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "lif_fi_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as data_file:
        writer = csv.writer(data_file)
        writer.writerow(
            ["current_uA", "observed_frequency_Hz", "analytical_frequency_Hz", "spikes"]
        )
        writer.writerows(rows)

    current = np.array([row[0] for row in rows])
    observed = np.array([row[1] for row in rows])
    predicted = np.array([row[2] for row in rows])
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(current, predicted, "-", color="#2563eb", label="Solución LIF analítica")
    axis.scatter(current, observed, color="#dc2626", label="Simulación Euler")
    axis.set_xlabel("Corriente de entrada (µA)")
    axis.set_ylabel("Frecuencia de disparo (Hz)")
    axis.set_title("Curva frecuencia-corriente de la neurona LIF")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    for path in (
        figure_dir / "lif_fi_curve.png",
        thesis_figure_dir / "lif_fi_curve.png",
    ):
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"CSV: {csv_path}")
    print(f"Figure: {thesis_figure_dir / 'lif_fi_curve.png'}")


if __name__ == "__main__":
    main()
