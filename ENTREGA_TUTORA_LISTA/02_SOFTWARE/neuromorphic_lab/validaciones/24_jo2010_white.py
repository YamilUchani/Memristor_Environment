import csv
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

from neurolab.validation.jo2010_stdp import (
    load_jo2010_stdp_data,
)
from neurolab.validation.jo2010_ltp_ltd import validate_jo2010_ltp_ltd
from neurolab.synapses import STDPRule
from neurolab.core.validation_metrics import (
    coefficient_of_determination,
    compute_metrics,
)


def _fit_stdp_parameters(train_mask=None):
    dt_exp, dw_exp, _ = load_jo2010_stdp_data()
    if train_mask is None:
        train_mask = np.ones(dt_exp.shape, dtype=bool)
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape != dt_exp.shape:
        raise ValueError("La máscara de entrenamiento debe coincidir con los datos STDP")
    ltp = (dt_exp > 0) & train_mask
    ltd = (dt_exp < 0) & train_mask
    if not ltp.any() or not ltd.any():
        raise ValueError("La referencia STDP debe incluir puntos LTP y LTD")

    def fit_ltp(dt_ms, amplitude, tau_ms):
        return amplitude * np.exp(-dt_ms / tau_ms)

    def fit_ltd(dt_ms, amplitude, tau_ms):
        return -amplitude * np.exp(dt_ms / tau_ms)

    a_plus, tau_plus = curve_fit(
        fit_ltp, dt_exp[ltp], dw_exp[ltp], p0=(15.0, 30.0),
        bounds=([0.0, 0.01], [100.0, 1000.0]),
    )[0]
    a_minus, tau_minus = curve_fit(
        fit_ltd, dt_exp[ltd], dw_exp[ltd], p0=(20.0, 30.0),
        bounds=([0.0, 0.01], [100.0, 1000.0]),
    )[0]
    return a_plus, a_minus, tau_plus, tau_minus


def _pearson_metrics(reference, prediction):
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    reference_scale = max(1.0, float(np.max(np.abs(reference))))
    prediction_scale = max(1.0, float(np.max(np.abs(prediction))))
    if (
        reference.size < 2
        or np.ptp(reference) <= np.finfo(float).eps * reference_scale
        or np.ptp(prediction) <= np.finfo(float).eps * prediction_scale
    ):
        return {"pearson_r": None, "pearson_r_squared": None}
    pearson_r = float(np.corrcoef(reference, prediction)[0, 1])
    return {
        "pearson_r": pearson_r,
        "pearson_r_squared": pearson_r**2,
    }


def plot_stdp():
    dt_exp, dw_exp, _ = load_jo2010_stdp_data()
    test_mask = np.zeros(dt_exp.shape, dtype=bool)
    for branch_mask in (dt_exp < 0, dt_exp > 0):
        branch_indices = np.flatnonzero(branch_mask)
        if branch_indices.size < 4:
            raise ValueError("Cada rama STDP debe tener al menos cuatro puntos")
        test_mask[branch_indices[[1, -2]]] = True
    train_mask = ~test_mask
    A_plus_pct, A_minus_pct, tau_plus_ms, tau_minus_ms = _fit_stdp_parameters(
        train_mask
    )

    stdp = STDPRule(
        A_plus=A_plus_pct / 100.0,
        A_minus=-A_minus_pct / 100.0,
        tau_plus=tau_plus_ms * 1e-3,
        tau_minus=tau_minus_ms * 1e-3,
    )
    dt_values = np.linspace(-80.0, 80.0, 161)
    dw_sim = np.array([stdp.delta_w(dt * 1e-3) * 100.0 for dt in dt_values])
    predictions_exp = np.array(
        [stdp.delta_w(dt * 1e-3) * 100.0 for dt in dt_exp]
    )
    metrics = compute_metrics(
        dw_exp[test_mask],
        predictions_exp[test_mask],
        name="STDP holdout",
        units="pp",
    )

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.plot(dt_values, dw_sim, color='#2ca02c', linewidth=2, label='Modelo analítico STDP')
    
    mask_ltp = dt_exp > 0
    mask_ltd = dt_exp < 0

    for branch_mask, color, branch_name in (
        (mask_ltp, "#1f77b4", "LTP"),
        (mask_ltd, "#d62728", "LTD"),
    ):
        train_branch = branch_mask & train_mask
        test_branch = branch_mask & test_mask
        ax.scatter(
            dt_exp[train_branch],
            dw_exp[train_branch],
            s=80,
            color=color,
            edgecolors="white",
            label=f"Jo 2010 (ajuste) — {branch_name}",
            zorder=3,
        )
        ax.scatter(
            dt_exp[test_branch],
            dw_exp[test_branch],
            s=90,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
            label=f"Jo 2010 (holdout) — {branch_name}",
            zorder=4,
        )
    ax.scatter(
        dt_exp[test_mask],
        predictions_exp[test_mask],
        marker="x",
        s=65,
        color="black",
        label="Predicción holdout",
        zorder=5,
    )

    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel(r'$\Delta t = t_{\mathrm{post}} - t_{\mathrm{pre}}$ (ms)', fontsize=12)
    ax.set_ylabel(r'$\Delta W$ (%)', fontsize=12)
    ax.set_title(
        'Jo 2010 — STDP, $\\Delta t=t_{post}-t_{pre}$ (positivo = LTP)',
        fontsize=13, fontweight='bold', color='#1f77b4',
    )

    ax.legend(fontsize=8, facecolor='white', framealpha=0.9)

    text_str = (
        f"Holdout $R^2$ = {metrics.r2:.4f}\n"
        f"MAE = {metrics.mae:.2f} pp\n"
        f"RMSE = {metrics.rmse:.2f} pp\n"
        f"ERP = {metrics.erp_range_pct:.2f}%"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.95, 0.05, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    out_path = Path(__file__).parent / 'figuras' / 'fig_15b_stdp_jo2010.png'
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    thesis_path = (
        Path(__file__).resolve().parents[2]
        / 'PLANTILLA_PDF_FINAL' / 'figuras' / out_path.name
    )
    thesis_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(thesis_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

    data_dir = Path(__file__).parent / "datos"
    data_dir.mkdir(exist_ok=True)
    csv_name = "jo2010_stdp_holdout.csv"
    with (data_dir / csv_name).open("w", newline="", encoding="utf-8") as data_file:
        writer = csv.writer(data_file)
        writer.writerow(
            [
                "delta_t_ms",
                "reference_delta_w_percent",
                "split",
                "predicted_delta_w_percent",
            ]
        )
        for index, (dt, reference, prediction) in enumerate(
            zip(dt_exp, dw_exp, predictions_exp)
        ):
            writer.writerow(
                [
                    dt,
                    reference,
                    "test" if test_mask[index] else "train",
                    prediction,
                ]
            )

    metrics_json = {
        "source": "csv",
        "n_total": int(dt_exp.size),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "split_method": (
            "Hold out sorted indices 1 and -2 from each nonzero delta_t branch; "
            "fit each exponential on remaining points."
        ),
        "timing_convention": {
            "project_and_digitized_axis": "delta_t = t_post - t_pre",
            "positive_delta_t": "pre-before-post; LTP",
            "paper_prose_definition": "delta_t = t_pre - t_post",
            "conversion": "delta_t_project = -delta_t_paper",
            "note": (
                "The paper's prose definition has the opposite sign to the "
                "positive-LTP digitized axis and the implemented rule; this "
                "comparison uses the explicit project/digitized convention."
            ),
        },
        "test_indices_zero_based": np.flatnonzero(test_mask).tolist(),
        "parameters_fitted_on_training_points": {
            "A_plus_percent": A_plus_pct,
            "A_minus_magnitude_percent": A_minus_pct,
            "tau_plus_ms": tau_plus_ms,
            "tau_minus_ms": tau_minus_ms,
        },
        "test_metrics": {
            "residual_r2": metrics.r2,
            "mae_percentage_points": metrics.mae,
            "rmse_percentage_points": metrics.rmse,
            "erp_percent": metrics.erp_range_pct,
        },
        "csv": csv_name,
    }
    with (data_dir / "jo2010_stdp_holdout_metrics.json").open(
        "w", encoding="utf-8"
    ) as metrics_file:
        json.dump(metrics_json, metrics_file, indent=2)

    print(
        f"STDP fit: A+={A_plus_pct:.4f}%, A-={A_minus_pct:.4f}%, "
        f"tau+={tau_plus_ms:.4f} ms, tau-={tau_minus_ms:.4f} ms"
    )
    print(
        f"STDP holdout (n={metrics.n_points}): R2={metrics.r2:.6f}, "
        f"MAE={metrics.mae:.6f} pp, RMSE={metrics.rmse:.6f} pp, "
        f"ERP_range={metrics.erp_range_pct:.6f}%"
    )

def plot_ltp_ltd():
    results = validate_jo2010_ltp_ltd()
    exp_pulses = np.asarray(results["pulse_num"], dtype=float)
    exp_current = np.asarray(results["current_exp"], dtype=float)
    idx_peak = int(results["idx_peak"])
    in_range = (exp_pulses >= 0.0) & (exp_pulses <= 200.0)
    split = np.full(exp_pulses.shape, "excluded", dtype=object)

    for branch_indices in (
        np.flatnonzero(in_range & (np.arange(len(exp_pulses)) <= idx_peak)),
        np.flatnonzero(in_range & (np.arange(len(exp_pulses)) > idx_peak)),
    ):
        ordered = branch_indices[np.argsort(exp_pulses[branch_indices])]
        if ordered.size < 5:
            raise ValueError("LTP y LTD requieren al menos cinco puntos dentro del rango")
        train_end = int(np.floor(0.60 * ordered.size))
        validation_end = int(np.floor(0.80 * ordered.size))
        if train_end < 2 or validation_end <= train_end or validation_end >= ordered.size:
            raise ValueError("No se pueden construir particiones LTP/LTD train/validation/test")
        split[ordered[:train_end]] = "train"
        split[ordered[train_end:validation_end]] = "validation"
        split[ordered[validation_end:]] = "test"

    def simulate(parameters):
        r_on = 10.0 ** parameters[0]
        r_off = r_on + 10.0 ** parameters[1]
        drift_coefficient = 10.0 ** parameters[2]
        state = float(parameters[3])
        v_read = 1.0
        pulse_width = 300e-6
        current_read_100na = [
            v_read / (r_on * state + r_off * (1.0 - state)) * 1e7
        ]

        for voltage in [3.2] * 100 + [-2.8] * 100:
            resistance = r_on * state + r_off * (1.0 - state)
            drive = drift_coefficient * voltage * pulse_width
            discriminant = (
                resistance**2 + 2.0 * (r_on - r_off) * drive
            )
            if discriminant <= 0.0:
                state = 1.0 if voltage > 0.0 else 0.0
            else:
                delta_state = (2.0 * drive) / (
                    resistance + np.sqrt(discriminant)
                )
                state = float(np.clip(state + delta_state, 0.0, 1.0))
            resistance = r_on * state + r_off * (1.0 - state)
            current_read_100na.append(v_read / resistance * 1e7)

        return np.arange(201, dtype=float), np.asarray(current_read_100na)

    train = split == "train"
    validation = split == "validation"
    test = split == "test"
    lower_bounds = np.array([6.0, 6.0, 6.0, 0.0])
    upper_bounds = np.array([9.0, 10.0, 14.0, 1.0])
    initial_guesses = (
        (7.0, 8.0, 10.0, 0.05),
        (7.5, 8.5, 11.0, 0.10),
        (8.0, 9.0, 12.0, 0.02),
    )
    candidates = []
    for initial_guess in initial_guesses:
        def train_residual(parameters):
            pulse_sim, current_sim = simulate(parameters)
            predicted = np.interp(exp_pulses[train], pulse_sim, current_sim)
            return predicted - exp_current[train]

        fit = least_squares(
            train_residual,
            x0=np.asarray(initial_guess, dtype=float),
            bounds=(lower_bounds, upper_bounds),
            max_nfev=500,
        )
        pulse_sim, current_sim = simulate(fit.x)
        predictions = np.interp(exp_pulses, pulse_sim, current_sim)
        candidates.append(
            {
                "initial_guess": list(initial_guess),
                "parameters": fit.x,
                "pulse_sim": pulse_sim,
                "current_sim": current_sim,
                "predictions": predictions,
                "train_rmse": float(np.sqrt(np.mean(train_residual(fit.x) ** 2))),
                "validation_rmse": float(
                    np.sqrt(
                        np.mean(
                            (predictions[validation] - exp_current[validation]) ** 2
                        )
                    )
                ),
                "optimizer_success": bool(fit.success),
            }
        )

    selected = min(candidates, key=lambda candidate: candidate["validation_rmse"])
    selected_parameters = selected["parameters"]
    train_metrics = compute_metrics(
        exp_current[train], selected["predictions"][train],
        name="LTP/LTD train", units="100 nA",
    )
    validation_metrics = compute_metrics(
        exp_current[validation], selected["predictions"][validation],
        name="LTP/LTD validation", units="100 nA",
    )
    test_metrics = compute_metrics(
        exp_current[test],
        selected["predictions"][test],
        name="LTP/LTD test",
        units="100 nA",
    )
    test_pearson = _pearson_metrics(
        exp_current[test], selected["predictions"][test]
    )
    pearson_label = (
        f"{test_pearson['pearson_r']:.4f}"
        if test_pearson["pearson_r"] is not None
        else "n/d"
    )
    r_on = 10.0 ** selected_parameters[0]
    r_off = r_on + 10.0 ** selected_parameters[1]
    drift_coefficient = 10.0 ** selected_parameters[2]

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.6)

    exp_I = exp_current
    ax.plot(exp_pulses[:idx_peak+1], exp_I[:idx_peak+1], 's-', color='#1f77b4', markersize=6, linewidth=1.5, label='LTP (Jo 2010, exp.)', zorder=3)
    ax.plot(exp_pulses[idx_peak:], exp_I[idx_peak:], 'o-', color='#d62728', markersize=6, linewidth=1.5, label='LTD (Jo 2010, exp.)', zorder=3)

    for name, color, marker in (
        ("train", "#1f77b4", "o"),
        ("validation", "#ff7f0e", "s"),
        ("test", "#2ca02c", "D"),
    ):
        mask = split == name
        ax.scatter(
            exp_pulses[mask],
            exp_current[mask],
            s=35,
            marker=marker,
            facecolors="none" if name == "test" else color,
            edgecolors=color,
            label=f"Referencia {name}",
            zorder=4,
        )
    ax.plot(
        selected["pulse_sim"], selected["current_sim"],
        "--",
        color="#111827",
        linewidth=2.2,
        label="Modelo Strukov; protocolo publicado",
        zorder=2,
    )

    ax.set_xlabel('Pulso', fontsize=12)
    ax.set_ylabel('Corriente de lectura (100 nA)', fontsize=12)
    ax.set_title('Jo et al. (2010), Fig. 2a — protocolo publicado', fontsize=14, fontweight='bold', color='#1f77b4')

    ax.legend(fontsize=10, facecolor='white', framealpha=0.9, loc='upper left')

    text_str = (
        f"Test $R^2$ = {test_metrics.r2:.4f}\n"
        f"Pearson $r$ = {pearson_label}\n"
        f"MAE = {test_metrics.mae * 100:.2f} nA\n"
        f"RMSE = {test_metrics.rmse * 100:.2f} nA\n"
        f"ERP = {test_metrics.erp_range_pct:.2f}%\n"
        f"$n_{{test}}$ = {test_metrics.n_points}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    out_path = Path(__file__).parent / 'figuras' / 'fig_14b_ltp_ltd_jo2010.png'
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    thesis_path = (
        Path(__file__).resolve().parents[2]
        / 'PLANTILLA_PDF_FINAL' / 'figuras' / out_path.name
    )
    thesis_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(thesis_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

    data_dir = Path(__file__).parent / "datos"
    data_dir.mkdir(exist_ok=True)
    csv_name = "jo2010_ltp_ltd_train_validation_test.csv"
    with (data_dir / csv_name).open("w", newline="", encoding="utf-8") as data_file:
        writer = csv.writer(data_file)
        writer.writerow(
            [
                "pulse_number",
                "reference_current_100nA",
                "split",
                "simulated_read_current_100nA",
            ]
        )
        for pulse, reference, split_name, prediction in zip(
            exp_pulses, exp_current, split, selected["predictions"]
        ):
            writer.writerow([pulse, reference, split_name, prediction])

    candidate_summary = [
        {
            "initial_guess": candidate["initial_guess"],
            "optimizer_success": candidate["optimizer_success"],
            "train_rmse_100nA": candidate["train_rmse"],
            "validation_rmse_100nA": candidate["validation_rmse"],
        }
        for candidate in candidates
    ]
    metrics_json = {
        "source": "Jo et al. (2010), Fig. 2a digitized CSV",
        "n_total": int(len(exp_pulses)),
        "n_excluded_outside_simulated_pulse_range": int(np.sum(split == "excluded")),
        "n_train": int(train.sum()),
        "n_validation": int(validation.sum()),
        "n_test": int(test.sum()),
        "split_method": (
            "Separately within LTP/LTD, sort covered points by pulse count; "
            "first 60% train, next 20% validation, last 20% test."
        ),
        "simulator": {
            "device": "Strukov linear-drift model",
            "initial_state_x": float(selected_parameters[3]),
            "r_on_ohm": r_on,
            "r_off_ohm": r_off,
            "effective_drift_coefficient_per_amp_second": drift_coefficient,
            "set_pulse_voltage_V": 3.2,
            "reset_pulse_voltage_V": -2.8,
            "pulse_width_s": 300e-6,
            "pulse_count_per_phase": 100,
            "read_voltage_V": 1.0,
            "switching_threshold_magnitude_V": 2.2,
            "read_state_update": "disabled; read voltage is below the reported threshold",
        },
        "parameter_fit": (
            "Fit R_on, R_off-R_on, effective drift coefficient, and x0 on "
            "training points only. No post-fit current scale or offset is "
            "applied. Select among three predeclared optimizer initializations "
            "using validation RMSE."
        ),
        "candidates": candidate_summary,
        "selected_parameters": {
            "r_on_ohm": r_on,
            "r_off_ohm": r_off,
            "initial_state_x": float(selected_parameters[3]),
            "effective_drift_coefficient_per_amp_second": drift_coefficient,
        },
        "metrics": {
            split_name: {
                "r2": metric.r2,
                "mae_nA": metric.mae * 100.0,
                "rmse_nA": metric.rmse * 100.0,
                "erp_range_pct": metric.erp_range_pct,
                "n_points": metric.n_points,
            }
            for split_name, metric in (
                ("train", train_metrics),
                ("validation", validation_metrics),
                ("test", test_metrics),
            )
        },
        "test_metrics_by_phase": {
            phase_name: {
                "r2": phase_metrics.r2,
                "mae_nA": phase_metrics.mae * 100.0,
                "rmse_nA": phase_metrics.rmse * 100.0,
                "erp_range_pct": phase_metrics.erp_range_pct,
                "n_points": phase_metrics.n_points,
                **_pearson_metrics(
                    exp_current[phase_mask],
                    selected["predictions"][phase_mask],
                ),
            }
            for phase_name, phase_mask in (
                ("LTP", test & (np.arange(len(exp_pulses)) <= idx_peak)),
                ("LTD", test & (np.arange(len(exp_pulses)) > idx_peak)),
            )
            for phase_metrics in (
                compute_metrics(
                    exp_current[phase_mask],
                    selected["predictions"][phase_mask],
                    name=f"LTP/LTD {phase_name} test",
                    units="100 nA",
                ),
            )
        },
        "test_metrics": {
            "residual_r2": test_metrics.r2,
            **test_pearson,
            "mae_100nA_units": test_metrics.mae,
            "mae_nA": test_metrics.mae * 100.0,
            "rmse_100nA_units": test_metrics.rmse,
            "rmse_nA": test_metrics.rmse * 100.0,
            "erp_range_pct": test_metrics.erp_range_pct,
            "n_points": test_metrics.n_points,
        },
        "csv": csv_name,
        "interpretation": (
            "The fitted parameters are calibrated to this digitized device "
            "trace; held-out points test the protocol simulation against "
            "the same experiment, not prediction of an independent device."
        ),
    }
    with (data_dir / "jo2010_ltp_ltd_train_validation_test.json").open(
        "w", encoding="utf-8"
    ) as metrics_file:
        json.dump(metrics_json, metrics_file, indent=2)

    print(
        f"LTP/LTD split n={train.sum()}/{validation.sum()}/{test.sum()} "
        "(train/validation/test); applied +3.2/-2.8 V, 300 us pulses "
        "and 1 V reads"
    )
    print(
        f"LTP/LTD test (n={test_metrics.n_points}): "
        f"R2={test_metrics.r2:.6f}, "
        f"Pearson_r={pearson_label}, "
        f"MAE={test_metrics.mae * 100:.6f} nA, "
        f"RMSE={test_metrics.rmse * 100:.6f} nA, "
        f"ERP_range={test_metrics.erp_range_pct:.6f}%"
    )

if __name__ == "__main__":
    plot_stdp()
    plot_ltp_ltd()
    print("Figuras de validación Jo 2010 en blanco generadas correctamente.")
