"""Run a paired Auto 2D REINFORCE/R-STDP comparison batch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_car
import run_car_rstdp
from car_env import ACTION_NAMES, CarEnv

EPISODES = 4000
SEEDS = (0, 1, 2)
INITIAL_STATES = (0.1, 0.9)
ALGORITHMS = ("REINFORCE", "R-STDP")
EVAL_INTERVAL = 200
if (
    REPOSITORY_ROOT.name == "02_SOFTWARE"
    and REPOSITORY_ROOT.parent.name == "ENTREGA_TUTORA_LISTA"
):
    DELIVERY_ROOT = REPOSITORY_ROOT.parent
    OUTPUT_DIR = DELIVERY_ROOT / "03_EVIDENCIAS" / "PILOTO_AUTO2D_4000"
else:
    OUTPUT_DIR = (
        REPOSITORY_ROOT
        / "ENTREGA_TUTORA_LISTA"
        / "03_EVIDENCIAS"
        / "PILOTO_AUTO2D_4000"
    )


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _write_json(path: Path, data: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, default=_json_value)
        stream.write("\n")
    temporary_path.replace(path)


def _write_source_manifest(output_dir: Path) -> None:
    source_files = [
        ROOT / "run_car.py",
        ROOT / "run_car_rstdp.py",
        ROOT / "car_env.py",
        ROOT / "neurobot" / "reinforce_agent.py",
        ROOT / "neurobot" / "rstdp_cartpole_agent.py",
        ROOT / "neurobot" / "crossbar_brain.py",
        ROOT / "neurobot" / "encoder.py",
        ROOT / "neurobot" / "decoder.py",
        ROOT / "neurobot" / "lab_bridge.py",
        ROOT / "neurobot" / "__init__.py",
        Path(__file__).resolve(),
        ROOT / "requirements.txt",
        ROOT / "tests" / "test_car_reinforce.py",
        ROOT / "tests" / "test_car_rstdp.py",
        REPOSITORY_ROOT / "neuromorphic_lab" / "requirements.txt",
        REPOSITORY_ROOT
        / "neuromorphic_lab"
        / "configs"
        / "strukov_ideal.json",
        REPOSITORY_ROOT
        / "neuromorphic_lab"
        / "configs"
        / "memristor_volatile.json",
        REPOSITORY_ROOT
        / "neuromorphic_lab"
        / "configs"
        / "lif_config.json",
    ]
    lab_source = REPOSITORY_ROOT / "neuromorphic_lab" / "neurolab"
    source_files.extend(sorted(lab_source.rglob("*.py")))
    manifest_files = {}
    for source in sorted(set(source_files)):
        if not source.is_file():
            raise FileNotFoundError(f"Pilot source file is missing: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest_files[source.relative_to(REPOSITORY_ROOT).as_posix()] = digest
    _write_json(output_dir / "source_manifest.json", {
        "source_tree": (
            "ENTREGA_TUTORA_LISTA/02_SOFTWARE robot_sim and neuromorphic_lab"
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "sha256": manifest_files,
    })

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_reinforce_agent(seed: int, initial_x: float):
    env = CarEnv(seed=seed)
    return run_car.ReinforceAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        N=run_car.N,
        M=run_car.M,
        V_read=0.05,
        eta=run_car.ETA,
        encoder_mode=run_car.ENCODER_MODE,
        temp_init=run_car.TEMP_INIT_HD,
        temp_min=run_car.TEMP_MIN,
        temp_decay=run_car.TEMP_DECAY,
        dt=1e-3,
        k_wta=3,
        wta_temperature=1.0,
        baseline_window=200,
        seed=seed,
        encoder_seed=seed,
        initial_X=initial_x,
        json_memristor_nv=run_car_rstdp.DEFAULT_NV_PROFILE,
        json_memristor_v="memristor_volatile.json",
        json_lif=run_car_rstdp.DEFAULT_LIF_PROFILE,
        d2d_enabled=False,
        c2c_enabled=False,
    )


def _run_one(algorithm: str, seed: int, initial_x: float, episodes: int,
             output_dir: Path, verbose: bool) -> dict:
    label = f"{algorithm.lower().replace('-', '')}_seed{seed}_X0_{initial_x:g}"
    run_dir = output_dir / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("episodes_completed") == episodes
            and existing.get("algorithm") == algorithm
            and existing.get("seed") == seed
            and existing.get("initial_X") == initial_x
        ):
            print(f"[omitida, completa] {label}")
            return existing
        raise FileExistsError(
            f"Existing run does not match this protocol: {result_path}"
        )

    env = CarEnv(seed=seed)
    if algorithm == "REINFORCE":
        agent = _make_reinforce_agent(seed, initial_x)
        initial_weights = agent.brain.X.copy()
        tracker = run_car.GreedyPolicyTracker(seed)
        log = run_car.run_headless(
            env,
            agent,
            episodes=episodes,
            seed=seed,
            verbose=verbose,
            policy_tracker=tracker,
            eval_interval=EVAL_INTERVAL,
            adapt_exploration=False,
        )
        log["policy_evaluations"] = tracker.history
        test_tracker = run_car.GreedyPolicyTracker(
            seed,
            eval_episodes=run_car_rstdp.POLICY_TEST_EPISODES,
            map_seed_offset=run_car.POLICY_TEST_SEED_OFFSET,
        )
        weights_before_test = agent.brain.X.copy()
        log["held_out_test_evaluation"] = test_tracker.evaluate(
            agent, episodes
        )
        if not np.array_equal(agent.brain.X, weights_before_test):
            raise RuntimeError("Held-out evaluation modified REINFORCE weights.")
        training_map_seed = run_car._training_map_seed(seed)
        agent_config = {
            "N": run_car.N,
            "M": run_car.M,
            "eta": run_car.ETA,
            "dt": agent.dt,
            "steps_per_action": agent.steps_per_action,
            "k_wta": agent.k_wta,
            "wta_temperature": agent.wta_temperature,
            "V_read": agent.encoder.V_read,
            "memristor_nv_profile": run_car_rstdp.DEFAULT_NV_PROFILE,
            "lif_profile": run_car_rstdp.DEFAULT_LIF_PROFILE,
            "d2d_enabled": False,
            "c2c_enabled": False,
            "temperature_initial": run_car.TEMP_INIT_HD,
            "temperature_min": run_car.TEMP_MIN,
            "temperature_decay": run_car.TEMP_DECAY,
            "baseline_window": 200,
            "exploration_epsilon": agent.epsilon,
            "max_weight_update": run_car.MAX_WEIGHT_UPDATE,
            "temperature_adaptation_during_training": False,
        }
    elif algorithm == "R-STDP":
        agent = run_car_rstdp.make_agent(
            seed=seed,
            eta=run_car_rstdp.DEFAULT_ETA,
            initial_x=initial_x,
        )
        initial_weights = agent.brain.X.copy()
        log = run_car_rstdp.run_training(
            agent,
            env,
            episodes=episodes,
            seed=seed,
            verbose=verbose,
            policy_eval_interval=EVAL_INTERVAL,
        )
        training_map_seed = run_car_rstdp._training_map_seed(seed)
        agent_config = {
            "N": agent.brain.N,
            "M": agent.brain.M,
            "eta": agent.eta,
            "dt": agent.dt,
            "steps_per_action": agent.steps_per_action,
            "k_wta": agent.k_wta,
            "wta_temperature": agent.wta_temperature,
            "V_read": agent.encoder.V_read,
            "memristor_nv_profile": run_car_rstdp.DEFAULT_NV_PROFILE,
            "lif_profile": run_car_rstdp.DEFAULT_LIF_PROFILE,
            "d2d_enabled": False,
            "c2c_enabled": False,
            "temperature_initial": agent.temp_init,
            "temperature_min": agent.temp_min,
            "temperature_decay": agent.temp_decay,
            "baseline_window": agent.reward_history.maxlen,
            "exploration_epsilon": run_car_rstdp.EXPLORATION_EPSILON,
            "max_abs_delta_x": agent.max_abs_delta_x,
        }
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if log.get("episodes_completed") != episodes:
        raise RuntimeError(
            f"{algorithm} seed={seed} X0={initial_x} completed "
            f"{log.get('episodes_completed')} of {episodes} episodes"
        )
    expected_initial_weights = np.clip(
        initial_x
        + np.random.default_rng(seed).normal(
            0.0, 0.01, initial_weights.shape
        ),
        0.0,
        1.0,
    )
    if not np.array_equal(initial_weights, expected_initial_weights):
        raise RuntimeError(
            "The seeded crossbar initialization does not match the requested X0."
        )

    final_weights = agent.brain.X.copy()
    np.save(run_dir / "crossbar_initial_X.npy", initial_weights)
    np.save(run_dir / "crossbar_final_X.npy", final_weights)
    log.update({
        "algorithm": algorithm,
        "encoder": "auto2d",
        "action_names": list(ACTION_NAMES),
        "seed": int(seed),
        "initial_X": float(initial_x),
        "episodes_requested": int(episodes),
        "episodes_completed": int(episodes),
        "training_map_seed": int(training_map_seed),
        "policy_eval_interval": EVAL_INTERVAL,
        "policy_eval_episodes": run_car.POLICY_EVAL_EPISODES,
        "held_out_test_episodes": run_car_rstdp.POLICY_TEST_EPISODES
        if algorithm == "REINFORCE"
        else run_car_rstdp.POLICY_TEST_EPISODES,
        "agent_config": agent_config,
        "initial_crossbar_X_mean": float(initial_weights.mean()),
        "initial_crossbar_X_min": float(initial_weights.min()),
        "initial_crossbar_X_max": float(initial_weights.max()),
        "final_crossbar_X_mean": float(final_weights.mean()),
        "final_crossbar_X_min": float(final_weights.min()),
        "final_crossbar_X_max": float(final_weights.max()),
        "crossbar_shape": list(final_weights.shape),
    })
    _write_json(result_path, log)
    _write_episode_csv(run_dir / "episodes.csv", log)
    print(f"[guardado] {result_path}")
    return log


def _write_episode_csv(path: Path, log: dict) -> None:
    success = log.get("success", [])
    count = len(success)
    episodes = log.get("episode", list(range(1, count + 1)))
    keys = {
        "episode": episodes,
        "success": success,
        "reward": log.get("reward", []),
        "distance": log.get("distance", log.get("dist_end", [])),
        "steps": log.get("steps", log.get("episode_steps", [])),
        "termination_reason": log.get("termination_reason", []),
        "collision_type": log.get("collision_type", []),
        "path_length": log.get("path_length", []),
        "path_efficiency": log.get("path_efficiency", []),
        "X_mean": log.get("X_mean", []),
        "temperature": log.get("temperature", []),
        "delta_x_abs_mean": log.get(
            "delta_x_abs_mean", log.get("actual_delta_x_abs_mean", [])
        ),
        "eligibility_abs_mean": log.get("eligibility_abs_mean", []),
        "pre_events": log.get("pre_events", []),
        "post_spikes": log.get("post_spikes", []),
        "update_applied": log.get("update_applied", []),
    }
    rows = []
    for index in range(count):
        row = {}
        for key, values in keys.items():
            row[key] = values[index] if index < len(values) else ""
        rows.append(row)
    _write_csv(path, rows, list(keys))


def _run_label(log: dict) -> str:
    return (
        f"{log['algorithm'].lower().replace('-', '')}"
        f"_seed{log['seed']}_X0_{log['initial_X']:g}"
    )


def _crossbar_paths(log: dict, output_dir: Path) -> tuple[Path, Path]:
    run_dir = output_dir / "runs" / _run_label(log)
    return run_dir / "crossbar_initial_X.npy", run_dir / "crossbar_final_X.npy"


def _summary_row(log: dict, output_dir: Path) -> dict:
    validation = log.get("policy_evaluations", [])
    test = log.get("held_out_test_evaluation") or {}
    training_success = np.asarray(log.get("success", []), dtype=float)
    validation_rates = [
        float(item["success_rate"])
        for item in validation
        if item.get("success_rate") is not None
    ]
    pre_events = log.get("pre_events", [])
    post_spikes = log.get("post_spikes", [])
    update_applied = log.get("update_applied", [])
    delta_x = log.get("delta_x_abs_mean", [])
    initial_path, final_path = _crossbar_paths(log, output_dir)
    crossbar_change = np.abs(
        np.load(final_path) - np.load(initial_path)
    )
    return {
        "algorithm": log["algorithm"],
        "seed": log["seed"],
        "initial_X": log["initial_X"],
        "episodes_completed": log["episodes_completed"],
        "train_success_last_100": (
            float(training_success[-100:].mean())
            if training_success.size else None
        ),
        "validation_success_final": (
            validation[-1].get("success_rate") if validation else None
        ),
        "validation_success_best": (
            max(validation_rates) if validation_rates else None
        ),
        "test_success_rate": test.get("success_rate"),
        "test_successes": test.get("successes"),
        "test_episodes": test.get("episodes"),
        "test_mean_path_efficiency": test.get("mean_path_efficiency"),
        "test_mean_distance": test.get("mean_distance"),
        "final_X_mean": log["final_crossbar_X_mean"],
        "crossbar_mean_change": (
            log["final_crossbar_X_mean"] - log["initial_crossbar_X_mean"]
        ),
        "final_X_min": log["final_crossbar_X_min"],
        "final_X_max": log["final_crossbar_X_max"],
        "total_pre_events": sum(pre_events) if pre_events else None,
        "total_post_spikes": sum(post_spikes) if post_spikes else None,
        "episodes_with_weight_update": (
            sum(bool(value) for value in update_applied)
            if update_applied else None
        ),
        "mean_episode_abs_delta_X": (
            float(np.mean(delta_x)) if delta_x else None
        ),
        "mean_abs_crossbar_change": float(crossbar_change.mean()),
        "fraction_crossbar_cells_changed": float(
            np.count_nonzero(crossbar_change > 1e-12) / crossbar_change.size
        ),
        "training_map_seed": log["training_map_seed"],
        "policy_eval_interval": log["policy_eval_interval"],
    }


def _write_aggregate_tables(logs: list[dict], output_dir: Path) -> None:
    summary = [_summary_row(log, output_dir) for log in logs]
    fields = list(summary[0])
    _write_csv(output_dir / "runs_summary.csv", summary, fields)

    condition_summary = []
    for algorithm in ALGORITHMS:
        for initial_x in INITIAL_STATES:
            group = [
                row for row in summary
                if row["algorithm"] == algorithm
                and row["initial_X"] == initial_x
            ]
            for metric in (
                "train_success_last_100",
                "validation_success_final",
                "test_success_rate",
                "test_mean_path_efficiency",
                "test_mean_distance",
                "mean_abs_crossbar_change",
                "fraction_crossbar_cells_changed",
            ):
                values = np.asarray(
                    [row[metric] for row in group if row[metric] is not None],
                    dtype=float,
                )
                condition_summary.append({
                    "algorithm": algorithm,
                    "initial_X": initial_x,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()) if values.size else None,
                    "sample_std": (
                        float(values.std(ddof=1)) if values.size > 1 else None
                    ),
                })
    _write_csv(
        output_dir / "conditions_summary.csv",
        condition_summary,
        ["algorithm", "initial_X", "metric", "n", "mean", "sample_std"],
    )

    validation_rows = []
    for log in logs:
        for item in log.get("policy_evaluations", []):
            validation_rows.append({
                "algorithm": log["algorithm"],
                "seed": log["seed"],
                "initial_X": log["initial_X"],
                "episode": item["episode"],
                "success_rate": item["success_rate"],
            })
    _write_csv(
        output_dir / "validation.csv",
        validation_rows,
        ["algorithm", "seed", "initial_X", "episode", "success_rate"],
    )

    episode_rows = []
    for log in logs:
        success = log.get("success", [])
        rewards = log.get("reward", [])
        distances = log.get("distance", log.get("dist_end", []))
        for index, result in enumerate(success):
            episode_rows.append({
                "algorithm": log["algorithm"],
                "seed": log["seed"],
                "initial_X": log["initial_X"],
                "episode": index + 1,
                "success": int(result),
                "reward": rewards[index] if index < len(rewards) else None,
                "distance": (
                    distances[index] if index < len(distances) else None
                ),
                "pre_events": (
                    log["pre_events"][index]
                    if index < len(log.get("pre_events", [])) else None
                ),
                "post_spikes": (
                    log["post_spikes"][index]
                    if index < len(log.get("post_spikes", [])) else None
                ),
                "update_applied": (
                    log["update_applied"][index]
                    if index < len(log.get("update_applied", [])) else None
                ),
                "delta_x_abs_mean": (
                    log["delta_x_abs_mean"][index]
                    if index < len(log.get("delta_x_abs_mean", [])) else None
                ),
            })
    _write_csv(
        output_dir / "episodes_all.csv",
        episode_rows,
        [
            "algorithm", "seed", "initial_X", "episode",
            "success", "reward", "distance", "pre_events", "post_spikes",
            "update_applied", "delta_x_abs_mean",
        ],
    )


def _plot_training_success(logs: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {0.1: "tab:blue", 0.9: "tab:orange"}
    for ax, algorithm in zip(axes, ALGORITHMS):
        for initial_x in INITIAL_STATES:
            group = [
                log for log in logs
                if log["algorithm"] == algorithm
                and log["initial_X"] == initial_x
            ]
            curves = []
            for log in group:
                values = np.asarray(log.get("success", []), dtype=float)
                if values.size == 0:
                    continue
                window = min(100, values.size)
                curve = np.convolve(
                    values, np.ones(window) / window, mode="valid"
                )
                curves.append(curve)
            if not curves:
                continue
            matrix = np.vstack(curves)
            window = min(100, len(group[0].get("success", [])))
            x = np.arange(window, window + matrix.shape[1])
            mean = matrix.mean(axis=0)
            spread = matrix.std(axis=0, ddof=1) if len(curves) > 1 else 0
            label = f"X0={initial_x:g}, media de {len(curves)} semillas"
            ax.plot(x, mean, color=colors[initial_x], label=label)
            if isinstance(spread, np.ndarray):
                ax.fill_between(
                    x, mean - spread, mean + spread,
                    color=colors[initial_x], alpha=0.18,
                )
        ax.set_title(algorithm)
        ax.set_xlabel("Episodio")
        ax.set_xlim(100, EPISODES)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Éxito de entrenamiento (media móvil de 100)")
    seed_count = len({log["seed"] for log in logs})
    fig.suptitle(
        f"Auto 2D: REINFORCE frente a R-STDP, {seed_count} semillas"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "exito_entrenamiento.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, algorithm in zip(axes, ALGORITHMS):
        for initial_x in INITIAL_STATES:
            group = [
                log for log in logs
                if log["algorithm"] == algorithm
                and log["initial_X"] == initial_x
            ]
            series = []
            for log in group:
                evaluations = log.get("policy_evaluations", [])
                if evaluations:
                    series.append({
                        item["episode"]: item["success_rate"]
                        for item in evaluations
                    })
            if not series:
                continue
            episodes = sorted(set.intersection(*[
                set(row) for row in series
            ]))
            values = np.asarray([
                [row[episode] for row in series] for episode in episodes
            ], dtype=float)
            mean = values.mean(axis=1)
            spread = values.std(axis=1, ddof=1) if len(series) > 1 else 0
            label = f"X0={initial_x:g}, validación"
            ax.plot(episodes, mean, marker=".", color=colors[initial_x], label=label)
            if isinstance(spread, np.ndarray):
                ax.fill_between(
                    episodes, mean - spread, mean + spread,
                    color=colors[initial_x], alpha=0.18,
                )
        ax.set_title(algorithm)
        ax.set_xlabel("Episodio")
        ax.set_xlim(0, EPISODES)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Éxito greedy en mapa de entrenamiento")
    fig.suptitle("Validación greedy (20 orientaciones, pesos congelados)")
    fig.tight_layout()
    fig.savefig(output_dir / "exito_validacion.png", dpi=180)
    plt.close(fig)


def _plot_crossbars(logs: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    image = None
    action_labels = (
        "Acel.", "Coast", "Izq. suave", "Der. suave",
        "Izq. fuerte", "Der. fuerte", "Freno izq.", "Freno der.",
    )
    for row, algorithm in enumerate(ALGORITHMS):
        for column, initial_x in enumerate(INITIAL_STATES):
            matrices = [
                np.load(_crossbar_paths(log, output_dir)[1])
                for log in logs
                if log["algorithm"] == algorithm
                and log["initial_X"] == initial_x
            ]
            if not matrices:
                continue
            matrix = np.mean(np.stack(matrices), axis=0)
            image = axes[row, column].imshow(
                matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0
            )
            axes[row, column].set_title(
                f"{algorithm}, X0={initial_x:g} (media de {len(matrices)} semillas)"
            )
            axes[row, column].set_xlabel("Acción")
            axes[row, column].set_ylabel("Sensor")
            axes[row, column].set_xticks(range(len(action_labels)))
            axes[row, column].set_xticklabels(
                action_labels, rotation=35, ha="right", fontsize=8
            )
    if image is None:
        raise RuntimeError("No crossbar matrices available for plotting.")
    colorbar_axis = fig.add_axes([0.93, 0.2, 0.018, 0.65])
    fig.colorbar(image, cax=colorbar_axis, label="Estado normalizado X")
    fig.suptitle("Crossbar final X (24×8); media entre semillas")
    fig.subplots_adjust(top=0.91, right=0.9, wspace=0.24, hspace=0.32)
    fig.savefig(output_dir / "crossbar_final_promedio.png", dpi=180)
    plt.close(fig)

    change_groups = {}
    vmax = 0.0
    for algorithm in ALGORITHMS:
        for initial_x in INITIAL_STATES:
            matrices = []
            for log in logs:
                if log["algorithm"] != algorithm or log["initial_X"] != initial_x:
                    continue
                initial_path, final_path = _crossbar_paths(log, output_dir)
                change = np.abs(np.load(final_path) - np.load(initial_path))
                matrices.append(change)
            if matrices:
                mean_change = np.mean(np.stack(matrices), axis=0)
                change_groups[(algorithm, initial_x)] = mean_change
                vmax = max(vmax, float(mean_change.max()))
    if not change_groups:
        raise RuntimeError("No crossbar changes available for plotting.")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    image = None
    for row, algorithm in enumerate(ALGORITHMS):
        for column, initial_x in enumerate(INITIAL_STATES):
            matrix = change_groups.get((algorithm, initial_x))
            if matrix is None:
                continue
            image = axes[row, column].imshow(
                matrix, aspect="auto", cmap="magma", vmin=0.0,
                vmax=max(vmax, 1e-12),
            )
            axes[row, column].set_title(
                f"{algorithm}, X0={initial_x:g}; "
                f"media |ΔX|={matrix.mean():.4f}"
            )
            axes[row, column].set_xlabel("Acción")
            axes[row, column].set_ylabel("Sensor")
            axes[row, column].set_xticks(range(len(action_labels)))
            axes[row, column].set_xticklabels(
                action_labels, rotation=35, ha="right", fontsize=8
            )
    colorbar_axis = fig.add_axes([0.93, 0.2, 0.018, 0.65])
    fig.colorbar(
        image, cax=colorbar_axis,
        label="Cambio absoluto medio |Xfinal − Xinicial|",
    )
    fig.suptitle("Plasticidad del crossbar por sensor y acción")
    fig.subplots_adjust(top=0.91, right=0.9, wspace=0.24, hspace=0.32)
    fig.savefig(output_dir / "crossbar_cambio_abs_promedio.png", dpi=180)
    plt.close(fig)


def _check_complete(logs: list[dict], episodes: int) -> None:
    seed_count = len({log.get("seed") for log in logs})
    expected = len(ALGORITHMS) * len(INITIAL_STATES) * seed_count
    if len(logs) != expected:
        raise RuntimeError(f"Expected {expected} runs, found {len(logs)}.")
    for log in logs:
        if log.get("episodes_completed") != episodes:
            raise RuntimeError(
                f"Incomplete run: {log.get('algorithm')} "
                f"seed={log.get('seed')} X0={log.get('initial_X')}"
            )
        if log.get("crossbar_shape") != [24, 8]:
            raise RuntimeError("Pilot crossbar shape differs from 24×8.")
        if len(log.get("success", [])) != episodes:
            raise RuntimeError("Per-episode success data is incomplete.")
    for seed in {log["seed"] for log in logs}:
        for initial_x in INITIAL_STATES:
            paired = [
                log for log in logs
                if log["seed"] == seed and log["initial_X"] == initial_x
            ]
            if len(paired) != len(ALGORITHMS):
                raise RuntimeError(
                    f"Missing paired algorithm runs for seed={seed}, X0={initial_x}."
                )
            if len({log["training_map_seed"] for log in paired}) != 1:
                raise RuntimeError("Paired algorithms used different training maps.")
            if len({
                log["initial_crossbar_X_mean"] for log in paired
            }) != 1:
                raise RuntimeError(
                    "Paired algorithms did not receive identical seeded X0 matrices."
                )


def _run_batch(output_dir: Path, episodes: int, seeds: tuple[int, ...],
               verbose: bool, resume: bool = False) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=resume)
    logs = []
    for algorithm in ALGORITHMS:
        for initial_x in INITIAL_STATES:
            for seed in seeds:
                print(
                    f"\n[comienza] {algorithm} | X0={initial_x:g} "
                    f"| semilla={seed} | {episodes} episodios",
                    flush=True,
                )
                log = _run_one(
                    algorithm, seed, initial_x, episodes, output_dir, verbose
                )
                logs.append(log)
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for all run artifacts (defaults to the thesis evidence folder).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Paired non-negative seeds; each seed runs all four algorithm/X0 conditions.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 4 short conditions in a temporary directory, then clean up.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the pilot from completed run files in its output folder.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write source checksums for an existing pilot without rerunning it.",
    )
    parser.add_argument(
        "--rebuild-reports",
        action="store_true",
        help="Recreate aggregate tables and figures from complete run JSON files.",
    )
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("All seeds must be non-negative integers.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seed values must be unique.")
    output_dir = args.output_dir.resolve()
    if args.rebuild_reports:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Pilot output directory not found: {output_dir}")
        logs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output_dir / "runs").glob("*/result.json"))
        ]
        _check_complete(logs, EPISODES)
        for log in logs:
            label = (
                f"{log['algorithm'].lower().replace('-', '')}"
                f"_seed{log['seed']}_X0_{log['initial_X']:g}"
            )
            _write_episode_csv(
                output_dir / "runs" / label / "episodes.csv",
                log,
            )
        _write_aggregate_tables(logs, output_dir)
        _plot_training_success(logs, output_dir)
        _plot_crossbars(logs, output_dir)
        _write_source_manifest(output_dir)
        protocol_path = output_dir / "protocol.json"
        if protocol_path.is_file():
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["action_names"] = list(ACTION_NAMES)
            protocol["reports_rebuilt_with"] = (
                "per-episode spike/update metrics, labeled crossbar figures, "
                "and per-cell plasticity magnitudes"
            )
            protocol["crossbar_plots"] = [
                "crossbar_final_promedio.png",
                "crossbar_cambio_abs_promedio.png",
            ]
            _write_json(protocol_path, protocol)
        print(f"[reportes regenerados] {output_dir}")
        return
    if args.manifest_only:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Pilot output directory not found: {output_dir}")
        _write_source_manifest(output_dir)
        print(f"[manifest] {output_dir / 'source_manifest.json'}")
        return
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="auto2d-pilot-smoke-") as temp:
            output_dir = Path(temp) / "pilot"
            logs = _run_batch(output_dir, episodes=2, seeds=(987654,), verbose=False)
            _check_complete(logs, 2)
            _write_aggregate_tables(logs, output_dir)
            _plot_training_success(logs, output_dir)
            _plot_crossbars(logs, output_dir)
            required = (
                "runs_summary.csv",
                "conditions_summary.csv",
                "exito_entrenamiento.png",
                "crossbar_final_promedio.png",
                "crossbar_cambio_abs_promedio.png",
            )
            missing = [name for name in required if not (output_dir / name).is_file()]
            if missing:
                raise RuntimeError(f"Smoke test outputs missing: {missing}")
        print("[smoke] passed; temporary outputs cleaned.")
        return

    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )
    if (output_dir / "protocol.json").exists():
        raise FileExistsError(
            f"The pilot already has a completion record: {output_dir / 'protocol.json'}"
        )
    logs = _run_batch(
        output_dir, episodes=EPISODES, seeds=SEEDS, verbose=True,
        resume=args.resume,
    )
    _check_complete(logs, EPISODES)
    _write_aggregate_tables(logs, output_dir)
    _plot_training_success(logs, output_dir)
    _plot_crossbars(logs, output_dir)
    _write_source_manifest(output_dir)
    _write_json(output_dir / "protocol.json", {
        "task": "Auto 2D",
        "algorithms": list(ALGORITHMS),
        "action_names": list(ACTION_NAMES),
        "initial_X": list(INITIAL_STATES),
        "seeds": list(args.seeds),
        "episodes_per_run": EPISODES,
        "total_runs": len(logs),
        "total_training_episodes": EPISODES * len(logs),
        "validation_interval": EVAL_INTERVAL,
        "validation_episodes": run_car.POLICY_EVAL_EPISODES,
        "test_episodes": run_car_rstdp.POLICY_TEST_EPISODES,
        "validation_map": "same deterministic training map per seed",
        "test_map_seeds": [
            run_car.POLICY_TEST_SEED_OFFSET + index
            for index in range(run_car_rstdp.POLICY_TEST_EPISODES)
        ],
        "test_checkpoint": "final weights only; no test-based checkpoint selection",
        "crossbar_plots": [
            "crossbar_final_promedio.png",
            "crossbar_cambio_abs_promedio.png",
        ],
        "initial_X_definition": (
            "nominal mean of the seeded initial matrix; per-cell Gaussian "
            "perturbation sigma=0.01, clipped to [0,1]"
        ),
        "shared_dynamics": {
            "N": 24,
            "M": 8,
            "encoder": "auto2d",
            "dt": 1e-3,
            "steps_per_action": 1,
            "k_wta": 3,
            "wta_temperature": 1.0,
            "V_read": 0.05,
            "memristor_nv_profile": run_car_rstdp.DEFAULT_NV_PROFILE,
            "lif_profile": run_car_rstdp.DEFAULT_LIF_PROFILE,
            "d2d_enabled": False,
            "c2c_enabled": False,
            "eta": run_car.ETA,
            "max_abs_weight_change_per_episode": 0.01,
        },
        "algorithm_specific_exploration": {
            "REINFORCE": {
                "temperature_initial": run_car.TEMP_INIT_HD,
                "temperature_adaptation_during_training": False,
            },
            "R-STDP": {
                "temperature_initial": 1.0,
                "epsilon": run_car_rstdp.EXPLORATION_EPSILON,
            },
        },
    })
    print(f"\n[completo] artefactos del piloto: {output_dir}")


if __name__ == "__main__":
    main()
