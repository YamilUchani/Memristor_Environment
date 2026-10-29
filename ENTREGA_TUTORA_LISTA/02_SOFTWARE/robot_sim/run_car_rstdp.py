"""Entrena Auto 2D con plasticidad R-STDP pre/post modulada por recompensa.

Uso:
    python run_car_rstdp.py
    python run_car_rstdp.py --headless --episodes 3000 --seed 0 --initial-X 0.1

Cada corrida comienza desde cero, salvo que se indique --weights.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import monotonic

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from car_env import (
    ACTION_NAMES,
    CarEnv,
    GOAL_POS,
    GOAL_R,
    MAP_SIZE,
    N_ACTIONS,
    REWARD_STEERING_SMOOTHNESS,
    REWARD_UNPRODUCTIVE_TURN,
    TURN_PROGRESS_REQUIRED,
)
from neurobot.rstdp_cartpole_agent import NeuromorphicAgent


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figuras"
NO_WEIGHTS = "--- (Empezar desde cero) ---"
N_INPUTS = 24
DEFAULT_EPISODES = 3000
DEFAULT_ETA = 0.01
DEFAULT_INITIAL_X = 0.1
EXPLORATION_EPSILON = 0.15
DEFAULT_NV_PROFILE = "strukov_ideal.json"
DEFAULT_LIF_PROFILE = "lif_config.json"
POLICY_EVAL_INTERVAL = 100
POLICY_EVAL_EPISODES = 20
POLICY_TEST_EPISODES = 50
POLICY_TEST_SEED_OFFSET = 1_100_000_000
TRAIN_MAP_SEED_OFFSET = 2_000_000_000
TRAIN_MAP_SEED_STRIDE = 2**32


def _training_map_seed(seed: int) -> int:
    return TRAIN_MAP_SEED_OFFSET + int(seed) * TRAIN_MAP_SEED_STRIDE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena Auto 2D con elegibilidad pre/post R-STDP."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument(
        "--initial-X",
        type=float,
        choices=(0.1, 0.9),
        default=DEFAULT_INITIAL_X,
    )
    parser.add_argument(
        "--weights",
        help="Archivo .npy existente en robot_sim/outputs para continuar.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Entrena sin ventana gráfica."
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed debe ser no negativo")
    if args.episodes < 1:
        parser.error("--episodes debe ser positivo")
    if not np.isfinite(args.eta) or args.eta <= 0.0:
        parser.error("--eta debe ser finito y positivo")
    if args.weights:
        path = Path(args.weights)
        if (
            path.name != args.weights
            or path.suffix.lower() != ".npy"
            or not (OUTPUT_DIR / path).is_file()
        ):
            parser.error("--weights debe ser un .npy existente en outputs")
    return args


def make_agent(
    seed: int,
    eta: float = DEFAULT_ETA,
    initial_x: float = DEFAULT_INITIAL_X,
    weights_file: str | None = None,
    nv_profile: str = DEFAULT_NV_PROFILE,
    lif_profile: str = DEFAULT_LIF_PROFILE,
) -> NeuromorphicAgent:
    agent = NeuromorphicAgent(
        obs_dim=11,
        n_actions=N_ACTIONS,
        N=N_INPUTS,
        M=N_ACTIONS,
        V_read=0.05,
        eta=eta,
        dt=1e-3,
        steps_per_action=1,
        seed=seed,
        crossbar_seed=seed,
        encoder_seed=seed,
        action_seed=seed,
        encoder_mode="auto2d",
        k_wta=3,
        wta_temperature=1.0,
        temp_init=1.0,
        temp_min=1.0,
        temp_decay=1.0,
        epsilon=EXPLORATION_EPSILON,
        initial_X=initial_x,
        json_memristor_nv=nv_profile,
        json_memristor_v="memristor_volatile.json",
        json_lif=lif_profile,
        d2d_enabled=False,
        c2c_enabled=False,
        max_abs_delta_x=0.01,
    )
    if weights_file and weights_file != NO_WEIGHTS:
        weight_path = OUTPUT_DIR / weights_file
        if not weight_path.is_file():
            raise FileNotFoundError(f"No se encontraron los pesos: {weight_path}")
        metadata_path = Path(f"{weight_path}.meta.json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            encoder = metadata.get(
                "encoder_mode", metadata.get("encoder")
            )
            if encoder not in (None, "auto2d"):
                raise ValueError(
                    f"Los pesos usan el codificador {encoder!r}; "
                    "Auto 2D R-STDP requiere 'auto2d'."
                )
            action_names = metadata.get("action_names")
            if action_names is not None and tuple(action_names) != ACTION_NAMES:
                raise ValueError("Los pesos tienen otro mapeo de acciones.")
        agent.brain.load_weights(str(weight_path))
    return agent


def _path_metrics(env: CarEnv) -> dict:
    path = np.asarray(env.trail, dtype=float)
    if path.ndim == 2 and path.shape[0] > 1:
        direct = float(np.linalg.norm(path[-1] - path[0]))
    else:
        direct = 0.0
    return {
        "path_length": float(env.path_length),
        "path_efficiency": float(
            np.clip(direct / env.path_length, 0.0, 1.0)
            if env.path_length > 0.0 else 0.0
        ),
        "steering_reversals": int(env.steering_reversals),
        "steering_smoothness_penalty": float(
            env.steering_smoothness_penalty
        ),
        "turning_angle": float(env.turning_angle),
        "unproductive_turns": int(env.unproductive_turns),
        "unproductive_turn_penalty": float(env.unproductive_turn_penalty),
    }


def _greedy_evaluate(
    agent: NeuromorphicAgent,
    seeds: list[int],
    same_map: bool,
) -> dict:
    """Evaluate frozen weights and restore exploration state after rollouts."""
    weights_before = agent.brain.X.copy()
    rng_state = agent.decoder.rng.bit_generator.state
    sample_mode = agent.decoder.sample
    outcomes = []
    best_path = []
    best_path_steps = float("inf")

    try:
        agent.decoder.sample = False
        env = CarEnv(seed=seeds[0])
        for index, seed in enumerate(seeds):
            obs = env.reset(seed=seed) if not same_map or index == 0 else env.reset()
            agent.reset_episode_state()
            done = False
            while not done:
                action = agent.act_deterministic(obs)
                obs, _, done, info = env.step(action)
            metrics = _path_metrics(env)
            outcomes.append({
                "success": info["reason"] == "goal",
                "reason": info["reason"],
                "collision_type": info["collision_type"],
                "steps": int(info["steps"]),
                "distance": float(info["dist"]),
                **metrics,
            })
            if info["reason"] == "goal" and info["steps"] < best_path_steps:
                best_path_steps = info["steps"]
                best_path = [list(point) for point in env.trail]
        agent.reset_episode_state()
    finally:
        agent.decoder.sample = sample_mode
        agent.decoder.rng.bit_generator.state = rng_state

    if not np.array_equal(agent.brain.X, weights_before):
        raise RuntimeError("Greedy R-STDP evaluation changed crossbar weights.")
    successes = [item["success"] for item in outcomes]
    return {
        "episodes": len(outcomes),
        "same_map": bool(same_map),
        "map_seeds": (
            [int(seeds[0])] * len(outcomes)
            if same_map else [int(seed) for seed in seeds]
        ),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "successes": int(sum(successes)),
        "collisions": sum(item["reason"] == "collision" for item in outcomes),
        "timeouts": sum(item["reason"] == "timeout" for item in outcomes),
        "mean_path_length": float(np.mean([x["path_length"] for x in outcomes])),
        "mean_path_efficiency": float(
            np.mean([x["path_efficiency"] for x in outcomes])
        ),
        "mean_distance": float(
            np.mean([x["distance"] for x in outcomes])
        ),
        "mean_unproductive_turns": float(
            np.mean([x["unproductive_turns"] for x in outcomes])
        ),
        "mean_success_steps": (
            float(np.mean([x["steps"] for x in outcomes if x["success"]]))
            if any(successes) else None
        ),
        "outcomes": outcomes,
        "best_success_path": best_path,
    }


def run_training(
    agent: NeuromorphicAgent,
    env: CarEnv,
    episodes: int = DEFAULT_EPISODES,
    seed: int = 0,
    verbose: bool = True,
    on_episode=None,
    policy_eval_interval: int = POLICY_EVAL_INTERVAL,
) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if policy_eval_interval < 1:
        raise ValueError("policy_eval_interval must be positive")

    training_seed = _training_map_seed(seed)
    log = {
        "episodes_requested": int(episodes),
        "episode": [],
        "reward": [],
        "success": [],
        "termination_reason": [],
        "collision_type": [],
        "steps": [],
        "distance": [],
        "X_mean": [],
        "baseline": [],
        "eligibility_abs_mean": [],
        "delta_x_abs_mean": [],
        "pre_events": [],
        "post_spikes": [],
        "selected_post_events": [],
        "path_length": [],
        "path_efficiency": [],
        "steering_reversals": [],
        "steering_smoothness_penalty": [],
        "turning_angle": [],
        "unproductive_turns": [],
        "unproductive_turn_penalty": [],
        "policy_evaluations": [],
    }
    started = monotonic()
    first_path = []
    last_path = []
    held_out_eval = None

    for episode in range(episodes):
        obs = env.reset(seed=training_seed) if episode == 0 else env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            reward = float(reward)
            episode_reward += reward
            agent.observe_reward(reward)

        if episode == 0:
            first_path = [list(point) for point in env.trail]
        last_path = [list(point) for point in env.trail]
        success = info["reason"] == "goal"
        learning = agent.finish_episode(episode_reward)
        agent.end_episode()
        metrics = _path_metrics(env)
        row = {
            "episode": episode + 1,
            "reward": float(episode_reward),
            "success": bool(success),
            "termination_reason": info["reason"],
            "collision_type": info["collision_type"],
            "steps": int(info["steps"]),
            "distance": float(info["dist"]),
            "X_mean": float(agent.brain.X.mean()),
            "baseline": float(agent.baseline),
            "eligibility_abs_mean": float(
                learning["reward_modulated_eligibility_abs_mean"]
            ),
            "delta_x_abs_mean": float(learning["actual_delta_x_abs_mean"]),
            "delta_x_abs_max": float(learning["actual_delta_x_abs_max"]),
            "delta_g_abs_mean": float(learning["actual_delta_g_abs_mean"]),
            "update_applied": bool(learning["update_applied"]),
            "pre_events": int(learning["pre_events"]),
            "post_spikes": int(learning["post_spikes"]),
            "selected_post_events": int(learning["selected_post_events"]),
            **metrics,
        }
        for key, value in row.items():
            log.setdefault(key, []).append(value)

        completed = episode + 1
        if completed % policy_eval_interval == 0 or completed == episodes:
            evaluation = _greedy_evaluate(
                agent, [training_seed] * POLICY_EVAL_EPISODES, same_map=True
            )
            evaluation["episode"] = completed
            log["policy_evaluations"].append(evaluation)
            if verbose:
                print(
                    f"[validación greedy ep {completed}] "
                    f"éxito={evaluation['success_rate']:.2f} "
                    f"distancia={evaluation['mean_distance']:.2f} "
                    f"choques={evaluation['collisions']}"
                )
        if verbose and (
            episode % max(1, episodes // 10) == 0 or episode == episodes - 1
        ):
            recent = float(np.mean(log["success"][-100:]))
            print(
                f"[ep {completed:4d}] exito(100)={recent:.3f} "
                f"R={episode_reward:+7.2f} pasos={info['steps']:3d} "
                f"X_mean={agent.brain.X.mean():.4f} "
                f"dX={learning['actual_delta_x_abs_mean']:.2e} "
                f"dG={learning['actual_delta_g_abs_mean']:.2e}"
            )
        if on_episode is not None and not on_episode(
            episode, success, env, agent, log
        ):
            break

    completed = len(log["episode"])
    if completed:
        held_out_eval = _greedy_evaluate(
            agent,
            [POLICY_TEST_SEED_OFFSET + i for i in range(POLICY_TEST_EPISODES)],
            same_map=False,
        )
    log["episodes_completed"] = completed
    log["training_map_seed"] = training_seed
    log["training_map_mode"] = "fixed_map_per_run"
    log["initial_path"] = first_path
    log["final_training_path"] = last_path
    log["held_out_test_evaluation"] = held_out_eval
    log["elapsed_seconds"] = float(monotonic() - started)
    log["final_success_rate"] = (
        float(np.mean(log["success"][-100:])) if completed else 0.0
    )
    log["final_greedy_success_rate"] = (
        log["policy_evaluations"][-1]["success_rate"]
        if log["policy_evaluations"] else None
    )
    return log


def plot_learning(log: dict, path: Path) -> None:
    episodes = np.asarray(log["episode"], dtype=int)
    success = np.asarray(log["success"], dtype=float)
    distance = np.asarray(log["distance"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    if success.size:
        window = min(50, success.size)
        smoothed = np.convolve(
            success, np.ones(window) / window, mode="valid"
        )
        axes[0].plot(
            episodes[window - 1 :], smoothed, color="tab:blue",
            label=f"éxito entrenamiento (media {window})"
        )
    evaluations = log["policy_evaluations"]
    if evaluations:
        axes[0].plot(
            [item["episode"] for item in evaluations],
            [item["success_rate"] for item in evaluations],
            "o--",
            color="tab:green",
            label="éxito greedy (validación)",
        )
    axes[0].set_ylabel("Éxito")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    if distance.size:
        window = min(50, distance.size)
        smoothed_distance = np.convolve(
            distance, np.ones(window) / window, mode="valid"
        )
        axes[1].plot(
            episodes[window - 1 :],
            smoothed_distance,
            color="tab:purple",
            label=f"distancia a meta (media {window})",
        )
    axes[1].set_ylabel("Distancia final a meta")
    axes[1].set_xlabel("Episodio")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Auto 2D — R-STDP")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _launch_config(args: argparse.Namespace) -> dict:
    import tkinter as tk
    from tkinter import messagebox, ttk

    weights = [NO_WEIGHTS] + sorted(path.name for path in OUTPUT_DIR.glob("*.npy"))
    root = tk.Tk()
    root.title("Auto 2D — configuración R-STDP")
    root.geometry("480x330")
    root.eval("tk::PlaceWindow . center")
    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    def entry(row: int, label: str, value: str):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
        widget = ttk.Entry(frame)
        widget.insert(0, value)
        widget.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=6)
        return widget

    seed = entry(0, "Semilla:", str(args.seed))
    episodes = entry(1, "Episodios:", str(args.episodes))
    eta = entry(2, "Tasa eta:", str(args.eta))
    ttk.Label(frame, text="X₀:").grid(row=3, column=0, sticky="w", pady=6)
    initial_x = ttk.Combobox(frame, values=("0.1", "0.9"), state="readonly")
    initial_x.set(str(args.initial_X))
    initial_x.grid(row=3, column=1, sticky="ew", padx=(12, 0), pady=6)
    ttk.Label(frame, text="Pesos previos (.npy):").grid(
        row=4, column=0, sticky="w", pady=6
    )
    weights_box = ttk.Combobox(frame, values=weights, state="readonly")
    weights_box.set(args.weights or NO_WEIGHTS)
    weights_box.grid(row=4, column=1, sticky="ew", padx=(12, 0), pady=6)
    config = {}

    def start():
        try:
            seed_value = int(seed.get())
            episode_value = int(episodes.get())
            eta_value = float(eta.get())
            if (
                seed_value < 0
                or episode_value < 1
                or not np.isfinite(eta_value)
                or eta_value <= 0.0
            ):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Configuración inválida",
                "Semilla/episodios deben ser enteros válidos y eta positiva.",
                parent=root,
            )
            return
        config.update(
            seed=seed_value,
            episodes=episode_value,
            eta=eta_value,
            initial_X=float(initial_x.get()),
            weights=None if weights_box.get() == NO_WEIGHTS else weights_box.get(),
        )
        root.destroy()

    ttk.Button(
        root, text="Iniciar R-STDP", command=start
    ).pack(pady=(0, 14))
    root.mainloop()
    if not config:
        raise SystemExit("Configuración cancelada.")
    return config


def run_live(
    agent: NeuromorphicAgent,
    env: CarEnv,
    episodes: int,
    seed: int,
) -> dict:
    state = {"paused": False, "stop": False, "speed": 1}
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Auto 2D — entrenamiento R-STDP")
    ax_map, ax_weights = axes[0]
    ax_success, ax_distance = axes[1]
    im_weights = ax_weights.imshow(
        agent.brain.X, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0
    )
    ax_weights.set_title("Crossbar X (24×8)")
    ax_weights.set_xlabel("Acción")
    ax_weights.set_ylabel("Sensor")
    fig.colorbar(im_weights, ax=ax_weights)
    line_success, = ax_success.plot([], [], color="tab:blue")
    line_greedy, = ax_success.plot(
        [], [], "o--", color="tab:green", label="validación greedy"
    )
    ax_success.set_xlim(0, max(10, episodes))
    ax_success.set_ylim(-0.05, 1.05)
    ax_success.set_ylabel("Éxito (media móvil 50)")
    ax_success.legend()
    ax_success.grid(alpha=0.25)
    line_distance, = ax_distance.plot([], [], color="tab:purple")
    ax_distance.set_xlim(0, max(10, episodes))
    ax_distance.set_xlabel("Episodio")
    ax_distance.set_ylabel("Distancia final a meta (media 50)")
    ax_distance.grid(alpha=0.25)
    status = fig.text(0.02, 0.98, "", ha="left", va="top", family="monospace")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key in ("q", "escape"):
            state["stop"] = True
            plt.close(fig)
        elif event.key in ("+", "=", "plus"):
            state["speed"] = min(100, state["speed"] * 2)
        elif event.key in ("-", "_", "minus"):
            state["speed"] = max(1, state["speed"] // 2)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda _event: state.update(stop=True))

    def on_episode(episode, success, env_, agent_, log):
        while state["paused"] and not state["stop"]:
            plt.pause(0.05)
        if state["stop"] or not plt.fignum_exists(fig.number):
            return False
        if episode % state["speed"] != 0 and episode != episodes - 1:
            return True
        ax_map.clear()
        env_.draw(ax_map, title="")
        ax_map.add_patch(
            Circle(
                GOAL_POS,
                GOAL_R,
                fill=False,
                linewidth=2,
                edgecolor="darkgreen",
            )
        )
        ax_map.set_title(
            f"{'META' if success else log['termination_reason'][-1]} | "
            f"pasos {env_.steps}",
            color="tab:green" if success else "tab:red",
        )
        im_weights.set_data(agent_.brain.X)
        successes = np.asarray(log["success"], dtype=float)
        if successes.size >= 50:
            smooth = np.convolve(successes, np.ones(50) / 50, mode="valid")
            line_success.set_data(np.arange(50, 50 + len(smooth)), smooth)
        evaluations = log["policy_evaluations"]
        if evaluations:
            line_greedy.set_data(
                [item["episode"] for item in evaluations],
                [item["success_rate"] for item in evaluations],
            )
        distances = np.asarray(log["distance"], dtype=float)
        if distances.size >= 50:
            mean_distance = np.convolve(
                distances, np.ones(50) / 50, mode="valid"
            )
            line_distance.set_data(
                np.arange(50, 50 + len(mean_distance)), mean_distance
            )
            ax_distance.relim()
            ax_distance.autoscale_view(scalex=False, scaley=True)
        status.set_text(
            f"Ep {episode + 1}/{episodes} | éxito(100) "
            f"{np.mean(successes[-100:]):.3f} | X_mean "
            f"{agent_.brain.X.mean():.4f} | dX "
            f"{log['delta_x_abs_mean'][-1]:.2e} | dG "
            f"{log['delta_g_abs_mean'][-1]:.2e} | dibujar x{state['speed']}"
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
        return True

    print("[controles] espacio pausa | q/Esc salir | +/- velocidad de dibujo")
    log = run_training(
        agent, env, episodes=episodes, seed=seed, on_episode=on_episode
    )
    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    return log


def main() -> None:
    args = parse_args()
    config = (
        {
            "seed": args.seed,
            "episodes": args.episodes,
            "eta": args.eta,
            "initial_X": args.initial_X,
            "weights": args.weights,
        }
        if args.headless else _launch_config(args)
    )
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    FIGURE_DIR.mkdir(exist_ok=True, parents=True)
    env = CarEnv(seed=config["seed"])
    agent = make_agent(
        seed=config["seed"],
        eta=config["eta"],
        initial_x=config["initial_X"],
        weights_file=config["weights"],
    )
    weights_source = config["weights"]
    initial_state = {
        "X_mean": float(agent.brain.X.mean()),
        "weights_source": weights_source,
    }
    print(
        f"[Auto 2D R-STDP] seed={config['seed']} "
        f"episodes={config['episodes']} eta={config['eta']} "
        f"X0={config['initial_X']} "
        f"pesos={'nuevos' if not weights_source else weights_source}"
    )
    if config["weights"]:
        stem = Path(config["weights"]).stem
    else:
        stem = (
            f"run_car_rstdp_seed{config['seed']}"
            f"_A8_X0_{config['initial_X']:g}"
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_stem = f"{stem}_{stamp}"
    start = monotonic()
    log = (
        run_training(
            agent,
            env,
            episodes=config["episodes"],
            seed=config["seed"],
            verbose=True,
        )
        if args.headless
        else run_live(
            agent,
            env,
            episodes=config["episodes"],
            seed=config["seed"],
        )
    )
    log.update({
        "algorithm": "R-STDP",
        "encoder": "auto2d",
        "action_names": list(ACTION_NAMES),
        "seed": int(config["seed"]),
        "eta": float(config["eta"]),
        "initial_X": float(config["initial_X"]),
        "weights_source": weights_source,
        "initial_state": initial_state,
        "exploration_epsilon": EXPLORATION_EPSILON,
        "crossbar_shape": [N_INPUTS, N_ACTIONS],
        "environment_config": {
            "observation_dim": 11,
            "crossbar_rows": N_INPUTS,
            "actions": list(ACTION_NAMES),
            "training_map_seed": _training_map_seed(config["seed"]),
            "policy_evaluation_interval": POLICY_EVAL_INTERVAL,
            "policy_evaluation_episodes": POLICY_EVAL_EPISODES,
            "held_out_test_episodes": POLICY_TEST_EPISODES,
            "exploration_epsilon": EXPLORATION_EPSILON,
            "steering_smoothness_reward_weight": REWARD_STEERING_SMOOTHNESS,
            "unproductive_turn_progress_required": TURN_PROGRESS_REQUIRED,
            "unproductive_turn_reward_weight": REWARD_UNPRODUCTIVE_TURN,
        },
        "elapsed_seconds_total": float(monotonic() - start),
    })
    weights_path = OUTPUT_DIR / f"{run_stem}.npy"
    json_path = OUTPUT_DIR / f"{run_stem}.json"
    figure_path = FIGURE_DIR / f"{run_stem}.png"
    agent.brain.save_weights(str(weights_path))
    metadata_path = Path(f"{weights_path}.meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "task": "auto2d",
        "algorithm": "R-STDP",
        "encoder": "auto2d",
        "encoder_mode": "auto2d",
        "training_seed": int(config["seed"]),
        "action_names": list(ACTION_NAMES),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    json_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    plot_learning(log, figure_path)
    print(
        f"[resultado] train(100)={log['final_success_rate']:.3f} "
        f"greedy final={log['final_greedy_success_rate'] or 0.0:.3f} "
        f"held-out="
        f"{log['held_out_test_evaluation']['success_rate']:.3f}"
    )
    print(f"[ok] pesos: {weights_path}")
    print(f"[ok] log: {json_path}")
    print(f"[ok] figura: {figure_path}")


if __name__ == "__main__":
    main()
