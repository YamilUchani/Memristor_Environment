"""Entrena con R-STDP en el gridworld 2D con obstaculos.

Uso desde la raiz del repositorio:
    python robot_sim/run_obstacle_rstdp.py
    python robot_sim/run_obstacle_rstdp.py --headless --episodes 1500 --seed 0

Cada corrida comienza con pesos nuevos salvo que se indique --weights.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import monotonic

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from neurobot.rstdp_cartpole_agent import NeuromorphicAgent
from obstacle_env import ObstacleGridworld


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figuras"
CONFIG_DIR = ROOT.parent / "neuromorphic_lab" / "configs"
NO_WEIGHTS = "--- (Empezar desde cero) ---"
N_INPUTS = 16
N_ACTIONS = 4
DEFAULT_EPISODES = 1500
DEFAULT_ETA = 0.05
DEFAULT_INITIAL_X = 0.9
EXPLORATION_EPSILON = 0.15
DEFAULT_NV_PROFILE = "strukov_ideal.json"
DEFAULT_LIF_PROFILE = "lif_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena la navegacion del laberinto con R-STDP."
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
        help="Nombre de un archivo .npy existente en robot_sim/outputs.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Entrena sin ventanas gráficas."
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed debe ser no negativo")
    if args.episodes < 1:
        parser.error("--episodes debe ser positivo")
    if not np.isfinite(args.eta) or args.eta <= 0.0:
        parser.error("--eta debe ser finito y positivo")
    if args.weights:
        weight_name = Path(args.weights)
        if (
            weight_name.name != args.weights
            or weight_name.suffix.lower() != ".npy"
            or not (OUTPUT_DIR / weight_name).is_file()
        ):
            parser.error("--weights debe ser un .npy existente en robot_sim/outputs")
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
        encoder_mode="random",
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
        max_abs_delta_x=0.02,
    )
    if weights_file and weights_file != NO_WEIGHTS:
        weight_path = OUTPUT_DIR / weights_file
        if not weight_path.is_file():
            raise FileNotFoundError(f"No se encontraron los pesos: {weight_path}")
        agent.brain.load_weights(str(weight_path))
    return agent


def _greedy_rollout(agent: NeuromorphicAgent, env: ObstacleGridworld):
    agent.reset_episode_state()
    obs = env.reset()
    done = False
    while not done:
        action = agent.act_deterministic(obs)
        obs, _, done, _ = env.step(action)
    path = list(env.path)
    success = bool(env.agent == env.goal)
    steps = int(env.steps)
    agent.reset_episode_state()
    return path, success, steps


def run_training(
    agent: NeuromorphicAgent,
    env: ObstacleGridworld,
    episodes: int = DEFAULT_EPISODES,
    verbose: bool = True,
    on_step=None,
) -> tuple[dict, list, list, bool]:
    if episodes < 1:
        raise ValueError("episodes must be positive")

    log = {
        "episodes_requested": int(episodes),
        "episode": [],
        "reward": [],
        "success": [],
        "steps": [],
        "X_mean": [],
        "baseline": [],
        "temperature": [],
        "eligibility_abs_mean": [],
        "delta_x_abs_mean": [],
    }
    early_path: list = []
    started = monotonic()

    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        rewards = []
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            reward = float(reward)
            total_reward += reward
            rewards.append(reward)
            agent.observe_reward(reward)

        if episode == 0:
            early_path = list(env.path)
        success = bool(env.agent == env.goal)
        metrics = agent.finish_episode(total_reward)
        agent.end_episode()

        log["episode"].append(episode)
        log["reward"].append(float(total_reward))
        log["success"].append(float(success))
        log["steps"].append(int(env.steps))
        log["X_mean"].append(float(agent.brain.X.mean()))
        log["baseline"].append(float(agent.baseline))
        log["temperature"].append(float(agent.decoder.temperature))
        log["eligibility_abs_mean"].append(
            float(metrics["reward_modulated_eligibility_abs_mean"])
        )
        log["delta_x_abs_mean"].append(float(metrics["actual_delta_x_abs_mean"]))

        if verbose and (
            episode % max(1, episodes // 10) == 0 or episode == episodes - 1
        ):
            recent_success = float(np.mean(log["success"][-100:]))
            print(
                f"[ep {episode + 1:4d}] exito(100)={recent_success:.3f} "
                f"R={total_reward:+7.2f} pasos={env.steps:3d} "
                f"X_mean={agent.brain.X.mean():.4f}"
            )
        if on_step is not None and not on_step(
            episode, success, env, agent, log
        ):
            break

    learned_path, final_success, final_steps = _greedy_rollout(agent, env)
    log["episodes_completed"] = len(log["episode"])
    log["elapsed_seconds"] = float(monotonic() - started)
    log["final_success_rate"] = float(np.mean(log["success"][-100:]))
    log["greedy_evaluation_success"] = final_success
    log["greedy_evaluation_steps"] = final_steps
    if verbose:
        print(
            f"[evaluacion greedy] "
            f"{'EXITO' if final_success else 'FRACASO'} en {final_steps} pasos"
        )
    return log, early_path, learned_path, final_success


def plot_learning(log: dict, path: Path) -> None:
    episodes = np.asarray(log["episode"], dtype=int) + 1
    success = np.asarray(log["success"], dtype=float)
    reward = np.asarray(log["reward"], dtype=float)
    x_mean = np.asarray(log["X_mean"], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    window = min(50, success.size)
    if window:
        smoothed = np.convolve(
            success, np.ones(window, dtype=float) / window, mode="valid"
        )
        axes[0].plot(
            episodes[window - 1 :], smoothed, color="tab:blue", linewidth=1.8
        )
    axes[0].set_ylabel(f"Exito (media movil {window or 50})")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(alpha=0.25)
    axes[1].plot(episodes, reward, color="tab:purple", linewidth=1.0)
    axes[1].set_ylabel("Recompensa")
    axes[1].grid(alpha=0.25)
    axes[2].plot(episodes, x_mean, color="tab:green", linewidth=1.3)
    axes[2].set_ylabel("Media X")
    axes[2].set_xlabel("Episodio")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.25)
    fig.suptitle("Aprendizaje R-STDP — laberinto")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def launch_config_gui(args: argparse.Namespace) -> dict:
    import tkinter as tk
    from tkinter import messagebox, ttk

    weights = [NO_WEIGHTS] + sorted(
        path.name for path in OUTPUT_DIR.glob("*.npy")
    )
    root = tk.Tk()
    root.title("Robot Sim — Laberinto R-STDP")
    root.geometry("480x370")
    root.minsize(440, 340)
    root.eval("tk::PlaceWindow . center")

    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    def add_entry(row: int, label: str, value: str):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
        entry = ttk.Entry(frame)
        entry.insert(0, value)
        entry.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=6)
        return entry

    seed = add_entry(0, "Semilla:", str(args.seed))
    episodes = add_entry(1, "Episodios:", str(args.episodes))
    eta = add_entry(2, "Tasa eta:", str(args.eta))

    ttk.Label(frame, text="X₀ (si empiezas desde cero):").grid(
        row=3, column=0, sticky="w", pady=6
    )
    initial_x = ttk.Combobox(frame, values=("0.1", "0.9"), state="readonly")
    initial_x.set(str(args.initial_X))
    initial_x.grid(row=3, column=1, sticky="ew", padx=(12, 0), pady=6)

    ttk.Label(frame, text="Pesos previos (.npy):").grid(
        row=4, column=0, sticky="w", pady=6
    )
    previous_weights = ttk.Combobox(
        frame, values=weights, state="readonly"
    )
    previous_weights.set(args.weights or NO_WEIGHTS)
    previous_weights.grid(row=4, column=1, sticky="ew", padx=(12, 0), pady=6)

    result: dict = {}

    def start():
        try:
            selected_seed = int(seed.get())
            selected_episodes = int(episodes.get())
            selected_eta = float(eta.get())
            if (
                selected_seed < 0
                or selected_episodes < 1
                or not np.isfinite(selected_eta)
                or selected_eta <= 0
            ):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Configuracion invalida",
                "Semilla y eta deben ser validas; episodios debe ser positivo.",
                parent=root,
            )
            return
        result.update(
            seed=selected_seed,
            episodes=selected_episodes,
            eta=selected_eta,
            initial_X=float(initial_x.get()),
            weights=(
                None
                if previous_weights.get() == NO_WEIGHTS
                else previous_weights.get()
            ),
        )
        root.destroy()

    ttk.Button(
        root, text="Iniciar entrenamiento R-STDP", command=start
    ).pack(pady=(0, 14))
    root.mainloop()
    if not result:
        raise SystemExit("Configuracion cancelada.")
    return result


def run_live(
    agent: NeuromorphicAgent,
    env: ObstacleGridworld,
    episodes: int,
) -> tuple[dict, list, list, bool]:
    state = {"paused": False, "stop": False, "speed": 1}
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Laberinto — entrenamiento R-STDP")
    ax_grid, ax_x = axes[0]
    ax_success, ax_reward = axes[1]

    env.draw(ax_grid, path=[], title="Entrenamiento")
    im_x = ax_x.imshow(
        agent.brain.X, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0
    )
    ax_x.set_title("Matriz X del crossbar")
    ax_x.set_xlabel("Accion")
    ax_x.set_ylabel("Sensor")
    fig.colorbar(im_x, ax=ax_x)

    line_success, = ax_success.plot([], [], color="tab:blue", linewidth=1.6)
    ax_success.set_xlim(0, max(10, episodes))
    ax_success.set_ylim(-0.05, 1.05)
    ax_success.set_xlabel("Episodio")
    ax_success.set_ylabel("Exito (media movil 50)")
    ax_success.grid(alpha=0.25)
    line_reward, = ax_reward.plot([], [], color="tab:purple", linewidth=1.2)
    ax_reward.set_xlim(0, max(10, episodes))
    ax_reward.set_xlabel("Episodio")
    ax_reward.set_ylabel("Recompensa")
    ax_reward.grid(alpha=0.25)
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

    def on_step(episode, success, env_, agent_, log):
        while state["paused"] and not state["stop"]:
            plt.pause(0.05)
        if state["stop"] or not plt.fignum_exists(fig.number):
            return False
        if episode % state["speed"] != 0 and episode != episodes - 1:
            return True

        ax_grid.clear()
        env_.draw(ax_grid, path=list(env_.path), title="")
        color = "tab:green" if success else "tab:red"
        ax_grid.add_patch(
            Rectangle(
                (-0.5, -0.5),
                env_.size,
                env_.size,
                fill=False,
                linewidth=4,
                edgecolor=color,
            )
        )
        ax_grid.set_title(
            f"{'EXITO' if success else 'FRACASO'} | pasos {env_.steps}",
            color=color,
            fontweight="bold",
        )
        im_x.set_data(agent_.brain.X)
        successes = np.asarray(log["success"], dtype=float)
        if successes.size >= 50:
            smooth = np.convolve(successes, np.ones(50) / 50, mode="valid")
            line_success.set_data(np.arange(50, 50 + len(smooth)), smooth)
        line_reward.set_data(np.arange(1, len(log["reward"]) + 1), log["reward"])
        status.set_text(
            f"Ep {episode + 1}/{episodes} | exito(100) "
            f"{np.mean(successes[-100:]):.3f} | X_mean "
            f"{agent_.brain.X.mean():.4f} | velocidad x{state['speed']}"
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
        return True

    print("[controles] espacio pausa | q/Esc salir | +/- velocidad de dibujo")
    result = run_training(
        agent, env, episodes=episodes, verbose=True, on_step=on_step
    )
    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    return result


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
        if args.headless
        else launch_config_gui(args)
    )

    env = ObstacleGridworld(
        seed=config["seed"],
        goal_reward=10.0,
        shaping_w=0.6,
        collision_penalty=-1.0,
    )
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
        f"[laberinto R-STDP] seed={config['seed']} "
        f"episodes={config['episodes']} eta={config['eta']} "
        f"X0={config['initial_X']} "
        f"pesos={'nuevos' if not weights_source else weights_source}"
    )

    if config["weights"]:
        weights_stem = Path(config["weights"]).stem
    else:
        weights_stem = (
            f"run_obstacle_rstdp_seed{config['seed']}"
            f"_X0_{config['initial_X']:g}"
        )
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{weights_stem}_{run_stamp}"

    if args.headless:
        started = monotonic()
        result = run_training(
            agent, env, episodes=config["episodes"], verbose=True
        )
    else:
        result = run_live(agent, env, episodes=config["episodes"])
        started = monotonic()
    log, early_path, learned_path, final_success = result

    log.update(
        {
            "algorithm": "R-STDP",
            "encoder": "random",
            "seed": int(config["seed"]),
            "eta": float(config["eta"]),
            "initial_X": float(config["initial_X"]),
            "weights_source": weights_source,
            "initial_state": initial_state,
            "exploration_epsilon": EXPLORATION_EPSILON,
            "environment_config": {
                "goal_reward": 10.0,
                "collision_penalty": -1.0,
                "progress_shaping_weight": 0.6,
                "max_steps": env.max_steps,
            },
            "greedy_evaluation_success": bool(final_success),
        }
    )
    if args.headless:
        log["elapsed_seconds_total"] = float(monotonic() - started)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = OUTPUT_DIR / f"{stem}.npy"
    json_path = OUTPUT_DIR / f"{stem}.json"
    figure_path = FIGURE_DIR / f"{stem}.png"
    paths_path = OUTPUT_DIR / f"{stem}_paths.json"
    paths = {
        "seed": config["seed"],
        "obstacles": [list(cell) for cell in sorted(env.obstacles)],
        "start": list(env.start),
        "goal": list(env.goal),
        "early_path": [list(cell) for cell in early_path],
        "learned_path": [list(cell) for cell in learned_path],
        "final_success": bool(final_success),
    }

    agent.brain.save_weights(str(weights_path))
    json_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    paths_path.write_text(json.dumps(paths, indent=2), encoding="utf-8")
    plot_learning(log, figure_path)

    print(
        f"[resultado] exito(100)={log['final_success_rate']:.3f}; "
        f"evaluacion greedy={'EXITO' if final_success else 'FRACASO'}"
    )
    print(f"[ok] pesos: {weights_path}")
    print(f"[ok] log: {json_path}")
    print(f"[ok] trayectorias: {paths_path}")
    print(f"[ok] grafica: {figure_path}")


if __name__ == "__main__":
    main()
