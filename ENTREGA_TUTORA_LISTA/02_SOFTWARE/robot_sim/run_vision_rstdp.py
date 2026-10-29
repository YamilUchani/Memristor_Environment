"""Reward-modulated pre/post learning for the 8x8 visual pattern task.

Usage: python robot_sim/run_vision_rstdp.py
       python robot_sim/run_vision_rstdp.py --headless --episodes 5000
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
from run_vision import make_patterns


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figuras"
CONFIG_DIR = ROOT.parent / "neuromorphic_lab" / "configs"
NO_WEIGHTS = "--- (Empezar desde cero) ---"
N_PIXELS = 64
N_CLASSES = 8
DEFAULT_EPISODES = 5000
DEFAULT_ETA = 0.2
DEFAULT_INITIAL_X = 0.9
EXPLORATION_EPSILON = 0.30
CLASS_BASELINE_ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena los patrones 8x8 con elegibilidad pre/post R-STDP."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument(
        "--initial-X", type=float, choices=(0.1, 0.9), default=DEFAULT_INITIAL_X
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
    return args


def make_agent(
    seed: int,
    eta: float,
    initial_x: float,
    nv_profile: str,
    weights_file: str | None = None,
) -> NeuromorphicAgent:
    agent = NeuromorphicAgent(
        obs_dim=N_PIXELS,
        n_actions=N_CLASSES,
        N=N_PIXELS,
        M=N_CLASSES,
        V_read=0.05,
        eta=eta,
        dt=1e-3,
        steps_per_action=1,
        seed=seed,
        crossbar_seed=seed,
        encoder_seed=seed,
        action_seed=seed,
        encoder_mode="vision_bipolar",
        k_wta=1,
        wta_temperature=0.0,
        temp_init=1.0,
        temp_min=1.0,
        temp_decay=1.0,
        epsilon=EXPLORATION_EPSILON,
        initial_X=initial_x,
        json_memristor_nv=nv_profile,
        json_memristor_v="memristor_volatile.json",
        json_lif="lif_config.json",
        d2d_enabled=False,
        c2c_enabled=False,
        max_abs_delta_x=0.02,
    )
    if weights_file and weights_file != NO_WEIGHTS:
        agent.brain.load_weights(str(OUTPUT_DIR / weights_file))
    return agent


def run_classifier(
    patterns: np.ndarray,
    episodes: int = DEFAULT_EPISODES,
    eta: float = DEFAULT_ETA,
    seed: int = 0,
    initial_x: float = DEFAULT_INITIAL_X,
    nv_profile: str = "strukov_ideal.json",
    weights_file: str | None = None,
    verbose: bool = True,
    on_step=None,
) -> tuple[object, dict]:
    """Train one supervised contextual-bandit decision per presented image."""
    if patterns.shape != (N_CLASSES, 8, 8):
        raise ValueError(f"patterns must have shape ({N_CLASSES}, 8, 8)")
    if episodes < 1:
        raise ValueError("episodes must be positive")

    rng = np.random.default_rng(seed)
    class_order = np.tile(
        np.arange(N_CLASSES),
        (episodes + N_CLASSES - 1) // N_CLASSES,
    )
    rng.shuffle(class_order)
    class_order = class_order[:episodes]

    agent = make_agent(seed, eta, initial_x, nv_profile, weights_file)
    brain = agent.brain
    initial_crossbar_X = brain.X.copy()
    initial_state = {
        "X_mean": float(brain.X.mean()),
        "G_mean_siemens": float(brain.G.mean()),
        "weights_source": weights_file if weights_file != NO_WEIGHTS else None,
    }
    log = {
        "experiment": "vision_8x8_reward_modulated_pre_post_rstdp",
        "algorithm": "R-STDP",
        "encoder": "vision_bipolar",
        "episodes_requested": int(episodes),
        "seed": int(seed),
        "eta": float(eta),
        "initial_X": float(initial_x),
        "weights_source": initial_state["weights_source"],
        "initial_state": initial_state,
        "initial_crossbar_X": initial_crossbar_X.tolist(),
        "exploration_epsilon": EXPLORATION_EPSILON,
        "class_baseline_alpha": CLASS_BASELINE_ALPHA,
        "ep": [],
        "reward": [],
        "correct": [],
        "sampled_correct": [],
        "greedy_correct": [],
        "class_true": [],
        "class_pred": [],
        "class_pred_greedy": [],
        "X_mean": [],
        "baseline": [],
        "spikes": [],
        "eligibility_abs_mean": [],
        "delta_x_abs_mean": [],
    }
    class_baseline = np.zeros(N_CLASSES, dtype=float)
    start = monotonic()

    for episode, label in enumerate(class_order):
        label = int(label)
        action = int(agent.act(patterns[label].ravel()))
        step = agent.last_step()
        probabilities = np.asarray(step["probs"], dtype=float)
        greedy_prediction = int(np.argmax(probabilities))
        sampled_correct = int(action == label)
        reward = 1.0 if sampled_correct else -1.0
        agent.observe_reward(reward)
        metrics = agent.finish_episode(
            reward,
            baseline_override=float(class_baseline[label]),
        )
        class_baseline[label] += CLASS_BASELINE_ALPHA * (
            reward - class_baseline[label]
        )
        agent.end_episode()

        greedy_correct = int(greedy_prediction == label)
        log["ep"].append(episode)
        log["reward"].append(reward)
        log["correct"].append(greedy_correct)
        log["sampled_correct"].append(sampled_correct)
        log["greedy_correct"].append(greedy_correct)
        log["class_true"].append(label)
        log["class_pred"].append(action)
        log["class_pred_greedy"].append(greedy_prediction)
        log["X_mean"].append(float(brain.X.mean()))
        log["baseline"].append(float(class_baseline[label]))
        log["spikes"].append(int(np.count_nonzero(step["spikes"])))
        log["eligibility_abs_mean"].append(
            float(metrics["reward_modulated_eligibility_abs_mean"])
        )
        log["delta_x_abs_mean"].append(
            float(metrics["actual_delta_x_abs_mean"])
        )

        if on_step is not None and not on_step(
            episode, label, probabilities, brain, log
        ):
            break
        if verbose and (episode % 500 == 0 or episode == episodes - 1):
            recent = float(np.mean(log["correct"][-200:]))
            print(
                f"[ep {episode + 1:5d}] greedy_acc(200)={recent:.3f} "
                f"sampled_acc(200)={np.mean(log['sampled_correct'][-200:]):.3f} "
                f"X_mean={brain.X.mean():.4f}"
            )

    log["episodes_completed"] = len(log["ep"])
    log["elapsed_seconds"] = float(monotonic() - start)
    log["final_accuracy_greedy"] = (
        float(np.mean(log["greedy_correct"][-500:]))
        if log["greedy_correct"] else 0.0
    )
    log["final_accuracy_sampled"] = (
        float(np.mean(log["sampled_correct"][-500:]))
        if log["sampled_correct"] else 0.0
    )
    if verbose:
        print(
            f"[resultado] greedy accuracy últimos episodios="
            f"{log['final_accuracy_greedy']:.3f}; "
            f"tiempo={log['elapsed_seconds']:.1f}s"
        )
    return brain, log


def plot_learning(log: dict, path: Path) -> None:
    accuracy = np.asarray(log["greedy_correct"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    window = 100
    if accuracy.size >= window:
        smooth = np.convolve(
            accuracy, np.ones(window) / window, mode="valid"
        )
        ax.plot(np.arange(window, window + len(smooth)), smooth, lw=1.8)
    elif accuracy.size:
        ax.plot(np.arange(1, accuracy.size + 1), accuracy, lw=1.2)
    ax.axhline(1 / N_CLASSES, ls="--", color="gray", label="azar (1/8)")
    ax.set_xlabel("Ensayo")
    ax.set_ylabel("Accuracy greedy (media móvil 100)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Aprendizaje R-STDP — patrones 8×8")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def launch_config_gui(args: argparse.Namespace) -> dict:
    import tkinter as tk
    from tkinter import messagebox, ttk

    configs = sorted(path.name for path in CONFIG_DIR.glob("*.json"))
    if not configs:
        raise FileNotFoundError(f"No hay perfiles JSON en {CONFIG_DIR}")
    weight_files = [NO_WEIGHTS] + sorted(
        path.name for path in OUTPUT_DIR.glob("*.npy")
    )
    root = tk.Tk()
    root.title("Visión 8×8 — entrenamiento R-STDP")
    root.geometry("560x430")
    root.minsize(520, 400)
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

    def add_combo(row: int, label: str, choices: list[str], default: str):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
        combo = ttk.Combobox(frame, values=choices, state="readonly")
        combo.set(default if default in choices else choices[0])
        combo.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=6)
        return combo

    seed = add_entry(0, "Semilla:", str(args.seed))
    episodes = add_entry(1, "Ensayos:", str(args.episodes))
    eta = add_entry(2, "Tasa R-STDP (eta):", str(args.eta))
    initial_x = add_combo(3, "X₀:", ["0.1", "0.9"], str(args.initial_X))
    nv = add_combo(
        4, "Memristor no volátil:", configs,
        "strukov_ideal.json" if "strukov_ideal.json" in configs else configs[0],
    )
    weights = add_combo(5, "Pesos previos (.npy):", weight_files, NO_WEIGHTS)
    if args.weights in weight_files:
        weights.set(args.weights)
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
                or selected_eta <= 0.0
            ):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Configuración inválida",
                "Semilla ≥ 0, ensayos > 0 y eta finita > 0.",
                parent=root,
            )
            return
        result.update(
            seed=selected_seed,
            episodes=selected_episodes,
            eta=selected_eta,
            initial_x=float(initial_x.get()),
            nv_profile=nv.get(),
            weights_file=weights.get(),
        )
        root.destroy()

    ttk.Button(root, text="Iniciar entrenamiento", command=start).pack(pady=10)
    root.mainloop()
    if not result:
        raise SystemExit("Configuración cancelada.")
    return result


def run_live(patterns: np.ndarray, cfg: dict) -> tuple[object, dict]:
    state = {"paused": False, "stop": False, "speed": 1, "brain": None}
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Visión 8×8 — R-STDP")
    ax_pattern, ax_x = axes[0]
    ax_accuracy, ax_prob = axes[1]

    image = ax_pattern.imshow(patterns[0], cmap="gray_r", vmin=0, vmax=1)
    border = Rectangle(
        (-0.5, -0.5), 8, 8, fill=False, linewidth=4, edgecolor="tab:gray"
    )
    ax_pattern.add_patch(border)
    ax_pattern.set_title("Entrada")
    ax_pattern.axis("off")

    preview = np.full((N_PIXELS, N_CLASSES), cfg["initial_x"], dtype=float)
    if cfg["weights_file"] != NO_WEIGHTS:
        loaded = np.load(OUTPUT_DIR / cfg["weights_file"])
        if loaded.shape != preview.shape:
            raise ValueError(
                f"El .npy elegido tiene forma {loaded.shape}; "
                f"se esperaba {preview.shape}."
            )
        preview = np.clip(loaded, 0.0, 1.0)
    image_x = ax_x.imshow(preview, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax_x.set_title("Matriz X (64×8)")
    ax_x.set_xlabel("Columna / clase")
    ax_x.set_ylabel("Fila / píxel")
    fig.colorbar(image_x, ax=ax_x)

    line_accuracy, = ax_accuracy.plot([], [], lw=1.6, color="tab:blue")
    ax_accuracy.axhline(1 / N_CLASSES, ls="--", color="gray", label="azar (1/8)")
    ax_accuracy.set_ylim(0, 1.05)
    ax_accuracy.set_xlabel("Ensayo")
    ax_accuracy.set_ylabel("Accuracy greedy (100)")
    ax_accuracy.set_title("Aprendizaje")
    ax_accuracy.legend(fontsize=8)
    ax_accuracy.grid(alpha=0.25)

    status = fig.text(
        0.02, 0.98, "", ha="left", va="top", family="monospace", fontsize=9
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)

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
        elif event.key == "s" and state["brain"] is not None:
            state["brain"].save_weights(
                str(OUTPUT_DIR / f"vision_rstdp_manual_seed{cfg['seed']}.npy")
            )

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect(
        "close_event", lambda _event: state.update(stop=True)
    )

    def on_step(episode, label, probabilities, brain, log):
        state["brain"] = brain
        while state["paused"] and plt.fignum_exists(fig.number):
            plt.pause(0.05)
        if state["stop"] or not plt.fignum_exists(fig.number):
            return False
        if episode % state["speed"] != 0 and episode != cfg["episodes"] - 1:
            fig.canvas.flush_events()
            return True

        prediction = int(np.argmax(probabilities))
        is_correct = prediction == label
        color = "tab:green" if is_correct else "tab:red"
        image.set_data(patterns[label])
        border.set_edgecolor(color)
        ax_pattern.set_facecolor("#d4f5d4" if is_correct else "#f5d4d4")
        ax_pattern.set_title(
            f"Esperado: {label} | Predicho: {prediction}",
            color=color,
            fontweight="bold",
        )
        image_x.set_data(brain.X)
        accuracy = np.asarray(log["greedy_correct"], dtype=float)
        if len(accuracy) >= 100:
            smooth = np.convolve(
                accuracy, np.ones(100) / 100, mode="valid"
            )
            line_accuracy.set_data(
                np.arange(100, len(accuracy) + 1), smooth
            )
        else:
            line_accuracy.set_data(
                np.arange(1, len(accuracy) + 1), accuracy
            )
        ax_accuracy.set_xlim(0, max(10, cfg["episodes"]))
        ax_prob.clear()
        colors = ["tab:gray"] * N_CLASSES
        colors[label] = "tab:green"
        colors[prediction] = "tab:green" if is_correct else "tab:blue"
        ax_prob.bar(range(N_CLASSES), probabilities, color=colors)
        ax_prob.set_xticks(range(N_CLASSES))
        ax_prob.set_ylim(0, 1)
        ax_prob.set_xlabel("Clase")
        ax_prob.set_ylabel("Probabilidad")
        ax_prob.set_title("Probabilidades del decoder")
        ax_prob.grid(alpha=0.25, axis="y")
        acc100 = float(np.mean(log["greedy_correct"][-100:]))
        status.set_text(
            f"Ep {episode + 1:5d} | acc(100) {acc100:.3f} | "
            f"spikes {log['spikes'][-1]:2d} | X_mean {brain.X.mean():.4f} | "
            f"{'PAUSA' if state['paused'] else 'running x' + str(state['speed'])}"
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
        return True

    print("[vision R-STDP] espacio=pausa, q/Esc=salir, +/-=velocidad, s=guardar X")
    result = run_classifier(
        patterns,
        episodes=cfg["episodes"],
        eta=cfg["eta"],
        seed=cfg["seed"],
        initial_x=cfg["initial_x"],
        nv_profile=cfg["nv_profile"],
        weights_file=cfg["weights_file"],
        on_step=on_step,
    )
    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    return result


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    patterns = make_patterns()
    if args.headless:
        cfg = {
            "seed": args.seed,
            "episodes": args.episodes,
            "eta": args.eta,
            "initial_x": args.initial_X,
            "nv_profile": "strukov_ideal.json",
            "weights_file": args.weights or NO_WEIGHTS,
        }
        brain, log = run_classifier(
            patterns,
            episodes=cfg["episodes"],
            eta=cfg["eta"],
            seed=cfg["seed"],
            initial_x=cfg["initial_x"],
            nv_profile=cfg["nv_profile"],
            weights_file=cfg["weights_file"],
        )
    else:
        cfg = launch_config_gui(args)
        brain, log = run_live(patterns, cfg)

    log["config"] = {
        "N": N_PIXELS,
        "M": N_CLASSES,
        "seed": cfg["seed"],
        "initial_X": cfg["initial_x"],
        "eta": cfg["eta"],
        "nv_profile": cfg["nv_profile"],
        "weights_source": log["weights_source"],
        "exploration_epsilon": EXPLORATION_EPSILON,
        "class_baseline_alpha": CLASS_BASELINE_ALPHA,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"run_vision_rstdp_seed{cfg['seed']}_X0_{cfg['initial_x']:g}_{timestamp}"
    log["weights_file"] = f"{stem}.npy"
    (OUTPUT_DIR / f"{stem}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    brain.save_weights(str(OUTPUT_DIR / f"{stem}.npy"))
    plot_learning(log, FIGURE_DIR / f"{stem}.png")

    truth = np.asarray(log["class_true"], dtype=int)
    predicted = np.asarray(log["class_pred_greedy"], dtype=int)
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(confusion, (truth, predicted), 1)
    print("[matriz de confusión greedy]")
    print(confusion)
    print(f"[salida] {OUTPUT_DIR / f'{stem}.json'}")


if __name__ == "__main__":
    main()
