"""Auto 2D continuo sobre crossbar 24x8 con REINFORCE.

Mapa 10x10 con obstaculos, meta en (8.5, 8.5). El auto se mueve de forma
continua (avance y giro), entrenado con REINFORCE.

La configuración incluye semilla, algoritmo, X0 y parámetros físicos.

Uso:  python run_car.py              (abre GUI en vivo — modo normal)
      python run_car.py --headless   (sin ventana, para automatizacion)

Guarda: outputs/car_seed{N}.json (log completo + trayectoria ganadora),
        outputs/car_seed{N}_X.npy (matriz final),
        figuras/car_learning_seed{N}.png (curva de exito).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from neurobot.reinforce_agent import NeuromorphicAgent as ReinforceAgent
from car_env import (
    ACTION_NAMES,
    CarEnv,
    MAP_SIZE,
    N_ACTIONS,
    REWARD_STEERING_SMOOTHNESS,
    REWARD_UNPRODUCTIVE_TURN,
    TURN_PROGRESS_REQUIRED,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR = Path(__file__).resolve().parent / "figuras"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------
# Configuracion
# ------------------------------------------------------------------
N, M = 24, N_ACTIONS
ENCODER_MODE = "auto2d"
ETA = 0.01
N_EPISODES = 3000
SPEED_INIT = 3          # velocidad inicial; [+/-] permite llegar hasta x500
SEED = 0
# Exploración: el decoder aplica z-score a I_col, así que con T->0.05 la
# softmax colapsa a one-hot (pi ~= delta) y la elegibilidad E = pre*(delta-pi)
# se anula -> dX ~ 1e-17 y el aprendizaje se congela. La temperatura sigue las
# validaciones greedy para no enfriar por éxitos casuales durante entrenamiento.
TEMP_INIT_HD = 1.5
TEMP_MIN = 0.3            # Evita congelar la exploración y el gradiente.
TEMP_DECAY = 1.0         # La temperatura se controla solo por la validación greedy.
TEMP_TREND_WINDOW = 3
TEMP_TREND_DOWN_STEP = 0.05
TEMP_TREND_UP_STEP = 0.05
TEMP_TREND_UP_CONFIRMATIONS = 2
GUI_EVENT_BATCH = 10  # poll close/keyboard events at most every 10 sim steps
CKPT_EVERY = 200        # checkpoint cada N episodios (sobrevive kills externos)
POLICY_EVAL_INTERVAL = 100
POLICY_EVAL_EPISODES = 20
POLICY_EVAL_SEED_OFFSET = 1_000_000_000
POLICY_TEST_SEED_OFFSET = 1_100_000_000
TRAIN_MAP_SEED_OFFSET = 2_000_000_000
TRAIN_MAP_SEED_STRIDE = 2**32
POLICY_SOLVED_SUCCESS = 0.95
POLICY_SOLVED_CONFIRMATIONS = 3
MAX_WEIGHT_UPDATE = 0.01


def _training_map_seed(seed):
    return TRAIN_MAP_SEED_OFFSET + int(seed) * TRAIN_MAP_SEED_STRIDE


def _evaluation_map_seeds(count, seed_offset=POLICY_EVAL_SEED_OFFSET):
    return [
        int(seed_offset) + index
        for index in range(int(count))
    ]


def _parse_positive_integer(value, field_name):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} debe ser un entero positivo.") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} debe ser un entero positivo.")
    return parsed


def _episode_path_metrics(env, info):
    trail = np.asarray(getattr(env, "trail", []), dtype=float)
    if trail.ndim == 2 and trail.shape[0] > 1:
        segment_lengths = np.linalg.norm(np.diff(trail, axis=0), axis=1)
        measured_path_length = float(segment_lengths.sum())
        direct_distance = float(np.linalg.norm(trail[-1] - trail[0]))
    else:
        measured_path_length = 0.0
        direct_distance = 0.0
    path_length = float(info.get(
        "path_length",
        getattr(env, "path_length", measured_path_length),
    ))
    efficiency = (
        direct_distance / path_length if path_length > 0.0 else 0.0
    )
    return {
        "path_length": path_length,
        "path_efficiency": float(np.clip(efficiency, 0.0, 1.0)),
        "steering_reversals": int(info.get(
            "steering_reversals",
            getattr(env, "steering_reversals", 0),
        )),
        "steering_smoothness_penalty": float(info.get(
            "total_steering_smoothness_penalty",
            getattr(env, "steering_smoothness_penalty", 0.0),
        )),
        "turning_angle": float(info.get(
            "turning_angle", getattr(env, "turning_angle", 0.0)
        )),
        "unproductive_turns": int(info.get(
            "unproductive_turns", getattr(env, "unproductive_turns", 0)
        )),
        "unproductive_turn_penalty": float(info.get(
            "unproductive_turn_penalty",
            getattr(env, "unproductive_turn_penalty", 0.0),
        )),
    }


def _read_encoder_seed(weights_path, fallback_seed):
    metadata_path = Path(str(weights_path) + ".meta.json")
    if not metadata_path.is_file():
        return int(fallback_seed)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata.get(
        "encoder_seed", metadata.get("crossbar_seed", fallback_seed)
    ))


def _read_training_seed(weights_path, fallback_seed):
    metadata_path = Path(str(weights_path) + ".meta.json")
    if not metadata_path.is_file():
        return int(fallback_seed)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata.get("training_seed", fallback_seed))


def _validate_car_weights(weights_path):
    weights = np.load(weights_path, mmap_mode="r")
    if weights.shape != (N, M):
        raise ValueError(
            f"Auto 2D with eight actions requires a ({N}, {M}) crossbar; "
            f"{Path(weights_path).name} has shape {weights.shape}. "
            "Four-action weights cannot be used by this runner."
        )
    metadata_path = Path(str(weights_path) + ".meta.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("task") not in (None, "auto2d"):
            raise ValueError(
                f"{Path(weights_path).name} is marked for task "
                f"{metadata['task']!r}, not Auto 2D."
            )
        encoder_mode = metadata.get("encoder_mode", metadata.get("encoder"))
        if encoder_mode not in (None, ENCODER_MODE):
            raise ValueError(
                f"{Path(weights_path).name} uses encoder "
                f"{encoder_mode!r}; {ENCODER_MODE!r} is required."
            )
        action_names = metadata.get("action_names")
        if action_names is not None and tuple(action_names) != ACTION_NAMES:
            raise ValueError(
                f"{Path(weights_path).name} uses a different action mapping."
            )


def _save_run_metadata(weights_path, agent, seed, evaluation_only=False):
    metadata_path = Path(str(weights_path) + ".meta.json")
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "encoder": ENCODER_MODE,
        "encoder_mode": ENCODER_MODE,
        "encoder_seed": int(agent.seed_config["encoder"]),
        "training_seed": int(seed),
        "task": "auto2d",
        "action_names": list(ACTION_NAMES),
        "evaluation_only": bool(evaluation_only),
        "policy_success_threshold": POLICY_SOLVED_SUCCESS,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _pick_action(agent, sensors, episode=None):
    # La temperatura del decoder controla la exploracion; no mezclar otro
    # muestreo epsilon-greedy que vuelva opaca la politica.
    return int(agent.act(sensors))


def _episode_action(agent, sensors, evaluation_only):
    if evaluation_only:
        return int(agent.act_deterministic(sensors))
    return _pick_action(agent, sensors)


def _training_finished(episodes_completed, episode_limit):
    return episodes_completed >= episode_limit


def _adapt_temperature(agent, validation_success):
    """Adapt exploration only to changes in the frozen greedy policy."""
    if len(validation_success) < TEMP_TREND_WINDOW:
        return

    previous = float(np.mean(validation_success[-TEMP_TREND_WINDOW:-1]))
    recent = float(validation_success[-1])
    slope = recent - previous
    current = float(agent.current_temperature())
    initial = float(getattr(agent, "temp_init", TEMP_INIT_HD))
    floor = min(TEMP_MIN, initial)
    agent.temp_min = floor
    if slope > 0.0:
        # Progress gradually shifts the policy toward exploitation.
        agent._temp_negative_trend_count = 0
        updated = max(floor, current - TEMP_TREND_DOWN_STEP)
    else:
        # A flat or falling greedy score is not a reason to reduce exploration.
        count = int(getattr(agent, "_temp_negative_trend_count", 0)) + 1
        agent._temp_negative_trend_count = count
        if count < TEMP_TREND_UP_CONFIRMATIONS:
            return
        agent._temp_negative_trend_count = 0
        updated = min(initial, current + TEMP_TREND_UP_STEP)
    agent.decoder.set_temperature(updated)


class GreedyPolicyTracker:
    """Track greedy success on repeatable orientations of the training map."""

    def __init__(
        self,
        seed,
        eval_episodes=POLICY_EVAL_EPISODES,
        map_seed_offset=None,
    ):
        self.seed = int(seed)
        self.eval_episodes = int(eval_episodes)
        self.training_map_seed = _training_map_seed(self.seed)
        self.map_seeds = (
            [self.training_map_seed] * self.eval_episodes
            if map_seed_offset is None
            else _evaluation_map_seeds(
                self.eval_episodes, seed_offset=map_seed_offset
            )
        )
        self.history = []
        self.best_success = -1.0
        self.best_mean_steps = float("inf")
        self.best_episode = None
        self.last_trail = []
        self.success_confirmations = 0
        self.validation_regressed = False
        self.solved = False

    def evaluate(self, agent, episode):
        memo = {}
        factorization = getattr(agent.brain, "_mna_factorization", None)
        if factorization is not None:
            # SuperLU owns native state and the evaluation copy rebuilds its cache.
            memo[id(factorization)] = None
        eval_agent = copy.deepcopy(agent, memo)
        invalidate_cache = getattr(
            eval_agent.brain, "_invalidate_network_cache", None
        )
        if invalidate_cache is not None:
            invalidate_cache()
        eval_agent.decoder.sample = False
        successes = []
        successful_steps = []
        path_lengths = []
        successful_path_lengths = []
        steering_reversals = []
        successful_steering_reversals = []
        steering_penalties = []
        unproductive_turns = []
        termination_reasons = []
        best_trail = []
        best_trail_steps = float("inf")

        for map_seed in dict.fromkeys(self.map_seeds):
            eval_env = CarEnv(seed=map_seed)
            for index in range(self.map_seeds.count(map_seed)):
                sensors = (
                    eval_env.reset(seed=map_seed)
                    if index == 0 else eval_env.reset()
                )
                eval_agent.reset_episode_state()
                done = False
                while not done:
                    action = eval_agent.act_deterministic(sensors)
                    sensors, _, done, info = eval_env.step(action)
                arrived = info["reason"] == "goal"
                successes.append(float(arrived))
                termination_reasons.append(info["reason"])
                metrics = _episode_path_metrics(eval_env, info)
                path_lengths.append(metrics["path_length"])
                steering_reversals.append(metrics["steering_reversals"])
                steering_penalties.append(
                    metrics["steering_smoothness_penalty"]
                )
                unproductive_turns.append(metrics["unproductive_turns"])
                if arrived:
                    successful_steps.append(int(eval_env.steps))
                    successful_path_lengths.append(metrics["path_length"])
                    successful_steering_reversals.append(
                        metrics["steering_reversals"]
                    )
                    if eval_env.steps < best_trail_steps:
                        best_trail_steps = eval_env.steps
                        best_trail = [
                            list(point) for point in eval_env.trail
                        ]
        self.last_trail = [list(point) for point in best_trail]

        success_rate = float(np.mean(successes))
        mean_steps = (
            float(np.mean(successful_steps))
            if successful_steps else float("inf")
        )
        improved = (
            success_rate > self.best_success
            or (
                success_rate == self.best_success
                and mean_steps < self.best_mean_steps
            )
        )
        if improved:
            self.best_success = success_rate
            self.best_mean_steps = mean_steps
            self.best_episode = int(episode)

        self.success_confirmations = (
            self.success_confirmations + 1
            if success_rate >= POLICY_SOLVED_SUCCESS
            else 0
        )
        self.solved = (
            self.success_confirmations >= POLICY_SOLVED_CONFIRMATIONS
        )
        self.validation_regressed = self.validation_regressed or (
            self.success_confirmations == 0
            and any(
                item["success_rate"] >= POLICY_SOLVED_SUCCESS
                for item in self.history
            )
        )
        result = {
            "episode": int(episode),
            "success_rate": success_rate,
            "successful_episodes": int(sum(successes)),
            "episodes": self.eval_episodes,
            "map_seeds": list(self.map_seeds),
            "collisions": termination_reasons.count("collision"),
            "timeouts": termination_reasons.count("timeout"),
            "mean_success_steps": (
                mean_steps if np.isfinite(mean_steps) else None
            ),
            "mean_path_length": float(np.mean(path_lengths)),
            "mean_success_path_length": (
                float(np.mean(successful_path_lengths))
                if successful_path_lengths else None
            ),
            "mean_steering_reversals": float(
                np.mean(steering_reversals)
            ),
            "mean_success_steering_reversals": (
                float(np.mean(successful_steering_reversals))
                if successful_steering_reversals else None
            ),
            "mean_steering_smoothness_penalty": float(
                np.mean(steering_penalties)
            ),
            "mean_unproductive_turns": float(
                np.mean(unproductive_turns)
            ),
            "best_success_rate": self.best_success,
            "best_episode": self.best_episode,
            "is_best": improved,
            "success_confirmations": self.success_confirmations,
            "confirmations_required": POLICY_SOLVED_CONFIRMATIONS,
            "solved": self.solved,
            "validation_regressed": self.validation_regressed,
        }
        self.history.append(result)
        return result


def launch_config_gui(seed_default=0, episodes_default=N_EPISODES):
    """Dialogo tkinter con semilla, algoritmo, estado X0 y parámetros físicos.

    En Auto 2D se configuran semilla, episodios, X0 y parámetros físicos.
    """
    import tkinter as tk
    from tkinter import ttk

    config_dir = Path(__file__).resolve().parents[1] / "neuromorphic_lab" / "configs"
    if not config_dir.exists():
        config_dir = Path(__file__).resolve().parent
    jsons = [p.name for p in config_dir.glob("*.json")]
    if not jsons:
        jsons = ["last_session.json", "lif_config.json", "memristor_volatile.json"]

    npys = []
    for path in OUT_DIR.glob("*.npy"):
        if np.load(path, mmap_mode="r").shape != (N, M):
            continue
        metadata_path = Path(str(path) + ".meta.json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("task") not in (None, "auto2d"):
                continue
            action_names = metadata.get("action_names")
            if (action_names is not None
                    and tuple(action_names) != ACTION_NAMES):
                continue
        npys.append(path.name)
    npys.insert(0, "--- (Empezar desde cero) ---")

    out = {"nv": "strukov_ideal.json" if "strukov_ideal.json" in jsons else jsons[0],
           "v": ("memristor_volatile.json"
                 if "memristor_volatile.json" in jsons else jsons[0]),
           "lif": "lif_config.json" if "lif_config.json" in jsons else jsons[0],
           "weights": npys[0], "temp": TEMP_INIT_HD,
           "evaluation_only": True,
           "out_name": (
               f"run_car_reinforce_seed{seed_default}_A8_X0_0.9.npy"
           ),
           "seed": seed_default, "episodes": episodes_default,
           "algorithm": "REINFORCE", "initial_X": 0.9}

    root = tk.Tk()
    root.title("Robot Sim - Configuración (Auto 2D)")
    root.geometry("650x680")
    root.minsize(650, 600)
    root.columnconfigure(0, weight=1)
    root.eval('tk::PlaceWindow . center')
    root.attributes("-topmost", True)

    tk.Label(root, text="Cargar Archivos JSON desde el Laboratorio",
             font=("Arial", 11, "bold")).pack(pady=10)
    frame = tk.Frame(root)
    frame.pack(padx=20, pady=10, fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    tk.Label(frame, text="Semilla:").grid(row=0, column=0, sticky="w", pady=5)
    entry_seed = ttk.Entry(frame, width=33)
    entry_seed.insert(0, str(seed_default))
    entry_seed.grid(row=0, column=1, pady=5, padx=10, sticky="ew")

    tk.Label(frame, text="Episodios máximos:").grid(
        row=1, column=0, sticky="w", pady=5)
    entry_episodes = ttk.Entry(frame, width=33)
    entry_episodes.insert(0, str(episodes_default))
    entry_episodes.grid(row=1, column=1, pady=5, padx=10, sticky="ew")

    # 1) Memristor No Volátil [APLICA: sustrato fisico del crossbar]
    tk.Label(frame, text="Memristor No Volátil (Pesos):").grid(
        row=2, column=0, sticky="w", pady=5)
    cbo_nv = ttk.Combobox(frame, values=jsons, state="readonly", width=30)
    cbo_nv.set(out["nv"])
    cbo_nv.grid(row=2, column=1, pady=5, padx=10, sticky="ew")

    # Se conserva el campo por compatibilidad visual, pero no se usa en el agente.
    tk.Label(frame, text="Memristor Volátil (no usado):").grid(row=3, column=0, sticky="w", pady=5)
    cbo_v = ttk.Combobox(frame, values=jsons, state="disabled", width=30)
    cbo_v.set(out["v"])
    cbo_v.grid(row=3, column=1, pady=5, padx=10, sticky="ew")

    # 3) Neurona LIF [APLICA: decision por spikes via WTA]
    tk.Label(frame, text="Neurona LIF:").grid(row=4, column=0, sticky="w", pady=5)
    cbo_lif = ttk.Combobox(frame, values=jsons, state="readonly", width=30)
    cbo_lif.set(out["lif"])
    cbo_lif.grid(row=4, column=1, pady=5, padx=10, sticky="ew")

    # 4) Pesos Previos [APLICA: continua un entrenamiento previo]
    tk.Label(frame, text="Pesos Previos (.npy):").grid(row=5, column=0, sticky="w", pady=5)
    cbo_w = ttk.Combobox(frame, values=npys, state="readonly", width=30)
    cbo_w.set(npys[0])
    cbo_w.grid(row=5, column=1, pady=5, padx=10, sticky="ew")

    tk.Label(frame, text="Modo con pesos cargados:").grid(
        row=6, column=0, sticky="w", pady=5)
    modes = [
        "Evaluar política (greedy, sin aprendizaje)",
        "Continuar entrenamiento desde pesos",
    ]
    cbo_mode = ttk.Combobox(frame, values=modes, state="readonly", width=38)
    cbo_mode.set(modes[0])
    cbo_mode.grid(row=6, column=1, pady=5, padx=10, sticky="ew")

    # 5) Temperatura Inicial [APLICA: RL, muestreo exploratorio]
    tk.Label(frame, text="Temperatura inicial de entrenamiento:").grid(
        row=7, column=0, sticky="w", pady=5)
    temps = ["1.5 (Explorar y entrenar)", "0.05 (Explotar pesos entrenados)"]
    cbo_t = ttk.Combobox(frame, values=temps, state="readonly", width=30)
    cbo_t.set(temps[0])
    cbo_t.grid(row=7, column=1, pady=5, padx=10, sticky="ew")

    # 6) Guardar como [APLICA: nombre del archivo de salida]
    tk.Label(frame, text="Guardar al terminar como:").grid(
        row=8, column=0, sticky="w", pady=5)
    entry_out = ttk.Entry(frame, width=33)
    default_out_name = (
        f"run_car_reinforce_seed{seed_default}_A8_X0_0.9.npy"
    )
    entry_out.insert(0, default_out_name)
    entry_out.grid(row=8, column=1, pady=5, padx=10, sticky="ew")

    tk.Label(frame, text="X₀ inicial (si empiezas desde cero):").grid(
        row=9, column=0, sticky="w", pady=5)
    cbo_x0 = ttk.Combobox(frame, values=["0.1", "0.9"],
                          state="readonly", width=30)
    cbo_x0.set("0.9")
    cbo_x0.grid(row=9, column=1, pady=5, padx=10, sticky="ew")

    def on_start():
        try:
            selected_seed = int(entry_seed.get())
            if selected_seed < 0:
                raise ValueError
        except ValueError:
            from tkinter import messagebox
            messagebox.showerror("Semilla inválida", "Use un entero no negativo.", parent=root)
            return
        try:
            selected_episodes = _parse_positive_integer(
                entry_episodes.get(), "El número de episodios"
            )
        except ValueError as exc:
            from tkinter import messagebox
            messagebox.showerror("Episodios inválidos", str(exc), parent=root)
            return
        out["seed"] = selected_seed
        out["episodes"] = selected_episodes
        out["algorithm"] = "REINFORCE"
        out["nv"] = cbo_nv.get()
        out["v"] = cbo_v.get()
        out["lif"] = cbo_lif.get()
        out["weights"] = cbo_w.get()
        out["evaluation_only"] = (
            cbo_w.get() != npys[0] and cbo_mode.get() == modes[0]
        )
        # A low temperature is only for exploiting an already trained policy.
        has_weights = cbo_w.get() != npys[0]
        out["temp"] = (0.05 if has_weights and "0.05" in cbo_t.get()
                       else TEMP_INIT_HD)
        out["initial_X"] = float(cbo_x0.get())
        name = entry_out.get().strip()
        if name == default_out_name:
            name = (
                f"run_car_reinforce_seed{selected_seed}_A8_"
                f"X0_{out['initial_X']:g}.npy"
            )
        if not name.endswith(".npy"):
            name += ".npy"
        out["out_name"] = name
        out["out_name"] = name
        root.destroy()

    btn = tk.Button(root, text="Entrenar", command=on_start, bg="#4CAF50",
                    fg="white", font=("Arial", 10, "bold"), padx=15, pady=5)
    btn.pack(pady=10)

    root.mainloop()
    return (out["nv"], out["v"], out["lif"], out["weights"], out["temp"],
            out["out_name"], out["seed"], out["algorithm"], out["initial_X"],
            out["episodes"], out["evaluation_only"])


def plot_curves(success, path, policy_evaluations=None):
    """Curva de exito (media movil 50) guardada como PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    acc = np.asarray(success, dtype=float)
    if len(acc) >= 50:
        smooth = np.convolve(acc, np.ones(50) / 50, mode="valid")
        ax.plot(
            np.arange(50, 50 + len(smooth)),
            smooth,
            lw=1.8,
            color="tab:blue",
            label="Éxito durante entrenamiento",
        )
    if policy_evaluations:
        ax.plot(
            [item["episode"] for item in policy_evaluations],
            [item["success_rate"] for item in policy_evaluations],
            "o--",
            lw=1.4,
            ms=3,
            color="tab:green",
            label="Evaluación greedy (20 orientaciones, mismo mapa)",
        )
    ax.axhline(0.0, ls="--", color="gray", label="azar (~0)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Exito (media movil 50)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Curva de aprendizaje — Auto 2D")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_gui(env, agent, episodes, seed, policy_tracker,
            evaluation_only=False):
    """Run training or immutable greedy evaluation in the live window."""
    state = {
        "paused": False, "stop": False, "speed": SPEED_INIT, "episode": 0,
        "ep_reward": 0.0, "done": True, "action": 0,
        "step_rewards": [],
        "success": [], "dist_end": [], "rewards": [],
        "path_length": [], "path_efficiency": [],
        "steering_reversals": [], "steering_smoothness_penalty": [],
        "turning_angle": [], "unproductive_turns": [],
        "unproductive_turn_penalty": [],
        "termination_reason": [], "episode_steps": [],
        "collision_types": [],
        "update_limit": [],
        "map_seeds": [],
        "current_map_seed": None,
        "last_result": None, "win_trail": [], "seed": seed,
        "temperature": [], "policy_evaluations": [],
        "x_episode_start": agent.brain.X.copy(),
        "algorithm": "REINFORCE",
        "front_clearance": env.front_clearance(),
        "safety_clearance": env.safety_clearance(),
        "collision_type": None,
        "evaluation_only": bool(evaluation_only),
    }

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.canvas.manager.set_window_title(
        "Neuromorphic Car 2D — Crossbar 24x8 — sensor coding v3"
    )

    ax_map, ax_X = axes[0, 0], axes[0, 1]
    ax_succ, ax_dist = axes[1, 0], axes[1, 1]

    # P1: mapa 2D (borde gris inicial)
    env.draw(ax_map, title="Listo")
    ax_map.set_xlim(-0.7, MAP_SIZE + 0.7)
    ax_map.set_ylim(-0.7, MAP_SIZE + 0.7)

    # P2: heatmap X escala fija 0-1 (NO auto-escalado: con X~0.1 parejo
    # el auto-escalado min/max amplificaba el ruido tiny a toda la paleta)
    im_X = ax_X.imshow(agent.brain.X, aspect="auto", vmin=0.0, vmax=1.0)
    ax_X.set_title("Matriz X (memristores, 24×8)")
    ax_X.set_xlabel("Columna (acción)")
    ax_X.set_ylabel("Fila (sensor)")
    fig.colorbar(im_X, ax=ax_X)

    # P3: curva de exito (media movil 50)
    line_succ, = ax_succ.plot(
        [], [], color="tab:blue", lw=1.6,
        label=(
            "Éxito de la política cargada"
            if evaluation_only else "Éxito durante entrenamiento"
        ),
    )
    line_greedy, = ax_succ.plot(
        [], [], "o--", color="tab:green", lw=1.3, ms=3,
        label="Evaluación greedy (20 orientaciones, mismo mapa)",
    )
    ax_succ.axhline(0.0, ls="--", color="gray", label="azar (~0%)")
    ax_succ.set_xlim(0, max(10, episodes))
    ax_succ.set_ylim(-0.05, 1.05)
    ax_succ.set_xlabel("Episodio")
    ax_succ.set_ylabel("Éxito (media móvil 50)")
    ax_succ.set_title(
        "Evaluación de política cargada" if evaluation_only
        else "Curva de aprendizaje"
    )
    ax_succ.legend(fontsize=8, loc="lower right")
    ax_succ.grid(alpha=0.25)

    # P4: distancia a meta (ultimos 100) + trayectoria ganadora superpuesta
    line_dist, = ax_dist.plot([], [], "o-", color="tab:blue", ms=3, lw=1.0)
    ax_dist.set_xlim(0, 100)
    ax_dist.set_ylim(0, MAP_SIZE * 1.5)
    ax_dist.set_xlabel("Episodio (ventana 100)")
    ax_dist.set_ylabel("Dist. a meta al terminar")
    ax_dist.set_title("Distancia a meta + trayectoria ganadora")
    ax_dist.grid(alpha=0.25)
    # trayectoria ganadora normalizada en coords. de ejes (overlay visual)
    line_win, = ax_dist.plot([], [], color="0.55", lw=1.4, ls=":",
                             transform=ax_dist.transAxes, zorder=1)
    sc_start, = ax_dist.plot([], [], "s", color="tab:blue", ms=5,
                             transform=ax_dist.transAxes)
    sc_goal, = ax_dist.plot([], [], "*", color="tab:green", ms=9,
                            transform=ax_dist.transAxes)

    info_text = fig.text(0.02, 0.98, "", ha="left", va="top",
                         family="monospace", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()
    fig.canvas.flush_events()

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
            print(f"[pausa] {state['paused']}")
        elif event.key in ("q", "escape"):
            state["stop"] = True
            plt.close(fig)
        elif event.key in ("+", "=", "plus"):
            state["speed"] = min(500, state["speed"] * 2)
            print(f"[speed] running x{state['speed']}")
        elif event.key in ("-", "_", "minus"):
            state["speed"] = max(1, state["speed"] // 2)
            print(f"[speed] running x{state['speed']}")
        elif event.key == "r":
            state["done"] = True   # reinicia el episodio en el proximo paso
            print("[reset] episodio reiniciado")
        elif event.key == "s":
            import tkinter as tk
            from tkinter import simpledialog
            was_paused = state["paused"]
            state["paused"] = True
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            name = simpledialog.askstring(
                "Guardar Pesos", "Nombre del archivo (.npy):",
                initialvalue="car_weights.npy", parent=root)
            root.destroy()
            if name:
                if not name.endswith(".npy"):
                    name += ".npy"
                agent.brain.save_weights(str(OUT_DIR / name))
                _save_run_metadata(OUT_DIR / name, agent, seed)
                print(f"[ok] crossbar actual guardado: {name}")
            state["paused"] = was_paused

    def on_close(_event):
        state["stop"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    def reset_episode():
        agent.reset_episode_state()
        if evaluation_only:
            if state["current_map_seed"] is None:
                state["current_map_seed"] = _training_map_seed(seed)
                sensors = env.reset(seed=state["current_map_seed"])
            else:
                sensors = env.reset()
        elif state["current_map_seed"] is None:
            state["current_map_seed"] = _training_map_seed(seed)
            sensors = env.reset(seed=state["current_map_seed"])
        else:
            sensors = env.reset()
        state["x_episode_start"] = agent.brain.X.copy()
        state["front_clearance"] = env.front_clearance()
        state["safety_clearance"] = env.safety_clearance()
        state["collision_type"] = None
        state["action"] = _episode_action(
            agent, sensors, evaluation_only
        )
        state["ep_reward"] = 0.0
        state["step_rewards"] = []
        state["done"] = False

    def do_step():
        if state["done"]:
            reset_episode()
            return
        sensors, r, done, info = env.step(state["action"])
        state["front_clearance"] = info["front_clearance"]
        state["safety_clearance"] = info["safety_clearance"]
        state["ep_reward"] += r
        state["step_rewards"].append(r)
        if done:
            state["done"] = True
            if evaluation_only:
                agent.reset_episode_state()
            else:
                agent.learn_from_rewards(state["step_rewards"])
                agent.end_episode()
            state["update_limit"].append(
                float(getattr(agent, "max_weight_update", MAX_WEIGHT_UPDATE))
            )
            arrived = info["reason"] == "goal"
            state["success"].append(1.0 if arrived else 0.0)
            state["termination_reason"].append(info["reason"])
            state["episode_steps"].append(int(info["steps"]))
            state["collision_types"].append(info["collision_type"])
            path_metrics = _episode_path_metrics(env, info)
            for key, value in path_metrics.items():
                state[key].append(value)
            state["map_seeds"].append(state["current_map_seed"])
            state["dist_end"].append(info["dist"])
            state["rewards"].append(state["ep_reward"])
            state["episode"] += 1
            state["last_result"] = info["reason"]
            state["collision_type"] = info.get("collision_type")
            state["temperature"].append(agent.current_temperature())
            if arrived:
                state["win_trail"] = [list(p) for p in env.trail]
            completed = state["episode"]
            if (
                completed % POLICY_EVAL_INTERVAL == 0
                or completed >= episodes
            ):
                evaluation = policy_tracker.evaluate(agent, completed)
                state["policy_evaluations"].append(evaluation)
                if not evaluation_only:
                    _adapt_temperature(
                        agent,
                        [
                            item["success_rate"]
                            for item in state["policy_evaluations"]
                        ],
                    )
                print(
                    f"[evaluación greedy ep {completed}] "
                    f"éxito={evaluation['success_rate']:.2f} "
                    f"mejor={evaluation['best_success_rate']:.2f}"
                )
        else:
            state["action"] = _episode_action(
                agent, sensors, evaluation_only
            )

    def draw_all():
        # --- P1: mapa con feedback del episodio ---
        res = state["last_result"]
        if state["done"] and res is not None:
            if res == "goal":
                title = (f"Ep {state['episode']} | Meta alcanzada en "
                         f"{env.steps} pasos | R_ep {state['ep_reward']:.2f}")
            elif res == "collision":
                kind = state["collision_type"] or "desconocido"
                title = (f"Ep {state['episode']} | Choque: {kind} | "
                         f"R_ep {state['ep_reward']:.2f}")
            else:
                title = (f"Ep {state['episode']} | Timeout ({env.steps} pasos) "
                         f"| R_ep {state['ep_reward']:.2f}")
        else:
            title = f"Ep {state['episode']} | Dist meta: {env._dist_goal():.2f}"
        env.draw(ax_map, title=title, result=res if state["done"] else None)
        shown = res if state["done"] else None
        edge = {"goal": "tab:green", "collision": "tab:red",
                "timeout": "tab:red"}.get(shown, "0.6")
        ax_map.set_facecolor({"tab:green": "#d4f5d4",
                              "tab:red": "#f5d4d4"}.get(edge, "white"))
        ax_map.add_patch(Rectangle((0, 0), MAP_SIZE, MAP_SIZE, fill=False,
                                   lw=4, edgecolor=edge))
        ax_map.set_xlim(-0.7, MAP_SIZE + 0.7)
        ax_map.set_ylim(-0.7, MAP_SIZE + 0.7)

        # --- P2: heatmap X (escala fija 0-1, sin auto-escalado) ---
        X = agent.brain.X
        im_X.set_data(X)

        # --- P3: exito movil 50 ---
        succ = np.asarray(state["success"], dtype=float)
        if len(succ) >= 50:
            smooth = np.convolve(succ, np.ones(50) / 50, mode="valid")
            line_succ.set_data(np.arange(50, 50 + len(smooth)), smooth)
        if state["policy_evaluations"]:
            line_greedy.set_data(
                [item["episode"] for item in state["policy_evaluations"]],
                [item["success_rate"] for item in state["policy_evaluations"]],
            )

        # --- P4: distancias ultimos 100 + trayectoria ganadora ---
        d = state["dist_end"][-100:]
        line_dist.set_data(np.arange(len(d)), d)
        ax_dist.set_xlim(0, max(10, len(d)))
        if state["win_trail"]:
            wt = np.asarray(state["win_trail"])
            xa = 0.06 + 0.88 * wt[:, 0] / MAP_SIZE
            ya = 0.06 + 0.88 * wt[:, 1] / MAP_SIZE
            line_win.set_data(xa, ya)
            sc_start.set_data([xa[0]], [ya[0]])
            sc_goal.set_data([xa[-1]], [ya[-1]])

        # --- status bar ---
        s100 = (float(np.mean(state["success"][-100:]))
                if state["success"] else 0.0)
        greedy_rate = (
            state["policy_evaluations"][-1]["success_rate"]
            if state["policy_evaluations"] else float("nan")
        )
        speed = state["speed"]
        running = "PAUSA" if state["paused"] else f"running x{speed}"
        x_delta = np.abs(agent.brain.X - state["x_episode_start"])
        info_text.set_text(
            f"{state['algorithm']} | MNA optimizado\n"
            f"Ep {state['episode']:4d}  |  pasos {env.steps:3d}  |  "
            f"éxito(100) {s100:.3f}  |  dist_meta {env._dist_goal():.2f}  |  "
            f"greedy_eval {greedy_rate:.2f}  |  "
            f"X_mean {agent.brain.X.mean():.4f}  |  "
            f"ΔX max/episodio {x_delta.max():.1e} "
            f"({np.count_nonzero(x_delta)} celdas)  |  "
            f"margen {state['safety_clearance']:.2f}m  |  "
            f"R_ep {state['ep_reward']:.2f}  |  "
            f"giro_rev {getattr(env, 'steering_reversals', 0)}  |  "
            f"vueltas_improd {getattr(env, 'unproductive_turns', 0)}  |  "
            f"camino {getattr(env, 'path_length', 0.0):.2f}m  |  "
            f"T {agent.current_temperature():.3f}  |  {running}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    print(f"[run_car] N={N} M={M} eta={ETA} seed={seed} speed={SPEED_INIT} "
          "safety_shaping=v2 sensor_coding=random-projection-v1")
    print("[run_car] [Espacio]=pausa [q]=salir [+/-]=velocidad hasta x500 "
          "[r]=reset [s]=guardar.")
    reset_episode()
    try:
        while (
            plt.fignum_exists(fig.number)
            and not state["stop"]
            and not _training_finished(
                state["episode"], episodes
            )
        ):
            if not state["paused"]:
                remaining = state["speed"]
                while (
                    remaining > 0
                    and not state["paused"]
                    and not state["stop"]
                    and not _training_finished(
                        state["episode"], episodes
                    )
                ):
                    batch = min(GUI_EVENT_BATCH, remaining)
                    for _ in range(batch):
                        do_step()
                        if state["done"] or state["stop"]:
                            break
                    remaining -= batch
                    # Process close, pause, and stop events between batches.
                    fig.canvas.flush_events()
                    if not plt.fignum_exists(fig.number):
                        state["stop"] = True
            if state["stop"] or not plt.fignum_exists(fig.number):
                break
            draw_all()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    return state


def run_headless(
    env,
    agent,
    episodes,
    seed,
    verbose=True,
    ckpt_name=None,
    policy_tracker=None,
    eval_interval=POLICY_EVAL_INTERVAL,
    evaluation_only=False,
    adapt_exploration=True,
    on_episode=None,
):
    """Run training or immutable greedy evaluation; return the full log."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if eval_interval < 1:
        raise ValueError("eval_interval must be positive")
    policy_tracker = policy_tracker or GreedyPolicyTracker(seed)
    if not hasattr(agent, "max_weight_update"):
        agent.max_weight_update = MAX_WEIGHT_UPDATE
    success, dist_end, rewards = [], [], []
    termination_reason, episode_steps, collision_type = [], [], []
    temperature, policy_evaluations, update_limits = [], [], []
    path_length, path_efficiency = [], []
    steering_reversals, steering_smoothness_penalty = [], []
    turning_angle, unproductive_turns = [], []
    unproductive_turn_penalty = []
    win_trail = []
    info = {"reason": "timeout", "dist": 0.0}
    map_seeds = []
    training_map_seed = _training_map_seed(seed)
    for ep in range(episodes):
        if evaluation_only:
            map_seed = training_map_seed
            if ep == 0:
                sensors = env.reset(seed=training_map_seed)
            else:
                sensors = env.reset()
            agent.reset_episode_state()
        elif ep == 0:
            map_seed = training_map_seed
            sensors = env.reset(seed=training_map_seed)
        else:
            map_seed = training_map_seed
            sensors = env.reset()
        map_seeds.append(map_seed)
        done = False
        action = _episode_action(agent, sensors, evaluation_only)
        ep_r = 0.0
        step_rewards = []
        while not done:
            sensors, r, done, info = env.step(action)
            ep_r += r
            step_rewards.append(r)
            if not done:
                action = _episode_action(agent, sensors, evaluation_only)
        if evaluation_only:
            agent.reset_episode_state()
        else:
            agent.learn_from_rewards(step_rewards)
            agent.end_episode()
        update_limits.append(
            float(getattr(agent, "max_weight_update", MAX_WEIGHT_UPDATE))
        )
        arrived = info["reason"] == "goal"
        success.append(1.0 if arrived else 0.0)
        termination_reason.append(info["reason"])
        episode_steps.append(int(info["steps"]))
        collision_type.append(info["collision_type"])
        path_metrics = _episode_path_metrics(env, info)
        path_length.append(path_metrics["path_length"])
        path_efficiency.append(path_metrics["path_efficiency"])
        steering_reversals.append(path_metrics["steering_reversals"])
        steering_smoothness_penalty.append(
            path_metrics["steering_smoothness_penalty"]
        )
        turning_angle.append(path_metrics["turning_angle"])
        unproductive_turns.append(path_metrics["unproductive_turns"])
        unproductive_turn_penalty.append(
            path_metrics["unproductive_turn_penalty"]
        )
        temperature.append(agent.current_temperature())
        dist_end.append(info["dist"])
        rewards.append(ep_r)
        if arrived:
            win_trail = [list(p) for p in env.trail]
        completed = ep + 1
        if completed % eval_interval == 0 or completed == episodes:
            evaluation = policy_tracker.evaluate(agent, completed)
            policy_evaluations.append(evaluation)
            if not evaluation_only and adapt_exploration:
                _adapt_temperature(
                    agent,
                    [item["success_rate"] for item in policy_evaluations],
                )
            if verbose:
                print(
                    f"[evaluación greedy ep {completed}] "
                    f"éxito={evaluation['success_rate']:.2f} "
                    f"mejor={evaluation['best_success_rate']:.2f}"
                )
        if verbose and (ep % max(1, episodes // 10) == 0
                        or ep == episodes - 1):
            s = success[-100:]
            print(f"[ep {ep:4d}] exito(100)={float(np.mean(s)):.3f}  "
                  f"dist={info['dist']:.2f}  "
                  f"X_mean={agent.brain.X.mean():.4f}")
        if (ckpt_name and not evaluation_only
                and (ep + 1) % CKPT_EVERY == 0):
            agent.brain.save_weights(
                str(OUT_DIR / f"{ckpt_name}_X_ep{ep + 1}.npy")
            )
        if on_episode is not None and not on_episode(ep, bool(arrived)):
            break
    result = {"seed": seed, "success": success, "dist_end": dist_end,
              "reward": rewards, "win_trail": win_trail,
              "termination_reason": termination_reason,
              "episode_steps": episode_steps,
              "collision_type": collision_type,
              "path_length": path_length,
              "path_efficiency": path_efficiency,
              "steering_reversals": steering_reversals,
              "steering_smoothness_penalty": steering_smoothness_penalty,
              "turning_angle": turning_angle,
              "unproductive_turns": unproductive_turns,
              "unproductive_turn_penalty": unproductive_turn_penalty,
              "training_map_seeds": map_seeds,
              "training_map_seed": training_map_seed,
              "training_map_mode": (
                  "fixed_map_for_greedy_evaluation"
                  if evaluation_only else "fixed_map_per_run"
              ),
              "temperature": temperature,
              "max_weight_update": update_limits,
              "policy_evaluations": policy_evaluations,
              "episodes_requested": int(episodes),
              "episodes_completed": len(success),
              "evaluation_only": bool(evaluation_only),
              "best_greedy_success_rate": (
                  policy_tracker.best_success
                  if policy_tracker.best_success >= 0.0 else None
              ),
              "best_greedy_mean_steps": (
                  policy_tracker.best_mean_steps
                  if np.isfinite(policy_tracker.best_mean_steps) else None
              ),
              "policy_solved": bool(policy_tracker.solved),
              "validation_regressed": bool(
                  policy_tracker.validation_regressed
              ),
              "steering_smoothness_reward_weight": (
                  REWARD_STEERING_SMOOTHNESS
              ),
              "unproductive_turn_progress_required": TURN_PROGRESS_REQUIRED,
              "unproductive_turn_reward_weight": REWARD_UNPRODUCTIVE_TURN,
              "final_greedy_success_rate": (
                  policy_tracker.history[-1]["success_rate"]
                  if policy_tracker.history else None
              )}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--eta", type=float, default=ETA)
    ap.add_argument("--initial-X", type=float, choices=[0.1, 0.9], default=0.9,
                    help="Estado inicial del crossbar: X=0.1 o X=0.9.")
    ap.add_argument(
        "--weights",
        help="Archivo .npy dentro de outputs (o ruta completa) para cargar.",
    )
    ap.add_argument(
        "--continue-training",
        action="store_true",
        help="Entrena desde --weights; por defecto, cargarlos significa "
             "evaluación greedy sin aprendizaje.",
    )
    ap.add_argument("--headless", action="store_true",
                    help="Sin ventana: solo imprime y guarda artefactos "
                         "(automatización)")
    ap.add_argument("--gui", action="store_true",
                    help=argparse.SUPPRESS)  # compat: ahora es el default
    args = ap.parse_args()

    np.random.seed(args.seed)
    env = CarEnv(seed=args.seed)

    if not args.headless:
        (JSON_NV, JSON_V, JSON_LIF, WEIGHTS_FILE,
         TEMP_INIT, OUT_WEIGHTS, selected_seed, algorithm, initial_X,
         selected_episodes, evaluation_only) = launch_config_gui(
             args.seed, args.episodes
         )
        args.seed = selected_seed
        args.episodes = selected_episodes
        np.random.seed(args.seed)
        env = CarEnv(seed=args.seed)
        fname = Path(OUT_WEIGHTS).stem
        weights_name = OUT_WEIGHTS
        print(f"[car] algorithm={algorithm} seed={args.seed}  ep={args.episodes}  eta={args.eta}")
        print("[codificador] auto2d: ocho rayos direccionales, meta y velocidad")
        print(f"[config] NV={JSON_NV} | V={JSON_V} | LIF={JSON_LIF} | T={TEMP_INIT}")
        has_weights = WEIGHTS_FILE and WEIGHTS_FILE != "--- (Empezar desde cero) ---"
        fresh_training = not has_weights
        weights_path = OUT_DIR / WEIGHTS_FILE if has_weights else None
        if has_weights:
            _validate_car_weights(weights_path)
        if has_weights and evaluation_only:
            args.seed = _read_training_seed(weights_path, args.seed)
            np.random.seed(args.seed)
            env = CarEnv(seed=args.seed)
            fname = f"evaluation_{weights_path.stem}"
        encoder_seed = (
            _read_encoder_seed(weights_path, args.seed)
            if has_weights else args.seed
        )
        AgentClass = ReinforceAgent
        agent = AgentClass(
            obs_dim=env.obs_dim, n_actions=env.n_actions,
            N=N, M=M, V_read=0.05, eta=args.eta,
            encoder_mode=ENCODER_MODE,
            temp_init=(0.05 if evaluation_only else TEMP_INIT),
            temp_min=(TEMP_MIN if fresh_training else 0.05),
            temp_decay=TEMP_DECAY,
            baseline_window=200, seed=args.seed,
            encoder_seed=encoder_seed,
            json_memristor_nv=JSON_NV, json_memristor_v=JSON_V,
            json_lif=JSON_LIF, initial_X=initial_X,
        )
        if has_weights:
            agent.brain.load_weights(str(weights_path))
        agent.max_weight_update = MAX_WEIGHT_UPDATE
        policy_tracker = GreedyPolicyTracker(args.seed)
        t0 = time.time()
        state = run_gui(
            env,
            agent,
            episodes=args.episodes,
            seed=args.seed,
            policy_tracker=policy_tracker,
            evaluation_only=evaluation_only,
        )
        log = {"seed": args.seed, "initial_X": initial_X,
               "success": state["success"],
               "dist_end": state["dist_end"], "reward": state["rewards"],
               "termination_reason": state["termination_reason"],
               "episode_steps": state["episode_steps"],
               "collision_type": state["collision_types"],
               "path_length": state["path_length"],
               "path_efficiency": state["path_efficiency"],
               "steering_reversals": state["steering_reversals"],
               "steering_smoothness_penalty": (
                   state["steering_smoothness_penalty"]
               ),
               "turning_angle": state["turning_angle"],
               "unproductive_turns": state["unproductive_turns"],
               "unproductive_turn_penalty": (
                   state["unproductive_turn_penalty"]
               ),
               "training_map_seeds": state["map_seeds"],
               "training_map_seed": (
                   state["map_seeds"][0] if state["map_seeds"] else None
               ),
               "training_map_mode": (
                   "fixed_map_for_greedy_evaluation"
                   if evaluation_only else "fixed_map_per_run"
               ),
               "temperature": state["temperature"],
               "policy_evaluations": state["policy_evaluations"],
               "episodes_requested": int(args.episodes),
               "episodes_completed": len(state["success"]),
               "evaluation_only": bool(evaluation_only),
               "policy_solved": bool(policy_tracker.solved),
               "validation_regressed": bool(
                   policy_tracker.validation_regressed
               ),
               "max_weight_update": state["update_limit"],
               "steering_smoothness_reward_weight": (
                   REWARD_STEERING_SMOOTHNESS
               ),
               "unproductive_turn_progress_required": TURN_PROGRESS_REQUIRED,
               "unproductive_turn_reward_weight": REWARD_UNPRODUCTIVE_TURN,
               "win_trail": state["win_trail"], "encoder": ENCODER_MODE,
               "action_names": list(ACTION_NAMES)}
    else:
        weights_path = None
        if args.weights:
            weights_path = Path(args.weights)
            if not weights_path.is_absolute():
                weights_path = OUT_DIR / weights_path
            if weights_path.suffix.lower() != ".npy":
                raise ValueError("--weights debe apuntar a un archivo .npy")
            _validate_car_weights(weights_path)
            evaluation_only = not args.continue_training
            if evaluation_only:
                args.seed = _read_training_seed(weights_path, args.seed)
                np.random.seed(args.seed)
                env = CarEnv(seed=args.seed)
            encoder_seed = _read_encoder_seed(weights_path, args.seed)
            fname = (
                f"evaluation_{weights_path.stem}"
                if evaluation_only else
                f"car_reinforce_from_{weights_path.stem}_A8_seed{args.seed}"
            )
        else:
            evaluation_only = False
            encoder_seed = args.seed
            fname = (
                f"run_car_reinforce_seed{args.seed}_A8_"
                f"X0_{args.initial_X:g}"
            )
        weights_name = f"{fname}_X.npy"
        print(f"[car][headless] seed={args.seed}  ep={args.episodes}  "
              f"eta={args.eta}  "
              f"mode={'evaluation' if evaluation_only else 'training'}")
        print("[codificador] auto2d: ocho rayos direccionales, meta y velocidad")
        AgentClass = ReinforceAgent
        agent = AgentClass(
            obs_dim=env.obs_dim, n_actions=env.n_actions,
            N=N, M=M, V_read=0.05, eta=args.eta,
            encoder_mode=ENCODER_MODE,
            temp_init=TEMP_INIT_HD, temp_min=TEMP_MIN, temp_decay=TEMP_DECAY,
            baseline_window=200, seed=args.seed,
            encoder_seed=encoder_seed, initial_X=args.initial_X)
        if weights_path is not None:
            agent.brain.load_weights(str(weights_path))
        policy_tracker = GreedyPolicyTracker(args.seed)
        t0 = time.time()
        log = run_headless(env, agent, episodes=args.episodes, seed=args.seed,
                           ckpt_name=fname,
                           policy_tracker=policy_tracker,
                           evaluation_only=evaluation_only)
        log["initial_X"] = args.initial_X
        log["loaded_weights"] = (
            str(weights_path) if weights_path is not None else None
        )

    completed = int(log.get("episodes_completed", len(log["success"])))
    if (
        completed
        and (
            not policy_tracker.history
            or policy_tracker.history[-1]["episode"] != completed
        )
    ):
        policy_tracker.evaluate(agent, completed)
    test_tracker = GreedyPolicyTracker(
        args.seed,
        map_seed_offset=POLICY_TEST_SEED_OFFSET,
    )
    held_out_test_evaluation = (
        test_tracker.evaluate(agent, completed)
        if completed else None
    )
    log["policy_evaluations"] = policy_tracker.history
    log["final_greedy_success_rate"] = (
        policy_tracker.history[-1]["success_rate"]
        if policy_tracker.history else None
    )
    log["best_greedy_success_rate"] = (
        policy_tracker.best_success
        if policy_tracker.best_success >= 0.0 else None
    )
    log["best_greedy_episode"] = policy_tracker.best_episode
    log["best_greedy_mean_steps"] = (
        policy_tracker.best_mean_steps
        if np.isfinite(policy_tracker.best_mean_steps) else None
    )
    log["policy_solved"] = bool(policy_tracker.solved)
    log["validation_regressed"] = bool(
        policy_tracker.validation_regressed
    )
    log["policy_evaluation_config"] = {
        "episodes": POLICY_EVAL_EPISODES,
        "map_seeds": policy_tracker.map_seeds,
        "held_out_test_map_seeds": test_tracker.map_seeds,
        "interval": POLICY_EVAL_INTERVAL,
        "success_threshold": POLICY_SOLVED_SUCCESS,
        "stable_confirmations": POLICY_SOLVED_CONFIRMATIONS,
        "validation_map": "training_map_with_fixed_orientation_sequence",
        "checkpoint_selection": False,
        "base_seed": int(args.seed),
        "training_map_seed": _training_map_seed(args.seed),
    }
    log["held_out_test_evaluation"] = held_out_test_evaluation
    held_out_success_rate = (
        held_out_test_evaluation["success_rate"]
        if held_out_test_evaluation is not None else None
    )
    log["policy_qualified"] = bool(
        policy_tracker.solved
    )
    log["held_out_policy_qualified"] = bool(
        held_out_success_rate is not None
        and held_out_success_rate >= POLICY_SOLVED_SUCCESS
    )
    log["evaluation_only"] = bool(evaluation_only)
    log["episode_metrics"] = {
        "termination_reason": "goal, collision, or timeout",
        "episode_steps": "number of environment steps before termination",
        "collision_type": "wall, obstacle, wall+obstacle, or null",
        "turning_angle": "total absolute heading change in radians",
        "unproductive_turns": (
            "completed full turns with less than 0.75 map units of goal progress"
        ),
        "unproductive_turn_penalty": (
            "cumulative reward penalty for unproductive full turns"
        ),
    }
    log["training_stop_reason"] = (
        "not_training"
        if evaluation_only else (
            "episode_limit" if completed >= args.episodes else "user_stopped"
        )
    )
    log["win_trail"] = policy_tracker.last_trail

    elapsed = time.time() - t0
    algorithm = "REINFORCE"
    log["algorithm"] = algorithm
    log["encoder"] = ENCODER_MODE
    log["action_names"] = list(ACTION_NAMES)
    succ = log["success"][-200:] if log["success"] else [0.0]
    print(f"\n[resultado] exito(200)={float(np.mean(succ)):.3f}  "
          f"evaluacion greedy final="
          f"{log['final_greedy_success_rate'] if log['final_greedy_success_rate'] is not None else 0.0:.3f}  "
          f"maximo="
          f"{log['best_greedy_success_rate'] if log['best_greedy_success_rate'] is not None else 0.0:.3f}  "
          f"episodios={completed}/{args.episodes}  tiempo={elapsed / 60:.1f} min")

    (OUT_DIR / f"{fname}.json").write_text(json.dumps(log, indent=2),
                                           encoding="utf-8")
    if not evaluation_only:
        weights_path = OUT_DIR / weights_name
        agent.brain.save_weights(str(weights_path))
        _save_run_metadata(weights_path, agent, args.seed)
    plot_name = (
        f"car_evaluation_{Path(log['loaded_weights']).stem}.png"
        if evaluation_only and log.get("loaded_weights")
        else f"car_evaluation_{Path(WEIGHTS_FILE).stem}.png"
        if evaluation_only and not args.headless
        else f"car_learning_reinforce_seed{args.seed}.png"
    )
    plot_curves(
        log["success"],
        FIG_DIR / plot_name,
        policy_evaluations=log["policy_evaluations"],
    )
    if not evaluation_only:
        print(
            f"[ok] guardado en {weights_path}, {OUT_DIR / fname}.json "
            "y curva PNG"
        )
    else:
        print(
            f"[evaluación] pesos preservados sin cambios; "
            f"resultado={OUT_DIR / fname}.json"
        )


if __name__ == "__main__":
    main()
