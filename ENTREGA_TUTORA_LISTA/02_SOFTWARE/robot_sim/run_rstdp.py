"""Entrenamiento CartPole con R-STDP modulada por recompensa paso a paso.

Este runner es deliberadamente independiente de run_robot.py. Usa el mismo
encoder, CrossbarBrain (incluido el memristor volátil), LIF y decoder compartidos;
cada recompensa por paso modula la elegibilidad pre/post; X se programa al cerrar el episodio.

Uso desde la raíz del repositorio: python robot_sim/run_rstdp.py
Sin seleccionar un .npy, cada ejecución empieza con X0; los pesos finales
se guardan en outputs para cargarlos explícitamente en otra ejecución.
"""
from __future__ import annotations

import json
import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
from time import monotonic
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from neurobot.env_wrapper import CartPoleWrapper
from neurobot.rstdp_cartpole_agent import NeuromorphicAgent


# Mantener las dimensiones y opciones físicas de la configuración habitual.
N, M = 24, 12
V_READ = 0.05
ACTION_SEED = 0
DEFAULT_EPISODES = 1000
DEFAULT_TEMPERATURE = 1.0
R_STDP_ETA = 0.2
EARLY_STOP_THRESHOLD = 430.0
SPEED = 100  # pasos globales entre refrescos; +/- ajusta este intervalo
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
CONFIG_DIR = ROOT.parent / "neuromorphic_lab" / "configs"


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento R-STDP CartPole modulado por recompensa en cada paso."
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Episodios máximos de entrenamiento antes del límite.")
    parser.add_argument("--crossbar-seed", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--encoder-seed", type=int, default=0)
    parser.add_argument("--initial-x", type=float, choices=(0.1, 0.9), default=None,
                        help="Fuerza X0 aunque se seleccione otro valor en la ventana.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes debe ser positivo")
    return args


CLI = parse_cli()


def choose_config(cli: argparse.Namespace) -> dict:
    """Pantalla separada con los mismos parámetros de dispositivo/semillas."""
    configs = sorted(path.name for path in CONFIG_DIR.glob("*.json"))
    if not configs:
        raise FileNotFoundError(f"No hay perfiles JSON en {CONFIG_DIR}")
    volatile_configs = []
    for name in configs:
        profile = json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))
        if profile.get("enable_volatile", False) or "volatil" in str(
            profile.get("model_name", "")
        ).lower():
            volatile_configs.append(name)
    if not volatile_configs:
        raise FileNotFoundError(
            f"No hay perfiles con enable_volatile=true en {CONFIG_DIR}"
        )
    weights = ["--- (Empezar desde cero) ---"] + sorted(
        path.name for path in OUTPUT_DIR.glob("*.npy")
    )

    root = tk.Tk()
    root.title("Run R-STDP — entrenamiento")
    root.geometry("680x700")
    root.minsize(620, 600)
    root.columnconfigure(0, weight=1)
    root.eval("tk::PlaceWindow . center")

    tk.Label(
        root,
        text="R-STDP: elegibilidad pre/post modulada por recompensa en cada paso",
        font=("Arial", 12, "bold"),
    ).pack(pady=(14, 4))
    tk.Label(
        root,
        text="Cada recompensa modula la elegibilidad; X se actualiza al cerrar el episodio.",
    ).pack(pady=(0, 10))

    frame = ttk.Frame(root, padding=14)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    def add_entry(row: int, label: str, value: str):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
        entry = ttk.Entry(frame)
        entry.insert(0, value)
        entry.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=5)
        return entry

    crossbar_seed = add_entry(0, "Semilla del crossbar:", str(cli.crossbar_seed))
    env_seed = add_entry(1, "Semilla del entorno:", str(cli.env_seed))
    encoder_seed = add_entry(2, "Semilla del codificador:", str(cli.encoder_seed))
    episodes = add_entry(3, "Episodios:", str(cli.episodes))

    def add_combo(row: int, label: str, values: list[str], default: str):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
        combo = ttk.Combobox(frame, values=values, state="readonly")
        combo.set(default if default in values else values[0])
        combo.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=5)
        return combo

    nv = add_combo(4, "Memristor no volátil:", configs,
                   "strukov_ideal.json" if "strukov_ideal.json" in configs else configs[0])
    volatile = add_combo(
        5, "Memristor volátil (entrada LIF):", volatile_configs,
        "memristor_volatile.json"
        if "memristor_volatile.json" in volatile_configs else volatile_configs[0],
    )
    lif = add_combo(6, "Perfil LIF:", configs,
                    "lif_config.json" if "lif_config.json" in configs else configs[0])
    weights = add_combo(
        7, "Pesos previos (.npy):", weights, "--- (Empezar desde cero) ---"
    )
    temp = add_combo(8, "Temperatura del decoder:", ["1.0", "0.05"], "1.0")
    x0_default = "0.9" if cli.initial_x is None else str(cli.initial_x)
    x0 = add_combo(9, "X₀ (si empiezas desde cero):", ["0.1", "0.9"], x0_default)

    d2d = tk.BooleanVar(value=True)
    c2c = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="D2D (sigma=8%)", variable=d2d).grid(
        row=10, column=0, sticky="w", pady=6
    )
    ttk.Checkbutton(frame, text="C2C (sigma=5%)", variable=c2c).grid(
        row=10, column=1, sticky="w", padx=(12, 0), pady=6
    )
    result: dict = {}

    def start():
        try:
            seeds = [int(crossbar_seed.get()), int(env_seed.get()), int(encoder_seed.get())]
            n_episodes = int(episodes.get())
            if min(seeds) < 0 or n_episodes < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Configuración inválida",
                "Las semillas deben ser enteros ≥ 0 y los episodios un entero > 0.",
                parent=root,
            )
            return
        result.update(
            crossbar_seed=seeds[0], env_seed=seeds[1], encoder_seed=seeds[2],
            episodes=n_episodes, nv=nv.get(), volatile=volatile.get(), lif=lif.get(),
            weights=weights.get(), temperature=float(temp.get()),
            initial_X=(float(cli.initial_x) if cli.initial_x is not None else float(x0.get())),
            d2d=bool(d2d.get()), c2c=bool(c2c.get()),
        )
        root.destroy()

    ttk.Button(root, text="Iniciar entrenamiento R-STDP", command=start).pack(pady=12)
    root.mainloop()
    if not result:
        raise SystemExit("Configuración cancelada.")
    return result


cfg = choose_config(CLI)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
stem = (
    f"run_rstdp_train_crossbar{cfg['crossbar_seed']}_env{cfg['env_seed']}_"
    f"encoder{cfg['encoder_seed']}_{cfg['episodes']}ep_{timestamp}"
)

env = CartPoleWrapper(seed=cfg["env_seed"], death_penalty=50.0)
environment_dt = float(getattr(env.env.unwrapped, "tau", 0.015))
neural_substeps = 3
neural_dt = environment_dt / neural_substeps
agent = NeuromorphicAgent(
    obs_dim=env.obs_dim,
    n_actions=env.n_actions,
    N=N,
    M=M,
    V_read=V_READ,
    eta=R_STDP_ETA,
    dt=neural_dt,
    steps_per_action=neural_substeps,
    k_wta=3,
    wta_temperature=1.0,
    a_plus=0.01,
    a_minus=0.01,
    reward_baseline_window=200,
    return_baseline_alpha=0.05,
    max_abs_delta_x=0.02,
    temp_init=cfg["temperature"],
    temp_min=min(cfg["temperature"], 0.05),
    temp_decay=0.995,
    baseline_window=200,
    initial_X=cfg["initial_X"],
    seed=ACTION_SEED,
    crossbar_seed=cfg["crossbar_seed"],
    encoder_seed=cfg["encoder_seed"],
    action_seed=ACTION_SEED,
    json_memristor_nv=cfg["nv"],
    json_memristor_v=cfg["volatile"],
    json_lif=cfg["lif"],
    d2d_enabled=cfg["d2d"],
    c2c_enabled=cfg["c2c"],
    d2d_sigma=0.08,
    c2c_sigma=0.05,
    c2c_theta=1.0,
    c2c_dt=0.02,
    encoder_mode="random",
)
if cfg["weights"] != "--- (Empezar desde cero) ---":
    agent.brain.load_weights(str(OUTPUT_DIR / cfg["weights"]))

volatile_device = agent.brain.volatile_devices[0]
volatile_threshold = float(getattr(volatile_device.model_config, "v_th", 0.1))
volatile_decay = next(
    modifier for modifier in volatile_device.modifiers
    if hasattr(modifier, "tau_relax") and hasattr(modifier, "x0_override")
)
X_start = agent.brain.X.copy()
G_start = agent.brain.G.copy()
volatile_start = np.asarray([device.x for device in agent.brain.volatile_devices])
print(
    f"[run_rstdp] entrenamiento R-STDP | N={N} M={M} "
    f"semillas crossbar/env/encoder={cfg['crossbar_seed']}/{cfg['env_seed']}/"
    f"{cfg['encoder_seed']} | temp={cfg['temperature']}"
)
print(
    f"[aprendizaje] R-STDP: retorno futuro modula pares pre/post; "
    f"LIF integra {environment_dt:.3f}s por acción en {neural_substeps} subpasos."
)
print("[controles] espacio pausa | q/Esc salir | +/- cambia la frecuencia de dibujo")
started_at = monotonic()

# Cuatro paneles, siguiendo la disposición familiar de Robot Sim.
plt.ion()
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.canvas.manager.set_window_title("Run R-STDP — entrenamiento")
ax_scene, ax_X = axes[0]
ax_reward, ax_diag = axes[1]
ax_scene.set_xlim(-2.6, 2.6)
ax_scene.set_ylim(-0.6, 1.6)
ax_scene.set_aspect("equal")
ax_scene.set_title("Robot (CartPole)")
ax_scene.grid(alpha=0.2)
cart = Rectangle((0, 0), 0.4, 0.2, color="tab:blue")
pole, = ax_scene.plot([], [], "o-", lw=3, color="tab:orange", markersize=6)
ax_scene.add_patch(cart)
im_X = ax_X.imshow(agent.brain.X, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax_X.set_title("Matriz X (memristores; actualización R-STDP)")
ax_X.set_xlabel("Columna")
ax_X.set_ylabel("Fila")
fig.colorbar(im_X, ax=ax_X)
line_reward, = ax_reward.plot([], [], color="tab:blue", lw=1.4)
ax_reward.set_title("Retorno por episodio")
ax_reward.set_xlabel("Episodio")
ax_reward.set_ylabel("Retorno")
ax_reward.grid(alpha=0.25)
ax_diag.set_title("Eventos recientes por columna (últimas 500 acciones)")
ax_diag.set_xlabel("Columna / acción")
ax_diag.set_ylabel("Conteo en ventana")
bar_width = 0.25
bar_x = np.arange(M)
bars_action = ax_diag.bar(bar_x - bar_width, np.zeros(M), bar_width, alpha=0.8, label="acciones")
bars_spike = ax_diag.bar(bar_x, np.zeros(M), bar_width, alpha=0.8, label="spikes LIF")
bars_wta = ax_diag.bar(bar_x + bar_width, np.zeros(M), bar_width, alpha=0.8, label="ganadores WTA")
ax_diag.set_xticks(bar_x)
ax_diag.legend(fontsize=8)
info_text = fig.text(0.02, 0.98, "", ha="left", va="top", family="monospace", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.canvas.draw()
plt.show(block=False)
fig.canvas.flush_events()
try:
    fig.canvas.manager.window.raise_()
    fig.canvas.manager.window.activateWindow()
except (AttributeError, RuntimeError):
    # Some Matplotlib backends do not expose a native window handle.
    pass
plt.pause(0.05)

episode_rows: list[dict] = []
rewards: list[float] = []
action_counts = np.zeros(M, dtype=np.int64)
spike_counts = np.zeros(M, dtype=np.int64)
wta_counts = np.zeros(M, dtype=np.int64)
total_steps = 0
frozen_update_episodes = 0
exploration_restarts = 0
volatile_voltage_abs_sum = 0.0
volatile_voltage_samples = 0
volatile_voltage_abs_max = 0.0
volatile_voltage_over_threshold = 0
selected_post_events_total = 0
last_update = {}
partial_episode = None
state = {"stop": False, "paused": False, "speed": SPEED}
activity_window = deque(maxlen=500)


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

try:
    for episode in range(cfg["episodes"]):
        if state["stop"] or not plt.fignum_exists(fig.number):
            break
        obs = env.reset()
        ep_return = 0.0
        steps = 0
        ep_actions = np.zeros(M, dtype=np.int64)
        ep_spikes = np.zeros(M, dtype=np.int64)
        ep_winners = np.zeros(M, dtype=np.int64)
        ep_selected_post_events = 0
        ep_v_abs_sum = 0.0
        ep_v_samples = 0
        ep_v_abs_max = 0.0
        ep_v_over_threshold = 0
        volatile_ep_start = np.asarray([device.x for device in agent.brain.volatile_devices])
        done = False

        while not done and not state["stop"]:
            if state["paused"]:
                fig.canvas.draw_idle()
                plt.pause(0.05)
                continue
            action = agent.act(obs)
            record = agent.last_step()
            v_volatile = np.asarray(agent.brain.last_volatile_voltage, dtype=float)
            v_abs = np.abs(v_volatile)
            ep_v_abs_sum += float(v_abs.sum())
            ep_v_samples += int(v_abs.size)
            ep_v_abs_max = max(ep_v_abs_max, float(v_abs.max(initial=0.0)))
            ep_v_over_threshold += int(np.count_nonzero(v_abs > volatile_threshold))
            volatile_voltage_abs_sum += float(v_abs.sum())
            volatile_voltage_samples += int(v_abs.size)
            volatile_voltage_abs_max = max(
                volatile_voltage_abs_max, float(v_abs.max(initial=0.0))
            )
            volatile_voltage_over_threshold += int(np.count_nonzero(v_abs > volatile_threshold))
            col = int(record["col"])
            ep_actions[col] += 1
            action_counts[col] += 1
            selected_post_events_total += int(
                np.count_nonzero(record["rstdp_post_credit"])
            )
            ep_selected_post_events += int(
                np.count_nonzero(record["rstdp_post_credit"])
            )
            spike_events = np.asarray(
                record["spikes_by_substep"], dtype=np.int64
            ).sum(axis=0)
            winner_events = np.asarray(
                record["wta_winners_by_substep"], dtype=np.int64
            ).sum(axis=0)
            ep_spikes += spike_events
            ep_winners += winner_events
            activity_window.append((
                col,
                spike_events,
                winner_events,
            ))
            obs, reward, done, info = env.step(action)
            step_reward = agent.observe_reward(reward)
            ep_return += float(reward)
            steps += 1
            total_steps += 1

            raw = info.get("obs_raw")
            if raw is not None:
                x, theta = float(raw[0]), float(raw[2])
                cart.set_xy((x - 0.2, -0.1))
                pole.set_data([x, x + 0.5 * np.sin(theta)], [0.0, 0.5 * np.cos(theta)])

            spike_counts += spike_events
            wta_counts += winner_events
            im_X.set_data(agent.brain.X)
            info_text.set_text(
                f"Ep {episode + 1:4d} | paso {steps:3d} | R_ep {ep_return:7.1f} | "
                f"temp {agent.current_temperature():.3f} | "
                f"spikes {int(spike_events.sum()):2d} | "
                f"WTA {int(winner_events.sum()):2d} | "
                f"pre {int(np.count_nonzero(record['pre'])):2d} | "
                f"post {int(record['col']):2d} | "
                f"r {step_reward:+.2f} | "
                f"x_vol {np.mean([d.x for d in agent.brain.volatile_devices]):.3f} | "
                f"V_vol_max {v_abs.max(initial=0.0):.3f} V | "
                f"dX_ep {last_update.get('actual_delta_x_abs_mean', 0.0):.2e} | "
                f"{'PAUSA' if state['paused'] else 'entrenando R-STDP'}"
            )
            if total_steps % state["speed"] == 0:
                recent_actions = np.zeros(M, dtype=np.int64)
                recent_spikes = np.zeros(M, dtype=np.int64)
                recent_winners = np.zeros(M, dtype=np.int64)
                for recent_col, recent_spike_mask, recent_wta_mask in activity_window:
                    recent_actions[recent_col] += 1
                    recent_spikes += recent_spike_mask
                    recent_winners += recent_wta_mask
                for bars, values in zip(
                    (bars_action, bars_spike, bars_wta),
                    (recent_actions, recent_spikes, recent_winners),
                ):
                    for bar, value in zip(bars, values):
                        bar.set_height(int(value))
                line_reward.set_data(np.arange(len(rewards)), rewards)
                ax_reward.set_xlim(0, max(10, len(rewards) + 1))
                ax_reward.relim()
                ax_reward.autoscale_view(scalex=False, scaley=True)
                ax_diag.set_ylim(0, max(1, int(max(
                    recent_actions.max(), recent_spikes.max(), recent_winners.max()
                ) * 1.1)))
                fig.canvas.draw_idle()
                plt.pause(0.001)

        if state["stop"] and not done:
            if steps:
                partial_episode = {
                    "episode": episode + 1,
                    "steps": int(steps),
                    "return": float(ep_return),
                    "interrupted": True,
                    "learning_applied": False,
                    "terminated": bool(info.get("terminated", False)),
                    "truncated": bool(info.get("truncated", False)),
                    "action_counts_by_column": ep_actions.tolist(),
                    "lif_spikes_by_column": ep_spikes.tolist(),
                    "wta_winners_by_column": ep_winners.tolist(),
                    "selected_post_events": ep_selected_post_events,
                }
            break

        learning_metrics = agent.finish_episode(
            ep_return, freeze_update_above=EARLY_STOP_THRESHOLD
        )
        frozen_update_episodes += int(learning_metrics["update_frozen"])
        agent.end_episode()
        if (
            agent.reward_history
            and ep_return - learning_metrics["baseline_before"] < -30.0
            and agent.current_temperature() < 0.10
        ):
            agent.decoder.set_temperature(0.10)
            exploration_restarts += 1
        last_update = learning_metrics
        rewards.append(float(ep_return))
        im_X.set_data(agent.brain.X)
        # Per-episode counts are stored compactly; no per-step spike arrays.
        episode_rows.append({
            "episode": episode + 1,
            "return": float(ep_return),
            "steps": int(steps),
            "learning": learning_metrics,
            "X_mean_after_update": float(agent.brain.X.mean()),
            "action_counts_by_column": ep_actions.tolist(),
            "lif_spikes_by_column": ep_spikes.tolist(),
            "wta_winners_by_column": ep_winners.tolist(),
            "volatile_x_mean_start": float(np.mean(volatile_ep_start)),
            "volatile_x_mean_end": float(np.mean([device.x for device in agent.brain.volatile_devices])),
            "volatile_voltage_abs_mean": float(ep_v_abs_sum / max(ep_v_samples, 1)),
            "volatile_voltage_abs_max": float(ep_v_abs_max),
            "volatile_voltage_samples_above_threshold": int(ep_v_over_threshold),
            "terminated": bool(info.get("terminated", False)),
            "truncated": bool(info.get("truncated", False)),
        })

        # finish_episode reinicia trazas/LIF; X actualizada y volátil siguen vivas.
        line_reward.set_data(np.arange(len(rewards)), rewards)
        ax_reward.set_xlim(0, max(10, len(rewards) + 1))
        ax_reward.relim()
        ax_reward.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if len(episode_rows) % 10 == 0:
            elapsed = monotonic() - started_at
            print(
                f"[progreso] episodios={len(episode_rows)} "
                f"pasos={total_steps} "
                f"tiempo={elapsed:.1f}s; X_mean={agent.brain.X.mean():.4f}; "
                f"dX_ep={learning_metrics['actual_delta_x_abs_mean']:.2e}; "
                f"temp={agent.current_temperature():.3f}; "
                f"R20={np.mean(rewards[-20:]):.1f}"
            )


except KeyboardInterrupt:
    state["stop"] = True
    print("\n[interrupción] guardando el registro parcial.")
finally:
    env.close()
    x_changed = not np.array_equal(agent.brain.X, X_start)

    volatile_end = np.asarray([device.x for device in agent.brain.volatile_devices])
    report = {
        "experiment": "rstdp_return_advantage_modulated_cartpole",
        "description": "Pre/post action eligibility snapshots are modulated by discounted return advantages against a per-time-step baseline, then applied as bounded X updates.",
        "episodes_requested": int(cfg["episodes"]),
        "episodes_completed": len(episode_rows),
        "total_environment_steps": total_steps,
        "total_selected_post_events": selected_post_events_total,
        "aggregate_activity_includes_partial_episode": partial_episode is not None,
        "partial_episode": partial_episode,
        "elapsed_seconds": float(monotonic() - started_at),
        "config": {
            "N": N, "M": M, "V_read": V_READ,
            "crossbar_seed": cfg["crossbar_seed"], "environment_seed": cfg["env_seed"],
            "encoder_seed": cfg["encoder_seed"], "action_seed": ACTION_SEED,
            "encoder": "random", "temperature_initial": cfg["temperature"],
            "temperature_min": agent.temp_min, "temperature_decay": agent.temp_decay,
            "early_stop_threshold": EARLY_STOP_THRESHOLD,
            "temperature_end": agent.current_temperature(),
            "exploration_restarts": exploration_restarts,
            "initial_X": cfg["initial_X"], "weights_source": cfg["weights"],
            "memristor_nonvolatile": cfg["nv"], "memristor_volatile": cfg["volatile"],
            "lif": cfg["lif"], "d2d_enabled": cfg["d2d"], "c2c_enabled": cfg["c2c"],
            "dt": agent.dt, "steps_per_action": agent.steps_per_action,
            "environment_dt": environment_dt,
            "k_wta": agent.k_wta, "wta_temperature": agent.wta_temperature,
        },
        "learning": {
            "rule": "reward_modulated_pre_post_action_eligibility",
            "reward_signal": "discounted return-to-go at each action minus a running baseline for that time step; baseline updates after the episode",
            "post_credit": "Decoded action event centered by its softmax probabilities; competing columns receive negative eligibility.",
            "stdp_polarity": "active pre -> selected post: +A_plus; active pre -> competing posts: -A_minus",
            "updates_applied_per_completed_episode": "always when reward-modulated eligibility is nonzero",
            "X_changed": x_changed,
            "episodes_with_nonzero_update": int(sum(
                row["learning"]["actual_delta_x_abs_max"] > 0.0
                for row in episode_rows
            )),
            "episodes_with_frozen_update": frozen_update_episodes,
            "episode_return_baseline_diagnostic_only": float(agent.baseline),
            "return_baseline_by_step": {str(k): float(v) for k, v in agent._return_baseline_by_step.items()},
            "parameters": {
                "a_plus": agent.a_plus, "a_minus": agent.a_minus,
                "eta": agent.eta, "gamma": agent.gamma,
                "return_baseline_alpha": agent.return_baseline_alpha,
                "max_abs_delta_x": agent.max_abs_delta_x,
                "freeze_update_above": EARLY_STOP_THRESHOLD,
            },
        },
        "X_start": {"mean": float(X_start.mean()), "min": float(X_start.min()), "max": float(X_start.max())},
        "X_end": {"mean": float(agent.brain.X.mean()), "min": float(agent.brain.X.min()), "max": float(agent.brain.X.max())},
        "X_mean_by_column_start": X_start.mean(axis=0).tolist(),
        "X_mean_by_column_end": agent.brain.X.mean(axis=0).tolist(),
        "X_total_delta_by_column": (agent.brain.X - X_start).mean(axis=0).tolist(),
        "G_start_siemens": {"mean": float(G_start.mean()), "min": float(G_start.min()), "max": float(G_start.max())},
        "G_end_siemens": {"mean": float(agent.brain.G.mean()), "min": float(agent.brain.G.min()), "max": float(agent.brain.G.max())},
        "volatile_x_start_mean": float(volatile_start.mean()),
        "volatile_x_end_mean": float(volatile_end.mean()),
        "volatile_model": {
            "mu_v": float(volatile_device.model_config.mu_v),
            "v_th_v": volatile_threshold,
            "tau_relax_s": float(volatile_decay.tau_relax),
            "x_eq": float(volatile_decay.x0_override),
            "r_on_ohm": float(volatile_device.electrical.r_on),
            "r_off_ohm": float(volatile_device.electrical.r_off),
        },
        "volatile_voltage": {
            "compliance_source": "abs(I_col) * R_sense per output column",
            "R_sense_ohm": float(agent.brain.R_sense),
            "switching_threshold_v": volatile_threshold,
            "absolute_mean_v": float(volatile_voltage_abs_sum / max(volatile_voltage_samples, 1)),
            "absolute_max_v": float(volatile_voltage_abs_max),
            "samples_above_switching_threshold": int(volatile_voltage_over_threshold),
        },
        "total_action_counts_by_column": action_counts.tolist(),
        "total_lif_spikes_by_column": spike_counts.tolist(),
        "total_wta_winners_by_column": wta_counts.tolist(),
        "neural_activity_per_environment_action": {
            "lif_spikes": float(spike_counts.sum() / max(total_steps, 1)),
            "wta_winner_events": float(wta_counts.sum() / max(total_steps, 1)),
            "selected_post_events": float(
                selected_post_events_total / max(total_steps, 1)
            ),
        },
        "episodes": episode_rows,
    }
    weights_path = OUTPUT_DIR / f"{stem}.npy"
    agent.brain.save_weights(str(weights_path))
    report["weights_file"] = weights_path.name
    json_path = OUTPUT_DIR / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[run_rstdp] episodios completados: {len(episode_rows)}")
    print(f"[pesos] guardados para uso opcional en otra corrida: {weights_path}")
    print(f"[verificación] X cambió: {x_changed}; X_mean {X_start.mean():.6f} -> {agent.brain.X.mean():.6f}; "
          f"episodios con actualización: {report['learning']['episodes_with_nonzero_update']}")
    print(f"[registro] {json_path}")
    print("[registro] entrenamiento R-STDP terminado; revisa learning y episodios[].learning en el JSON.")
