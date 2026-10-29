"""Entrena con REINFORCE en el gridworld 2D con obstaculos (3.3.7).

Reusa el agente REINFORCE de CartPole/Vision. Solo cambian el entorno y las
dimensiones del crossbar (N=16 entradas, M=4 acciones).

Guarda: outputs/run_obstacle_seed{N}.json (log completo),
        outputs/run_obstacle_seed{N}_X.npy (matriz final),
        outputs/obstacle_seed{N}_paths.json (layout + trayectorias antes/despues).

Uso:  python run_obstacle.py              (abre GUI en vivo — modo normal)
      python run_obstacle.py --headless   (sin ventana, para automatización)

La configuración incluye semilla, algoritmo, X0 y parámetros físicos.
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from neurobot.reinforce_agent import NeuromorphicAgent as ReinforceAgent

from obstacle_env import ObstacleGridworld

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def run_episodes(agent, env, n_episodes, verbose=True, on_step=None):
    log = {"episode": [], "reward": [], "success": [], "steps": [],
           "X_mean": [], "baseline": [], "temperature": []}
    early_path = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        step_rewards = []
        while not done:
            a = agent.act(obs)
            obs, r, done, _ = env.step(a)
            total_r += r
            step_rewards.append(float(r))
        if ep == 0:
            early_path = list(env.path)
        arrived = bool(env.agent == env.goal)

        # Both policies receive the same per-decision discounted returns.
        # REINFORCE uses discounted returns and the sampled policy scores.
        agent.learn_from_rewards(step_rewards)
        agent.end_episode()

        log["episode"].append(ep)
        log["reward"].append(total_r)
        log["success"].append(1.0 if arrived else 0.0)
        log["steps"].append(env.steps)
        log["X_mean"].append(float(agent.brain.X.mean()))
        log["baseline"].append(agent.baseline)
        log["temperature"].append(agent.decoder.temperature)

        if verbose and (ep % max(1, n_episodes // 8) == 0 or ep == n_episodes - 1):
            s = log["success"][-100:] if ep >= 100 else log["success"]
            succ = float(np.mean(s))
            print(f"[ep {ep:4d}] exito(100)={succ:.2f}  "
                  f"R={total_r:+6.1f}  X_mean={agent.brain.X.mean():.4f}")

        if on_step is not None:
            if not on_step(ep, arrived, env, agent, log):
                # Ventana cerrada por el usuario: se omite el rollout final
                return log, early_path, list(env.path), bool(env.agent == env.goal)

    # Rollout deterministico con la politica aprendida
    obs = env.reset()
    done = False
    while not done:
        a = agent.act_deterministic(obs)
        obs, r, done, _ = env.step(a)
    learned_path = list(env.path)
    final_success = bool(env.agent == env.goal)
    return log, early_path, learned_path, final_success


def launch_config_gui(seed_default=0):
    """Diálogo con semilla, algoritmo, estado X0 y parámetros físicos.

    En Navegación se configuran semilla, algoritmo, X0 y parámetros físicos.
    """
    import tkinter as tk
    from tkinter import ttk

    config_dir = Path(__file__).resolve().parents[1] / "neuromorphic_lab" / "configs"
    if not config_dir.exists():
        config_dir = Path(__file__).resolve().parent
    jsons = [p.name for p in config_dir.glob("*.json")]
    if not jsons:
        jsons = ["last_session.json", "lif_config.json", "memristor_volatile.json"]

    npys = [p.name for p in OUT_DIR.glob("*.npy")]
    npys.insert(0, "--- (Empezar desde cero) ---")

    out = {"nv": "strukov_ideal.json" if "strukov_ideal.json" in jsons else jsons[0],
           "v": ("memristor_volatile.json"
                 if "memristor_volatile.json" in jsons else jsons[0]),
           "lif": "lif_config.json" if "lif_config.json" in jsons else jsons[0],
           "weights": npys[0], "temp": 1.0,
           "out_name": f"run_obstacle_seed{seed_default}.npy",
           "seed": seed_default, "algorithm": "REINFORCE", "initial_X": 0.9}

    root = tk.Tk()
    root.title("Robot Sim - Configuración (Navegación)")
    root.geometry("560x540")
    root.minsize(560, 460)
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

    # 1) Memristor No Volátil [APLICA: sustrato físico del crossbar]
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

    # 3) Neurona LIF [APLICA: la decisión sale de spikes vía WTA]
    tk.Label(frame, text="Neurona LIF:").grid(row=4, column=0, sticky="w", pady=5)
    cbo_lif = ttk.Combobox(frame, values=jsons, state="readonly", width=30)
    cbo_lif.set(out["lif"])
    cbo_lif.grid(row=4, column=1, pady=5, padx=10, sticky="ew")

    # 4) Pesos Previos [APLICA: continúa un entrenamiento previo]
    tk.Label(frame, text="Pesos Previos (.npy):").grid(row=5, column=0, sticky="w", pady=5)
    cbo_w = ttk.Combobox(frame, values=npys, state="readonly", width=30)
    cbo_w.set(npys[0])
    cbo_w.grid(row=5, column=1, pady=5, padx=10, sticky="ew")

    # 5) Temperatura Inicial [APLICA: RL, muestreo exploratorio del decoder]
    tk.Label(frame, text="Temperatura Inicial:").grid(row=6, column=0, sticky="w", pady=5)
    temps = ["1.0 (Aprender desde cero)", "0.05 (Modo Experto)"]
    cbo_t = ttk.Combobox(frame, values=temps, state="readonly", width=30)
    cbo_t.set(temps[0])
    cbo_t.grid(row=6, column=1, pady=5, padx=10, sticky="ew")

    # 6) Guardar como [APLICA: nombre del archivo de salida]
    tk.Label(frame, text="Guardar al terminar como:").grid(row=7, column=0, sticky="w", pady=5)
    entry_out = ttk.Entry(frame, width=33)
    entry_out.insert(0, f"run_obstacle_seed{seed_default}.npy")
    entry_out.grid(row=7, column=1, pady=5, padx=10, sticky="ew")

    tk.Label(frame, text="X₀ inicial (si empiezas desde cero):").grid(
        row=8, column=0, sticky="w", pady=5)
    cbo_x0 = ttk.Combobox(frame, values=["0.1", "0.9"],
                          state="readonly", width=30)
    cbo_x0.set("0.9")
    cbo_x0.grid(row=8, column=1, pady=5, padx=10, sticky="ew")

    def on_start():
        try:
            selected_seed = int(entry_seed.get())
            if selected_seed < 0:
                raise ValueError
        except ValueError:
            from tkinter import messagebox
            messagebox.showerror("Semilla inválida", "Use un entero no negativo.", parent=root)
            return
        out["seed"] = selected_seed
        out["algorithm"] = "REINFORCE"
        out["nv"] = cbo_nv.get()
        out["v"] = cbo_v.get()
        out["lif"] = cbo_lif.get()
        out["weights"] = cbo_w.get()
        out["temp"] = 1.0 if "1.0" in cbo_t.get() else 0.05
        out["initial_X"] = float(cbo_x0.get())
        name = entry_out.get().strip()
        if name == f"run_obstacle_seed{seed_default}.npy":
            name = f"run_obstacle_reinforce_seed{selected_seed}.npy"
        if name == f"run_obstacle_{out['algorithm'].lower()}_seed{selected_seed}.npy":
            name = name[:-4] + f"_X0_{out['initial_X']:g}.npy"
        if not name.endswith(".npy"):
            name += ".npy"
        out["out_name"] = name
        root.destroy()

    btn = tk.Button(root, text="Entrenar", command=on_start, bg="#4CAF50",
                    fg="white", font=("Arial", 10, "bold"), padx=15, pady=5)
    btn.pack(pady=10)

    root.mainloop()
    return (out["nv"], out["v"], out["lif"], out["weights"], out["temp"],
            out["out_name"], out["seed"], out["algorithm"], out["initial_X"])


def run_gui(agent, env, n_episodes):
    """Loop de entrenamiento con ventana en vivo de 4 paneles."""
    state = {"paused": False, "stop": False, "speed": 1}

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.canvas.manager.set_window_title(
        "Neuromorphic Navigation 10x10 — Crossbar 16x4")

    ax_grid, ax_X = axes[0, 0], axes[0, 1]
    ax_succ, ax_rew = axes[1, 0], axes[1, 1]

    # Panel 1: grid del entorno
    env.draw(ax_grid, path=[], title="Entrenamiento...")
    ax_grid.add_patch(Rectangle((-0.5, -0.5), env.size, env.size, fill=False,
                                linewidth=4, edgecolor="tab:gray"))

    # Keep a fixed 0-1 scale. Auto-scaling a nearly uniform X0 made both
    # starting states look alike and concealed that X0=0.1 and 0.9 differ.
    im_X = ax_X.imshow(agent.brain.X, aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=1.0)
    ax_X.set_title("Estado X del crossbar (0 = G alta, 1 = G baja)")
    ax_X.set_xlabel("Columna (acción)")
    ax_X.set_ylabel("Fila (sensor)")
    fig.colorbar(im_X, ax=ax_X)

    # Panel 3: curva de éxito (media móvil 50)
    line_succ, = ax_succ.plot([], [], color="tab:blue", lw=1.6)
    ax_succ.axhline(0.042, ls="--", color="gray", label="azar (~4.2%)")
    ax_succ.set_xlim(0, max(10, n_episodes))
    ax_succ.set_ylim(0, 1.05)
    ax_succ.set_xlabel("Episodio")
    ax_succ.set_ylabel("Éxito (media móvil 50)")
    ax_succ.set_title("Curva de aprendizaje")
    ax_succ.legend(fontsize=8, loc="lower right")
    ax_succ.grid(alpha=0.25)

    # Panel 4: recompensa por episodio
    line_rew, = ax_rew.plot([], [], color="tab:purple", lw=1.5)
    ax_rew.axhline(0.0, color="gray", ls="--", lw=1)
    ax_rew.set_xlim(0, max(10, n_episodes))
    ax_rew.set_xlabel("Episodio")
    ax_rew.set_ylabel("Recompensa")
    ax_rew.set_title("Recompensa por episodio")
    ax_rew.grid(alpha=0.25)

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
            state["speed"] = min(100, state["speed"] * 2)
            print(f"[speed] running x{state['speed']}")
        elif event.key in ("-", "_", "minus"):
            state["speed"] = max(1, state["speed"] // 2)
            print(f"[speed] running x{state['speed']}")
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
                initialvalue="obstacle_weights.npy", parent=root)
            root.destroy()
            if name:
                if not name.endswith(".npy"):
                    name += ".npy"
                agent.brain.save_weights(str(OUT_DIR / name))
                print(f"[ok] Pesos guardados: {name}")
            state["paused"] = was_paused

    def on_close(_event):
        state["stop"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    def on_step(ep, arrived, env_, agent_, log):
        if state["stop"] or not plt.fignum_exists(fig.number):
            return False
        while state["paused"] and plt.fignum_exists(fig.number):
            plt.pause(0.05)
        if state["stop"] or not plt.fignum_exists(fig.number):
            return False
        if ep % state["speed"] != 0 and ep != n_episodes - 1:
            fig.canvas.flush_events()
            return True
        if ep % state["speed"] == 0 or ep == n_episodes - 1:
            # Panel 1: grid + trayectoria + feedback verde/rojo
            ax_grid.cla()
            env_.draw(ax_grid, path=list(env_.path), title="")
            color = "tab:green" if arrived else "tab:red"
            ax_grid.add_patch(Rectangle((-0.5, -0.5), env_.size, env_.size,
                                        fill=False, linewidth=4, edgecolor=color))
            ax_grid.set_xlim(-0.9, env_.size - 0.1)
            ax_grid.set_ylim(env_.size - 0.1, -0.9)
            ax_grid.set_facecolor("#d4f5d4" if arrived else "#f5d4d4")
            ax_grid.set_title(
                f"Episodio: {'ÉXITO' if arrived else 'FRACASO'} | pasos: {env_.steps}",
                color=color, fontweight="bold", fontsize=10)
            # Panel 2: estado X con escala fija 0-1
            X = agent_.brain.X
            im_X.set_data(X)
            # Panel 3: éxito móvil 50
            succ = np.asarray(log["success"], dtype=float)
            if len(succ) >= 50:
                smooth = np.convolve(succ, np.ones(50) / 50, mode="valid")
                line_succ.set_data(np.arange(len(smooth)), smooth)
            # Panel 4: recompensa
            line_rew.set_data(np.arange(len(log["reward"])), log["reward"])
            ax_rew.set_xlim(0, max(10, len(log["reward"])))
            # Status bar
            s100 = float(np.mean(log["success"][-100:])) if log["success"] else 0.0
            run_status = "PAUSA" if state["paused"] else f"running x{state['speed']}"
            info_text.set_text(
                f"Ep {ep:4d}  |  éxito(100) {s100:.3f}  |  "
                f"X_mean {agent_.brain.X.mean():.4f}  |  "
                f"{run_status}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
        return True

    print("[obstacle] [+/-]=velocidad de visualización por lotes hasta x100; "
          "[Espacio]=pausa [q/Esc]=salir [X]=cerrar [s]=guardar pesos.")
    out = run_episodes(agent, env, n_episodes=n_episodes, verbose=True,
                       on_step=on_step)
    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--eta", type=float, default=1e-2)
    ap.add_argument("--initial-X", type=float, choices=[0.1, 0.9], default=0.9,
                    help="Estado inicial del crossbar: X=0.1 o X=0.9.")
    ap.add_argument("--headless", action="store_true",
                    help="Sin ventana: solo imprime y guarda artefactos (automatización)")
    ap.add_argument("--gui", action="store_true",
                    help=argparse.SUPPRESS)  # compat: ahora es el default
    args = ap.parse_args()

    env = ObstacleGridworld(seed=args.seed, goal_reward=10.0,
                            shaping_w=0.6, collision_penalty=-1.0)

    if not args.headless:
        (JSON_NV, JSON_V, JSON_LIF, WEIGHTS_FILE,
         TEMP_INIT, OUT_WEIGHTS, selected_seed, algorithm, initial_X) = launch_config_gui(args.seed)
        args.seed = selected_seed
        env = ObstacleGridworld(seed=args.seed, goal_reward=10.0,
                                shaping_w=0.6, collision_penalty=-1.0)
        fname = Path(OUT_WEIGHTS).stem
        weights_name = OUT_WEIGHTS
        print(f"[obstacle] algorithm={algorithm} seed={args.seed}  N={args.N}  M={args.M}  "
              f"ep={args.episodes}  eta={args.eta}")
        print("[codificador] random: proyección gaussiana fija por semilla hacia las filas del crossbar")
        print(f"[config] NV={JSON_NV} | V={JSON_V} | LIF={JSON_LIF} | T={TEMP_INIT}")
        agent = ReinforceAgent(
            obs_dim=env.obs_dim, n_actions=env.n_actions,
            N=args.N, M=args.M, eta=args.eta, seed=args.seed,
            temp_init=TEMP_INIT,
            temp_min=(0.3 if TEMP_INIT >= 0.3 else 0.05),
            initial_X=initial_X,
            encoder_mode="random",
            json_memristor_nv=JSON_NV,
            json_memristor_v=JSON_V, json_lif=JSON_LIF)
        start_from_scratch = (
            not WEIGHTS_FILE
            or WEIGHTS_FILE == "--- (Empezar desde cero) ---"
        )
        if start_from_scratch:
            # Make the GUI choice explicit on both the crossbar array and the
            # physical memristor cells; never carry state from an earlier run.
            agent.brain.reset(keep_weights=False)
            print(f"[pesos] desde cero | X0={initial_X:.3f} | "
                  f"X inicial mean={agent.brain.X.mean():.4f} "
                  f"range=[{agent.brain.X.min():.4f}, {agent.brain.X.max():.4f}]")
        else:
            agent.brain.load_weights(str(OUT_DIR / WEIGHTS_FILE))
            print(f"[pesos] cargados desde {WEIGHTS_FILE} | "
                  f"X inicial mean={agent.brain.X.mean():.4f}")
        t0 = time.time()
        log, early_path, learned_path, final_success = run_gui(
            agent, env, n_episodes=args.episodes)
    else:
        fname = f"run_obstacle_reinforce_seed{args.seed}_X0_{args.initial_X:g}"
        weights_name = f"{fname}_X.npy"
        print(f"[obstacle][headless] seed={args.seed}  N={args.N}  M={args.M}  "
              f"ep={args.episodes}  eta={args.eta}")
        print("[codificador] random: proyección gaussiana fija por semilla hacia las filas del crossbar")
        agent = ReinforceAgent(obs_dim=env.obs_dim, n_actions=env.n_actions,
                               N=args.N, M=args.M, eta=args.eta, seed=args.seed,
                               initial_X=args.initial_X,
                               encoder_mode="random")
        t0 = time.time()
        log, early_path, learned_path, final_success = run_episodes(
            agent, env, n_episodes=args.episodes, verbose=True
        )
    elapsed = time.time() - t0
    algorithm = "REINFORCE"
    log["algorithm"] = algorithm
    log["encoder"] = "random"
    log["initial_X"] = initial_X if not args.headless else args.initial_X
    log["environment_config"] = {
        "goal_reward": 10.0,
        "collision_penalty": -1.0,
        "progress_shaping_weight": 0.6,
    }
    if args.episodes >= 200:
        succ_last = float(np.mean(log["success"][-200:]))
    else:
        succ_last = float(np.mean(log["success"]))
    print(f"\n[resultado] exito(200)={succ_last:.3f}  "
          f"rollout final: {'EXITO' if final_success else 'fracaso'}  "
          f"tiempo={elapsed/60:.1f} min")

    (OUT_DIR / f"{fname}.json").write_text(json.dumps(log, indent=2),
                                           encoding="utf-8")
    weights_path = OUT_DIR / weights_name
    agent.brain.save_weights(str(weights_path))

    paths = {
        "seed": args.seed,
        "obstacles": [list(o) for o in sorted(env.obstacles)],
        "start": list(env.start),
        "goal": list(env.goal),
        "early_path": [list(p) for p in early_path],
        "learned_path": [list(p) for p in learned_path],
        "final_success": final_success,
    }
    (OUT_DIR / f"obstacle_{algorithm.lower()}_seed{args.seed}_paths.json").write_text(
        json.dumps(paths, indent=2), encoding="utf-8"
    )
    print(f"[ok] guardado en {weights_path}, {OUT_DIR / fname}.json y _paths.json")


if __name__ == "__main__":
    main()
