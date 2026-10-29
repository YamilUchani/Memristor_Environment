"""
Clasificador de patrones 8×8 con crossbar memristivo.
Prueba de concepto: crossbar 64×8 entrenado con REINFORCE supervisado, 8 clases.

Uso:  python run_vision.py              (abre GUI en vivo — modo normal)
      python run_vision.py --headless   (sin ventana, para automatización)

=================================================================
GUÍA DE LOS 6 CAMPOS DEL DIÁLOGO DE CONFIGURACIÓN
=================================================================

El diálogo de configuración es idéntico al de CartPole (run_robot.py)
para mantener consistencia visual entre aplicaciones. Sin embargo,
no todos los campos aplican a esta tarea.

ESTADO DE CADA CAMPO EN VISIÓN:

  ┌─────────────────────────┬──────────┬────────────────────────────┐
  │ Campo                   │ ¿Aplica? │ Motivo                     │
  ├─────────────────────────┼──────────┼────────────────────────────┤
  │ Memristor No Volátil    │ SÍ       │ Sustrato físico del        │
  │                         │          │ crossbar (r_on, r_off).    │
  │ Memristor Volátil       │ NO       │ La tarea es estática: no   │
  │                         │          │ hay dependencia temporal.  │
  │ Neurona LIF             │ NO       │ Salida analógica (softmax) │
  │                         │          │ en vez de eventos discretos│
  │ Pesos Previos (.npy)    │ SÍ       │ Carga de modelo entrenado. │
  │ Temperatura Inicial     │ NO       │ Aprendizaje supervisado,   │
  │                         │          │ no RL. No hay exploración. │
  │ Guardar como            │ SÍ       │ Nombre del .npy y .json.   │
  └─────────────────────────┴──────────┴────────────────────────────┘

REGLA GENERAL — ¿Cuándo se activa cada componente?

  • Memristor Volátil → no se usa en el agente Robot Sim actual.
      El modelo de dispositivo sigue disponible en Neuromorphic Lab, pero
      ninguna tarea de este agente integra una traza volátil de elegibilidad.

  • Neurona LIF → depende del TIPO DE SALIDA
      ¿La salida es analógica (valores continuos) o discreta (eventos)?
      Analógica: softmax → NO usa LIF.
      Discreta: spikes → SÍ usa LIF (ej. SNN con imágenes estáticas).

  • Temperatura Inicial → depende del TIPO DE APRENDIZAJE
      ¿Aprendizaje supervisado (sabés la respuesta) o RL (no la sabés)?
      Supervisado: NO usa temperatura.
      RL: SÍ usa temperatura (exploración necesaria).

COMPARATIVA ENTRE LAS 3 APLICACIONES:

  ┌──────────────────┬──────────────┬─────────┬───────────────┐
  │ Aplicación       │ Volátil      │ LIF     │ Temperatura   │
  ├──────────────────┼──────────────┼─────────┼───────────────┤
  │ Visión 8×8       │ NO           │ NO      │ NO            │
  │ (estático,       │              │         │               │
  │  supervisado)    │              │         │               │
  ├──────────────────┼──────────────┼─────────┼───────────────┤
  │ CartPole         │ SÍ (traza)   │ SÍ      │ SÍ (RL)       │
  │ (continuo, RL)   │              │ (spikes)│               │
  ├──────────────────┼──────────────┼─────────┼───────────────┤
  │ Navegación       │ SÍ (traza)   │ SÍ      │ SÍ (RL)       │
  │ (continuo, RL)   │              │ (spikes)│               │
  └──────────────────┴──────────────┴─────────┴───────────────┘

CASOS FUTUROS que activarían los 3 componentes en clasificación:

  • Video classification   → temporal + spikes + exploración
  • DVS (event cameras)    → temporal + spikes
  • MNIST secuencial       → temporal
  • Imagen + SNN           → spikes (sin temporalidad)
  • Bandit problem         → exploración (sin temporalidad)

=================================================================
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from neurobot.reinforce_agent import NeuromorphicAgent as ReinforceAgent

OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = Path(__file__).resolve().parent / "figuras"
THESIS_FIG_DIR = Path(__file__).resolve().parents[1] / "PLANTILLA_PDF_FINAL" / "figuras"
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)
THESIS_FIG_DIR.mkdir(exist_ok=True, parents=True)


# ==================================================================
# Dataset: 8 patrones binarios 8×8
# ==================================================================
def make_patterns():
    P = np.zeros((8, 8, 8), dtype=float)

    # 0: horizontal
    P[0, 3, :] = 1.0

    # 1: vertical
    P[1, :, 3] = 1.0

    # 2: diagonal \
    for i in range(8):
        P[2, i, i] = 1.0

    # 3: diagonal /
    for i in range(8):
        P[3, i, 7 - i] = 1.0

    # 4: cruz
    P[4, 3, :] = 1.0
    P[4, :, 3] = 1.0

    # 5: marco
    P[5, 0, :] = 1.0
    P[5, 7, :] = 1.0
    P[5, :, 0] = 1.0
    P[5, :, 7] = 1.0

    # 6: X
    for i in range(8):
        P[6, i, i] = 1.0
        P[6, i, 7 - i] = 1.0

    # 7: cuadrado relleno
    P[7, 2:6, 2:6] = 1.0

    return P


def _vision_policy_eligibility(agent):
    """REINFORCE score mapped through the vision crossbar readout to X.

    The decoder softmaxes z-scored column currents. Account for that
    normalization and for dG/dX instead of treating X as a linear weight.
    The cell derivative uses the ideal crossbar expression; the forward read
    remains the configured nodal/MNA read used by the simulator.
    """
    step = agent.last_step()
    current = np.asarray(step["I_col"], dtype=float)
    centered = current - float(np.mean(current))
    sigma = float(np.std(current))
    scale = sigma + 1e-8  # same stabilizer as ActionDecoder.decode

    probabilities = np.asarray(step["probs"], dtype=float)
    behavior = np.asarray(step["behavior_probs"], dtype=float)
    col = int(step["col"])
    epsilon = float(step["exploration_epsilon"])
    posterior = ((1.0 - epsilon) * probabilities[col]
                 / max(float(behavior[col]), 1e-15))
    action_score = -posterior * probabilities
    action_score[col] += posterior

    # Jacobian of the decoder's z-scored logits with respect to each current.
    dlogp_dI = action_score / scale
    if sigma > 1e-15:
        dlogp_dI -= (
            centered * float(np.dot(action_score, centered))
            / (agent.brain.M * sigma * scale * scale)
        )

    # G(X)=1/(Ron + X*(Roff-Ron)); hence -dI/dX is positive
    # for positive row voltage and negative for negative row voltage.
    conductance = agent.brain.G
    delta_r = agent.brain._cell_r_off - agent.brain._cell_r_on
    read_steps = max(1, agent.steps_per_action)
    minus_dI_dX = (
        step["V_rows"][:, None] * delta_r * conductance**2 * read_steps
    )
    # learn() applies -eta * eligibility * advantage, so eligibility is
    # -d(log pi)/dX here.
    return minus_dI_dX * dlogp_dI[None, :]


# ==================================================================
# Clasificador
# ==================================================================
def run_classifier(patterns, episodes=5000, V_read=0.05, eta=None,
                   baseline_window=200, seed=0, verbose=True, on_step=None,
                   json_memristor_nv="last_session.json",
                   json_memristor_v="memristor_volatile.json",
                   json_lif="lif_config.json", weights_file=None, algorithm="REINFORCE", initial_X=None):
    n_classes = patterns.shape[0]
    N = 64
    M = n_classes
    rng = np.random.default_rng(seed)
    class_order = np.concatenate([
        rng.permutation(n_classes)
        for _ in range((episodes + n_classes - 1) // n_classes)
    ])[:episodes]
    if eta is None:
        eta = 5e-4

    AgentClass = ReinforceAgent
    agent_kwargs = dict(
        obs_dim=N, n_actions=M, N=N, M=M, V_read=V_read, dt=1e-3,
        encoder_mode="random",
        steps_per_action=1,
        seed=seed, eta=eta, temp_init=1.0,
        temp_min=1.0, temp_decay=1.0, baseline_window=baseline_window,
        epsilon=0.30,
        json_memristor_nv=json_memristor_nv,
        json_memristor_v=json_memristor_v, json_lif=json_lif,
        initial_X=initial_X,
    )
    agent = AgentClass(**agent_kwargs)
    # Sample actions during training so the policy receives corrective feedback.
    agent.decoder.sample = True
    brain = agent.brain
    initial_crossbar_X = brain.X.copy()
    if weights_file and weights_file != "--- (Empezar desde cero) ---":
        brain.load_weights(str(OUT_DIR / weights_file))
    initial_state = {"X_mean": float(brain.X.mean()),
                     "G_mean_S": float(brain.G.mean()),
                     "G_min_S": float(brain.G.min()),
                     "G_max_S": float(brain.G.max())}

    log = {
        "ep": [], "reward": [], "correct": [],
        "class_true": [], "class_pred": [],
        "greedy_correct": [], "X_mean": [], "baseline": [],
        "encoder": "random",
        "initial_state": initial_state,
        "initial_crossbar_X": initial_crossbar_X.tolist(),
    }
    t0 = time.time()
    # State-dependent baselines are unbiased because each one depends only on
    # the presented pattern, not on the sampled action.
    class_baseline = np.zeros(n_classes, dtype=float)
    class_baseline_rate = 0.05
    for ep in range(episodes):
        idx = int(class_order[ep])
        image = patterns[idx].ravel()

        # Lectura analógica
        pred = int(agent.act(image))
        agent._E_accum[:] = _vision_policy_eligibility(agent)
        p = agent.decoder.last_probs()

        sampled_correct = int(pred == idx)
        greedy_pred = int(np.argmax(p))
        greedy_correct = int(greedy_pred == idx)
        reward = 1.0 if sampled_correct else -1.0

        # Each image is one contextual-bandit decision; this is its full return.
        agent.learn(reward, baseline_override=class_baseline[idx])
        class_baseline[idx] += class_baseline_rate * (
            reward - class_baseline[idx]
        )
        baseline = agent.baseline
        agent.end_episode()

        # Log
        log["ep"].append(ep)
        log["reward"].append(reward)
        # Report policy quality with the greedy action shown by the GUI;
        # retain sampled reward accuracy separately for the learning signal.
        log["correct"].append(greedy_correct)
        log.setdefault("sampled_correct", []).append(sampled_correct)
        log["greedy_correct"].append(greedy_correct)
        log["class_true"].append(idx)
        log["class_pred"].append(pred)
        log.setdefault("class_pred_greedy", []).append(greedy_pred)
        log["X_mean"].append(float(brain.X.mean()))
        log["baseline"].append(baseline)

        if on_step is not None:
            if not on_step(ep, idx, p, brain, log):
                break

        if verbose and (ep % 500 == 0 or ep == episodes - 1):
            acc = float(np.mean(log["correct"][-200:])) if ep >= 200 else float(np.mean(log["correct"]))
            print(f"[ep {ep:5d}]  acc={acc:.3f}  "
                  f"reward={np.mean(log['reward'][-200:]):+.3f}  "
                  f"X_mean={brain.X.mean():.4f}")

    elapsed = time.time() - t0
    acc_final = float(np.mean(log["correct"][-500:])) if episodes >= 500 else float(np.mean(log["correct"]))
    print(f"\n[resultado] accuracy últimos episodios = {acc_final:.3f}  "
          f"tiempo = {elapsed:.1f}s")

    return brain, log


# ==================================================================
# Figuras
# ==================================================================
def plot_dataset(patterns, path):
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(patterns[i], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"Clase {i}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_learning(log, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    acc = np.array(log["correct"], dtype=float)
    window = 100
    if len(acc) >= window:
        smooth = np.convolve(acc, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(len(smooth)), smooth, lw=1.8, color="tab:blue")
    ax.axhline(1 / 8, ls="--", color="gray", label="random (1/8)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Accuracy (media móvil 100)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Curva de aprendizaje — Visión 8×8")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# GUI en vivo (opcional: --gui)
# ==================================================================
def launch_config_gui(seed_default=0):
    """Diálogo de semilla, algoritmo, X0 y configuración del crossbar.

    Visión es clasificación supervisada estática con salida analógica
    (softmax): Volátil/LIF/Temperatura NO aplican (ver docstring del módulo).
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
           "v": "memristor_volatile.json", "lif": "lif_config.json",
           "weights": npys[0], "temp": 1.0, "seed": seed_default,
           "algorithm": "REINFORCE",
           "out_name": f"run_vision_seed{seed_default}.npy",
           "initial_X": 0.9}

    root = tk.Tk()
    root.title("Clasificador de patrones 8×8")
    root.geometry("560x500")
    root.minsize(560, 420)
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

    # 1) Memristor No Volátil [APLICA: define la física del crossbar]
    tk.Label(frame, text="Memristor No Volátil (Pesos):").grid(
        row=2, column=0, sticky="w", pady=5)
    cbo_nv = ttk.Combobox(frame, values=jsons, state="readonly", width=30)
    cbo_nv.set(out["nv"])
    cbo_nv.grid(row=2, column=1, pady=5, padx=10, sticky="ew")

    # 2) Pesos Previos (.npy) [APLICA: carga modelo entrenado]
    tk.Label(frame, text="Pesos Previos (.npy):").grid(row=3, column=0, sticky="w", pady=5)
    cbo_w = ttk.Combobox(frame, values=npys, state="readonly", width=30)
    cbo_w.set(npys[0])
    cbo_w.grid(row=3, column=1, pady=5, padx=10, sticky="ew")

    # 3) Guardar como [APLICA: nombre del archivo de salida]
    tk.Label(frame, text="Guardar al terminar como:").grid(row=4, column=0, sticky="w", pady=5)
    entry_out = ttk.Entry(frame, width=33)
    entry_out.insert(0, f"run_vision_seed{seed_default}.npy")
    entry_out.grid(row=4, column=1, pady=5, padx=10, sticky="ew")

    tk.Label(frame, text="X₀ inicial (si empiezas desde cero):").grid(
        row=5, column=0, sticky="w", pady=5)
    cbo_x0 = ttk.Combobox(frame, values=["0.1", "0.9"],
                          state="readonly", width=30)
    cbo_x0.set("0.9")
    cbo_x0.grid(row=5, column=1, pady=5, padx=10, sticky="ew")

    def on_start():
        try:
            out["seed"] = int(entry_seed.get())
            if out["seed"] < 0: raise ValueError
        except ValueError:
            from tkinter import messagebox
            messagebox.showerror("Semilla inválida", "Use un entero no negativo.", parent=root)
            return
        out["algorithm"] = "REINFORCE"
        out["nv"] = cbo_nv.get()
        out["weights"] = cbo_w.get()
        out["initial_X"] = float(cbo_x0.get())
        name = entry_out.get().strip()
        if name == f"run_vision_seed{seed_default}.npy":
            name = f"run_vision_reinforce_seed{out['seed']}.npy"
        if name == f"run_vision_{out['algorithm'].lower()}_seed{out['seed']}.npy":
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


def run_gui(patterns, episodes, eta, seed, json_nv="last_session.json",
            json_v="memristor_volatile.json", json_lif="lif_config.json",
    weights_file=None, algorithm="REINFORCE", initial_X=None):
    """Loop de entrenamiento con ventana en vivo de 4 paneles."""
    state = {"paused": False, "stop": False, "brain": None, "speed": 1}

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Neuromorphic Vision 8x8 — Crossbar 64x8")

    ax_pat, ax_X = axes[0, 0], axes[0, 1]
    ax_acc, ax_prob = axes[1, 0], axes[1, 1]

    # Panel 1: patrón de entrada actual (con borde de acierto/error)
    im_pat = ax_pat.imshow(patterns[0], cmap="gray_r", vmin=0, vmax=1)
    border = Rectangle((-0.5, -0.5), 8, 8, fill=False,
                       linewidth=4, edgecolor="tab:gray")
    ax_pat.add_patch(border)
    ax_pat.set_xlim(-0.9, 7.9)   # anillo fino para el fondo verde/rojo
    ax_pat.set_ylim(7.9, -0.9)
    ax_pat.set_facecolor("white")
    ax_pat.set_title("Entrada: clase ?")
    ax_pat.axis("off")

    # Panel 2: matriz memristiva X (64×8)
    # Show the requested starting state immediately; the previous all-zero
    # placeholder made both X0 choices look like a dark, empty crossbar until
    # the first training callback. A loaded matrix takes precedence.
    preview_X = np.full((64, 8), float(0.9 if initial_X is None else initial_X))
    if weights_file and weights_file != "--- (Empezar desde cero) ---":
        candidate = np.load(OUT_DIR / weights_file)
        if candidate.shape == preview_X.shape:
            preview_X = np.clip(candidate, 0.0, 1.0)
    im_X = ax_X.imshow(preview_X, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax_X.set_title("Estado X del crossbar (0 = G alta, 1 = G baja)")
    ax_X.set_xlabel("Columna (clase)")
    ax_X.set_ylabel("Fila (píxel)")
    fig.colorbar(im_X, ax=ax_X)

    # Panel 3: accuracy móvil
    line_acc, = ax_acc.plot([], [], color="tab:blue", lw=1.6)
    ax_acc.axhline(1 / 8, ls="--", color="gray", label="azar (1/8)")
    ax_acc.set_xlim(0, max(10, episodes))
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_xlabel("Episodio")
    ax_acc.set_ylabel("Acc (media móvil 100)")
    ax_acc.set_title("Curva de aprendizaje")
    ax_acc.legend(fontsize=8, loc="lower right")
    ax_acc.grid(alpha=0.25)

    # Panel 4: probabilidades del último paso
    ax_prob.set_xticks(range(8))
    ax_prob.set_ylim(0, 1)
    ax_prob.set_xlabel("Clase")
    ax_prob.set_ylabel("P(clase | entrada)")
    ax_prob.set_title("Softmax (último paso)")
    ax_prob.grid(alpha=0.25, axis="y")

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
                initialvalue="vision_weights.npy", parent=root)
            root.destroy()
            if name:
                if not name.endswith(".npy"):
                    name += ".npy"
                if state["brain"] is not None:
                    state["brain"].save_weights(str(OUT_DIR / name))
                    print(f"[ok] Pesos guardados: {name}")
            state["paused"] = was_paused

    def on_close(_event):
        state["stop"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    def on_step(ep, idx, p, brain_, log):
        state["brain"] = brain_
        if state["stop"] or not plt.fignum_exists(fig.number):
            return False
        while state["paused"] and plt.fignum_exists(fig.number):
            plt.pause(0.05)
        if state["stop"] or not plt.fignum_exists(fig.number):
            return False
        if ep % state["speed"] != 0 and ep != episodes - 1:
            fig.canvas.flush_events()
            return True
        if ep % state["speed"] == 0 or ep == episodes - 1:
            y_pred = int(np.argmax(p))
            ok = (y_pred == idx)
            # --- Panel 1: feedback verde (acierto) / rojo (error) ---
            im_pat.set_data(patterns[idx])
            color_borde = "tab:green" if ok else "tab:red"
            border.set_edgecolor(color_borde)
            ax_pat.set_facecolor("#d4f5d4" if ok else "#f5d4d4")
            ax_pat.set_title(f"Esperado: {idx} | Predicho: {y_pred}",
                             color=color_borde, fontweight="bold", fontsize=11)
            # --- Panel 2: matriz X ---
            im_X.set_data(brain_.X)
            # --- Panel 3: accuracy móvil ---
            acc = np.asarray(log["correct"], dtype=float)
            if len(acc) >= 100:
                smooth = np.convolve(acc, np.ones(100) / 100, mode="valid")
                line_acc.set_data(np.arange(len(smooth)), smooth)
            # --- Panel 4: softmax coloreado por rol ---
            n_classes = len(p)
            colors = ["tab:gray"] * n_classes
            colors[idx] = "tab:green"                      # respuesta correcta
            colors[y_pred] = "tab:blue" if not ok else "tab:green"  # predicción
            ax_prob.cla()
            ax_prob.bar(range(n_classes), p, color=colors)
            ax_prob.set_xticks(range(n_classes))
            ax_prob.set_ylim(0, 1)
            ax_prob.set_xlabel("Clase")
            ax_prob.set_ylabel("P(clase | entrada)")
            ax_prob.set_title(f"Softmax — Predicción: {y_pred}",
                              color=color_borde, fontweight="bold")
            ax_prob.grid(alpha=0.25, axis="y")
            # --- Status bar ---
            recent = float(np.mean(log["correct"][-500:])) if log["correct"] else 0.0
            last100 = log["correct"][-100:]
            hits = int(np.sum(last100))
            misses = len(last100) - hits
            acc100 = hits / max(1, len(last100))
            run_status = "PAUSA" if state["paused"] else f"running x{state['speed']}"
            info_text.set_text(
                f"Ep {ep:5d}  |  acc(500) {recent:.3f}  |  acc(100) {acc100:.2f}  |  "
                f"OK {hits} / ERR {misses}  |  X_mean {brain_.X.mean():.4f}  |  "
                f"{run_status}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
        return True

    print("[vision] [+/-]=velocidad de visualización por lotes hasta x100; "
          "[Espacio]=pausa [q/Esc]=salir [X]=cerrar.")
    brain, log = run_classifier(patterns, episodes=episodes, eta=eta, seed=seed,
                                verbose=True, on_step=on_step,
                                json_memristor_nv=json_nv,
                                json_memristor_v=json_v,
                                json_lif=json_lif, weights_file=weights_file,
                                algorithm=algorithm, initial_X=initial_X)
    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    return brain, log


# ==================================================================
# Main
# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--eta", type=float, default=None,
                    help="Tasa de aprendizaje; predeterminada: 5e-4.")
    ap.add_argument("--initial-X", type=float, choices=[0.1, 0.9], default=0.9,
                    help="Estado inicial del crossbar: X=0.1 o X=0.9.")
    ap.add_argument("--headless", action="store_true",
                    help="Sin ventana: solo imprime y guarda artefactos (automatización)")
    ap.add_argument("--gui", action="store_true", help=argparse.SUPPRESS)  # compat: ahora es el default
    args = ap.parse_args()
    algorithm = "REINFORCE"

    patterns = make_patterns()

    if not args.headless:
        (JSON_NV, JSON_V, JSON_LIF, WEIGHTS_FILE,
         TEMP_INIT, OUT_WEIGHTS, seed, algorithm, initial_X) = launch_config_gui(args.seed)
        episodes = args.episodes
        eta = args.eta if args.eta is not None else 5e-4
        fname = Path(OUT_WEIGHTS).stem
        weights_name = OUT_WEIGHTS
        print(f"[vision] algorithm={algorithm} seed={seed}  ep={episodes}  eta={eta}")
        print("[codificador] random: proyección gaussiana fija por semilla hacia las filas del crossbar")
        print(f"[config] NV={JSON_NV} | V={JSON_V} | LIF={JSON_LIF}")
        if WEIGHTS_FILE and WEIGHTS_FILE != "--- (Empezar desde cero) ---":
            print(f"[pesos] memoria previa: {WEIGHTS_FILE}")
        plot_dataset(patterns, FIG_DIR / "vision_dataset.png")
        brain, log = run_gui(patterns, episodes=episodes, eta=eta, seed=seed,
                             json_nv=JSON_NV, json_v=JSON_V, json_lif=JSON_LIF,
                             weights_file=WEIGHTS_FILE, algorithm=algorithm,
                             initial_X=initial_X)
    else:
        episodes, seed = args.episodes, args.seed
        eta = args.eta if args.eta is not None else 5e-4
        fname = f"run_vision_reinforce_seed{seed}_X0_{args.initial_X:g}"
        weights_name = f"{fname}_X.npy"
        print(f"[vision][headless] algorithm=REINFORCE seed={seed}  ep={episodes}  eta={eta}")
        plot_dataset(patterns, FIG_DIR / "vision_dataset.png")
        brain, log = run_classifier(
            patterns,
            episodes=episodes,
            eta=eta,
            seed=seed,
            verbose=True, algorithm="REINFORCE", initial_X=args.initial_X,
        )

    log["initial_X"] = initial_X if not args.headless else args.initial_X
    log["algorithm"] = "REINFORCE"
    log["eta"] = eta
    plot_learning(log, FIG_DIR / f"vision_learning_{log['algorithm'].lower()}_seed{seed}.png")
    plot_learning(log, THESIS_FIG_DIR / f"vision_learning_{log['algorithm'].lower()}_seed{seed}.png")

    # Matriz de confusión (últimos 500 episodios)
    last = slice(-500, None)
    true = np.array(log["class_true"][last])
    pred = np.array(log.get("class_pred_greedy", log["class_pred"])[last])
    conf = np.zeros((8, 8), dtype=int)
    for t, p in zip(true, pred):
        conf[t, p] += 1

    print("\n[matriz de confusión]")
    print(conf)

    # Guardar
    (OUT_DIR / f"{fname}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    weights_path = OUT_DIR / weights_name
    brain.save_weights(str(weights_path))
    print(f"[ok] guardado en {weights_path} y {OUT_DIR / fname}.json")


if __name__ == "__main__":
    main()
