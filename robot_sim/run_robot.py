"""
Programa principal del simulador robótico.
Abre una ventana con 4 paneles que se actualizan en vivo mientras el
agente aprende CartPole sobre el crossbar memristivo.

Uso:  python run_robot.py
Controles: [Espacio] pausa/continúa   [q] salir
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from neurobot.env_wrapper import CartPoleWrapper
from neurobot.agent import NeuromorphicAgent

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
N, M = 24, 12
ETA = 0.005
N_EPISODES = 3000
SPEED = 5          # 1 = paso a paso, 5 = rápido, 20 = muy rápido
SEED = 0

import tkinter as tk
from tkinter import ttk
import os

# ------------------------------------------------------------------
# Ventana de Configuración de Robot Sim
# ------------------------------------------------------------------
def launch_config_gui():
    config_dir = Path(__file__).resolve().parents[1] / "neuromorphic_lab" / "configs"
    if not config_dir.exists():
        config_dir = Path(__file__).resolve().parent
    
    jsons = [p.name for p in config_dir.glob("*.json")]
    if not jsons: 
        jsons = ["last_session.json", "lif_config.json", "memristor_volatile.json"]
        
    out_dir = Path(__file__).resolve().parent / "outputs"
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    npys = [p.name for p in out_dir.glob("*.npy")]
    npys.insert(0, "--- (Empezar desde cero) ---")

    root = tk.Tk()
    root.title("Robot Sim - Configuración")
    root.geometry("450x300")
    
    # Centrar en pantalla
    root.eval('tk::PlaceWindow . center')
    root.attributes("-topmost", True)

    tk.Label(root, text="Cargar Archivos JSON desde el Laboratorio", font=("Arial", 11, "bold")).pack(pady=10)
    
    frame = tk.Frame(root)
    frame.pack(padx=20, pady=10, fill="x")
    
    # 1) Memristor No Volátil
    tk.Label(frame, text="Memristor No Volátil (Pesos):").grid(row=0, column=0, sticky="w", pady=5)
    cbo_nv = ttk.Combobox(frame, values=jsons, state="readonly", width=30)
    cbo_nv.set("strukov_ideal.json" if "strukov_ideal.json" in jsons else jsons[0])
    cbo_nv.grid(row=0, column=1, pady=5, padx=10)

    # 2) Memristor Volátil
    tk.Label(frame, text="Memristor Volátil:").grid(row=1, column=0, sticky="w", pady=5)
    cbo_v = ttk.Combobox(frame, values=jsons, state="readonly", width=30)
    cbo_v.set("memristor_volatile.json" if "memristor_volatile.json" in jsons else jsons[0])
    cbo_v.grid(row=1, column=1, pady=5, padx=10)

    # 3) Neurona LIF
    tk.Label(frame, text="Neurona LIF:").grid(row=2, column=0, sticky="w", pady=5)
    cbo_lif = ttk.Combobox(frame, values=jsons, state="readonly", width=30)
    cbo_lif.set("lif_config.json" if "lif_config.json" in jsons else jsons[0])
    cbo_lif.grid(row=2, column=1, pady=5, padx=10)
    
    # 4) Cargar Pesos Guardados
    tk.Label(frame, text="Pesos Previos (.npy):").grid(row=3, column=0, sticky="w", pady=5)
    cbo_w = ttk.Combobox(frame, values=npys, state="readonly", width=30)
    cbo_w.set(npys[0])
    cbo_w.grid(row=3, column=1, pady=5, padx=10)
    
    # 5) Temperatura Inicial
    tk.Label(frame, text="Temperatura Inicial:").grid(row=4, column=0, sticky="w", pady=5)
    temps = ["1.0 (Aprender desde cero)", "0.05 (Modo Experto)"]
    cbo_t = ttk.Combobox(frame, values=temps, state="readonly", width=30)
    cbo_t.set(temps[0])
    cbo_t.grid(row=4, column=1, pady=5, padx=10)
    
    # 6) Guardar archivo como
    tk.Label(frame, text="Guardar al terminar como:").grid(row=5, column=0, sticky="w", pady=5)
    entry_out = ttk.Entry(frame, width=33)
    entry_out.insert(0, "last_weights.npy")
    entry_out.grid(row=5, column=1, pady=5, padx=10)
    
    result = {"nv": "strukov_ideal.json", "v": "memristor_volatile.json", "lif": "lif_config.json", "weights": "--- (Empezar desde cero) ---", "temp": 1.0, "out_name": "last_weights.npy"}
    
    def on_start():
        result["nv"] = cbo_nv.get()
        result["v"] = cbo_v.get()
        result["lif"] = cbo_lif.get()
        result["weights"] = cbo_w.get()
        result["temp"] = 1.0 if "1.0" in cbo_t.get() else 0.05
        out_name = entry_out.get().strip()
        if not out_name.endswith(".npy"): out_name += ".npy"
        result["out_name"] = out_name
        root.destroy()
        
    btn = tk.Button(root, text="▶ Iniciar Robot Sim", command=on_start, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), padx=15, pady=5)
    btn.pack(pady=10)
    
    root.mainloop()
    return result["nv"], result["v"], result["lif"], result["weights"], result["temp"], result["out_name"]

JSON_NV, JSON_V, JSON_LIF, WEIGHTS_FILE, TEMP_INIT, OUT_WEIGHTS = launch_config_gui()
print(f"\n[Configuración] No Volátil: {JSON_NV} | Volátil: {JSON_V} | Neurona: {JSON_LIF}")
if WEIGHTS_FILE and WEIGHTS_FILE != "--- (Empezar desde cero) ---":
    print(f"[Pesos] Cargando memoria previa: {WEIGHTS_FILE}")
print(f"[Temperatura] Inicial: {TEMP_INIT}")
print()

env = CartPoleWrapper(seed=SEED, death_penalty=50.0)
agent = NeuromorphicAgent(
    obs_dim=env.obs_dim, n_actions=env.n_actions,
    N=N, M=M, V_read=0.1,
    eta=ETA,
    temp_init=TEMP_INIT, temp_min=0.05, temp_decay=0.995,
    baseline_window=200,
    seed=SEED,
    json_memristor_nv=JSON_NV,
    json_memristor_v=JSON_V,
    json_lif=JSON_LIF
)

if WEIGHTS_FILE and WEIGHTS_FILE != "--- (Empezar desde cero) ---":
    from pathlib import Path
    out_dir = Path(__file__).resolve().parent / "outputs"
    agent.brain.load_weights(str(out_dir / WEIGHTS_FILE))

# ------------------------------------------------------------------
# Estado global del runner
# ------------------------------------------------------------------
state = {
    "obs": None,
    "obs_raw": None,
    "action": None,
    "episode": 0,
    "ep_reward": 0.0,
    "rewards": [],
    "actions_per_ep": [],
    "angles_per_ep": [],
    "vels_per_ep": [],
    "paused": False,
    "step_in_ep": 0,
    "done": True,
    "total_steps": 0,
}

# Historial para política P(Derecha|angulo, vel)
policy_angles = []
policy_actions = []
policy_vels = []

# ------------------------------------------------------------------
# Figura
# ------------------------------------------------------------------
plt.ion()
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.canvas.manager.set_window_title("Neuromorphic Robot Sim — CartPole")

ax_scene  = axes[0, 0]   # carrito + poste
ax_X      = axes[0, 1]   # heatmap X
ax_rew    = axes[1, 0]   # curva de recompensa
ax_pol    = axes[1, 1]   # P(a=1|θ)

# --- Panel escena ---
ax_scene.set_xlim(-2.6, 2.6)
ax_scene.set_ylim(-0.6, 1.6)
ax_scene.set_aspect("equal")
ax_scene.set_title("Robot (CartPole)")
ax_scene.grid(alpha=0.2)
cart = Rectangle((0, 0), 0.4, 0.2, color="tab:blue")
pole, = ax_scene.plot([], [], "o-", lw=3, color="tab:orange", markersize=6)
ax_scene.add_patch(cart)

# --- Panel X ---
im_X = ax_X.imshow(agent.brain.X, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax_X.set_title("Matriz X (memristores)")
ax_X.set_xlabel("Columna (acción)")
ax_X.set_ylabel("Fila (sensor)")
fig.colorbar(im_X, ax=ax_X)

# --- Panel recompensa ---
line_rew, = ax_rew.plot([], [], color="tab:blue", lw=1.5)
ax_rew.axhline(22, color="gray", ls="--", lw=1, label="aleatorio (~22)")
ax_rew.axhline(500, color="green", ls=":", lw=1, label="óptimo (~500)")
ax_rew.set_title("Curva de aprendizaje")
ax_rew.set_xlabel("Episodio")
ax_rew.set_ylabel("Recompensa")
ax_rew.set_ylim(0, 520)
ax_rew.legend(fontsize=8, loc="upper left")
ax_rew.grid(alpha=0.25)

# --- Panel política ---
line_pol_0, = ax_pol.plot([], [], "o-", color="tab:purple", lw=1.8, ms=4, label="Quieto (v~0)")
line_pol_pos, = ax_pol.plot([], [], "^--", color="tab:red", lw=1.2, ms=3, label="Cayendo Der (v>0)")
line_pol_neg, = ax_pol.plot([], [], "v--", color="tab:blue", lw=1.2, ms=3, label="Cayendo Izq (v<0)")
ax_pol.axhline(0.5, color="gray", ls="--", lw=1)
ax_pol.set_title("Política aprendida (Multidimensional)")
ax_pol.set_xlabel("Ángulo del poste (rad)")
ax_pol.set_ylabel("P(Derecha | ángulo, vel)")
ax_pol.set_xlim(-0.22, 0.22)
ax_pol.set_ylim(0, 1)
ax_pol.legend(fontsize=7, loc="lower right")
ax_pol.grid(alpha=0.25)

# Estado en texto
info_text = fig.text(0.02, 0.98, "", ha="left", va="top",
                     family="monospace", fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.canvas.draw()
fig.canvas.flush_events()

# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------
from tkinter import simpledialog

def on_key(event):
    if event.key == " ":
        state["paused"] = not state["paused"]
        print(f"[pausa] {state['paused']}")
    elif event.key in ("q", "escape"):
        plt.close("all")
    elif event.key == "s":
        # Pausar la simulación temporalmente para que no siga corriendo de fondo
        was_paused = state["paused"]
        state["paused"] = True
        
        root = tk.Tk()
        root.withdraw()
        # Forzar que la ventana de guardado aparezca por encima del matplotlib
        root.attributes("-topmost", True)
        name = simpledialog.askstring("Guardar Pesos", "Nombre del archivo (.npy):", initialvalue="mis_pesos_expertos.npy", parent=root)
        if name:
            if not name.endswith(".npy"):
                name += ".npy"
            out_dir = Path(__file__).resolve().parent / "outputs"
            agent.brain.save_weights(str(out_dir / name))
            print(f"\n✅ Pesos guardados manualmente en: {name}")
        root.destroy()
        
        state["paused"] = was_paused

fig.canvas.mpl_connect("key_press_event", on_key)

# ------------------------------------------------------------------
# Lógica de simulación
# ------------------------------------------------------------------
def reset_episode():
    obs = env.reset()
    state["obs"] = obs
    state["ep_raw"] = None
    state["ep_reward"] = 0.0
    state["actions_per_ep"] = []
    state["angles_per_ep"] = []
    state["vels_per_ep"] = []
    state["step_in_ep"] = 0
    state["done"] = False


def do_step():
    if state["done"]:
        reset_episode()
        return

    obs = state["obs"]
    action = agent.act(obs)
    obs, r, done, info = env.step(action)

    state["obs"] = obs
    state["obs_raw"] = info.get("obs_raw", None)
    state["action"] = action
    state["ep_reward"] += r
    state["step_in_ep"] += 1
    state["total_steps"] += 1

    state["actions_per_ep"].append(int(action))
    if state["obs_raw"] is not None:
        state["angles_per_ep"].append(float(state["obs_raw"][2]))
        state["vels_per_ep"].append(float(state["obs_raw"][3]))

    if done:
        state["done"] = True
        agent.learn(state["ep_reward"])
        agent.end_episode()
        state["rewards"].append(state["ep_reward"])
        state["episode"] += 1

        # Acumular para política
        policy_angles.extend(state["angles_per_ep"])
        policy_actions.extend(state["actions_per_ep"])
        policy_vels.extend(state["vels_per_ep"])
        # Mantener sólo últimos 50 eps
        max_pts = 50 * 200
        if len(policy_angles) > max_pts:
            policy_angles[:] = policy_angles[-max_pts:]
            policy_actions[:] = policy_actions[-max_pts:]
            policy_vels[:] = policy_vels[-max_pts:]


def draw_policy():
    if len(policy_angles) < 50:
        return
    angles = np.asarray(policy_angles[-5000:])
    actions = np.asarray(policy_actions[-5000:])
    vels = np.asarray(policy_vels[-5000:])
    
    edges = np.linspace(-0.22, 0.22, 12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p_0 = np.full(len(centers), np.nan)
    p_pos = np.full(len(centers), np.nan)
    p_neg = np.full(len(centers), np.nan)
    
    for i in range(len(centers)):
        mask_angle = (angles >= edges[i]) & (angles < edges[i + 1])
        
        m0 = mask_angle & (np.abs(vels) < 0.2)
        m_pos = mask_angle & (vels >= 0.2)
        m_neg = mask_angle & (vels <= -0.2)
        
        half_act = env.n_actions // 2
        if m0.sum() > 3: p_0[i] = float(np.mean(actions[m0] >= half_act))
        if m_pos.sum() > 3: p_pos[i] = float(np.mean(actions[m_pos] >= half_act))
        if m_neg.sum() > 3: p_neg[i] = float(np.mean(actions[m_neg] >= half_act))
        
    line_pol_0.set_data(centers, p_0)
    line_pol_pos.set_data(centers, p_pos)
    line_pol_neg.set_data(centers, p_neg)


def draw_scene():
    if state["obs_raw"] is None:
        return
    x = float(state["obs_raw"][0])
    theta = float(state["obs_raw"][2])
    cart.set_xy((x - 0.2, -0.1))
    px = x + 0.5 * np.sin(theta)
    py = 0.0 + 0.5 * np.cos(theta)
    pole.set_data([x, px], [0.0, py])


def draw_all():
    draw_scene()
    im_X.set_data(agent.brain.X)

    if state["rewards"]:
        line_rew.set_data(np.arange(len(state["rewards"])), state["rewards"])
        ax_rew.set_xlim(0, max(10, len(state["rewards"])))

    draw_policy()

    r_recent = (np.mean(state["rewards"][-20:])
                if len(state["rewards"]) >= 20 else 0.0)
    info_text.set_text(
        f"Ep {state['episode']:4d}  |  "
        f"paso {state['step_in_ep']:3d}  |  "
        f"R_ep {state['ep_reward']:6.1f}  |  "
        f"R̄(20) {r_recent:6.1f}  |  "
        f"T {agent.current_temperature():.3f}  |  "
        f"b {agent.baseline:+.2f}  |  "
        f"Ḡ {agent.brain.G.mean():.2e}  |  "
        f"{'PAUSA' if state['paused'] else 'running'}"
    )

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ------------------------------------------------------------------
# Loop principal
# ------------------------------------------------------------------
print(f"[run_robot] N={N} M={M} eta={ETA} seed={SEED}")
print(f"[run_robot] Ventana abierta. [Espacio]=pausa, [q]=salir.")

reset_episode()
try:
    while plt.fignum_exists(fig.number):
        if not state["paused"]:
            for _ in range(SPEED):
                do_step()
                if state["done"]:
                    break
        draw_all()
        plt.pause(0.001)
except KeyboardInterrupt:
    pass

env.close()
# Autoguardar pesos de la matriz del agente
out_dir = Path(__file__).resolve().parent / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / OUT_WEIGHTS
agent.brain.save_weights(str(out_path))
print(f"✅ Pesos (X) guardados en: {out_path}")

# Autoguardar log de la corrida completa (JSON)
log_path = out_path.with_suffix('.json')
log_data = {
    "N": N, "M": M,
    "eta": ETA, "temp_init": TEMP_INIT,
    "seed": SEED,
    "stats": agent.stats,
    "policy_angles": policy_angles,
    "policy_actions": policy_actions,
    "policy_vels": policy_vels
}
with open(log_path, 'w', encoding='utf-8') as f:
    json.dump(log_data, f)
print(f"✅ Log de la simulación guardado en: {log_path}")

print("[run_robot] cerrado")
