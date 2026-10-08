import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from neurobot.env_wrapper import CartPoleWrapper
from neurobot.agent import NeuromorphicAgent

print("Generando gráfico de prueba de 100 episodios...")

env = CartPoleWrapper(seed=0, death_penalty=50.0)
agent = NeuromorphicAgent(
    obs_dim=env.obs_dim,
    n_actions=env.n_actions,
    N=24, M=12,
    V_read=0.1,
    eta=0.005,
    temp_init=1.0,
    temp_min=0.05,
    temp_decay=0.995,
    baseline_window=200,
    seed=0,
    json_memristor_nv="strukov_ideal.json",
    json_memristor_v="memristor_volatile.json",
    json_lif="lif_config.json"
)

rewards = []
baselines = []

for ep in range(100):
    obs = env.reset()
    done = False
    ep_r = 0.0
    while not done:
        action = agent.act(obs)
        obs, r, done, info = env.step(action)
        ep_r += r
    agent.learn(ep_r)
    agent.end_episode()
    rewards.append(ep_r)
    baselines.append(agent.baseline)

# Graficar
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
ax.plot(rewards, color="tab:blue", alpha=0.35, label="Recompensa por episodio")
# Media móvil de 10 eps
smooth_10 = [np.mean(rewards[max(0, i-9):i+1]) for i in range(len(rewards))]
ax.plot(smooth_10, color="tab:blue", lw=2.0, label="Media móvil (10 eps)")
ax.plot(baselines, color="tab:orange", ls="--", lw=1.5, label="Línea base b(t)")

ax.set_xlabel("Episodio")
ax.set_ylabel("Recompensa acumulada")
ax.set_title("Curva de Aprendizaje NeuroBot 24x12 (Física Real R_off=16kΩ, 160x)")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()

out_png = ROOT / "figuras" / "probe_100ep_curve.png"
fig.savefig(out_png, dpi=200)
plt.close(fig)

print(f"[ok] Gráfico guardado en {out_png}")
