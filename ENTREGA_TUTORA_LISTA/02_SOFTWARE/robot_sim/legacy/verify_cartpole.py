"""
Primera corrida de CartPole con la arquitectura REINFORCE completa.
Solo 500 episodios para diagnóstico rápido.
"""
import json
from pathlib import Path

import numpy as np

from neurobot.env_wrapper import CartPoleWrapper
from neurobot.agent import NeuromorphicAgent
from neurobot.rl_loop import run_episodes, OUT_DIR

OUT_DIR.mkdir(exist_ok=True, parents=True)

env = CartPoleWrapper(seed=0, death_penalty=1.0)
agent = NeuromorphicAgent(
    obs_dim=env.obs_dim,
    n_actions=env.n_actions,
    N=12, M=4,
    V_read=0.1,
    eta=5e-4,
    temp_init=1.0,
    temp_min=0.3,
    temp_decay=0.999,
    baseline_window=200,
    seed=0,
)

print(f"[setup] obs_dim={env.obs_dim}  n_actions={env.n_actions}")
print(f"[setup] N={agent.brain.N}  M={agent.brain.M}")

log = run_episodes(agent, env,
                   n_episodes=2000,
                   learn_enabled=True,
                   verbose=False)

# Resumen por bloques de 200
print("\n[RESUMEN POR BLOQUES]")
for start in range(0, 2000, 200):
    block = log["reward"][start:start+200]
    print(f"  eps {start:4d}-{start+199:4d}  R_mean={np.mean(block):6.1f}  "
          f"max={np.max(block):6.1f}  min={np.min(block):6.1f}")

# --- Distribución de acciones últimos 100 eps ---
hist_recent = np.array(log["action_hist"][-100:]).sum(axis=0)
tot = hist_recent.sum()
print(f"\n  Distribución acciones (últimos 100 eps):")
for i, h in enumerate(hist_recent):
    print(f"    a={i}: {h/tot:.3f}")

(OUT_DIR / "run_A_base.json").write_text(
    json.dumps(log, indent=2), encoding="utf-8"
)
print(f"\n[ok] log en {OUT_DIR / 'run_A_base.json'}")

np.save(OUT_DIR / "run_A_base_X.npy", agent.brain.X)
print(f"[ok] X final shape={agent.brain.X.shape}  mean={agent.brain.X.mean():.4f}")

env.close()
