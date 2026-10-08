import json
import numpy as np
from pathlib import Path

from neurobot.env_wrapper import CartPoleWrapper
from neurobot.agent import NeuromorphicAgent
from neurobot.rl_loop import run_episodes, OUT_DIR

# --- 1) Agente CON elegibilidad ---
print("=" * 60)
print("A) R-STDP con elegibilidad")
print("=" * 60)
env = CartPoleWrapper(seed=0)
agent_a = NeuromorphicAgent(
    obs_dim=env.obs_dim, n_actions=env.n_actions,
    N=8, M=4, V_read=0.1,
    eta=1e-4, tau_e=0.05, baseline_alpha=0.01,
    use_eligibility=True, seed=0,
)
X0 = agent_a.brain.X.copy()
log_a = run_episodes(agent_a, env, n_episodes=2000, learn_enabled=True, verbose=True)
dX_total = float(np.abs(agent_a.brain.X - X0).mean())
print(f"\nCambio total en X: {dX_total:.4e}")
print(f"Recompensa media últimos 20 eps: "
      f"{np.mean(log_a['reward'][-20:]):.1f}")

# --- 2) Agente SIN elegibilidad (control) ---
print("\n" + "=" * 60)
print("B) Control: sin elegibilidad (hebbian puro)")
print("=" * 60)
env = CartPoleWrapper(seed=0)
agent_b = NeuromorphicAgent(
    obs_dim=env.obs_dim, n_actions=env.n_actions,
    N=8, M=4, V_read=0.1,
    eta=1e-4, use_eligibility=False, seed=0,
)
log_b = run_episodes(agent_b, env, n_episodes=2000, learn_enabled=True, verbose=False)
print(f"Recompensa media últimos 20 eps: "
      f"{np.mean(log_b['reward'][-20:]):.1f}")

# --- Guardar ---
out = OUT_DIR / "verify_rstdp.json"
out.write_text(json.dumps({
    "with_eligibility": log_a,
    "without_eligibility": log_b,
    "dX_total": dX_total,
}, indent=2), encoding="utf-8")
print(f"\n[ok] log en {out}")

env.close()
