import numpy as np
import time
from neurobot.env_wrapper import CartPoleWrapper, BACKEND
from neurobot.agent import NeuromorphicAgent
from neurobot.rl_loop import run_episodes, OUT_DIR
import json

print(f"[env] backend = {BACKEND}")

env = CartPoleWrapper(seed=0)
agent = NeuromorphicAgent(
    obs_dim=env.obs_dim,
    n_actions=env.n_actions,
    N=8, M=4,
    V_read=0.1,
    steps_per_action=1,
    seed=0,
)
print(f"[agent] N={agent.brain.N} M={agent.brain.M}  n_actions={agent.n_actions}")

# --- Throughput test ---
print("\n--- Iniciando prueba de throughput ---")
V_test = np.full(agent.brain.N, 0.1)
t0 = time.perf_counter()
n_steps = 500
for _ in range(n_steps):
    agent.brain.step(V_test, dt=agent.dt, k_wta=1)
elapsed = time.perf_counter() - t0
print(f"[throughput] {n_steps} pasos en {elapsed:.3f} s = {n_steps/elapsed:.0f} pasos/s")
print(f"[throughput] {elapsed/n_steps*1e6:.1f} µs por paso físico")
# -----------------------

print("\n--- Iniciando evaluación congelada ---")
log = run_episodes(agent, env, n_episodes=20,
                   learn_enabled=False, verbose=True)

# Diagnóstico de política congelada
h = np.array(log["action_hist"])   # (episodes, n_actions)
print("\nDistribución global de acciones:")
totals = h.sum(axis=0)
print(f"  acción 0: {totals[0]} veces")
print(f"  acción 1: {totals[1]} veces")

mean_r = float(np.mean(log["reward"]))
print(f"\nRecompensa media (20 eps): {mean_r:.1f}  (CartPole random ~ 22)")
print(f"sneak medio: {np.mean(log['sneak']):.3f}")

# Guardar log
(OUT_DIR / "verify_agent_log.json").write_text(
    json.dumps(log, indent=2), encoding="utf-8"
)
print(f"\n[ok] log en {OUT_DIR / 'verify_agent_log.json'}")

env.close()
