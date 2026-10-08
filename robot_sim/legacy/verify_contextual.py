"""
Prueba de sanidad #2: bandit contextual de 1 paso.
El agente debe aprender: s=0 → a=0, s=1 → a=1.
"""
import json
from pathlib import Path

import numpy as np

from neurobot.tasks import ContextualBanditTask
from neurobot.agent import NeuromorphicAgent

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

N_EPISODES = 5000

env = ContextualBanditTask(n_actions=2, seed=0)
agent = NeuromorphicAgent(
    obs_dim=env.obs_dim,
    n_actions=env.n_actions,
    N=12, M=4, V_read=0.1,
    eta=1e-3,
    temp_init=1.0, temp_min=0.3, temp_decay=0.999,
    baseline_window=50,
    seed=0,
)

log = {"ep": [], "s": [], "a": [], "correct": [], "reward": [],
       "temp": [], "X_mean": []}

correct_window = []

for ep in range(N_EPISODES):
    obs = env.reset()
    s = env._s
    a = agent.act(obs)
    _, r, _, info = env.step(a)

    agent.learn(r)
    agent.end_episode()

    correct_window.append(info["correct"])
    if len(correct_window) > 200:
        correct_window.pop(0)

    log["ep"].append(ep)
    log["s"].append(s)
    log["a"].append(int(a))
    log["correct"].append(int(info["correct"]))
    log["reward"].append(float(r))
    log["temp"].append(float(agent.current_temperature()))
    log["X_mean"].append(float(agent.brain.X.mean()))

    if ep % 500 == 0 or ep == N_EPISODES - 1:
        acc = float(np.mean(correct_window))
        print(f"[ep {ep:5d}]  T={agent.current_temperature():.3f}  "
              f"acc={acc:.3f}  b={agent.baseline:.3f}")

# --- Cierre ---
last_correct = np.array(log["correct"][-1000:])
acc_final = float(np.mean(last_correct))

# Desglose por contexto (para ver si aprendió ambos)
s_arr = np.array(log["s"][-1000:])
a_arr = np.array(log["a"][-1000:])
acc_s0 = float(np.mean(a_arr[s_arr == 0] == 0)) if (s_arr == 0).any() else 0.0
acc_s1 = float(np.mean(a_arr[s_arr == 1] == 1)) if (s_arr == 1).any() else 0.0

print(f"\n[RESULTADO]")
print(f"  Accuracy global últimos 1000:  {acc_final:.3f}   (random = 0.500)")
print(f"  Accuracy s=0:                  {acc_s0:.3f}   (esperado > 0.85)")
print(f"  Accuracy s=1:                  {acc_s1:.3f}   (esperado > 0.85)")

# Diagnóstico estructural
print(f"\n[X mean por fila] {agent.brain.X.mean(axis=1).round(3)}")
print(f"[X mean por col ] {agent.brain.X.mean(axis=0).round(3)}")
print(f"[X matrix      ]\n{np.round(agent.brain.X, 3)}")

(OUT / "verify_contextual.json").write_text(
    json.dumps(log, indent=2), encoding="utf-8"
)
