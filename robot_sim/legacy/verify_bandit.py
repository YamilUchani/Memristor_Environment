"""
Prueba de sanidad #1: ¿puede el agente aprender un bandit de 2 brazos?
Si NO lo aprende, ningún ajuste va a hacerlo funcionar en CartPole.
"""
import json
from pathlib import Path

import numpy as np

from neurobot.tasks import BanditTask
from neurobot.agent import NeuromorphicAgent

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

# --- Configuración del experimento ---
N_EPISODES = 5000
P = (0.2, 0.8)              # brazo 0 malo, brazo 1 bueno

env = BanditTask(p=P, n_actions=2, seed=0)
agent = NeuromorphicAgent(
    obs_dim=env.obs_dim,
    n_actions=env.n_actions,
    N=8, M=4, V_read=0.1,
    eta=5e-3,
    temp_init=1.0, temp_min=0.3, temp_decay=0.999,
    baseline_window=50,
    seed=0,
)

log = {"ep": [], "action": [], "reward": [], "temp": [], "X_mean": []}
action_hist = np.zeros(2, dtype=int)

for ep in range(N_EPISODES):
    obs = env.reset()
    a = agent.act(obs)
    _, r, _, info = env.step(a)

    agent.learn(r)
    current_pre_sum = agent.last_step()["pre"].sum()
    agent.end_episode()

    action_hist[a] += 1
    log["ep"].append(ep)
    log["action"].append(int(a))
    log["reward"].append(float(r))
    log["temp"].append(float(agent.current_temperature()))
    log["X_mean"].append(float(agent.brain.X.mean()))

    if ep % 500 == 0 or ep == N_EPISODES - 1:
        recent = np.array(log["action"][-200:])
        frac_1 = float(np.mean(recent == 1)) if recent.size else 0.0
        r_recent = float(np.mean(log["reward"][-200:]))
        p_last = agent.decoder.last_probs()
        print(f"[ep {ep:5d}]  T={agent.current_temperature():.3f}  "
              f"frac(a=1)={frac_1:.3f}  R_mean={r_recent:.3f}  "
              f"b={agent.baseline:.3f}  pre={int(current_pre_sum)}  "
              f"p={np.array2string(p_last, precision=3)}")

# --- Cierre ---
last = np.array(log["action"][-1000:])
frac_1_final = float(np.mean(last == 1))
r_final = float(np.mean(log["reward"][-1000:]))

print(f"\n[RESULTADO]")
print(f"  Brazo óptimo: 1   (p=0.8)")
print(f"  Frac(a=1) últimos 1000:  {frac_1_final:.3f}   (random = 0.500)")
print(f"  R_mean últimos 1000:          {r_final:.3f}   (óptimo = 0.600)")

(OUT / "verify_bandit.json").write_text(
    json.dumps(log, indent=2), encoding="utf-8"
)
