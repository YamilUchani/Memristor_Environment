import numpy as np
import sys
from pathlib import Path

ROOT = Path("g:/Github/Software de simulacion/Memristor_Environment")
sys.path.insert(0, str(ROOT))

from robot_sim.neurobot.env_wrapper import CartPoleWrapper
from robot_sim.neurobot.agent import NeuromorphicAgent

env = CartPoleWrapper(seed=0)
agent = NeuromorphicAgent(
    obs_dim=env.obs_dim, n_actions=env.n_actions,
    N=12, M=4, V_read=0.1,
    eta=0.05,
    temp_init=1.0, temp_min=0.3, temp_decay=0.999,
    baseline_window=200, seed=0,
    json_memristor_nv="strukov_ideal.json",
    json_memristor_v="memristor_volatile.json",
    json_lif="lif_config.json"
)

# Simulamos 150 episodios
for ep in range(150):
    obs = env.reset()
    done = False
    ep_r = 0.0
    while not done:
        action = agent.act(obs)
        obs, r, done, info = env.step(action)
        ep_r += r
    agent.learn(ep_r)
    agent.end_episode()

    if (ep+1) % 10 == 0:
        print(f"Ep {ep+1:3d} | R_ep: {ep_r:6.1f} | p1: {np.mean(agent.decoder.last_probs()[1::2]):.3f}")

print("Done headless test.")
