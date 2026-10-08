import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from neurobot.env_wrapper import CartPoleWrapper
from neurobot.agent import NeuromorphicAgent

print("="*60)
print("PROBA 100 EPISODIOS - CONFIGURACION A (24x12) POST-FIX FISICO")
print("="*60)

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

for ep in range(100):
    obs = env.reset()
    done = False
    ep_r = 0.0
    steps = 0
    while not done:
        action = agent.act(obs)
        obs, r, done, info = env.step(action)
        ep_r += r
        steps += 1
    
    agent.learn(ep_r)
    agent.end_episode()
    
    rewards.append(ep_r)

    if (ep + 1) % 10 == 0 or ep == 0:
        avg_10 = np.mean(rewards[-10:])
        print(f"Ep {ep+1:3d}/100 | R_ep: {ep_r:6.1f} | R_bar(10): {avg_10:6.1f} | T: {agent.current_temperature():.3f} | b: {agent.baseline:6.1f}")

print("\n" + "="*60)
print(f"RESULTADO FINAL: R_bar (ultimos 20 ep): {np.mean(rewards[-20:]):.1f}")
print("="*60)
