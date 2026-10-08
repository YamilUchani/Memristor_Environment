"""
Script headless de verificación de CartPole (2000 episodios).
Prueba con r_off=16000.0 y guarda resultados en JSON y NPY.
"""
import json
import time
from pathlib import Path

import numpy as np

from neurobot.env_wrapper import CartPoleWrapper
from neurobot.agent import NeuromorphicAgent

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

def main():
    print("="*60)
    print("VERIFICACION CARTPOLE — 2000 EPISODIOS (r_off=16k)")
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

    log = {
        "episode": [], "reward": [], "X_mean": [], "G_mean": [], "baseline": []
    }

    t0 = time.time()
    for ep in range(2000):
        obs = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done:
            a = agent.act(obs)
            obs, r, done, info = env.step(a)
            total_r += r
            steps += 1

        agent.learn(total_r)
        agent.end_episode()

        log["episode"].append(ep)
        log["reward"].append(float(total_r))
        log["X_mean"].append(float(agent.brain.X.mean()))
        log["G_mean"].append(float(agent.brain.G.mean()))
        log["baseline"].append(float(agent.baseline))

        if (ep + 1) % 200 == 0 or ep == 0:
            avg_20 = float(np.mean(log["reward"][-20:]))
            print(f"[ep {ep+1:4d}] R={total_r:6.1f} | R_bar(20)={avg_20:6.1f} | G_mean={agent.brain.G.mean():.2e} | X_mean={agent.brain.X.mean():.4f}")

    elapsed = time.time() - t0
    r_last50 = float(np.mean(log["reward"][-50:]))
    r_max = float(np.max(log["reward"]))
    g_final = float(log["G_mean"][-1])
    x_final = float(log["X_mean"][-1])

    print("\n" + "="*60)
    print(f"RESULTADO FINAL (2000 ep):")
    print(f"  R_bar(50) = {r_last50:.1f}")
    print(f"  R_max     = {r_max:.1f}")
    print(f"  G_mean    = {g_final:.4e}")
    print(f"  X_mean    = {x_final:.4f}")
    print(f"  Tiempo    = {elapsed/60:.1f} min")
    print("="*60)

    # Guardar NPY y JSON
    np.save(OUT_DIR / "cartpole_X.npy", agent.brain.X)
    (OUT_DIR / "verify_cartpole_2000.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[ok] Guardado en cartpole_X.npy y verify_cartpole_2000.json")

if __name__ == "__main__":
    main()
