"""Entrena al agente R-STDP compartido en el gridworld 2D con obstaculos (3.3.7).

Reusa el mismo agente de CartPole/Pendulo/Vision (CrossbarBrain + SensorEncoder
+ ActionDecoder + elegibilidad REINFORCE). Solo cambian el entorno y las
dimensiones del crossbar (N=16 entradas, M=4 acciones).

Guarda: outputs/run_obstacle_seed{N}.json (log completo),
        outputs/run_obstacle_seed{N}_X.npy (matriz final),
        outputs/obstacle_seed{N}_paths.json (layout + trayectorias antes/despues).
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from neurobot.crossbar_brain import CrossbarBrain
from neurobot.decoder import ActionDecoder
from neurobot.encoder import SensorEncoder

from obstacle_env import ObstacleGridworld

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)


class ObstacleAgent:
    """Mismo R-STDP del pipeline CartPole/Pendulo."""

    def __init__(self, env, N=16, M=4, V_read=0.1, eta=1e-2,
                 temp_init=1.0, temp_min=0.3, temp_decay=0.999,
                 baseline_window=50, seed=0, k_wta=1):
        self.brain = CrossbarBrain(N=N, M=M, seed=seed, scheme="V2")
        self.encoder = SensorEncoder(obs_dim=env.obs_dim, N=N,
                                     V_read=V_read, seed=seed)
        self.decoder = ActionDecoder(n_actions=env.n_actions, M=M,
                                     sample=True, temperature=temp_init)
        self.n_actions = env.n_actions
        self.dt = 1e-4
        self.k_wta = k_wta
        self.eta = eta
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        self.reward_history = deque(maxlen=baseline_window)
        self.baseline = 0.0
        self._E_accum = np.zeros((N, M))

    def act(self, obs):
        V_rows = self.encoder.encode(obs)
        pre = self.encoder.last_pre_spikes()
        spikes, I_col, winners, sneak = self.brain.step(
            V_rows, dt=self.dt, k_wta=self.k_wta
        )
        action = self.decoder.decode(I_col)
        col = self.decoder.last_col()
        probs = self.decoder.last_probs()

        indicator = np.zeros(self.brain.M)
        indicator[col] = 1.0
        delta = indicator - probs
        self._E_accum += np.outer(pre.astype(float), delta)
        return action

    def act_deterministic(self, obs):
        """Rollout determinista (greedy) para grabar la trayectoria final."""
        V_rows = self.encoder.encode(obs)
        spikes, I_col, winners, sneak = self.brain.step(
            V_rows, dt=self.dt, k_wta=self.k_wta
        )
        self.decoder.sample = False
        action = self.decoder.decode(I_col)
        self.decoder.sample = True
        return action

    def learn(self, R_ep):
        self.reward_history.append(R_ep)
        self.baseline = float(np.mean(self.reward_history))
        signal = R_ep - self.baseline
        dX = -self.eta * self._E_accum * signal
        max_grad = float(np.max(np.abs(dX)))
        if max_grad > 0.05:
            dX = dX * (0.05 / max_grad)
        self.brain.X = np.clip(self.brain.X + dX, 0.0, 1.0)

    def end_episode(self):
        T = max(self.temp_min, self.decoder.temperature * self.temp_decay)
        self.decoder.set_temperature(T)
        self.brain.reset(keep_weights=True)
        self._E_accum.fill(0.0)


def run_episodes(agent, env, n_episodes, verbose=True):
    log = {"episode": [], "reward": [], "success": [], "steps": [],
           "X_mean": [], "baseline": [], "temperature": []}
    early_path = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        while not done:
            a = agent.act(obs)
            obs, r, done, _ = env.step(a)
            total_r += r
        if ep == 0:
            early_path = list(env.path)
        arrived = bool(env.agent == env.goal)

        agent.learn(total_r)
        agent.end_episode()

        log["episode"].append(ep)
        log["reward"].append(total_r)
        log["success"].append(1.0 if arrived else 0.0)
        log["steps"].append(env.steps)
        log["X_mean"].append(float(agent.brain.X.mean()))
        log["baseline"].append(agent.baseline)
        log["temperature"].append(agent.decoder.temperature)

        if verbose and (ep % max(1, n_episodes // 8) == 0 or ep == n_episodes - 1):
            s = log["success"][-100:] if ep >= 100 else log["success"]
            succ = float(np.mean(s))
            print(f"[ep {ep:4d}] exito(100)={succ:.2f}  "
                  f"R={total_r:+6.1f}  X_mean={agent.brain.X.mean():.4f}")

    # Rollout deterministico con la politica aprendida
    obs = env.reset()
    done = False
    while not done:
        a = agent.act_deterministic(obs)
        obs, r, done, _ = env.step(a)
    learned_path = list(env.path)
    final_success = bool(env.agent == env.goal)
    return log, early_path, learned_path, final_success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--eta", type=float, default=1e-2)
    args = ap.parse_args()

    print(f"[obstacle] seed={args.seed}  N={args.N}  M={args.M}  "
          f"ep={args.episodes}  eta={args.eta}")

    env = ObstacleGridworld(seed=args.seed, goal_reward=10.0,
                            shaping_w=0.0, collision_penalty=-0.1)
    agent = ObstacleAgent(env, N=args.N, M=args.M, eta=args.eta, seed=args.seed)

    t0 = time.time()
    log, early_path, learned_path, final_success = run_episodes(
        agent, env, n_episodes=args.episodes, verbose=True
    )
    elapsed = time.time() - t0

    if args.episodes >= 200:
        succ_last = float(np.mean(log["success"][-200:]))
    else:
        succ_last = float(np.mean(log["success"]))
    print(f"\n[resultado] exito(200)={succ_last:.3f}  "
          f"rollout final: {'EXITO' if final_success else 'fracaso'}  "
          f"tiempo={elapsed/60:.1f} min")

    fname = f"run_obstacle_seed{args.seed}"
    (OUT_DIR / f"{fname}.json").write_text(json.dumps(log, indent=2),
                                           encoding="utf-8")
    np.save(OUT_DIR / f"{fname}_X.npy", agent.brain.X)

    paths = {
        "seed": args.seed,
        "obstacles": [list(o) for o in sorted(env.obstacles)],
        "start": list(env.start),
        "goal": list(env.goal),
        "early_path": [list(p) for p in early_path],
        "learned_path": [list(p) for p in learned_path],
        "final_success": final_success,
    }
    (OUT_DIR / f"obstacle_seed{args.seed}_paths.json").write_text(
        json.dumps(paths, indent=2), encoding="utf-8"
    )
    print(f"[ok] guardado en {OUT_DIR / fname}.json y _paths.json")


if __name__ == "__main__":
    main()