"""
Corrida headless del agente neuromórfico en Pendulum-v1.
Guarda: run_pendulum_seed{N}.json + run_pendulum_seed{N}_X.npy
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

from neurobot.encoder import SensorEncoder
from neurobot.decoder import ActionDecoder
from neurobot.crossbar_brain import CrossbarBrain

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)


# ------------------------------------------------------------------
# Wrapper del péndulo: convierte acción discreta en torque continuo
# ------------------------------------------------------------------
class PendulumWrapper:
    """
    Pendulum-v1 tiene acción continua en [-2, 2] (torque).
    Discretizamos en M niveles: 6 negativos, 6 positivos.
    """
    def __init__(self, n_levels: int = 12, max_torque: float = 2.0, seed: int = 0):
        self.env = gym.make("Pendulum-v1")
        self.n_levels = n_levels
        self.max_torque = max_torque
        self.obs_dim = 3
        self.n_actions = n_levels
        self.seed = seed

        # 6 niveles negativos + 6 positivos
        half = n_levels // 2
        self.torques = np.concatenate([
            np.linspace(-max_torque, 0, half, endpoint=False),
            np.linspace(0, max_torque, n_levels - half),
        ])

        # Media/desv para normalizar obs del péndulo
        self._mean = np.array([0.0, 0.0, 0.0])
        self._std = np.array([1.0, 1.0, 8.0])

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed)
        self.seed = None  # solo la primera vez
        return self._obs(obs)

    def step(self, action: int):
        torque = self.torques[int(action)]
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([torque], dtype=np.float32)
        )
        done = terminated or truncated
        return self._obs(obs), float(reward) * 0.1, done, info

    def _obs(self, obs):
        obs = np.asarray(obs, dtype=float)
        return np.clip((obs - self._mean) / self._std, -3.0, 3.0)

    def close(self):
        self.env.close()


# ------------------------------------------------------------------
# Agente mínimo (reutiliza piezas del que ya tenés)
# ------------------------------------------------------------------
class PendulumAgent:
    def __init__(self, env, N=24, M=12, V_read=0.1,
                 eta=5e-3, temp_init=1.0, temp_min=0.3, temp_decay=0.995,
                 baseline_window=200, seed=0):
        self.brain = CrossbarBrain(N=N, M=M, seed=seed, scheme="V2")
        self.encoder = SensorEncoder(obs_dim=env.obs_dim, N=N,
                                     V_read=V_read, seed=seed)
        self.decoder = ActionDecoder(n_actions=env.n_actions, M=M,
                                     sample=True, temperature=temp_init)
        self.n_actions = env.n_actions
        self.dt = 1e-4
        self.k_wta = 1
        self.eta = eta
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.temp_decay = temp_decay

        from collections import deque
        self.reward_history = deque(maxlen=baseline_window)
        self.baseline = 0.0
        self._E_accum = np.zeros((N, M))
        self._last = {}

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

        self._last = {"pre": pre, "spikes": spikes, "col": col}
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
        self._last = {}


# ------------------------------------------------------------------
# Loop principal
# ------------------------------------------------------------------
def run_episodes(agent, env, n_episodes, verbose=True):
    log = {"episode": [], "reward": [], "steps": [],
           "X_mean": [], "baseline": [], "temperature": []}

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        steps = 0

        while not done:
            a = agent.act(obs)
            obs, r, done, _ = env.step(a)
            total_r += r
            steps += 1

        agent.learn(total_r)
        agent.end_episode()

        log["episode"].append(ep)
        log["reward"].append(total_r)
        log["steps"].append(steps)
        log["X_mean"].append(float(agent.brain.X.mean()))
        log["baseline"].append(agent.baseline)
        log["temperature"].append(agent.decoder.temperature)

        if verbose and (ep % max(1, n_episodes // 10) == 0 or ep == n_episodes - 1):
            r_last = np.mean(log["reward"][-20:]) if len(log["reward"]) >= 20 else total_r
            print(f"[ep {ep:4d}] R={total_r:8.1f}  R_bar(20)={r_last:8.1f}  "
                  f"T={agent.decoder.temperature:.3f}")

    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--M", type=int, default=12)
    ap.add_argument("--eta", type=float, default=5e-4)
    args = ap.parse_args()

    print(f"[pendulum] seed={args.seed}  N={args.N}  M={args.M}  "
          f"ep={args.episodes}  eta={args.eta}")

    env = PendulumWrapper(n_levels=args.M, seed=args.seed)
    agent = PendulumAgent(
        env, N=args.N, M=args.M,
        eta=args.eta, temp_init=1.0, temp_min=0.3,
        baseline_window=200, seed=args.seed,
    )

    t0 = time.time()
    log = run_episodes(agent, env, n_episodes=args.episodes, verbose=True)
    elapsed = time.time() - t0

    # Resumen
    r_last20 = float(np.mean(log["reward"][-20:]))
    r_max = float(np.max(log["reward"]))
    print(f"\n[resultado] R_bar(20)={r_last20:.1f}  R_max={r_max:.1f}  "
          f"tiempo={elapsed/60:.1f} min")

    # Guardar
    fname = f"run_pendulum_seed{args.seed}"
    (OUT_DIR / f"{fname}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    np.save(OUT_DIR / f"{fname}_X.npy", agent.brain.X)
    print(f"[ok] guardado en {OUT_DIR / fname}.json")

    env.close()


if __name__ == "__main__":
    main()
