"""
Corridas de ablación con la física REAL del memristor (R_off=16kΩ, 160x dinámico).

Configuraciones:
  A  REINFORCE acumulado (base)         — seeds 0, 1, 2
  B  temp_min=0.10 (menos exploración)  — seed 0
  C  sin acumulación de elegibilidad    — seed 0
  D  sin modulación por recompensa      — seed 0

Uso:
    python run_ablations_real_physics.py
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys_path = sys = __import__("sys")
sys.path.insert(0, str(ROOT))

from neurobot.env_wrapper import CartPoleWrapper
from neurobot.agent import NeuromorphicAgent
from neurobot.rl_loop import run_episodes, OUT_DIR

OUT_DIR.mkdir(exist_ok=True, parents=True)

N_EPISODES = 300
N, M = 24, 12
ETA = 0.005


class AgentAblationC(NeuromorphicAgent):
    """Config C: Sin traza de elegibilidad (_E_accum = 0 constante)."""
    def act(self, obs: np.ndarray) -> int:
        action = super().act(obs)
        self._E_accum = np.zeros((self.brain.N, self.brain.M))
        return action


class AgentAblationD(NeuromorphicAgent):
    """Config D: Hebbiano no modulado por recompensa."""
    def learn(self, R_ep: float):
        R_ep = float(R_ep)
        self.reward_history.append(R_ep)
        self.baseline = float(np.mean(self.reward_history))
        if R_ep >= 430.0:
            return
        dX = -self.eta * self._E_accum
        max_grad = float(np.max(np.abs(dX)))
        if max_grad > 0.05:
            dX = dX * (0.05 / max_grad)
        self.brain.X = np.clip(self.brain.X + dX, 0.0, 1.0)
        self.stats["dX_abs_mean"].append(float(np.abs(dX).mean()))
        self.stats["signal"].append(0.0)
        self.stats["R_ep"].append(R_ep)


def run_config(mode: str, seed: int):
    print(f"\n{'='*60}")
    print(f"[{mode}] SEED={seed} | EPISODIOS={N_EPISODES}")
    print(f"{'='*60}")

    env = CartPoleWrapper(seed=seed, death_penalty=50.0)

    agent_cls = NeuromorphicAgent
    temp_min = 0.05
    if mode == "B":
        temp_min = 0.10
    elif mode == "C":
        agent_cls = AgentAblationC
    elif mode == "D":
        agent_cls = AgentAblationD

    agent = agent_cls(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        N=N, M=M, V_read=0.1,
        eta=ETA,
        temp_init=1.0,
        temp_min=temp_min,
        temp_decay=0.995,
        baseline_window=200,
        seed=seed,
        json_memristor_nv="strukov_ideal.json",
        json_memristor_v="memristor_volatile.json",
        json_lif="lif_config.json"
    )

    t0 = time.time()
    log = run_episodes(agent, env, n_episodes=N_EPISODES, learn_enabled=True, verbose=False)
    elapsed = time.time() - t0

    rewards = log["reward"]
    r_last50 = float(np.mean(rewards[-50:]))
    r_last20 = float(np.mean(rewards[-20:]))
    r_max = float(np.max(rewards))

    print(f"  Tiempo: {elapsed:.1f} s")
    print(f"  R_bar (ult 20 eps): {r_last20:.1f}")
    print(f"  R_bar (ult 50 eps): {r_last50:.1f}")
    print(f"  R_max:             {r_max:.1f}")

    # Estructura del log compatible con generador de figuras
    log_data = {
        "config_name": f"{mode}_base" if mode == "A" else mode,
        "seed": seed,
        "n_episodes": N_EPISODES,
        "rewards": rewards,
        "temperatures": log["temperature"],
        "baselines": log["baseline"],
        "final_mean_reward_last_20": r_last20
    }

    # Guardar json y weights
    json_name = f"run_24x12_{mode}_base_seed{seed}.json" if mode == "A" else f"run_24x12_{mode}_seed{seed}.json"
    npy_name = f"run_24x12_{mode}_base_seed{seed}_X.npy" if mode == "A" else f"run_24x12_{mode}_seed{seed}_X.npy"

    (OUT_DIR / json_name).write_text(json.dumps(log_data, indent=2), encoding="utf-8")
    agent.brain.save_weights(str(OUT_DIR / npy_name))
    
    print(f"  [json] {json_name}")
    print(f"  [npy]  {npy_name}")

    env.close()
    return {
        "mode": mode,
        "seed": seed,
        "r_last20": r_last20,
        "r_last50": r_last50,
        "r_max": r_max,
        "elapsed_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="Correr solo un modo: A | B | C | D")
    args = ap.parse_args()

    runs = []
    if args.only is None or args.only == "A":
        runs += [("A", 0), ("A", 1), ("A", 2), ("A", 3), ("A", 4)]
    if args.only is None or args.only == "B":
        runs += [("B", 0)]
    if args.only is None or args.only == "C":
        runs += [("C", 0)]
    if args.only is None or args.only == "D":
        runs += [("D", 0)]

    print(f"[run_ablations] {len(runs)} corridas de {N_EPISODES} eps cada una")
    print(f"[run_ablations] N={N} M={M} eta={ETA}")

    results = []
    t0 = time.time()
    for mode, seed in runs:
        try:
            res = run_config(mode, seed)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] {mode} seed={seed}: {e}")

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"RESUMEN RE-ENTRENAMIENTO ({elapsed/60:.1f} min total)")
    print(f"{'='*60}")
    print(f"{'Modo':>6} {'Seed':>5} {'R_bar(20)':>10} {'R_bar(50)':>10} {'R_max':>10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['mode']:>6} {r['seed']:>5} {r['r_last20']:>10.1f} {r['r_last50']:>10.1f} {r['r_max']:>10.1f}")

    summary_path = OUT_DIR / "ablations_real_physics_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[ok] resumen en {summary_path}")


if __name__ == "__main__":
    main()
