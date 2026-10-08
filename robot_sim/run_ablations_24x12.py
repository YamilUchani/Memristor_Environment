"""
Script de Ablaciones y Robustez (Seeds 0, 1, 2) en arquitectura 24x12.
Ejecuta de forma automática y limpia:
 - Config A: Control Base (Adaptación suave de T + Acumulador de elegibilidad)
 - Config B: Sin adaptación de T (T = 0.10 fijo)
 - Config C: Sin acumulación temporal de elegibilidad (E instantáneo)
 - Config D: Aprendizaje Hebbiano no modulado por recompensa
 - Multi-seed: SEED=0, SEED=1, SEED=2 para Config A

Guarda los logs estructurados en JSON y las matrices resultantes en outputs/
"""
import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from neurobot.env_wrapper import CartPoleWrapper
from neurobot.agent import NeuromorphicAgent

# ------------------------------------------------------------------
# Subclases del agente para Ablaciones
# ------------------------------------------------------------------
class AgentAblationC(NeuromorphicAgent):
    """Config C: Sin acumulación temporal de elegibilidad."""
    def act(self, obs: np.ndarray) -> int:
        action = super().act(obs)
        pre = self._last["pre"]
        col = self._last["col"]
        probs = self._last["probs"]
        indicator = np.zeros(self.brain.M)
        indicator[col] = 1.0
        delta = indicator - probs
        self._E_accum = np.outer(pre.astype(float), delta)
        return action

class AgentAblationD(NeuromorphicAgent):
    """Config D: Hebbiano no modulado por recompensa."""
    def learn(self, R_ep: float):
        R_ep = float(R_ep)
        self.reward_history.append(R_ep)
        self.baseline = float(np.mean(self.reward_history))
        if R_ep >= 430.0:
            return
        # Hebbiano directo sin signal
        dX = -self.eta * self._E_accum
        max_grad = float(np.max(np.abs(dX)))
        if max_grad > 0.05:
            dX = dX * (0.05 / max_grad)
        self.brain.X = np.clip(self.brain.X + dX, 0.0, 1.0)
        self.stats["dX_abs_mean"].append(float(np.abs(dX).mean()))
        self.stats["signal"].append(0.0)
        self.stats["R_ep"].append(R_ep)


def run_experiment(config_name="A_base", agent_cls=NeuromorphicAgent, seed=0, n_episodes=150, overwrite=False, **agent_kwargs):
    log_file = ROOT / "outputs" / f"run_24x12_{config_name}_seed{seed}.json"
    weights_file = ROOT / "outputs" / f"run_24x12_{config_name}_seed{seed}_X.npy"

    if log_file.exists() and weights_file.exists() and not overwrite:
        print(f"[skip] Saltando experimento {config_name} (Seed {seed}) - Ya completado previamente.")
        with open(log_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"\n" + "="*60)
    print(f" INICIANDO EXPERIMENTO: {config_name} | SEED: {seed} | EPISODIOS: {n_episodes}")
    print("="*60)

    env = CartPoleWrapper(seed=seed, death_penalty=50.0)
    
    default_kwargs = dict(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        N=24, M=12,
        V_read=0.1,
        eta=0.005,
        temp_init=1.0,
        temp_min=0.05,
        temp_decay=0.995,
        baseline_window=200,
        seed=seed,
        json_memristor_nv="strukov_ideal.json",
        json_memristor_v="memristor_volatile.json",
        json_lif="lif_config.json"
    )
    default_kwargs.update(agent_kwargs)
    
    agent = agent_cls(**default_kwargs)
    
    rewards = []
    temps = []
    baselines = []

    for ep in range(n_episodes):
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
        temps.append(agent.current_temperature())
        baselines.append(agent.baseline)

        if (ep + 1) % 10 == 0 or ep == 0:
            avg_10 = np.mean(rewards[-10:])
            print(f"Ep {ep+1:3d}/{n_episodes} | R_ep: {ep_r:6.1f} | R_bar(10): {avg_10:6.1f} | T: {agent.current_temperature():.3f} | b: {agent.baseline:6.1f}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    agent.brain.save_weights(str(weights_file))
    
    log_data = {
        "config_name": config_name,
        "seed": seed,
        "n_episodes": n_episodes,
        "rewards": rewards,
        "temperatures": temps,
        "baselines": baselines,
        "final_mean_reward_last_20": float(np.mean(rewards[-20:]))
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

    print(f"[ok] Experimento {config_name} guardado en {log_file.name}")
    return log_data

def run_all_ablations_and_seeds():
    # 1. Corridas de Ablación (Arquitectura 24x12, Seed 0)
    # A: Base Control
    run_experiment("A_base", NeuromorphicAgent, seed=0, n_episodes=150, overwrite=True)
    
    # B: Temperatura Fija 0.10
    run_experiment("B_tempfixed", NeuromorphicAgent, seed=0, n_episodes=150, temp_init=0.10, temp_min=0.10, temp_decay=1.0, overwrite=True)
    
    # C: Sin acumulación de elegibilidad
    run_experiment("C_noaccum", AgentAblationC, seed=0, n_episodes=150, overwrite=True)
    
    # D: Hebbiano sin modulación
    run_experiment("D_hebbian", AgentAblationD, seed=0, n_episodes=150, overwrite=True)
    
    # 2. Corridas de Reproducibilidad (Seeds 1 y 2)
    run_experiment("A_base", NeuromorphicAgent, seed=1, n_episodes=150, overwrite=True)
    run_experiment("A_base", NeuromorphicAgent, seed=2, n_episodes=150, overwrite=True)

if __name__ == "__main__":
    run_all_ablations_and_seeds()
