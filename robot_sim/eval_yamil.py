"""
Script de evaluación e inferencia para Yamil.npy y Yamil_1.npy (24x12).
Evaluación tanto con Softmax (T=0.25, T=0.10) como en modo Codicioso (Greedy / argmax).
"""
import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from neurobot.env_wrapper import CartPoleWrapper
from neurobot.agent import NeuromorphicAgent

def eval_policy_mode(weights_file="Yamil_1.npy", n_episodes=20, temp=0.25, greedy=False, seed=42):
    weights_path = ROOT / "outputs" / weights_file
    if not weights_path.exists():
        print(f"Error: {weights_path} no existe.")
        return

    env = CartPoleWrapper(seed=seed)
    agent = NeuromorphicAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        N=24, M=12,
        V_read=0.1,
        eta=0.005,
        temp_init=temp,
        temp_min=temp,
        baseline_window=200,
        seed=seed,
        json_memristor_nv="strukov_ideal.json",
        json_memristor_v="memristor_volatile.json",
        json_lif="lif_config.json"
    )

    agent.brain.load_weights(str(weights_path))
    
    rewards = []
    steps_per_ep = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        steps = 0
        while not done:
            if greedy:
                # Codicioso: argmax sobre corrientes de salida
                V_rows = agent.encoder.encode(obs)
                I_out, _ = agent.brain.read(V_rows)
                # Map M outputs to n_actions
                M = agent.decoder.M
                n_actions = agent.decoder.n_actions
                group_size = M // n_actions
                I_grouped = np.zeros(n_actions)
                for a in range(n_actions):
                    I_grouped[a] = np.mean(I_out[a*group_size : (a+1)*group_size])
                action = int(np.argmax(I_grouped))
            else:
                action = agent.act(obs)

            obs, r, done, info = env.step(action)
            ep_r += r
            steps += 1
        
        agent.end_episode()
        rewards.append(ep_r)
        steps_per_ep.append(steps)

    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    mean_steps = np.mean(steps_per_ep)
    mode_str = "GREEDY (argmax)" if greedy else f"SOFTMAX (T={temp})"
    print(f"[{weights_file}] Modo {mode_str:20s} | R_bar: {mean_r:6.2f} ± {std_r:5.2f} | Pasos: {mean_steps:5.1f}/500")
    return mean_r, std_r, mean_steps

if __name__ == "__main__":
    print("="*65)
    print(" EVALUACIÓN DE INFERENCIA EN MATRICES 24x12 GUARDADAS")
    print("="*65)
    for f in ["Yamil_1.npy", "Yamil.npy"]:
        eval_policy_mode(f, n_episodes=20, temp=0.25, greedy=False)
        eval_policy_mode(f, n_episodes=20, temp=0.10, greedy=False)
        eval_policy_mode(f, n_episodes=20, greedy=True)
        print("-" * 65)
