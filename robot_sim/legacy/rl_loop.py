import json
import numpy as np
from pathlib import Path

from neurobot.lab_bridge import get_device_config, get_neuron_config
from neurobot.encoder import SensorEncoder
from neurobot.decoder import ActionDecoder
from neurobot.crossbar_brain import CrossbarBrain
from neurobot.agent import NeuromorphicAgent
from neurobot.env_wrapper import EnvWrapper
from neurobot.logger import NeuroLogger

def load_calibrated_neuron():
    neuron_cfg = get_neuron_config()
    calib_path = Path(__file__).parent / "neurobot" / "configs" / "lif_calibrado.json"
    if calib_path.exists():
        with open(calib_path, "r") as f:
            calib_data = json.load(f)
        neuron_cfg.c_m = calib_data["c_m_value"] * 1e-9
        neuron_cfg.r_leak = calib_data["r_leak_value"] * 1e6
        neuron_cfg.r_series = calib_data["r_series_value"] * 1e3
        neuron_cfg.v_th = calib_data["v_th"]
    return neuron_cfg

def run_loop(is_training=False, episodes=100):
    print(f"\n--- Iniciando Bucle {'Entrenamiento R-STDP' if is_training else 'Política Congelada'} ---")
    env = EnvWrapper("CartPole-v1")
    obs_dim = 4
    n_actions = 2
    N = 8
    M = 8
    
    device_cfg = get_device_config("strukov")
    neuron_cfg = load_calibrated_neuron()
    
    # Inicialización
    brain = CrossbarBrain(N, M, device_cfg, neuron_cfg, {})
    encoder = SensorEncoder(obs_dim, N, V_read=0.5, pre_threshold=0.1)
    decoder = ActionDecoder(n_actions, M)
    
    # Parámetros R-STDP
    agent = NeuromorphicAgent(brain, encoder, decoder, tau_e=0.1, lr=5e-5)
    
    logger = NeuroLogger()
    
    dt = 1e-3
    rewards = []
    
    for ep in range(episodes):
        obs = env.reset(seed=ep if not is_training else None)
        agent.brain.reset()
        agent.E.fill(0.0)
        
        ep_reward = 0
        done = False
        step = 0
        
        sneak_ratios = []
        winner_counts = np.zeros(M)
        
        while not done and step < 500:
            action, sneak = agent.act(obs, dt=dt, is_training=is_training)
            obs, reward, done, _ = env.step(action)
            
            ep_reward += reward
            step += 1
            sneak_ratios.append(sneak)
            winner_counts += agent.last_winners
            
        if is_training:
            agent.learn(ep_reward)
            
        mean_sneak_ep = np.mean(sneak_ratios) if sneak_ratios else 0.0
        logger.log_episode(ep_reward, mean_sneak_ep, agent.brain.G, winner_counts, step)
        rewards.append(ep_reward)
        
        if (ep + 1) % 10 == 0:
            avg_rew = np.mean(rewards[-10:])
            print(f"Ep {ep+1:3d} | Avg Reward: {avg_rew:5.1f} | Sneak: {mean_sneak_ep:.4f} | Winners: {winner_counts.astype(int)}")
            
    env.close()
    
    # Guardar estado al finalizar
    prefix = "rl_training" if is_training else "rl_frozen"
    logger.save(prefix=prefix, G_final=agent.brain.G if is_training else None)
    print(f"Datos guardados en outputs/{prefix}.npz")
    
    return rewards

if __name__ == "__main__":
    # Fase 8: Bucle congelado
    run_loop(is_training=False, episodes=20)
    
    # Fase 9: R-STDP Learning
    run_loop(is_training=True, episodes=300)
