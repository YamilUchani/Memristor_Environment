import numpy as np
import json
from pathlib import Path

class NeuroLogger:
    def __init__(self, out_dir="outputs"):
        # Relativo a la raíz de robot_sim
        self.out_dir = Path(__file__).resolve().parent.parent / out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            "reward": [],
            "sneak_ratio": [],
            "g_mean": [],
            "g_std": [],
            "spike_rate": [],
            "entropy": []
        }
        
    def log_episode(self, reward, sneak_ratio, G, spike_counts, total_steps):
        self.history["reward"].append(reward)
        self.history["sneak_ratio"].append(sneak_ratio)
        self.history["g_mean"].append(float(np.mean(G)))
        self.history["g_std"].append(float(np.std(G)))
        
        spike_rate = spike_counts / max(1, total_steps)
        self.history["spike_rate"].append(spike_rate.tolist())
        
        total_spikes = np.sum(spike_counts)
        if total_spikes > 0:
            p = spike_counts / total_spikes
            p = p[p > 0]
            entropy = -np.sum(p * np.log2(p))
        else:
            entropy = 0.0
        self.history["entropy"].append(float(entropy))
        
    def save(self, prefix="rl_training", G_final=None):
        npz_path = self.out_dir / f"{prefix}.npz"
        np.savez(
            npz_path,
            reward=np.array(self.history["reward"]),
            sneak_ratio=np.array(self.history["sneak_ratio"]),
            g_mean=np.array(self.history["g_mean"]),
            g_std=np.array(self.history["g_std"]),
            spike_rate=np.array(self.history["spike_rate"]),
            entropy=np.array(self.history["entropy"])
        )
        
        if G_final is not None:
            json_path = self.out_dir / "crossbar_state_trained.json"
            data = {
                "_meta": {"type": "trained_G", "description": "Matriz G entrenada en robot_sim"},
                "G_shape": G_final.shape,
                "G_matrix": G_final.tolist()
            }
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
                
        return npz_path
