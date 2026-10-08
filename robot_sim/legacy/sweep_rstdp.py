import numpy as np
from neurobot.env_wrapper import CartPoleWrapper
from neurobot.agent import NeuromorphicAgent
from neurobot.rl_loop import run_episodes
import copy

def run_sweep():
    etas = [1e-4, 1e-3, 1e-2]
    penalties = [1.0, 5.0, 10.0]
    
    results = {}
    
    for dp in penalties:
        print(f"\n{'='*40}")
        print(f"Death Penalty = {dp}")
        print(f"{'='*40}")
        for eta in etas:
            env = CartPoleWrapper(seed=0, death_penalty=dp)
            agent = NeuromorphicAgent(
                obs_dim=env.obs_dim, n_actions=env.n_actions,
                N=8, M=4, V_read=0.1,
                eta=eta, tau_e=0.05, baseline_alpha=0.01,
                use_eligibility=True, seed=0,
            )
            log = run_episodes(agent, env, n_episodes=1000, learn_enabled=True, verbose=False)
            mean_r = np.mean(log['reward'][-50:])
            print(f"  eta={eta:.1e} -> R={mean_r:.1f}")
            results[(dp, eta)] = mean_r
            env.close()
            
    print("\nResumen:")
    for (dp, eta), mean_r in results.items():
        print(f"dp={dp}, eta={eta:.1e} -> {mean_r:.1f}")

if __name__ == "__main__":
    run_sweep()
