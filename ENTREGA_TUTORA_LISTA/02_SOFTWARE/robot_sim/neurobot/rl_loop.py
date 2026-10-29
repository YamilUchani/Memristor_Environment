import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def run_episodes(agent, env, n_episodes=20,
                 learn_enabled=False, verbose=True):

    log = {
        "episode": [], "reward": [], "steps": [],
        "sneak": [], "G_mean": [], "action_hist": [],
        "dX_abs_mean": [], "signal": [], "baseline": [],
        "temperature": [],
        "angles_per_ep": [], "actions_per_ep": [],
    }

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        sneak_acc = []
        actions = []
        angles_ep = []

        while not done:
            a = agent.act(obs)
            info = agent.last_step()
            sneak_acc.append(info["sneak"])

            obs, r, done, info = env.step(a)
            angles_ep.append(float(info.get("obs_raw", [0.0, 0.0, 0.0, 0.0])[2]))
            total_r += r
            steps += 1
            actions.append(a)

        # --- aprendizaje: una sola actualización por episodio ---
        if learn_enabled:
            agent.learn(total_r)

            # stats poblados por learn()
            if agent.stats["dX_abs_mean"]:
                dX_ep = float(np.mean(agent.stats["dX_abs_mean"]))
                sig_ep = float(np.mean(agent.stats["signal"]))
                agent.stats["dX_abs_mean"].clear()
                agent.stats["signal"].clear()
                agent.stats["R_ep"].clear()
            else:
                dX_ep, sig_ep = 0.0, 0.0
        else:
            dX_ep, sig_ep = 0.0, 0.0

        # --- cierre del episodio: decae temperatura, resetea LIF ---
        agent.end_episode()

        hist = np.bincount(actions, minlength=agent.n_actions).tolist()

        log["episode"].append(ep)
        log["reward"].append(total_r)
        log["steps"].append(steps)
        log["sneak"].append(float(np.mean(sneak_acc)) if sneak_acc else 0.0)
        log["G_mean"].append(float(agent.brain.G.mean()))
        log["action_hist"].append(hist)
        log["dX_abs_mean"].append(dX_ep)
        log["signal"].append(sig_ep)
        log["baseline"].append(float(agent.baseline))
        log["temperature"].append(float(agent.current_temperature()))
        log["angles_per_ep"].append(angles_ep)
        log["actions_per_ep"].append(actions)

        if verbose and (ep % max(1, n_episodes // 10) == 0 or ep == n_episodes - 1):
            print(f"[ep {ep:4d}] R={total_r:6.1f}  "
                  f"steps={steps:3d}  "
                  f"|dX|={dX_ep:.2e}  "
                  f"sig={sig_ep:+.3f}  "
                  f"b={agent.baseline:+.3f}  "
                  f"T={agent.current_temperature():.3f}  "
                  f"a={hist}")

    return log
