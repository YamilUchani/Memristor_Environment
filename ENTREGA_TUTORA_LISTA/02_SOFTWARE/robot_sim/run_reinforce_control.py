"""Controlled 24x12 CartPole REINFORCE experiment.

Run from the repository root with ``python robot_sim/run_reinforce_control.py``.
This standalone control mirrors run_robot.py's REINFORCE path: 24x12 crossbar,
random seeded encoder, one neural step per action, the same discrete CartPole
action mapping, episodic ``agent.learn`` update, and memristor profiles.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from neurobot.env_wrapper import CartPoleWrapper
from neurobot.reinforce_agent import NeuromorphicAgent as ReinforceAgent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=700)
    parser.add_argument("--initial-x", type=float, choices=(0.1, 0.9), default=0.9)
    parser.add_argument("--crossbar-seed", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--encoder-seed", type=int, default=0)
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, choices=(1.0, 0.05),
                        default=1.0,
                        help="Misma temperatura inicial seleccionable en run_robot.")
    parser.add_argument("--eta", type=float, default=0.005)
    parser.add_argument("--speed", type=int, default=1,
                        help="Acciones entre actualizaciones de la gráfica.")
    args = parser.parse_args()
    if args.episodes <= 0 or args.speed <= 0:
        parser.error("--episodes y --speed deben ser positivos")
    return args


def main():
    args = parse_args()
    n_inputs, n_actions = 24, 12
    force_magnitudes = np.linspace(1.0, 30.0, n_actions // 2)
    force_levels = np.concatenate((-force_magnitudes, force_magnitudes))
    output_dir = Path(__file__).resolve().parent / "outputs" / "reinforce_control"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = (f"run_reinforce_control_cb{args.crossbar_seed}_env{args.env_seed}_"
            f"enc{args.encoder_seed}_{args.episodes}ep_24x12_"
            f"X0_{args.initial_x:g}_{stamp}")

    env = CartPoleWrapper(seed=args.env_seed, death_penalty=50.0)
    agent = ReinforceAgent(
        obs_dim=env.obs_dim, n_actions=n_actions,
        N=n_inputs, M=n_actions, V_read=0.05,
        seed=args.action_seed, crossbar_seed=args.crossbar_seed,
        encoder_seed=args.encoder_seed, action_seed=args.action_seed,
        encoder_mode="random", eta=args.eta,
        temp_init=args.temperature, temp_min=0.05, temp_decay=0.995,
        baseline_window=200, epsilon=0.0,
        json_memristor_nv="strukov_ideal.json",
        json_memristor_v="memristor_volatile.json",
        json_lif="lif_config.json", initial_X=args.initial_x,
        d2d_enabled=True, c2c_enabled=True,
        d2d_sigma=0.08, c2c_sigma=0.05, c2c_theta=1.0, c2c_dt=0.02,
        record_history=True,
    )
    initial_x = agent.brain.X.copy()
    episodes = []
    running = {"stop": False, "speed": args.speed,
               "obs": None, "raw_obs": None,
               "ep_return": 0.0, "ep_rewards": [], "steps": [],
               "action_counts": np.zeros(n_actions, dtype=int),
               "winner_counts": np.zeros(n_actions, dtype=int)}

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.canvas.manager.set_window_title("REINFORCE control (run_robot) — CartPole 24x12")
    ax_scene, ax_x, ax_reward, ax_policy = axes.flat
    ax_scene.set(xlim=(-2.6, 2.6), ylim=(-0.6, 1.6), title="Robot (CartPole)",
                 aspect="equal")
    ax_scene.grid(alpha=0.2)
    cart = Rectangle((0, 0), 0.4, 0.2, color="tab:blue")
    pole, = ax_scene.plot([], [], "o-", lw=3, color="tab:orange", markersize=6)
    ax_scene.add_patch(cart)
    image = ax_x.imshow(np.zeros_like(initial_x), cmap="viridis", vmin=0, vmax=1,
                        aspect="auto")
    ax_x.set(title="Matriz X (memristores)", xlabel="Columna (acción)",
             ylabel="Fila (sensor)")
    fig.colorbar(image, ax=ax_x, label="X")
    ax_reward.set(title="Retorno por episodio", xlabel="Episodio",
                  ylabel="Recompensa")
    ax_reward.axhline(22, color="gray", ls="--", lw=1, label="aleatorio (~22)")
    line_reward, = ax_reward.plot([], [], color="tab:blue", label="retorno")
    ax_reward.legend(loc="upper left", fontsize=8)
    ax_policy.set(title="Acciones REINFORCE", xlabel="Episodio",
                  ylabel="Fracción de acciones")
    ax_policy.set_ylim(-0.03, 1.03)
    ax_policy.axhline(0.5, color="gray", ls="--", lw=1)
    line_left, = ax_policy.plot([], [], "o-", label="Izquierda (6 columnas)")
    line_right, = ax_policy.plot([], [], "o-", label="Derecha (6 columnas)")
    ax_policy.legend(loc="best", fontsize=8)

    def on_key(event):
        if event.key in ("q", "escape"):
            running["stop"] = True
        elif event.key in ("+", "=", "plus"):
            running["speed"] = min(100, running["speed"] + 1)
        elif event.key in ("-", "_", "minus"):
            running["speed"] = max(1, running["speed"] - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def refresh():
        raw = running["raw_obs"]
        if raw is not None:
            x, theta = float(raw[0]), float(raw[2])
            cart.set_xy((x - 0.2, -0.1))
            pole.set_data([x, x + 0.5 * np.sin(theta)],
                          [0.0, 0.5 * np.cos(theta)])
        image.set_data(agent.brain.X)
        returns = [ep["return"] for ep in episodes]
        if returns:
            xs = np.arange(1, len(returns) + 1)
            line_reward.set_data(xs, returns)
            ax_reward.relim()
            ax_reward.autoscale_view()
            half = n_actions // 2
            left = [sum(ep["action_share_by_column"][:half]) for ep in episodes]
            right = [sum(ep["action_share_by_column"][half:]) for ep in episodes]
            line_left.set_data(xs, left)
            line_right.set_data(xs, right)
            ax_policy.set_xlim(1, max(2, len(xs)))
        fig.suptitle(
            f"Ep {len(episodes)+1} | paso {len(running['steps'])+1} | "
            f"R_ep {running['ep_return']:.1f} | "
            f"velocidad x{running['speed']} | REINFORCE control 24x12")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    try:
        for episode_index in range(args.episodes):
            if running["stop"]:
                break
            episode_start_x = agent.brain.X.copy()
            running.update({"obs": env.reset(), "ep_return": 0.0,
                            "ep_rewards": [], "steps": [],
                            "action_counts": np.zeros(n_actions, dtype=int),
                            "winner_counts": np.zeros(n_actions, dtype=int)})
            done = False
            while not done and not running["stop"]:
                action = int(agent.act(running["obs"]))
                target_force = float(force_levels[action])
                obs, reward, done, info = env.step(action)
                step_result = agent.last_step()
                raw = np.asarray(info.get("obs_raw", obs), dtype=float)
                running["obs"] = obs
                running["raw_obs"] = raw
                running["ep_return"] += float(reward)
                running["ep_rewards"].append(float(reward))
                running["action_counts"][action] += 1
                winners = np.asarray(step_result["wta_winners_by_substep"],
                                     dtype=bool)
                running["winner_counts"] += winners.sum(axis=0)
                history = agent.brain.history
                recent = min(agent.steps_per_action, len(history["I_volatile"]))
                vcurrent = (np.mean(list(history["I_volatile"])[-recent:], axis=0)
                            if recent else np.zeros(n_actions))
                running["steps"].append({
                    "obs_raw": raw.tolist(), "action_column": action,
                    "force_target_N": target_force,
                    "force_applied_N": float(getattr(env, "_current_force", 0.0)),
                    "reward": float(reward),
                    "policy_probabilities": np.asarray(
                        step_result["probs"], dtype=float).tolist(),
                    "pre_spikes": int(np.count_nonzero(step_result["pre"])),
                    "lif_spikes_by_substep": np.count_nonzero(
                        step_result["spikes_by_substep"], axis=1).astype(int).tolist(),
                    "wta_winners_by_substep": winners.astype(int).tolist(),
                    "crossbar_current_by_column": np.asarray(
                        step_result["I_col"], dtype=float).tolist(),
                    "volatile_current_by_column": np.asarray(
                        vcurrent, dtype=float).tolist(),
                })
                if len(running["steps"]) % running["speed"] == 0:
                    refresh()
                    plt.pause(0.001)

            if running["stop"]:
                break
            update_before = len(agent.stats["dX_abs_mean"])
            agent.learn(running["ep_return"], freeze_update_above=430.0)
            learned_update = update_before < len(agent.stats["dX_abs_mean"])
            end_x = agent.brain.X.copy()
            learned_dX = (
                float(agent.stats["dX_abs_mean"][-1]) if learned_update else 0.0
            )
            agent.end_episode()
            counts = running["action_counts"]
            episodes.append({
                "episode": episode_index + 1,
                "return": float(running["ep_return"]),
                "steps": len(running["steps"]),
                "action_counts": counts.tolist(),
                "action_share_by_column": (
                    counts / max(1, counts.sum())
                ).tolist(),
                "wta_winner_counts": running["winner_counts"].tolist(),
                "delta_X_abs_mean": float(np.mean(np.abs(end_x - episode_start_x))),
                "X_start_mean": float(np.mean(episode_start_x)),
                "X_end_mean": float(np.mean(end_x)),
                "learned_dX_abs_mean": learned_dX,
                "steps_trace": running["steps"],
            })
            refresh()
            plt.pause(0.001)
    finally:
        final_x = agent.brain.X.copy()
        rows = [step for ep in episodes for step in ep["steps_trace"]]
        if rows:
            angle = np.asarray([step["obs_raw"][2] for step in rows])
            angular_velocity = np.asarray([step["obs_raw"][3] for step in rows])
            force = np.asarray([step["force_target_N"] for step in rows])
            corr_angle = float(np.corrcoef(angle, force)[0, 1])
            corr_velocity = float(np.corrcoef(angular_velocity, force)[0, 1])
        else:
            corr_angle = corr_velocity = 0.0
        result = {
            "schema": "reinforce_control_run_robot_v1",
            "algorithm": "REINFORCE (run_robot.py: agent.learn + end_episode)",
            "episodes_requested": args.episodes,
            "episodes_completed": len(episodes),
            "seeds": {"crossbar": args.crossbar_seed,
                      "environment": args.env_seed,
                      "encoder": args.encoder_seed,
                      "actions": args.action_seed},
            "initial_X": args.initial_x,
            "crossbar_shape": [n_inputs, n_actions],
            "encoder": "random",
            "force_levels_N": force_levels.tolist(),
            "learning": {"eta": args.eta, "gamma": agent.gamma,
                         "epsilon": agent.epsilon,
                         "temperature_initial": args.temperature,
                         "temperature_final": agent.decoder.temperature,
                         "temperature_min": agent.temp_min,
                         "temperature_decay": agent.temp_decay,
                         "steps_per_action": agent.steps_per_action,
                         "update": "learn(episode_return, freeze_update_above=430); direct REINFORCE X update"},
            "devices": {"nonvolatile": "strukov_ideal.json",
                        "volatile": "memristor_volatile.json",
                        "volatile_devices_in_lif_input_path": True,
                        "d2d_enabled": True, "c2c_enabled": True},
            "audit": {
                "mean_return_first_100": float(np.mean(
                    [ep["return"] for ep in episodes[:100]])) if episodes else 0.0,
                "mean_return_last_100": float(np.mean(
                    [ep["return"] for ep in episodes[-100:]])) if episodes else 0.0,
                "mean_steps_first_100": float(np.mean(
                    [ep["steps"] for ep in episodes[:100]])) if episodes else 0.0,
                "mean_steps_last_100": float(np.mean(
                    [ep["steps"] for ep in episodes[-100:]])) if episodes else 0.0,
                "target_force_correlation_with_theta": corr_angle,
                "target_force_correlation_with_theta_dot": corr_velocity,
                "final_X_delta_abs_mean": float(np.mean(np.abs(final_x - initial_x))),
            },
            "initial_X_matrix": initial_x.tolist(),
            "final_X_matrix": final_x.tolist(),
            "episodes": episodes,
        }
        output_path = output_dir / f"{stem}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        env.close()
        print(f"REINFORCE control guardado en: {output_path}")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
