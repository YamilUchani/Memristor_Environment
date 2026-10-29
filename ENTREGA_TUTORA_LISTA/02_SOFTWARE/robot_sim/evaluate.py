"""Select and test frozen CartPole policies on disjoint seeded episodes."""
import json
from pathlib import Path

import numpy as np

from neurobot.reinforce_agent import NeuromorphicAgent
from neurobot.env_wrapper import CartPoleWrapper

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TRAINING_SEEDS = (0, 1, 2, 3, 4)
VALIDATION_SEEDS = tuple(range(10_000, 10_050))
TEST_SEEDS = tuple(range(20_000, 20_050))
OUTPUT_PATH = OUT / "cartpole_train_validation_test.json"


def make_agent(training_seed: int, weights_path: Path) -> NeuromorphicAgent:
    metadata_path = Path(f"{weights_path}.meta.json")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Trained weights not found: {weights_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Weight metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("convention") != "X = 1 - x":
        raise ValueError(f"Incompatible state convention in {metadata_path}")
    if (metadata.get("N"), metadata.get("M")) != (24, 12):
        raise ValueError(f"Expected 24x12 weights in {metadata_path}")

    variability = metadata.get("variability")
    if not isinstance(variability, dict):
        raise ValueError(f"Missing variability configuration in {metadata_path}")

    agent = NeuromorphicAgent(
        obs_dim=4,
        n_actions=12,
        N=24,
        M=12,
        V_read=0.05,
        temp_init=0.05,
        temp_min=0.05,
        temp_decay=1.0,
        baseline_window=200,
        initial_X=float(metadata["initial_X"]),
        seed=training_seed,
        json_memristor_nv=metadata["json_memristor_nv"],
        d2d_enabled=bool(variability["d2d_enabled"]),
        c2c_enabled=bool(variability["c2c_enabled"]),
        d2d_sigma=float(variability["d2d_sigma"]),
        c2c_sigma=float(variability["c2c_sigma"]),
        c2c_theta=float(variability["c2c_theta"]),
        c2c_dt=float(variability["c2c_dt"]),
    )
    agent.brain.load_weights(str(weights_path))
    agent.decoder.sample = False
    return agent


def run_frozen_episodes(
    agent: NeuromorphicAgent, environment_seeds: tuple[int, ...]
) -> list[float]:
    initial_weights = agent.brain.X.copy()
    rewards = []
    for episode_seed in environment_seeds:
        env = CartPoleWrapper(seed=episode_seed, death_penalty=50.0)
        try:
            obs = env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            rewards.append(float(episode_reward))
        finally:
            env.close()
        agent.reset_episode_state()

    if not np.array_equal(agent.brain.X, initial_weights):
        raise RuntimeError("Crossbar weights changed during frozen evaluation")
    return rewards


def summarize_candidate(training_seed: int) -> dict:
    weights_path = OUT / f"run_24x12_A_d2d_c2c_seed{training_seed}_X.npy"
    agent = make_agent(training_seed, weights_path)
    rewards = run_frozen_episodes(agent, VALIDATION_SEEDS)
    values = np.asarray(rewards, dtype=float)
    metadata = json.loads(
        Path(f"{weights_path}.meta.json").read_text(encoding="utf-8")
    )
    variability = metadata["variability"]
    return {
        "training_seed": training_seed,
        "weights": weights_path.name,
        "validation_episodes": len(rewards),
        "validation_reward_mean": float(np.mean(values)),
        "validation_reward_std": float(np.std(values)),
        "validation_rewards": rewards,
        "d2d_enabled": bool(variability["d2d_enabled"]),
        "c2c_enabled": bool(variability["c2c_enabled"]),
        "weights_unchanged": True,
    }


def run_protocol() -> dict:
    if set(VALIDATION_SEEDS) & set(TEST_SEEDS):
        raise ValueError("CartPole validation and test seeds must be disjoint")

    validation = [summarize_candidate(seed) for seed in TRAINING_SEEDS]
    selected = max(
        validation,
        key=lambda result: result["validation_reward_mean"],
    )
    selected_seed = int(selected["training_seed"])
    test_agent = make_agent(selected_seed, OUT / selected["weights"])
    test_rewards = run_frozen_episodes(test_agent, TEST_SEEDS)
    test_values = np.asarray(test_rewards, dtype=float)
    test_by_training_seed = []
    for candidate in validation:
        candidate_seed = int(candidate["training_seed"])
        if candidate_seed == selected_seed:
            candidate_rewards = test_rewards
        else:
            candidate_agent = make_agent(candidate_seed, OUT / candidate["weights"])
            candidate_rewards = run_frozen_episodes(candidate_agent, TEST_SEEDS)
        candidate_values = np.asarray(candidate_rewards, dtype=float)
        test_by_training_seed.append(
            {
                "training_seed": candidate_seed,
                "weights": candidate["weights"],
                "episodes": len(candidate_rewards),
                "reward_mean": float(np.mean(candidate_values)),
                "reward_std": float(np.std(candidate_values)),
                "reward_min": float(np.min(candidate_values)),
                "reward_max": float(np.max(candidate_values)),
                "weights_unchanged": True,
            }
        )
    return {
        "protocol": {
            "environment": "CartPole-v1 with repository CartPoleWrapper",
            "training_artifacts": "Five independently trained 300-episode checkpoints",
            "training_seeds": list(TRAINING_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "test_seeds": list(TEST_SEEDS),
            "validation_and_test_disjoint": True,
            "selection_rule": "Highest mean frozen-policy reward on validation seeds",
            "decoder_policy": "greedy (decoder.sample=False)",
            "learning_enabled": False,
            "weights_checked_unchanged": True,
            "variability": "D2D/C2C settings retained from each checkpoint metadata",
            "test_set_used_for_selection_or_tuning": False,
            "all_training_seeds_tested": True,
        },
        "validation": validation,
        "selected_training_seed": selected_seed,
        "test_by_training_seed": test_by_training_seed,
        "test": {
            "selected_weights": selected["weights"],
            "episodes": len(test_rewards),
            "reward_mean": float(np.mean(test_values)),
            "reward_std": float(np.std(test_values)),
            "reward_min": float(np.min(test_values)),
            "reward_max": float(np.max(test_values)),
            "rewards": test_rewards,
        },
    }


def main() -> None:
    report = run_protocol()
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for candidate in report["validation"]:
        print(
            f"validation training_seed={candidate['training_seed']} "
            f"mean={candidate['validation_reward_mean']:.2f} "
            f"std={candidate['validation_reward_std']:.2f}"
        )
    test = report["test"]
    print(f"selected training seed: {report['selected_training_seed']}")
    print(
        f"test episodes={test['episodes']} mean={test['reward_mean']:.2f} "
        f"std={test['reward_std']:.2f} min={test['reward_min']:.2f} "
        f"max={test['reward_max']:.2f}"
    )
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
