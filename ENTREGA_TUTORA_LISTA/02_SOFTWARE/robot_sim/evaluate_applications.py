"""Select and test frozen policies for vision, grid navigation, and Car 2D."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from car_env import CarEnv
from neurobot.reinforce_agent import NeuromorphicAgent
from neurobot.crossbar_brain import CrossbarBrain
from obstacle_env import ObstacleGridworld
from run_vision import make_patterns

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

VALIDATION_SEEDS = tuple(range(10_000, 10_020))
TEST_SEEDS = tuple(range(20_000, 20_050))
VISION_VALIDATION_SEED = 10_000
VISION_TEST_SEED = 20_000
VISION_EXAMPLES_PER_CLASS = 50
VISION_PIXEL_FLIP_PROBABILITY = 0.05

TASK_DIMENSIONS = {
    "maze": (16, 4),
    "car": (24, 8),
    "vision": (64, 8),
}

TRAINING_DESCRIPTIONS = {
    "maze": (
        "Three independent final checkpoints from seeds 0-2; the saved logs "
        "contain 746 episodes for seed 0 and 800 episodes for seeds 1 and 2."
    ),
    "car": (
        "Auto 2D checkpoints use eight steering/speed actions and are "
        "evaluated as frozen policies on held-out environment seeds."
    ),
    "vision": (
        "Three independently initialized checkpoints (seeds 0-2), each "
        "trained for 5000 episodes on the eight canonical binary patterns."
    ),
}


def _training_seed(path: Path) -> int:
    match = re.search(r"(?:run_obstacle_reinforce_seed|run_car_reinforce_seed|car_seed|run_vision_reinforce_seed|run_obstacle_seed|run_car_seed|run_vision_seed)(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot determine training seed from checkpoint name: {path.name}")
    return int(match.group(1))


def _checkpoint_metadata(path: Path) -> str:
    metadata_path = Path(f"{path}.meta.json")
    if metadata_path.is_file():
        return "present"
    return "missing; loader assumes the legacy convention X = 1 - x"


def _candidate_checkpoints(task: str) -> list[Path]:
    if task == "maze":
        paths = list(OUT.glob("run_obstacle_reinforce_seed*_X*.npy"))
    elif task == "car":
        paths = (
            list(OUT.glob("car_reinforce_seed*_X*.npy"))
            + list(OUT.glob("run_car_reinforce_seed*_X*.npy"))
        )
    else:
        paths = list(OUT.glob("run_vision_reinforce_seed*_X*.npy"))

    candidates = sorted({p.resolve() for p in paths if p.is_file()})
    if not candidates:
        raise FileNotFoundError(
            f"No trained {task} checkpoints found under {OUT}. "
            "Run the corresponding training script first."
        )
    expected_shapes = (
        {(24, 4), TASK_DIMENSIONS[task]}
        if task == "car" else {TASK_DIMENSIONS[task]}
    )
    for path in candidates:
        weights = np.load(path, mmap_mode="r")
        if weights.shape not in expected_shapes:
            raise ValueError(
                f"Checkpoint {path.name} has shape {weights.shape}; "
                f"expected one of {sorted(expected_shapes)} for {task}."
            )
    return candidates


def _make_maze_agent(seed: int, weights: Path, env: ObstacleGridworld) -> NeuromorphicAgent:
    agent = NeuromorphicAgent(
        obs_dim=env.obs_dim, n_actions=env.n_actions,
        N=16, M=4, seed=seed, encoder_mode="random",
    )
    agent.brain.load_weights(str(weights))
    agent.decoder.sample = False
    return agent


def _make_car_agent(seed: int, weights: Path) -> NeuromorphicAgent:
    crossbar_columns = np.load(weights, mmap_mode="r").shape[1]
    metadata_path = Path(f"{weights}.meta.json")
    encoder_seed = seed
    encoder_mode = "auto2d" if crossbar_columns == 8 else "random"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        encoder_seed = int(metadata.get("encoder_seed", seed))
        encoder_mode = metadata.get(
            "encoder_mode", metadata.get("encoder", encoder_mode)
        )
    agent = NeuromorphicAgent(
        obs_dim=11,
        n_actions=crossbar_columns,
        N=24,
        M=crossbar_columns,
        V_read=0.05,
        eta=0.01,
        temp_init=1.0,
        temp_min=0.30,
        temp_decay=0.9995,
        baseline_window=200,
        seed=seed,
        encoder_seed=encoder_seed,
        encoder_mode=encoder_mode,
    )
    agent.brain.load_weights(str(weights))
    agent.decoder.sample = False
    return agent


def _make_vision_brain(seed: int, weights: Path) -> CrossbarBrain:
    brain = CrossbarBrain(N=64, M=8, seed=seed, scheme="V2")
    brain.load_weights(str(weights))
    return brain


def _evaluate_maze_checkpoint(
    weights: Path, environment_seeds: tuple[int, ...]
) -> dict:
    training_seed = _training_seed(weights)
    env_for_agent = ObstacleGridworld(
        seed=training_seed,
        goal_reward=10.0,
        shaping_w=0.0,
        collision_penalty=-0.1,
    )
    agent = _make_maze_agent(training_seed, weights, env_for_agent)
    before = agent.brain.X.copy()
    returns = []
    successes = []
    steps = []

    for seed in environment_seeds:
        env = ObstacleGridworld(
            seed=seed,
            goal_reward=10.0,
            shaping_w=0.0,
            collision_penalty=-0.1,
        )
        obs = env.reset()
        done = False
        total_return = 0.0
        while not done:
            action = agent.act_deterministic(obs)
            obs, reward, done, _ = env.step(action)
            total_return += reward
        returns.append(float(total_return))
        successes.append(float(env.agent == env.goal))
        steps.append(int(env.steps))
        agent.brain.reset(keep_weights=True)

    if not np.array_equal(agent.brain.X, before):
        raise RuntimeError(f"Weights changed during frozen maze evaluation: {weights.name}")
    return {
        "training_seed": training_seed,
        "checkpoint": weights.name,
        "checkpoint_metadata": _checkpoint_metadata(weights),
        "episodes": len(environment_seeds),
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(steps)),
        "successes": successes,
        "returns": returns,
        "steps": steps,
        "weights_unchanged": True,
    }


def _evaluate_car_checkpoint(
    weights: Path, environment_seeds: tuple[int, ...]
) -> dict:
    training_seed = _training_seed(weights)
    agent = _make_car_agent(training_seed, weights)
    before = agent.brain.X.copy()
    returns = []
    successes = []
    distances = []
    reasons = []
    path_lengths = []
    steering_reversals = []
    steering_penalties = []

    for seed in environment_seeds:
        env = CarEnv(seed=seed)
        obs = env.reset()
        done = False
        total_return = 0.0
        info = {"reason": "timeout", "dist": float("nan")}
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_return += reward
        returns.append(float(total_return))
        successes.append(float(info["reason"] == "goal"))
        distances.append(float(info["dist"]))
        reasons.append(str(info["reason"]))
        path_lengths.append(float(info["path_length"]))
        steering_reversals.append(int(info["steering_reversals"]))
        steering_penalties.append(
            float(info["total_steering_smoothness_penalty"])
        )
        agent.reset_episode_state()

    if not np.array_equal(agent.brain.X, before):
        raise RuntimeError(f"Weights changed during frozen Car 2D evaluation: {weights.name}")
    return {
        "training_seed": training_seed,
        "checkpoint": weights.name,
        "checkpoint_metadata": _checkpoint_metadata(weights),
        "episodes": len(environment_seeds),
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_final_distance": float(np.mean(distances)),
        "termination_reasons": reasons,
        "successes": successes,
        "returns": returns,
        "final_distances": distances,
        "mean_path_length": float(np.mean(path_lengths)),
        "mean_steering_reversals": float(np.mean(steering_reversals)),
        "mean_steering_smoothness_penalty": float(
            np.mean(steering_penalties)
        ),
        "weights_unchanged": True,
    }


def make_noisy_vision_set(
    patterns: np.ndarray,
    *,
    seed: int,
    examples_per_class: int = VISION_EXAMPLES_PER_CLASS,
    pixel_flip_probability: float = VISION_PIXEL_FLIP_PROBABILITY,
    excluded_examples: set[bytes] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if examples_per_class < 1:
        raise ValueError("examples_per_class must be at least 1")
    if not 0.0 <= pixel_flip_probability <= 1.0:
        raise ValueError("pixel_flip_probability must be in [0, 1]")

    rng = np.random.default_rng(seed)
    seen = set(excluded_examples or ())
    images = []
    labels = []
    for label, pattern in enumerate(patterns):
        class_examples = 0
        attempts = 0
        while class_examples < examples_per_class:
            attempts += 1
            if attempts > examples_per_class * 1000:
                raise RuntimeError("Could not generate enough unique noisy pattern examples")
            candidate = np.logical_xor(
                pattern > 0.5,
                rng.random(pattern.shape) < pixel_flip_probability,
            ).astype(float)
            key = candidate.tobytes()
            if key in seen:
                continue
            seen.add(key)
            images.append(candidate)
            labels.append(label)
            class_examples += 1
    return np.asarray(images), np.asarray(labels, dtype=int)


def make_disjoint_noisy_vision_sets(
    patterns: np.ndarray,
    *,
    validation_seed: int,
    test_seed: int,
    validation_examples_per_class: int,
    test_examples_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    training_examples = {pattern.astype(float).tobytes() for pattern in patterns}
    validation_images, validation_labels = make_noisy_vision_set(
        patterns,
        seed=validation_seed,
        examples_per_class=validation_examples_per_class,
        excluded_examples=training_examples,
    )
    validation_keys = {image.tobytes() for image in validation_images}
    test_images, test_labels = make_noisy_vision_set(
        patterns,
        seed=test_seed,
        examples_per_class=test_examples_per_class,
        excluded_examples=training_examples | validation_keys,
    )
    test_keys = {image.tobytes() for image in test_images}
    if not validation_keys.isdisjoint(test_keys):
        raise RuntimeError("Generated vision validation and test sets overlap")
    return (
        validation_images,
        validation_labels,
        test_images,
        test_labels,
        test_seed,
    )


def _evaluate_vision_checkpoint(
    weights: Path,
    images: np.ndarray,
    labels: np.ndarray,
) -> dict:
    training_seed = _training_seed(weights)
    brain = _make_vision_brain(training_seed, weights)
    before = brain.X.copy()
    predictions = []

    for image in images:
        currents, _ = brain.read(image.ravel() * 0.05)
        scale = np.mean(np.abs(currents)) + 1e-30
        predictions.append(int(np.argmax(currents / scale)))

    predictions_array = np.asarray(predictions, dtype=int)
    confusion = np.zeros((8, 8), dtype=int)
    for label, prediction in zip(labels, predictions_array):
        confusion[int(label), int(prediction)] += 1
    per_class_recall = np.diag(confusion) / confusion.sum(axis=1)

    if not np.array_equal(brain.X, before):
        raise RuntimeError(f"Weights changed during frozen vision evaluation: {weights.name}")
    return {
        "training_seed": training_seed,
        "checkpoint": weights.name,
        "checkpoint_metadata": _checkpoint_metadata(weights),
        "examples": int(len(labels)),
        "accuracy": float(np.mean(predictions_array == labels)),
        "per_class_recall": per_class_recall.tolist(),
        "confusion_matrix": confusion.tolist(),
        "weights_unchanged": True,
    }


def evaluate_task(
    task: str,
    *,
    validation_episodes: int = len(VALIDATION_SEEDS),
    test_episodes: int = len(TEST_SEEDS),
) -> dict:
    if validation_episodes < 1 or test_episodes < 1:
        raise ValueError("Validation and test episode counts must be positive")
    candidates = _candidate_checkpoints(task)

    if task == "vision":
        patterns = make_patterns()
        (
            validation_images,
            validation_labels,
            test_images,
            test_labels,
            test_seed,
        ) = make_disjoint_noisy_vision_sets(
            patterns,
            validation_seed=VISION_VALIDATION_SEED,
            test_seed=VISION_TEST_SEED,
            validation_examples_per_class=validation_episodes,
            test_examples_per_class=test_episodes,
        )

        validation = [
            _evaluate_vision_checkpoint(path, validation_images, validation_labels)
            for path in candidates
        ]
        selected = max(
            validation,
            key=lambda result: (
                result["accuracy"],
                -result["training_seed"],
            ),
        )
        selected_path = OUT / selected["checkpoint"]
        test = _evaluate_vision_checkpoint(selected_path, test_images, test_labels)
        metric = "accuracy"
        validation_seed = VISION_VALIDATION_SEED
    else:
        validation_seeds = VALIDATION_SEEDS[:validation_episodes]
        test_seeds = TEST_SEEDS[:test_episodes]
        if set(validation_seeds) & set(test_seeds):
            raise ValueError("Validation and test environment seeds must be disjoint")
        evaluator = (
            _evaluate_maze_checkpoint if task == "maze"
            else _evaluate_car_checkpoint
        )
        validation = [
            evaluator(path, validation_seeds)
            for path in candidates
        ]
        metric = "success_rate"
        selected = max(
            validation,
            key=lambda result: (
                result[metric],
                result["mean_return"],
                -result["training_seed"],
            ),
        )
        selected_path = OUT / selected["checkpoint"]
        test = evaluator(selected_path, test_seeds)
        validation_seed = list(validation_seeds)
        test_seed = list(test_seeds)

    report = {
        "protocol": {
            "task": task,
            "stages": {
                "training": TRAINING_DESCRIPTIONS[task],
                "validation": (
                    "Frozen checkpoints ranked using validation-only scenarios "
                    "or generated examples."
                ),
                "test": (
                    "Only the validation-selected checkpoint is evaluated; "
                    "the test set is not used for tuning or selection."
                ),
            },
            "validation_seed_or_set": validation_seed,
            "test_seed_or_set": test_seed,
            "vision_pixel_flip_probability": (
                VISION_PIXEL_FLIP_PROBABILITY if task == "vision" else None
            ),
            "vision_examples_per_class": (
                {
                    "validation": validation_episodes,
                    "test": test_episodes,
                }
                if task == "vision" else None
            ),
            "vision_training_examples_excluded": task == "vision",
            "validation_test_disjoint": True,
            "selection_metric": metric,
            "selection_tie_break": (
                "Higher mean return, then lower training seed; remaining ties "
                "use the first lexically sorted checkpoint"
                if task in ("maze", "car")
                else "Lower training seed"
            ),
            "learning_enabled_during_validation_and_test": False,
            "weights_unchanged_checked": True,
            "selected_checkpoint": selected["checkpoint"],
        },
        "validation_candidates": validation,
        "test": test,
    }
    output_path = OUT / f"{task}_train_validation_test.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["output"] = str(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select frozen application checkpoints on validation data, then test once."
    )
    parser.add_argument(
        "--task",
        choices=("maze", "car", "vision", "all"),
        default="all",
    )
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=50)
    args = parser.parse_args()

    tasks = ("maze", "car", "vision") if args.task == "all" else (args.task,)
    for task in tasks:
        report = evaluate_task(
            task,
            validation_episodes=args.validation_episodes,
            test_episodes=args.test_episodes,
        )
        test = report["test"]
        print(
            f"{task}: selected={report['protocol']['selected_checkpoint']} "
            f"test_{report['protocol']['selection_metric']}="
            f"{test[report['protocol']['selection_metric']]:.3f}; "
            f"report={report['output']}"
        )


if __name__ == "__main__":
    main()
