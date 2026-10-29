import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obstacle_env import ObstacleGridworld
from run_obstacle_rstdp import make_agent, run_training


def test_obstacle_rstdp_learns_a_greedy_path_and_updates_weights():
    env = ObstacleGridworld(
        seed=0,
        goal_reward=10.0,
        shaping_w=0.6,
        collision_penalty=-1.0,
    )
    agent = make_agent(seed=0, eta=0.05, initial_x=0.9)
    initial_weights = agent.brain.X.copy()

    log, early_path, learned_path, success = run_training(
        agent,
        env,
        episodes=600,
        verbose=False,
    )

    assert log["episodes_requested"] == 600
    assert log["episodes_completed"] == 600
    assert log["final_success_rate"] >= 0.9
    assert success
    assert log["greedy_evaluation_success"]
    assert learned_path[0] == env.start
    assert learned_path[-1] == env.goal
    assert early_path[0] == env.start
    assert np.isfinite(agent.brain.X).all()
    assert np.any(agent.brain.X != initial_weights)
    assert len(log["delta_x_abs_mean"]) == 600
