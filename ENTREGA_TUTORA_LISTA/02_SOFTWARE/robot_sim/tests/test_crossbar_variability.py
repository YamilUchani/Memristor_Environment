import numpy as np

from neurobot.agent import NeuromorphicAgent
from neurobot.crossbar_brain import CrossbarBrain


def make_brain(*, d2d_enabled, c2c_enabled):
    return CrossbarBrain(
        N=2,
        M=3,
        seed=17,
        initial_X=0.5,
        d2d_enabled=d2d_enabled,
        c2c_enabled=c2c_enabled,
        d2d_sigma=0.08,
        c2c_sigma=0.05,
        c2c_theta=1.0,
        c2c_dt=0.02,
    )


def test_d2d_c2c_are_seeded_per_cell_and_affect_learning_updates():
    both_a = make_brain(d2d_enabled=True, c2c_enabled=True)
    both_b = make_brain(d2d_enabled=True, c2c_enabled=True)
    d2d_only = make_brain(d2d_enabled=True, c2c_enabled=False)

    assert np.array_equal(both_a.G, both_b.G)
    assert np.unique(both_a.G).size > 1

    requested = np.full((2, 3), 0.01)
    initial_x = both_a.X.copy()
    actual_a = both_a.apply_learning_update(requested)
    actual_b = both_b.apply_learning_update(requested)
    actual_d2d = d2d_only.apply_learning_update(requested)

    assert np.array_equal(actual_a, actual_b)
    assert not np.allclose(actual_a, actual_d2d)
    assert np.allclose(both_a.X, initial_x + actual_a)


def test_disabled_variability_preserves_requested_update():
    brain = make_brain(d2d_enabled=False, c2c_enabled=False)
    requested = np.full((2, 3), -0.01)
    initial_x = brain.X.copy()

    applied = brain.apply_learning_update(requested)

    assert np.array_equal(applied, requested)
    assert np.allclose(brain.X, initial_x + requested)


