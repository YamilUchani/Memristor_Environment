import numpy as np
import pytest

from neurobot.crossbar_brain import CrossbarBrain


def test_two_by_two_mna_matches_documented_nonideality():
    target_g = np.array(
        [[0.001, 0.0005], [0.00033333, 0.00025]],
        dtype=float,
    )
    brain = CrossbarBrain(
        N=2,
        M=2,
        seed=4,
        initial_X=0.5,
        d2d_enabled=False,
        c2c_enabled=False,
        json_memristor_nv="strukov_ideal.json",
        R_wire=1e-3,
        R_sense=100.0,
    )
    brain.X = (1.0 / target_g - brain.elec.r_on) / (
        brain.elec.r_off - brain.elec.r_on
    )

    np.testing.assert_allclose(brain.G, target_g, rtol=1e-12, atol=0.0)
    voltage = np.array([0.05, 0.0])
    ideal_current = brain.G.T @ voltage
    mna_current, relative_difference = brain.read(voltage)

    np.testing.assert_allclose(ideal_current * 1e6, [50.0, 25.0], atol=1e-4)
    np.testing.assert_allclose(mna_current * 1e6, [44.1176, 23.2558], atol=1e-3)
    assert relative_difference == pytest.approx(0.1097556, abs=1e-6)


def test_set_weights_restores_crossbar_state_and_rejects_invalid_values():
    brain = CrossbarBrain(
        N=2,
        M=2,
        seed=4,
        initial_X=0.5,
        d2d_enabled=False,
        c2c_enabled=False,
        json_memristor_nv="strukov_ideal.json",
        R_wire=1e-3,
        R_sense=100.0,
    )
    weights = np.array([[0.2, 0.4], [0.6, 0.8]])

    brain.set_weights(weights)

    np.testing.assert_array_equal(brain.X, weights)
    for row in range(brain.N):
        for column in range(brain.M):
            assert brain.cells[row][column].x == pytest.approx(
                1.0 - weights[row, column]
            )
    with pytest.raises(ValueError, match="Weight shape"):
        brain.set_weights(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        brain.set_weights(np.full((2, 2), np.nan))
