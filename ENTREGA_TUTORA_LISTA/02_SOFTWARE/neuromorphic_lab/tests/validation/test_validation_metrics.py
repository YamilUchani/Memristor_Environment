import numpy as np
import pytest

from neurolab.core.validation_metrics import (
    coefficient_of_determination,
    compute_all_metrics,
    compute_metrics,
)


def test_coefficient_of_determination_uses_residual_sum_of_squares():
    reference = np.array([1.0, 2.0, 3.0])
    prediction = np.array([2.0, 4.0, 6.0])

    expected = 1.0 - np.sum((reference - prediction) ** 2) / np.sum(
        (reference - reference.mean()) ** 2
    )
    assert coefficient_of_determination(reference, prediction) == pytest.approx(expected)
    assert coefficient_of_determination(reference, prediction) == pytest.approx(-6.0)


def test_compute_metrics_uses_residual_r2():
    metrics = compute_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
        name="test",
        units="u",
    )

    assert metrics.r2 == pytest.approx(-6.0)
    assert metrics.n_points == 3


def test_compute_metrics_reports_peak_relative_error_and_range_erp_separately():
    metrics = compute_metrics(
        np.array([1.0, 3.0, 5.0]),
        np.array([1.0, 4.0, 5.0]),
        name="test",
        units="u",
    )

    assert metrics.max_rel_error_pct == pytest.approx(20.0)
    assert metrics.erp_range_pct == pytest.approx(25.0)


def test_hysteresis_metrics_require_reference_branch_or_time_alignment():
    with pytest.raises(ValueError, match="etiquetas de rama"):
        compute_all_metrics(
            t_sim=np.array([0.0, 1.0]),
            i_sim_mA=np.array([0.0, 1.0]),
            x_sim=np.array([0.0, 1.0]),
            r_sim_kohm=np.array([1.0, 2.0]),
            g_sim_us=np.array([1.0, 2.0]),
            val_data={
                "t_i": np.array([0.0, 1.0]),
                "i_val_mA": np.array([0.0, 0.5]),
                "v_interp": np.array([0.0, 1.0]),
                "wd_val": np.array([np.nan, np.nan]),
            },
            v_sim=np.array([0.0, 1.0]),
        )


def test_coefficient_of_determination_handles_constant_reference_and_shape():
    assert np.isnan(
        coefficient_of_determination(np.array([2.0, 2.0]), np.array([2.0, 3.0]))
    )
    with pytest.raises(ValueError, match="misma forma"):
        coefficient_of_determination(np.array([1.0, 2.0]), np.array([1.0]))
