"""
test_model_selection.py
-----------------------
Unit test suite for model_selection.py verifying BIC, AIC, MLE extraction, and peak addition logic.
"""

from typing import Dict, List, Union
import numpy as np
import pytest

from temp_dependent_pl_proccessing.model_selection import (
    compute_aic,
    compute_bic,
    compute_total_data_points,
    evaluate_model_fit,
    justify_peak_addition,
)


def test_compute_total_data_points() -> None:
    """ Tests total data point counting across multiple dataset dictionaries.
    """
    mock_series: List[Dict[str, Union[float, np.ndarray]]] = [
        {"energy_ev": np.zeros(100)},
        {"energy_ev": np.zeros(150)},
    ]
    assert compute_total_data_points(mock_series) == 250

    with pytest.raises(ValueError):
        compute_total_data_points([])


def test_compute_bic_and_aic() -> None:
    """ Validates numerical correctness of BIC and AIC calculations.
    """
    max_ll = -500.0
    num_params = 10
    total_points = 1000

    # BIC = k * ln(N) - 2 * LL
    expected_bic = 10 * np.log(1000) - 2.0 * (-500.0)
    assert compute_bic(max_ll, num_params, total_points) == pytest.approx(expected_bic)

    # AIC = 2 * k - 2 * LL
    expected_aic = 2.0 * 10 - 2.0 * (-500.0)
    assert compute_aic(max_ll, num_params) == pytest.approx(expected_aic)

    with pytest.raises(ValueError):
        compute_bic(max_ll, -1, total_points)


def test_evaluate_model_fit() -> None:
    """ Tests extraction of MLE parameter set and summary dictionary construction.
    """
    flat_samples = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    flat_log_prob = np.array([-100.0, -50.0, -80.0])  # Index 1 is best
    mock_series: List[Dict[str, Union[float, np.ndarray]]] = [{"energy_ev": np.zeros(100)}]

    eval_res = evaluate_model_fit(flat_samples, flat_log_prob, mock_series, num_peaks=1)

    assert eval_res["max_log_prob"] == -50.0
    np.testing.assert_allclose(eval_res["mle_params"], [3.0, 4.0])
    assert eval_res["num_params"] == 10
    assert eval_res["total_points"] == 100


def test_justify_peak_addition() -> None:
    """ Verifies peak addition criteria based on ΔBIC threshold.
    """
    eval_k1 = {"num_peaks": 1, "bic": 1000.0}
    eval_k2_better = {"num_peaks": 2, "bic": 950.0}  # ΔBIC = -50 < -10 -> True
    eval_k2_worse = {"num_peaks": 2, "bic": 995.0}   # ΔBIC = -5 < -10 -> False

    assert justify_peak_addition(eval_k1, eval_k2_better, bic_threshold=10.0) is True
    assert justify_peak_addition(eval_k1, eval_k2_worse, bic_threshold=10.0) is False