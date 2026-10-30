"""
test_bayesian_sampler.py
------------------------
Unit tests for bayesian_sampler.py verifying log-priors, log-likelihoods, log-posteriors,
walker initialization, and emcee sampler execution.
"""

from typing import Dict, List, Union
import numpy as np
import pytest

from bayesian_sampler import (
    initialize_walker_positions,
    log_likelihood,
    log_posterior,
    log_prior,
    run_mcmc_sampling,
)


@pytest.fixture
def mock_processed_series() -> List[Dict[str, Union[float, np.ndarray]]]:
    """ Generates mock processed temperature-dependent spectra datasets for test fixtures.
    ------
    OUTPUT - List[Dict[str, Union[float, np.ndarray]]]: Mock datasets at two distinct temperatures.
    ------
    """
    energy_grid = np.linspace(1.8, 2.4, 100)
    # Simple simulated Gaussian intensity profiles
    int_100k = np.exp(-0.5 * ((energy_grid - 2.00) / 0.05) ** 2)
    int_200k = np.exp(-0.5 * ((energy_grid - 2.01) / 0.06) ** 2)

    return [
        {
            "temperature": 100.0,
            "energy_ev": energy_grid,
            "intensity": int_100k,
            "norm_factor": 1e4,
        },
        {
            "temperature": 200.0,
            "energy_ev": energy_grid,
            "intensity": int_200k,
            "norm_factor": 1e4,
        },
    ]


@pytest.fixture
def valid_single_peak_params() -> np.ndarray:
    """ Generates valid flat parameter array for a single peak plus log_sigma noise term.
    ------
    OUTPUT - np.ndarray: 10-element parameter vector [9 peak params, log_sigma].
    ------
    """
    return np.array(
        [
            0.04,   # gamma_0
            1e-4,   # gamma_ac
            0.03,   # gamma_lo
            0.025,  # e_lo
            0.1,    # i_0
            10.0,   # a_factor
            0.03,   # e_a
            2.00,   # e_0_zero
            1e-5,   # alpha_coeff
            -3.0,   # log_sigma
        ],
        dtype=np.float64,
    )


def test_log_prior(valid_single_peak_params: np.ndarray) -> None:
    """ Tests log_prior computation for valid parameters, out-of-bounds parameters, and invalid dimensions.
    ------
    OUTPUT - None: Returns None upon successful assertion verification.
    ------
    valid_single_peak_params - np.ndarray: Fixture supplying valid flat parameters for 1 peak.
    ------
    """
    num_peaks = 1
    absorption_centers = [2.00]

    # Valid parameter set should evaluate to finite float
    lp_valid = log_prior(valid_single_peak_params, num_peaks, absorption_centers)
    assert np.isfinite(lp_valid)

    # Out-of-bounds parameter (negative gamma_0) should return -np.inf
    invalid_params = valid_single_peak_params.copy()
    invalid_params[0] = -0.01  # gamma_0 < 0
    assert log_prior(invalid_params, num_peaks, absorption_centers) == -np.inf

    # Out-of-bounds center position (exceeds absorption window)
    invalid_center = valid_single_peak_params.copy()
    invalid_center[7] = 2.50  # Center shift > 0.15 eV window from 2.00 eV
    assert log_prior(invalid_center, num_peaks, absorption_centers) == -np.inf

    # Exception check for mismatched dimensions
    with pytest.raises(ValueError):
        log_prior(valid_single_peak_params[:5], num_peaks, absorption_centers)
    with pytest.raises(ValueError):
        log_prior(valid_single_peak_params, num_peaks, [2.00, 2.20])


def test_log_likelihood(
    valid_single_peak_params: np.ndarray,
    mock_processed_series: List[Dict[str, Union[float, np.ndarray]]],
) -> None:
    """ Verifies joint log-likelihood computation across temperature series datasets.
    ------
    OUTPUT - None: Returns None upon successful assertion verification.
    ------
    valid_single_peak_params - np.ndarray: Fixture supplying valid flat parameters for 1 peak.
    mock_processed_series - List[Dict[str, Union[float, np.ndarray]]]: Mock temperature series fixture.
    ------
    """
    num_peaks = 1

    ll = log_likelihood(valid_single_peak_params, mock_processed_series, num_peaks)
    assert np.isfinite(ll)
    assert isinstance(ll, float)

    # Exception checks
    with pytest.raises(ValueError):
        log_likelihood(valid_single_peak_params[:3], mock_processed_series, num_peaks)
    with pytest.raises(ValueError):
        log_likelihood(valid_single_peak_params, [], num_peaks)


def test_log_posterior(
    valid_single_peak_params: np.ndarray,
    mock_processed_series: List[Dict[str, Union[float, np.ndarray]]],
) -> None:
    """ Evaluates log_posterior as the sum of log_prior and log_likelihood.
    ------
    OUTPUT - None: Returns None upon successful assertion verification.
    ------
    valid_single_peak_params - np.ndarray: Fixture supplying valid flat parameters for 1 peak.
    mock_processed_series - List[Dict[str, Union[float, np.ndarray]]]: Mock temperature series fixture.
    ------
    """
    num_peaks = 1
    absorption_centers = [2.00]

    lp = log_prior(valid_single_peak_params, num_peaks, absorption_centers)
    ll = log_likelihood(valid_single_peak_params, mock_processed_series, num_peaks)
    lpost = log_posterior(
        valid_single_peak_params, mock_processed_series, num_peaks, absorption_centers
    )

    assert lpost == pytest.approx(lp + ll)

    # When prior is -inf, posterior must return -inf without evaluating likelihood
    invalid_params = valid_single_peak_params.copy()
    invalid_params[0] = -1.0
    assert (
        log_posterior(invalid_params, mock_processed_series, num_peaks, absorption_centers)
        == -np.inf
    )


def test_initialize_walker_positions() -> None:
    """ Tests MCMC walker position initialization, array shape, and prior validity.
    ------
    OUTPUT - None: Returns None upon successful assertion verification.
    ------
    """
    num_peaks = 1
    absorption_centers = [2.00]
    ndim = 9 * num_peaks + 1
    num_walkers = 24  # Must be >= 2 * ndim = 20

    walker_pos = initialize_walker_positions(num_walkers, num_peaks, absorption_centers)

    assert walker_pos.shape == (num_walkers, ndim)

    # All initial positions must pass log_prior check
    for pos in walker_pos:
        assert np.isfinite(log_prior(pos, num_peaks, absorption_centers))

    # Insufficient walkers should raise ValueError
    with pytest.raises(ValueError):
        initialize_walker_positions(10, num_peaks, absorption_centers)


def test_run_mcmc_sampling(
    mock_processed_series: List[Dict[str, Union[float, np.ndarray]]],
) -> None:
    """ Performs a short integration test run of emcee MCMC sampling.
    ------
    OUTPUT - None: Returns None upon successful assertion verification.
    ------
    mock_processed_series - List[Dict[str, Union[float, np.ndarray]]]: Mock temperature series fixture.
    ------
    """
    num_peaks = 1
    absorption_centers = [2.00]
    n_walkers = 20
    n_steps = 10
    burn_in = 2

    sampler, flat_samples, flat_log_prob = run_mcmc_sampling(
        mock_processed_series,
        num_peaks,
        absorption_centers,
        n_walkers=n_walkers,
        n_steps=n_steps,
        burn_in=burn_in,
        use_parallel=False,
    )

    expected_samples_count = n_walkers * (n_steps - burn_in)
    assert len(flat_samples) == expected_samples_count
    assert len(flat_log_prob) == expected_samples_count
    assert flat_samples.shape[1] == 10  # 9 peak params + 1 log_sigma

    # Invalid steps setup check
    with pytest.raises(ValueError):
        run_mcmc_sampling(
            mock_processed_series,
            num_peaks,
            absorption_centers,
            n_steps=5,
            burn_in=10,
        )