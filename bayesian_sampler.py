"""
bayesian_sampler.py
-------------------
Module for formulating log-priors, joint log-likelihoods, log-posteriors, and executing
MCMC sampling using emcee for temperature-dependent PL peak analysis.
"""

import logging
import multiprocessing
from typing import Dict, List, Tuple, Union

import emcee
import numpy as np

from physics import evaluate_composite_spectrum

logger = logging.getLogger("pl_analysis.bayesian_sampler")


def log_prior(
    params: np.ndarray,
    num_peaks: int,
    absorption_centers: List[float],
    center_window_ev: float = 0.15,
) -> float:
    """ Computes the log-prior probability density for physical temperature-dependent peak parameters.
    ------
    OUTPUT - float: Calculated log-prior probability density (0.0 or prior density value if valid, -np.inf if outside bounds).
    ------
    params - np.ndarray: Flat parameter vector of length (9 * num_peaks + 1) containing physical parameters and log_sigma.
    num_peaks - int: Number of peaks comprising the physical model.
    absorption_centers - List[float]: Initial peak position energy estimates in eV derived from absorption spectra.
    center_window_ev - float: Maximum allowed deviation (eV) from absorption center for initial peak position E_0(0).
    ------
    ValueError: Raised if params length does not match (9 * num_peaks + 1) or absorption_centers count does not equal num_peaks.
    """
    expected_param_count = 9 * num_peaks + 1
    if len(params) != expected_param_count:
        logger.error(
            "Parameter count mismatch in log_prior: expected %d, got %d",
            expected_param_count,
            len(params),
        )
        raise ValueError(
            f"Expected {expected_param_count} parameters for {num_peaks} peaks, got {len(params)}."
        )

    if len(absorption_centers) != num_peaks:
        logger.error(
            "Absorption centers count (%d) does not match num_peaks (%d)",
            len(absorption_centers),
            num_peaks,
        )
        raise ValueError("Length of absorption_centers must equal num_peaks.")

    log_prior_val = 0.0

    for k in range(num_peaks):
        peak_params = params[k * 9 : (k + 1) * 9]
        gamma_0, gamma_ac, gamma_lo, e_lo, i_0, a_factor, e_a, e_0_zero, alpha_coeff = peak_params
        center_target = absorption_centers[k]

        # Linewidth bounds (eV and eV/K)
        if not (0.001 <= gamma_0 <= 0.5):
            return -np.inf
        if not (0.0 <= gamma_ac <= 0.01):
            return -np.inf
        if not (0.0 <= gamma_lo <= 0.5):
            return -np.inf
        if not (0.005 <= e_lo <= 0.2):
            return -np.inf

        # Thermal Quenching bounds
        if not (0.0 < i_0 <= 500.0):
            return -np.inf
        if not (0.0 <= a_factor <= 1e7):
            return -np.inf
        if not (0.001 <= e_a <= 2.0):
            return -np.inf

        # Peak Center Energy bounds (eV) and Bandgap Shift Coefficient (eV/K)
        if not (center_target - center_window_ev <= e_0_zero <= center_target + center_window_ev):
            return -np.inf
        if not (-0.002 <= alpha_coeff <= 0.002):
            return -np.inf

        # Informative Gaussian prior anchoring peak center at 0 K to absorption estimate
        sigma_center = center_window_ev / 3.0
        log_prior_val += -0.5 * ((e_0_zero - center_target) / sigma_center) ** 2

    # Residual noise log_sigma bounds
    log_sigma = params[-1]
    if not (-10.0 <= log_sigma <= 1.0):
        return -np.inf

    return float(log_prior_val)


def log_likelihood(
    params: np.ndarray,
    processed_series: List[Dict[str, Union[float, np.ndarray]]],
    num_peaks: int,
) -> float:
    """ Computes joint Gaussian log-likelihood across all temperature datasets simultaneously.
    ------
    OUTPUT - float: Joint log-likelihood value summed across all temperature spectra and energy bins.
    ------
    params - np.ndarray: Flat parameter vector of length (9 * num_peaks + 1).
    processed_series - List[Dict[str, Union[float, np.ndarray]]]: List of processed dataset dicts ordered by temperature.
    num_peaks - int: Number of peaks comprising the model.
    ------
    ValueError: Raised if params length is invalid or processed_series is empty.
    """
    expected_param_count = 9 * num_peaks + 1
    if len(params) != expected_param_count:
        logger.error(
            "Parameter count mismatch in log_likelihood: expected %d, got %d",
            expected_param_count,
            len(params),
        )
        raise ValueError(f"Expected {expected_param_count} parameters, got {len(params)}.")

    if not processed_series:
        logger.error("Processed series empty in log_likelihood computation.")
        raise ValueError("processed_series list cannot be empty.")

    flat_peak_params = params[:-1]
    log_sigma = params[-1]
    sigma2 = np.exp(2.0 * log_sigma)

    total_log_like = 0.0

    for dataset in processed_series:
        temp_k = float(dataset["temperature"])
        energy_ev = dataset["energy_ev"]
        intensity_exp = dataset["intensity"]

        try:
            intensity_model = evaluate_composite_spectrum(
                energy_ev, temp_k, flat_peak_params, num_peaks
            )
        except ValueError:
            return -np.inf

        residual_sq = (intensity_exp - intensity_model) ** 2
        dataset_log_like = -0.5 * np.sum(residual_sq / sigma2 + np.log(2.0 * np.pi * sigma2))
        total_log_like += dataset_log_like

    return float(total_log_like)


def log_posterior(
    params: np.ndarray,
    processed_series: List[Dict[str, Union[float, np.ndarray]]],
    num_peaks: int,
    absorption_centers: List[float],
    center_window_ev: float = 0.15,
) -> float:
    """ Computes the total log-posterior probability density by summing log-prior and joint log-likelihood.
    ------
    OUTPUT - float: Calculated unnormalized log-posterior probability density.
    ------
    params - np.ndarray: Flat parameter vector of length (9 * num_peaks + 1).
    processed_series - List[Dict[str, Union[float, np.ndarray]]]: List of processed dataset dicts.
    num_peaks - int: Number of peaks comprising the model.
    absorption_centers - List[float]: Initial peak center energy estimates in eV.
    center_window_ev - float: Maximum allowed peak center shift window (default: 0.15 eV).
    ------
    ValueError: Raised if parameter dimensions or series inputs are invalid.
    """
    lp = log_prior(params, num_peaks, absorption_centers, center_window_ev=center_window_ev)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(params, processed_series, num_peaks)
    if not np.isfinite(ll):
        return -np.inf

    return float(lp + ll)


def initialize_walker_positions(
    num_walkers: int,
    num_peaks: int,
    absorption_centers: List[float],
    jitter_std: float = 1e-3,
) -> np.ndarray:
    """ Generates initial parameter walker starting positions centered around physically plausible guesses with Gaussian noise jitter.
    ------
    OUTPUT - np.ndarray: Walker starting array of shape (num_walkers, 9 * num_peaks + 1).
    ------
    num_walkers - int: Total number of ensemble MCMC walkers.
    num_peaks - int: Number of peaks comprising the model.
    absorption_centers - List[float]: Absorption peak center estimates in eV.
    jitter_std - float: Relative standard deviation of Gaussian noise jitter applied to initial parameters.
    ------
    ValueError: Raised if num_walkers is less than twice the parameter dimension.
    """
    ndim = 9 * num_peaks + 1
    min_walkers = 2 * ndim
    if num_walkers < min_walkers:
        logger.error(
            "Insufficient walker count: %d provided, minimum required is %d",
            num_walkers,
            min_walkers,
        )
        raise ValueError(
            f"num_walkers ({num_walkers}) must be at least twice the parameter count ({min_walkers})."
        )

    base_guess: List[float] = []
    for k in range(num_peaks):
        base_guess.extend([
            0.04,                  # gamma_0 (eV)
            1e-4,                  # gamma_ac (eV/K)
            0.03,                  # gamma_lo (eV)
            0.025,                 # e_lo (eV)
            0.1,                   # i_0
            10.0,                  # a_factor
            0.03,                  # e_a (eV)
            absorption_centers[k], # e_0_zero (eV)
            1e-5,                  # alpha_coeff (eV/K)
        ])
    base_guess.append(-3.0)  # log_sigma initial guess

    initial_pos = np.array(base_guess, dtype=np.float64)
    walker_positions = initial_pos + jitter_std * np.random.randn(num_walkers, ndim) * np.abs(initial_pos)

    # Re-sample any walkers that fall outside physical prior bounds
    for i in range(num_walkers):
        while not np.isfinite(log_prior(walker_positions[i], num_peaks, absorption_centers)):
            walker_positions[i] = initial_pos + jitter_std * np.random.randn(ndim) * np.abs(initial_pos)

    logger.info("Initialized %d walkers for %d dimensions (%d peaks)", num_walkers, ndim, num_peaks)
    return walker_positions


def run_mcmc_sampling(
    processed_series: List[Dict[str, Union[float, np.ndarray]]],
    num_peaks: int,
    absorption_centers: List[float],
    n_walkers: int = 60,
    n_steps: int = 3000,
    burn_in: int = 1000,
    use_parallel: bool = True,
) -> Tuple[emcee.EnsembleSampler, np.ndarray, np.ndarray]:
    """ Runs affine-invariant ensemble MCMC sampling using emcee for temperature series joint model fit.
    ------
    OUTPUT - Tuple[emcee.EnsembleSampler, np.ndarray, np.ndarray]: Tuple of (sampler instance, flat samples array post burn-in, log probabilities).
    ------
    processed_series - List[Dict[str, Union[float, np.ndarray]]]: Processed temperature series dataset dicts.
    num_peaks - int: Number of emission peaks to fit.
    absorption_centers - List[float]: Initial peak position energy targets in eV.
    n_walkers - int: Number of ensemble walkers (default: 60).
    n_steps - int: Total sampling steps per walker (default: 3000).
    burn_in - int: Number of initial steps to discard as burn-in (default: 1000).
    use_parallel - bool: Flag enabling CPU multiprocessing (default: True).
    ------
    ValueError: Raised if n_steps <= burn_in or input datasets are empty.
    """
    if n_steps <= burn_in:
        logger.error("n_steps (%d) must exceed burn_in (%d)", n_steps, burn_in)
        raise ValueError("n_steps must be strictly greater than burn_in steps.")

    ndim = 9 * num_peaks + 1
    init_positions = initialize_walker_positions(n_walkers, num_peaks, absorption_centers)
    log_prob_args = (processed_series, num_peaks, absorption_centers)

    logger.info(
        "Starting emcee MCMC sampling: ndim=%d, walkers=%d, steps=%d, burn_in=%d",
        ndim,
        n_walkers,
        n_steps,
        burn_in,
    )

    if use_parallel:
        num_cores = multiprocessing.cpu_count()
        logger.info("Enabling parallel CPU execution using %d cores.", num_cores)
        with multiprocessing.Pool(processes=num_cores) as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers,
                ndim,
                log_posterior,
                args=log_prob_args,
                pool=pool,
            )
            sampler.run_mcmc(init_positions, n_steps, progress=True)
    else:
        logger.info("Executing MCMC sampling sequentially.")
        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            log_posterior,
            args=log_prob_args,
        )
        sampler.run_mcmc(init_positions, n_steps, progress=True)

    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=burn_in, flat=True)

    logger.info("MCMC sampling complete. Extracted %d post-burn-in sample vectors.", len(flat_samples))
    return sampler, flat_samples, flat_log_prob