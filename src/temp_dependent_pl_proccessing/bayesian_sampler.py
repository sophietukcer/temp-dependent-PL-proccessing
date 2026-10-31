"""
bayesian_sampler.py
-------------------
Module for formulating log-priors, joint log-likelihoods, log-posteriors, and executing
MCMC sampling using emcee for temperature-dependent PL peak analysis.
"""

import logging
import multiprocessing
from typing import Dict, List, Tuple, Union, Optional

import emcee
import numpy as np

from temp_dependent_pl_proccessing.physics import evaluate_composite_spectrum

logger = logging.getLogger("pl_analysis.bayesian_sampler")

KB_EV: float = 8.617333262145e-5  # Boltzmann constant in eV/K


def log_prior(
    params: np.ndarray,
    num_peaks: int,
    absorption_centers: List[float],
    temperatures: List[float],
    center_window_ev: float = 0.015,
    max_fwhm_ev: float = 0.150,
    min_energy_ev: float = 1.50,
    max_energy_ev: float = 2.30,
) -> float:
    """ Calculates joint log-prior probability density enforcing both global parameter bounds and temperature-evaluated spectral bounds.
    ------
    OUTPUT - float: Natural log-prior probability density (-np.inf if global or single-T bounds are violated).
    ------
    params - np.ndarray: Joint model parameter vector of shape (9 * num_peaks + 1,).
    num_peaks - int: Number of emission peaks fitted.
    absorption_centers - List[float]: Target absorption peak centers in eV.
    temperatures - List[float]: Array of experimental temperatures in Kelvin present in the dataset.
    center_window_ev - float: Gaussian anchor standard deviation in eV for E0(0).
    max_fwhm_ev - float: Maximum physical linewidth limit in eV at any individual temperature (default: 0.250 eV).
    min_energy_ev - float: Lower detector/grid energy bound in eV (default: 1.20 eV).
    max_energy_ev - float: Upper detector/grid energy bound in eV (default: 2.80 eV).
    """
    if len(params) != 9 * num_peaks + 1:
        return -np.inf

    lp = 0.0

    # 1. Global energy ordering check at T = 0 K
    for k in range(num_peaks - 1):
        if params[k * 9 + 7] >= params[(k + 1) * 9 + 7]:
            return -np.inf

    # 2. Global parameter box bounds & Gaussian target priors
    for k in range(num_peaks):
        g0, g_ac, g_lo, e_lo, i0, a_fac, e_a, e0, alpha = params[k * 9 : (k + 1) * 9]

        if not (0.001 <= g0 <= 0.150):
            return -np.inf
        if not (1e-6 <= g_ac <= 1e-2):
            return -np.inf
        if not (0.001 <= g_lo <= 0.200):
            return -np.inf
        if not (0.005 <= e_lo <= 0.080):
            return -np.inf
        if not (1e-4 <= i0 <= 5.0):
            return -np.inf
        if not (1e-3 <= a_fac <= 1e4):
            return -np.inf
        if not (0.001 <= e_a <= 0.500):
            return -np.inf
        if not (1e-7 <= alpha <= 1e-3):
            return -np.inf

        target_center = absorption_centers[k]
        lp += -0.5 * ((e0 - target_center) / center_window_ev) ** 2

    log_sigma = params[-1]
    if not (-10.0 <= log_sigma <= 2.0):
        return -np.inf

    # 3. PER-TEMPERATURE BOUND EVALUATION across all experimental spectra
    for t_val in temperatures:
        for k in range(num_peaks):
            g0, g_ac, g_lo, e_lo, i0, a_fac, e_a, e0, alpha = params[k * 9 : (k + 1) * 9]

            # Evaluated peak center E_k(T)
            e_t = e0 - alpha * t_val
            if not (min_energy_ev <= e_t <= max_energy_ev):
                return -np.inf

            # Evaluated linewidth Gamma_k(T) via Segall formula
            exp_term = np.exp(e_lo / (KB_EV * t_val)) - 1.0
            gamma_t = g0 + g_ac * t_val + (g_lo / exp_term if exp_term > 0 else 0.0)

            # Reject if evaluated FWHM is unphysically broad or negative at this temperature
            if not (0.001 <= gamma_t <= max_fwhm_ev):
                return -np.inf

            # Evaluated intensity I_k(T) via Arrhenius quenching formula
            i_t = i0 / (1.0 + a_fac * np.exp(-e_a / (KB_EV * t_val)))
            if not (1e-6 <= i_t <= 10.0):
                return -np.inf

        # Optional: Enforce peak energy ordering at higher temperatures (prevents high-T peak crossover)
        for k in range(num_peaks - 1):
            e_curr_t = params[k * 9 + 7] - params[k * 9 + 8] * t_val
            e_next_t = params[(k + 1) * 9 + 7] - params[(k + 1) * 9 + 8] * t_val
            if e_curr_t >= e_next_t:
                return -np.inf

    return lp


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


# def log_posterior(
#     params: np.ndarray,
#     processed_series: List[Dict[str, Union[float, np.ndarray]]],
#     num_peaks: int,
#     absorption_centers: List[float],
#     center_window_ev: float = 0.08,
# ) -> float:
#     """ Computes the total log-posterior probability density by summing log-prior and joint log-likelihood.
#     ------
#     OUTPUT - float: Calculated unnormalized log-posterior probability density.
#     ------
#     params - np.ndarray: Flat parameter vector of length (9 * num_peaks + 1).
#     processed_series - List[Dict[str, Union[float, np.ndarray]]]: List of processed dataset dicts.
#     num_peaks - int: Number of peaks comprising the model.
#     absorption_centers - List[float]: Initial peak center energy estimates in eV.
#     center_window_ev - float: Maximum allowed peak center shift window (default: 0.15 eV).
#     ------
#     ValueError: Raised if parameter dimensions or series inputs are invalid.
#     """
#     lp = log_prior(params, num_peaks, absorption_centers, center_window_ev=center_window_ev)
#     if not np.isfinite(lp):
#         return -np.inf

#     ll = log_likelihood(params, processed_series, num_peaks)
#     if not np.isfinite(ll):
#         return -np.inf

#     return float(lp + ll)


def log_posterior(
    params: np.ndarray,
    processed_series: List[Dict],
    num_peaks: int,
    absorption_centers: List[float],
) -> float:
    # Extract temperature values from processed dataset list
    temperatures = [entry["temperature"] for entry in processed_series]

    lp = log_prior(
        params,
        num_peaks=num_peaks,
        absorption_centers=absorption_centers,
        temperatures=temperatures,
    )
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(params, processed_series, num_peaks)


DEFAULT_PEAK_GUESSES: Dict[str, float] = {
    "gamma_0": 0.04,     # Inhomogeneous broadening at 0 K (eV)
    "gamma_ac": 1e-4,    # Acoustic phonon coupling (eV/K)
    "gamma_lo": 0.03,    # LO phonon coupling (eV)
    "e_lo": 0.025,       # LO phonon energy (eV)
    "i_0": 0.01,          # Peak area at 0 K
    "a_factor": 10.0,    # Quenching pre-factor
    "e_a": 0.05,         # Quenching activation energy (eV)
    "alpha_coeff": 1e-5, # Bandgap temperature shift (eV/K)
}


def initialize_walker_positions(
    num_walkers: int,
    num_peaks: int,
    absorption_centers: List[float],
    temperatures: List[float],
    custom_peak_guesses: Optional[List[Dict[str, float]]] = None,
    log_sigma_guess: float = -3.0,
    jitter_std: float = 1e-4,
) -> np.ndarray:
    """ Generates parameter walker starting positions, resampling any walker that violates global or per-temperature physical prior bounds.
    ------
    OUTPUT - np.ndarray: Walker starting array of shape (num_walkers, 9 * num_peaks + 1).
    ------
    num_walkers - int: Total number of ensemble MCMC walkers.
    num_peaks - int: Number of peaks comprising the model.
    absorption_centers - List[float]: Absorption peak center estimates in eV.
    temperatures - List[float]: Experimental temperature series values in Kelvin.
    custom_peak_guesses - Optional[List[Dict[str, float]]]: Per-peak initial parameter guesses.
    log_sigma_guess - float: Initial guess for log residual noise variance (default: -3.0).
    jitter_std - float: Relative standard deviation of Gaussian noise jitter applied to initial parameters (default: 1e-4).
    ------
    ValueError: Raised if num_walkers is less than twice parameter dimensions or custom_peak_guesses length doesn't match num_peaks.
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

    if custom_peak_guesses is not None and len(custom_peak_guesses) != num_peaks:
        logger.error(
            "custom_peak_guesses length (%d) does not match num_peaks (%d)",
            len(custom_peak_guesses),
            num_peaks,
        )
        raise ValueError("Length of custom_peak_guesses must equal num_peaks.")

    # 1. Build initial parameter vector guess
    base_guess: List[float] = []
    for k in range(num_peaks):
        peak_dict = (
            custom_peak_guesses[k]
            if custom_peak_guesses is not None
            else DEFAULT_PEAK_GUESSES
        )

        base_guess.extend([
            peak_dict.get("gamma_0", DEFAULT_PEAK_GUESSES["gamma_0"]),
            peak_dict.get("gamma_ac", DEFAULT_PEAK_GUESSES["gamma_ac"]),
            peak_dict.get("gamma_lo", DEFAULT_PEAK_GUESSES["gamma_lo"]),
            peak_dict.get("e_lo", DEFAULT_PEAK_GUESSES["e_lo"]),
            peak_dict.get("i_0", DEFAULT_PEAK_GUESSES["i_0"]),
            peak_dict.get("a_factor", DEFAULT_PEAK_GUESSES["a_factor"]),
            peak_dict.get("e_a", DEFAULT_PEAK_GUESSES["e_a"]),
            absorption_centers[k],
            peak_dict.get("alpha_coeff", DEFAULT_PEAK_GUESSES["alpha_coeff"]),
        ])

    base_guess.append(log_sigma_guess)
    initial_pos = np.array(base_guess, dtype=np.float64)

    # 2. Populate walker array with prior validation
    walker_positions = np.zeros((num_walkers, ndim), dtype=np.float64)

    for i in range(num_walkers):
        candidate_pos = initial_pos + jitter_std * np.random.randn(ndim) * np.abs(initial_pos)
        
        attempts = 0
        max_attempts = 10000
        # Check against updated log_prior signature including temperatures
        while not np.isfinite(log_prior(candidate_pos, num_peaks, absorption_centers, temperatures)):
            candidate_pos = initial_pos + jitter_std * np.random.randn(ndim) * np.abs(initial_pos)
            attempts += 1
            if attempts > max_attempts:
                logger.warning("Walker %d failed to sample valid prior state after %d attempts; using base guess.", i, max_attempts)
                candidate_pos = initial_pos.copy()
                break

        walker_positions[i] = candidate_pos

    logger.info("Initialized %d walkers for %d dimensions (%d peaks)", num_walkers, ndim, num_peaks)
    return walker_positions


def run_mcmc_sampling(
    processed_series: List[Dict[str, Union[float, np.ndarray]]],
    num_peaks: int,
    absorption_centers: List[float],
    custom_peak_guesses: Optional[List[Dict[str, float]]] = None,
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
    custom_peak_guesses - Optional[List[Dict[str, float]]]: Optional list of dicts specifying initial guesses per peak.
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

    ndim = 9 * num_peaks + 1# Extract unique dataset temperatures
    temperatures = [entry["temperature"] for entry in processed_series]

# Pass temperatures to walker initialization
    init_positions = initialize_walker_positions(
    n_walkers,
    num_peaks,
    absorption_centers,
    temperatures=temperatures,
    custom_peak_guesses=custom_peak_guesses,
)
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
            a=1.2,
        )
        sampler.run_mcmc(init_positions, n_steps, progress=True)

    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=burn_in, flat=True)

    logger.info("MCMC sampling complete. Extracted %d post-burn-in sample vectors.", len(flat_samples))
    return sampler, flat_samples, flat_log_prob