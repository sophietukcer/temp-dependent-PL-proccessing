"""
model_selection.py
------------------
Module for evaluating model complexity metrics (BIC, AIC, Maximum Likelihood)
to determine whether adding additional photoluminescence emission peaks is justified.
"""

import logging
from typing import Dict, List, Union
import numpy as np

logger = logging.getLogger("pl_analysis.model_selection")


def compute_total_data_points(
    processed_series: List[Dict[str, Union[float, np.ndarray]]],
) -> int:
    """ Calculates the total number of spectral intensity data points across all temperature datasets.
    ------
    OUTPUT - int: Total number of data points N across the temperature series.
    ------
    processed_series - List[Dict[str, Union[float, np.ndarray]]]: Processed temperature series dataset dicts.
    ------
    ValueError: Raised if processed_series is empty.
    """
    if not processed_series:
        logger.error("Processed series empty in compute_total_data_points.")
        raise ValueError("processed_series list cannot be empty.")

    total_points = sum(len(dataset["energy_ev"]) for dataset in processed_series)
    return total_points


def compute_bic(
    max_log_likelihood: float,
    num_params: int,
    total_data_points: int,
) -> float:
    """ Computes the Bayesian Information Criterion (BIC) score for a given model fit.
    ------
    OUTPUT - float: Calculated BIC score (lower values indicate better model fit).
    ------
    max_log_likelihood - float: Maximum log-likelihood score evaluated at the best-fit parameters.
    num_params - int: Total number of fitted model parameters k.
    total_data_points - int: Total number of experimental data points N.
    ------
    ValueError: Raised if total_data_points <= 0 or num_params <= 0.
    """
    if total_data_points <= 0 or num_params <= 0:
        logger.error(
            "Invalid BIC inputs: num_params=%d, total_data_points=%d",
            num_params,
            total_data_points,
        )
        raise ValueError("num_params and total_data_points must be strictly positive.")

    bic_value = num_params * np.log(total_data_points) - 2.0 * max_log_likelihood
    return float(bic_value)


def compute_aic(
    max_log_likelihood: float,
    num_params: int,
) -> float:
    """ Computes the Akaike Information Criterion (AIC) score for a given model fit.
    ------
    OUTPUT - float: Calculated AIC score (lower values indicate better model fit).
    ------
    max_log_likelihood - float: Maximum log-likelihood score evaluated at the best-fit parameters.
    num_params - int: Total number of fitted model parameters k.
    ------
    ValueError: Raised if num_params <= 0.
    """
    if num_params <= 0:
        logger.error("Invalid num_params for AIC computation: %d", num_params)
        raise ValueError("num_params must be strictly positive.")

    aic_value = 2.0 * num_params - 2.0 * max_log_likelihood
    return float(aic_value)


def evaluate_model_fit(
    flat_samples: np.ndarray,
    flat_log_prob: np.ndarray,
    processed_series: List[Dict[str, Union[float, np.ndarray]]],
    num_peaks: int,
) -> Dict[str, Union[float, int, np.ndarray]]:
    """ Extracts maximum likelihood estimate (MLE) parameters and calculates BIC/AIC model scoring metrics.
    ------
    OUTPUT - Dict[str, Union[float, int, np.ndarray]]: Dict containing 'mle_params', 'max_log_like', 'num_params', 'total_points', 'bic', and 'aic'.
    ------
    flat_samples - np.ndarray: Flattened MCMC parameter chains post-burn-in of shape (samples, num_params).
    flat_log_prob - np.ndarray: Flattened MCMC log posterior probabilities of shape (samples,).
    processed_series - List[Dict[str, Union[float, np.ndarray]]]: Processed temperature series datasets.
    num_peaks - int: Number of emission peaks fitted in the model.
    ------
    ValueError: Raised if flat_samples and flat_log_prob lengths mismatch or processed_series is empty.
    """
    if len(flat_samples) != len(flat_log_prob):
        logger.error(
            "Length mismatch: flat_samples (%d) vs flat_log_prob (%d)",
            len(flat_samples),
            len(flat_log_prob),
        )
        raise ValueError("flat_samples and flat_log_prob must have matching lengths.")

    best_idx = int(np.argmax(flat_log_prob))
    mle_params = flat_samples[best_idx]
    max_log_prob = float(flat_log_prob[best_idx])

    num_params = 9 * num_peaks + 1
    total_points = compute_total_data_points(processed_series)

    bic = compute_bic(max_log_prob, num_params, total_points)
    aic = compute_aic(max_log_prob, num_params)

    logger.info(
        "Model evaluation for K=%d peaks: BIC=%.2f, AIC=%.2f, Max Log-Prob=%.2f (N=%d points, k=%d params)",
        num_peaks,
        bic,
        aic,
        max_log_prob,
        total_points,
        num_params,
    )

    return {
        "num_peaks": num_peaks,
        "mle_params": mle_params,
        "max_log_prob": max_log_prob,
        "num_params": num_params,
        "total_points": total_points,
        "bic": bic,
        "aic": aic,
    }


def justify_peak_addition(
    model_k_eval: Dict[str, Union[float, int, np.ndarray]],
    model_k_plus_1_eval: Dict[str, Union[float, int, np.ndarray]],
    bic_threshold: float = 10.0,
) -> bool:
    """ Evaluates whether adding an additional peak (K -> K+1) is statistically justified based on BIC reduction threshold.
    ------
    OUTPUT - bool: True if model K+1 is preferred over model K (ΔBIC < -bic_threshold), False otherwise.
    ------
    model_k_eval - Dict[str, Union[float, int, np.ndarray]]: Model evaluation dict for K peaks.
    model_k_plus_1_eval - Dict[str, Union[float, int, np.ndarray]]: Model evaluation dict for K+1 peaks.
    bic_threshold - float: Minimum required BIC drop to justify extra parameters (default: 10.0 for strong evidence).
    ------
    ValueError: Raised if model evaluation dictionaries lack required 'bic' key.
    """
    if "bic" not in model_k_eval or "bic" not in model_k_plus_1_eval:
        logger.error("Missing 'bic' key in model evaluation dictionaries.")
        raise ValueError("Both evaluation dictionaries must contain a 'bic' metric key.")

    bic_k = float(model_k_eval["bic"])
    bic_k_plus_1 = float(model_k_plus_1_eval["bic"])
    delta_bic = bic_k_plus_1 - bic_k

    is_justified = delta_bic < -bic_threshold

    logger.info(
        "Comparing K=%d vs K=%d peaks: ΔBIC = %.2f. Peak addition justified: %s",
        int(model_k_eval["num_peaks"]),
        int(model_k_plus_1_eval["num_peaks"]),
        delta_bic,
        is_justified,
    )

    return is_justified