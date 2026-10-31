"""
physics.py
-----------------
Module containing physical models for temperature-dependent photoluminescence peak broadening,
thermal quenching, peak center shifting, and composite spectrum generation.
"""

import logging
from typing import List, Tuple, Union
import numpy as np

logger = logging.getLogger("pl_analysis.physics_models")

K_B: float = 8.617333262145e-5


def calculate_linewidth(
    temp_k: float,
    gamma_0: float,
    gamma_ac: float,
    gamma_lo: float,
    e_lo: float,
) -> float:
    """ Calculates the temperature-dependent FWHM linewidth incorporating acoustic and LO phonon scattering.
    ------
    OUTPUT - float: Calculated linewidth (FWHM) in eV at temperature temp_k.
    ------
    temp_k - float: Temperature in Kelvin.
    gamma_0 - float: Inhomogeneous broadening linewidth at 0 K in eV.
    gamma_ac - float: Acoustic phonon scattering coefficient in eV/K.
    gamma_lo - float: Longitudinal Optical (LO) phonon coupling strength in eV.
    e_lo - float: LO phonon energy in eV.
    ------
    ValueError: Raised if temperature, gamma_0, or e_lo are non-positive.
    """
    if temp_k < 0:
        logger.error("Negative temperature provided: %.2f K", temp_k)
        raise ValueError("Temperature in Kelvin cannot be negative.")
    if gamma_0 <= 0 or e_lo <= 0:
        logger.error("Invalid parameters: gamma_0=%.4e, e_lo=%.4e must be > 0", gamma_0, e_lo)
        raise ValueError("Inhomogeneous broadening and LO phonon energy must be strictly positive.")

    if temp_k == 0:
        return float(gamma_0)

    exp_term = np.exp(e_lo / (K_B * temp_k)) - 1.0
    if exp_term <= 0:
        logger.warning(
            "LO phonon denominator exp_term <= 0 (%.4e) at T = %.2f K, e_lo = %.4e eV. Clamping exp_term to 1e-15.",
            exp_term,
            temp_k,
            e_lo,
        )
        exp_term = 1e-15

    lo_broadening = gamma_lo / exp_term
    fwhm = gamma_0 + gamma_ac * temp_k + lo_broadening
    return float(fwhm)


def calculate_peak_area(
    temp_k: float,
    i_0: float,
    a_factor: float,
    e_a: float,
) -> float:
    """ Calculates temperature-dependent integrated peak area via Arrhenius thermal quenching model.
    ------
    OUTPUT - float: Calculated integrated peak area at temperature temp_k.
    ------
    temp_k - float: Temperature in Kelvin.
    i_0 - float: Integrated peak area at 0 K.
    a_factor - float: Non-radiative recombination process probability pre-factor.
    e_a - float: Thermal activation energy for quenching in eV.
    ------
    ValueError: Raised if temp_k < 0, or if i_0, a_factor, or e_a are negative.
    """
    if temp_k < 0:
        logger.error("Negative temperature provided: %.2f K", temp_k)
        raise ValueError("Temperature in Kelvin cannot be negative.")
    if i_0 < 0 or a_factor < 0 or e_a < 0:
        logger.error("Invalid negative inputs: i_0=%.4e, a_factor=%.4e, e_a=%.4e", i_0, a_factor, e_a)
        raise ValueError("Intensity, pre-factor, and activation energy must be non-negative.")

    if temp_k == 0:
        return float(i_0)

    quenching_factor = 1.0 + a_factor * np.exp(-e_a / (K_B * temp_k))
    return float(i_0 / quenching_factor)


def calculate_peak_center(
    temp_k: float,
    e_0_zero: float,
    alpha_coeff: float,
) -> float:
    """ Calculates temperature-dependent emission peak center energy assuming a linear temperature coefficient.
    ------
    OUTPUT - float: Peak center energy in eV at temperature temp_k.
    ------
    temp_k - float: Temperature in Kelvin.
    e_0_zero - float: Unshifted emission peak center energy at 0 K in eV.
    alpha_coeff - float: Temperature coefficient of the bandgap shift in eV/K.
    ------
    ValueError: Raised if temperature in Kelvin is negative.
    """
    if temp_k < 0:
        logger.error("Negative temperature provided: %.2f K", temp_k)
        raise ValueError("Temperature in Kelvin cannot be negative.")

    return float(e_0_zero + alpha_coeff * temp_k)


def gaussian_profile(
    energy_ev: np.ndarray,
    center_e: float,
    fwhm_gamma: float,
    area: float,
) -> np.ndarray:
    """ Computes normalized Gaussian peak intensity profile across an energy array for given center, FWHM, and area.
    ------
    OUTPUT - np.ndarray: Gaussian peak spectrum intensity values corresponding to energy_ev.
    ------
    energy_ev - np.ndarray: Array of photon energy grid points in eV.
    center_e - float: Center energy of the peak in eV.
    fwhm_gamma - float: Full Width at Half Maximum (FWHM) of the peak in eV.
    area - float: Total integrated area under the Gaussian emission profile.
    ------
    ValueError: Raised if FWHM linewidth is non-positive.
    """
    if fwhm_gamma <= 0:
        logger.error("Non-positive FWHM provided to gaussian_profile: %.4e eV", fwhm_gamma)
        raise ValueError("FWHM linewidth must be strictly positive.")

    sigma = fwhm_gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    normalization = area / (sigma * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((energy_ev - center_e) / sigma) ** 2
    return normalization * np.exp(exponent)


def evaluate_single_peak_at_temperature(
    energy_ev: np.ndarray,
    temp_k: float,
    params: Union[Tuple[float, ...], List[float], np.ndarray],
) -> np.ndarray:
    """ Evaluates emission profile of a single peak at temperature temp_k using physical temperature dependency parameters.
    ------
    OUTPUT - np.ndarray: Evaluated spectral emission array for single peak at temperature temp_k.
    ------
    energy_ev - np.ndarray: Array of photon energy grid points in eV.
    temp_k - float: Temperature in Kelvin.
    params - Union[Tuple[float, ...], List[float], np.ndarray]: 9-element parameter sequence [gamma_0, gamma_ac, gamma_lo, e_lo, i_0, a_factor, e_a, e_0_zero, alpha_coeff].
    ------
    ValueError: Raised if params sequence length is not exactly 9.
    """
    if len(params) != 9:
        logger.error("Expected 9 peak parameters, received %d", len(params))
        raise ValueError(
            f"Expected 9 physical parameters per peak, but received {len(params)}."
        )

    gamma_0, gamma_ac, gamma_lo, e_lo, i_0, a_factor, e_a, e_0_zero, alpha_coeff = params

    fwhm = calculate_linewidth(temp_k, gamma_0, gamma_ac, gamma_lo, e_lo)
    area = calculate_peak_area(temp_k, i_0, a_factor, e_a)
    center = calculate_peak_center(temp_k, e_0_zero, alpha_coeff)

    return gaussian_profile(energy_ev, center, fwhm, area)


def evaluate_composite_spectrum(
    energy_ev: np.ndarray,
    temp_k: float,
    flat_params: np.ndarray,
    num_peaks: int,
) -> np.ndarray:
    """ Evaluates composite multi-peak emission spectrum at a given temperature by summing individual physical peak profiles.
    ------
    OUTPUT - np.ndarray: Composite spectrum intensity array summed across all num_peaks.
    ------
    energy_ev - np.ndarray: Array of photon energy grid points in eV.
    temp_k - float: Temperature in Kelvin.
    flat_params - np.ndarray: Flat vector containing 9 physical parameters for each of num_peaks.
    num_peaks - int: Number of peaks comprising the model.
    ------
    ValueError: Raised if flat_params length is not equal to num_peaks * 9.
    """
    expected_param_count = num_peaks * 9
    if len(flat_params) != expected_param_count:
        logger.error(
            "Flat parameters array size mismatch: expected %d, got %d",
            expected_param_count,
            len(flat_params),
        )
        raise ValueError(
            f"Parameter array size mismatch: expected {expected_param_count} ({num_peaks} peaks * 9 params), got {len(flat_params)}."
        )

    total_spectrum = np.zeros_like(energy_ev, dtype=np.float64)
    for k in range(num_peaks):
        peak_params = flat_params[k * 9 : (k + 1) * 9]
        total_spectrum += evaluate_single_peak_at_temperature(
            energy_ev, temp_k, peak_params
        )

    return total_spectrum