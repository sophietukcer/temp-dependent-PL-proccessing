"""
test_physics.py
----------------------
Unit test suite for physics_models.py verifying phonon broadening, thermal quenching,
bandgap center shifts, Gaussian peak shapes, and multi-peak composite models.
"""

import numpy as np
import pytest

from physics import (
    calculate_linewidth,
    calculate_peak_area,
    calculate_peak_center,
    evaluate_composite_spectrum,
    gaussian_profile,
)


def test_calculate_linewidth() -> None:
    """ Tests linewidth calculations under low-temperature, high-temperature, and invalid input conditions.
    ------
    OUTPUT - None: Returns None upon successful assertion verification.
    ------
    """
    # At 0 K, linewidth must equal gamma_0
    assert calculate_linewidth(0.0, 0.05, 1e-4, 0.03, 0.02) == 0.05

    # Linewidth should increase with increasing temperature
    fwhm_low = calculate_linewidth(10.0, 0.05, 1e-4, 0.03, 0.02)
    fwhm_high = calculate_linewidth(300.0, 0.05, 1e-4, 0.03, 0.02)
    assert fwhm_high > fwhm_low > 0.05

    # Exception checks
    with pytest.raises(ValueError):
        calculate_linewidth(-10.0, 0.05, 1e-4, 0.03, 0.02)
    with pytest.raises(ValueError):
        calculate_linewidth(100.0, -0.01, 1e-4, 0.03, 0.02)


def test_calculate_peak_area() -> None:
    """ Tests thermal quenching peak area calculations across temperatures and boundary edge cases.
    ------
    OUTPUT - None: Returns None upon successful assertion verification.
    ------
    """
    # At 0 K, area must equal i_0
    assert calculate_peak_area(0.0, 100.0, 10.0, 0.05) == 100.0

    # Quenching decreases peak area at elevated temperatures
    area_low = calculate_peak_area(50.0, 100.0, 10.0, 0.05)
    area_high = calculate_peak_area(300.0, 100.0, 10.0, 0.05)
    assert 100.0 > area_low > area_high > 0.0

    # Exception checks
    with pytest.raises(ValueError):
        calculate_peak_area(-5.0, 100.0, 10.0, 0.05)
    with pytest.raises(ValueError):
        calculate_peak_area(100.0, -1.0, 10.0, 0.05)


def test_calculate_peak_center() -> None:
    """ Verifies linear shift in peak center energy with temperature.
    ------
    OUTPUT - None: Returns None upon successful assertion verification.
    ------
    """
    center_0k = calculate_peak_center(0.0, 2.5, 1e-4)
    center_300k = calculate_peak_center(300.0, 2.5, 1e-4)

    assert center_0k == 2.5
    assert center_300k == pytest.approx(2.5 + 300.0 * 1e-4)

    with pytest.raises(ValueError):
        calculate_peak_center(-1.0, 2.5, 1e-4)


def test_gaussian_profile() -> None:
    """ Validates numerical integration area and peak maximum position for Gaussian profile function.
    ------
    OUTPUT - None: Returns None upon successful assertion verification.
    ------
    """
    energy_grid = np.linspace(1.5, 3.5, 1000)
    center = 2.5
    fwhm = 0.1
    target_area = 5.0

    profile = gaussian_profile(energy_grid, center, fwhm, target_area)

    # Integrated area should equal target_area
    calc_area = float(
        getattr(np, "trapezoid", getattr(np, "trapz", None))(profile, energy_grid)
    )
    assert calc_area == pytest.approx(target_area, rel=1e-3)

    # Maximum intensity should occur at center energy
    max_idx = int(np.argmax(profile))
    assert float(energy_grid[max_idx]) == pytest.approx(center, abs=0.005)

    with pytest.raises(ValueError):
        gaussian_profile(energy_grid, center, -0.05, target_area)


def test_evaluate_composite_spectrum() -> None:
    """ Evaluates multi-peak composite spectrum generation and array size validation.
    ------
    OUTPUT - None: Returns None upon successful assertion verification.
    ------
    """
    energy_grid = np.linspace(1.5, 3.5, 500)
    temp_k = 100.0
    num_peaks = 2

    params_peak1 = [0.05, 1e-4, 0.02, 0.03, 50.0, 5.0, 0.04, 2.2, 1e-4]
    params_peak2 = [0.06, 1e-4, 0.02, 0.03, 30.0, 3.0, 0.03, 2.6, 1e-4]
    flat_params = np.array(params_peak1 + params_peak2)

    spectrum = evaluate_composite_spectrum(energy_grid, temp_k, flat_params, num_peaks)

    assert len(spectrum) == len(energy_grid)
    assert np.all(spectrum >= 0.0)

    # Exception check for incorrect parameter count
    with pytest.raises(ValueError):
        evaluate_composite_spectrum(energy_grid, temp_k, flat_params[:10], num_peaks)