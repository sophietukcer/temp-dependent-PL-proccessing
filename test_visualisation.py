"""
test_visualisation.py
---------------------
Unit tests for visualisation.py module.
"""

from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pytest

from visualisation import (
    plot_corner_posterior,
    plot_parameter_temperature_trends,
    plot_temperature_series_fits,
)


@pytest.fixture
def mock_series() -> List[Dict[str, Union[float, np.ndarray]]]:
    """ Generates mock dataset fixture for visualization testing.
    """
    e_grid = np.linspace(1.8, 2.4, 50)
    return [
        {"temperature": 100.0, "energy_ev": e_grid, "intensity": np.exp(-((e_grid - 2.0) ** 2) / 0.01)},
        {"temperature": 200.0, "energy_ev": e_grid, "intensity": np.exp(-((e_grid - 2.01) ** 2) / 0.01)},
    ]


@pytest.fixture
def mock_params() -> np.ndarray:
    """ Generates valid 10-element parameter vector fixture (1 peak + log_sigma).
    """
    return np.array([0.04, 1e-4, 0.03, 0.025, 0.1, 10.0, 0.03, 2.0, 1e-5, -3.0])


def test_plot_temperature_series_fits(
    mock_series: List[Dict[str, Union[float, np.ndarray]]], mock_params: np.ndarray, tmp_path: Path
) -> None:
    """ Tests temperature series fit plot rendering and image output saving.
    """
    save_p = tmp_path / "fits.png"
    fig = plot_temperature_series_fits(mock_series, mock_params, num_peaks=1, save_path=save_p)

    assert isinstance(fig, plt.Figure)
    assert save_p.is_file()
    plt.close(fig)

    with pytest.raises(ValueError):
        plot_temperature_series_fits([], mock_params, num_peaks=1)


def test_plot_corner_posterior(mock_params: np.ndarray, tmp_path: Path) -> None:
    """ Tests MCMC corner plot rendering across single peak parameter samples.
    """
    samples = mock_params + 1e-3 * np.random.randn(100, 10)
    save_p = tmp_path / "corner.png"

    fig = plot_corner_posterior(samples, num_peaks=1, peak_index=0, save_path=save_p)
    assert isinstance(fig, plt.Figure)
    assert save_p.is_file()
    plt.close(fig)

    with pytest.raises(ValueError):
        plot_corner_posterior(samples, num_peaks=1, peak_index=5)


def test_plot_parameter_temperature_trends(mock_params: np.ndarray, tmp_path: Path) -> None:
    """ Tests rendering of physical parameter temperature dependence subplots.
    """
    temps = np.array([100.0, 150.0, 200.0])
    save_p = tmp_path / "trends.png"

    fig = plot_parameter_temperature_trends(temps, mock_params, num_peaks=1, save_path=save_p)
    assert isinstance(fig, plt.Figure)
    assert save_p.is_file()
    plt.close(fig)

    with pytest.raises(ValueError):
        plot_parameter_temperature_trends(np.array([]), mock_params, num_peaks=1)