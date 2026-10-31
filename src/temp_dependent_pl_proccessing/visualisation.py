"""
visualisation.py
----------------
Module for generating temperature-series spectral fit plots, parameter temperature dependency
curves, and MCMC posterior probability corner plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import emcee

import matplotlib.pyplot as plt
import numpy as np

try:
    import corner
except ImportError:
    corner = None

from temp_dependent_pl_proccessing.physics import (
    calculate_linewidth,
    calculate_peak_area,
    calculate_peak_center,
    evaluate_composite_spectrum,
    evaluate_single_peak_at_temperature,
)

logger = logging.getLogger("pl_analysis.visualization")

PARAM_NAMES_PER_PEAK: List[str] = [
    r"$\Gamma_0$",
    r"$\gamma_{\text{ac}}$",
    r"$\gamma_{\text{LO}}$",
    r"$E_{\text{LO}}$",
    r"$I_0$",
    r"$A$",
    r"$E_a$",
    r"$E_0(0)$",
    r"$\alpha$",
]


def plot_temperature_series_fits(
    processed_series: List[Dict[str, Union[float, np.ndarray]]],
    fit_params: np.ndarray,
    num_peaks: int,
    offset_step: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """ Generates stacked offset plots comparing experimental spectra against theoretical joint MCMC fits across temperature series.
    ------
    OUTPUT - plt.Figure: Matplotlib figure object containing stacked temperature fits.
    ------
    processed_series - List[Dict[str, Union[float, np.ndarray]]]: List of processed dataset dictionaries ordered by ascending temperature.
    fit_params - np.ndarray: Best-fit or posterior median parameter array of shape (9 * num_peaks + 1,).
    num_peaks - int: Number of emission peaks fitted in the model.
    offset_step - float: Vertical offset step applied between consecutive temperature spectra (default: 0.5).
    save_path - Optional[Union[str, Path]]: File path to save generated plot figure (default: None).
    ------
    ValueError: Raised if processed_series is empty or fit_params shape is invalid.
    """
    if not processed_series:
        logger.error("Processed series empty in plot_temperature_series_fits.")
        raise ValueError("processed_series list cannot be empty.")

    expected_param_count = 9 * num_peaks + 1
    if len(fit_params) != expected_param_count:
        logger.error("fit_params length %d != expected %d", len(fit_params), expected_param_count)
        raise ValueError(f"Expected {expected_param_count} parameters, got {len(fit_params)}.")

    flat_peak_params = fit_params[:-1]
    fig, ax = plt.subplots(figsize=(10, 2 + 0.6 * len(processed_series)))

    colors = plt.cm.viridis(np.linspace(0, 1, num_peaks))

    for idx, dataset in enumerate(processed_series):
        temp_k = float(dataset["temperature"])
        energy_ev = dataset["energy_ev"]
        intensity_exp = dataset["intensity"]
        offset = idx * offset_step

        intensity_model = evaluate_composite_spectrum(
            energy_ev, temp_k, flat_peak_params, num_peaks
        )

        ax.plot(
            energy_ev,
            intensity_exp + offset,
            "o",
            color="black",
            ms=2,
            alpha=0.5,
            label="Data" if idx == 0 else "",
        )
        ax.plot(
            energy_ev,
            intensity_model + offset,
            "r-",
            lw=1.5,
            label="Global Fit" if idx == 0 else "",
        )

        for k in range(num_peaks):
            single_peak_params = flat_peak_params[k * 9 : (k + 1) * 9]
            single_peak = evaluate_single_peak_at_temperature(
                energy_ev, temp_k, single_peak_params
            )
            ax.plot(
                energy_ev,
                single_peak + offset,
                "--",
                color=colors[k],
                lw=1.0,
                alpha=0.8,
                label=f"Peak {k+1}" if idx == 0 else "",
            )

        ax.text(
            energy_ev.min() + 0.02,
            offset + 0.1,
            f"{temp_k:.0f} K",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Energy (eV)", fontsize=12)
    ax.set_ylabel("Normalized Intensity + Offset", fontsize=12)
    ax.set_title(f"Temperature-Dependent PL Fits (K = {num_peaks} Peaks)", fontsize=14)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        logger.info("Saved temperature series fit plot to: %s", path)

    return fig


def plot_corner_posterior(
    flat_samples: np.ndarray,
    num_peaks: int,
    peak_index: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """ Generates MCMC posterior probability corner plots showing parameter covariances and marginal distributions.
    ------
    OUTPUT - plt.Figure: Matplotlib figure object containing corner plot layout.
    ------
    flat_samples - np.ndarray: Flattened MCMC parameter samples of shape (samples, 9 * num_peaks + 1).
    num_peaks - int: Number of peaks comprising the model.
    peak_index - Optional[int]: 0-indexed specific peak to plot (plots all parameters if None).
    save_path - Optional[Union[str, Path]]: File path to save corner plot image (default: None).
    ------
    ImportError: Raised if corner package is not installed.
    ValueError: Raised if peak_index exceeds num_peaks - 1 or flat_samples shape is invalid.
    """
    if corner is None:
        logger.error("The 'corner' Python package is required for posterior visualization.")
        raise ImportError("Please install 'corner' via: pip install corner")

    expected_ndim = 9 * num_peaks + 1
    if flat_samples.shape[1] != expected_ndim:
        logger.error("Samples dimension mismatch: expected %d, got %d", expected_ndim, flat_samples.shape[1])
        raise ValueError(f"Expected samples with {expected_ndim} columns.")

    if peak_index is not None:
        if not (0 <= peak_index < num_peaks):
            logger.error("Invalid peak_index %d for num_peaks %d", peak_index, num_peaks)
            raise ValueError(f"peak_index must be between 0 and {num_peaks - 1}.")

        sub_samples = flat_samples[:, peak_index * 9 : (peak_index + 1) * 9]
        labels = [f"{lbl} (P{peak_index+1})" for lbl in PARAM_NAMES_PER_PEAK]
        title = f"Posterior Distribution - Peak {peak_index+1}"
    else:
        sub_samples = flat_samples
        labels: List[str] = []
        for k in range(num_peaks):
            labels.extend([f"{lbl} (P{k+1})" for lbl in PARAM_NAMES_PER_PEAK])
        labels.append(r"$\ln \sigma$")
        title = "Global Posterior Distribution"

    fig = corner.corner(
        sub_samples,
        labels=labels,
        show_titles=True,
        title_fmt=".3e",
        quantiles=[0.16, 0.50, 0.84],
        plot_datapoints=False,
        fill_contours=True,
    )
    fig.suptitle(title, fontsize=14, y=1.02)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved corner plot to: %s", path)

    return fig


def plot_parameter_temperature_trends(
    temperatures: np.ndarray,
    fit_params: np.ndarray,
    num_peaks: int,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """ Generates 3-panel subplot tracking physical parameters (linewidth, peak area, peak center) across temperature series.
    ------
    OUTPUT - plt.Figure: Matplotlib figure object containing 3-panel temperature trend plots.
    ------
    temperatures - np.ndarray: Array of dataset measurement temperatures in Kelvin.
    fit_params - np.ndarray: Best-fit or posterior median parameter array of shape (9 * num_peaks + 1,).
    num_peaks - int: Number of peaks comprising the model.
    save_path - Optional[Union[str, Path]]: File path to save trend plot image (default: None).
    ------
    ValueError: Raised if temperatures array is empty or fit_params shape is invalid.
    """
    if len(temperatures) == 0:
        logger.error("Temperatures array empty in plot_parameter_temperature_trends.")
        raise ValueError("temperatures array cannot be empty.")

    expected_param_count = 9 * num_peaks + 1
    if len(fit_params) != expected_param_count:
        logger.error("fit_params length %d != expected %d", len(fit_params), expected_param_count)
        raise ValueError(f"Expected {expected_param_count} parameters, got {len(fit_params)}.")

    flat_peak_params = fit_params[:-1]
    t_smooth = np.linspace(temperatures.min(), temperatures.max(), 200)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 1, num_peaks))

    for k in range(num_peaks):
        g0, g_ac, g_lo, e_lo, i0, a_fac, e_a, e0, alpha = flat_peak_params[k * 9 : (k + 1) * 9]

        fwhm_smooth = [calculate_linewidth(t, g0, g_ac, g_lo, e_lo) for t in t_smooth]
        area_smooth = [calculate_peak_area(t, i0, a_fac, e_a) for t in t_smooth]
        center_smooth = [calculate_peak_center(t, e0, alpha) for t in t_smooth]

        label = f"Peak {k+1}"
        axes[0].plot(t_smooth, fwhm_smooth, "-", color=colors[k], lw=2, label=label)
        axes[1].plot(t_smooth, area_smooth, "-", color=colors[k], lw=2, label=label)
        axes[2].plot(t_smooth, center_smooth, "-", color=colors[k], lw=2, label=label)

    axes[0].set_ylabel("FWHM Linewidth (eV)")
    axes[0].set_title("Phonon Broadening $\Gamma(T)$")

    axes[1].set_ylabel("Integrated Area")
    axes[1].set_title("Thermal Quenching $I(T)$")

    axes[2].set_ylabel("Peak Center Energy (eV)")
    axes[2].set_title("Bandgap Shift $E_0(T)$")

    for ax in axes:
        ax.set_xlabel("Temperature (K)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        logger.info("Saved parameter temperature trends plot to: %s", path)

    return fig

def plot_mcmc_traceplots(
    sampler: emcee.EnsembleSampler,
    num_peaks: int,
    burn_in: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """ Generates walker traceplots showing parameter evolution over step iterations to diagnose MCMC convergence.
    ------
    OUTPUT - plt.Figure: Matplotlib figure containing trace subplots for each parameter.
    ------
    sampler - emcee.EnsembleSampler: Fitted emcee EnsembleSampler instance containing full chain history.
    num_peaks - int: Number of emission peaks fitted.
    burn_in - Optional[int]: Step iteration marking burn-in cutoff (drawn as a vertical dashed line).
    save_path - Optional[Union[str, Path]]: Path to save generated figure image file.
    """
    # Extract full chain without discarding burn-in: shape is (n_steps, n_walkers, ndim)
    chain = sampler.get_chain()
    n_steps, n_walkers, ndim = chain.shape

    # Construct parameter labels
    labels: List[str] = []
    for k in range(num_peaks):
        labels.extend([f"{lbl} (P{k+1})" for lbl in PARAM_NAMES_PER_PEAK])
    labels.append(r"$\ln \sigma$")

    fig, axes = plt.subplots(ndim, 1, figsize=(12, 1.8 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]

    for i in range(ndim):
        ax = axes[i]
        # Plot every walker's path over time
        ax.plot(chain[:, :, i], "k-", alpha=0.15, lw=0.5)
        ax.set_ylabel(labels[i], fontsize=10)
        ax.grid(True, alpha=0.3)

        if burn_in is not None:
            ax.axvline(x=burn_in, color="red", linestyle="--", lw=1.2, label="Burn-in Cutoff" if i == 0 else "")

    axes[-1].set_xlabel("Step Number", fontsize=12)
    axes[0].set_title(f"MCMC Walker Traceplots ({n_walkers} Walkers, {n_steps} Steps)", fontsize=14)
    if burn_in is not None:
        axes[0].legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        logger.info("Saved traceplots to: %s", path)

    return fig