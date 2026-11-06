"""
visualisations_pymc.py
Native ArviZ and Matplotlib visualization utilities for PyMC InferenceData.
"""

from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import arviz as az


def plot_pymc_traces(
    idata: az.InferenceData,
    var_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    compact: bool = False,
) -> plt.Figure:
    """
    Plots MCMC chain traces and posterior distributions using native ArviZ.
    """
    axes = az.plot_trace(
        idata,
        var_names=var_names,
        compact=compact,
        backend_kwargs={"layout": "constrained"},
    )
    fig = axes[0, 0].get_figure()
    
    if save_path:
        fig.savefig(Path(save_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_pymc_corner(
    idata: az.InferenceData,
    var_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generates a Bayesian corner/pair plot with KDE densities and marginal histograms.
    """
    axes = az.plot_pair(
        idata,
        var_names=var_names,
        kind="kde",
        marginals=True,
        divergence=True,
        textsize=10,
        backend_kwargs={"layout": "constrained"},
    )
    
    # Extract figure handle from axes grid
    if hasattr(axes, "flat"):
        fig = axes.flat[0].get_figure()
    else:
        fig = axes[0, 0].get_figure()

    if save_path:
        fig.savefig(Path(save_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_energy_diagnostics(
    idata: az.InferenceData,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plots NUTS energy transition vs. marginal energy distribution to assess sampling efficiency.
    """
    ax = az.plot_energy(idata, fill_alpha=0.5, fill_color=["C0", "C1"])
    fig = ax.get_figure()
    fig.suptitle("HMC Energy Diagnostic (Marginal Energy vs Energy Transition)")
    
    if save_path:
        fig.savefig(Path(save_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_posterior_predictive_fit(
    idata: az.InferenceData,
    save_path: Optional[Union[str, Path]] = None,
    num_pp_samples: int = 100,
) -> plt.Figure:
    """
    Plots Posterior Predictive Checks (PPC) comparing theoretical spectral fits to raw data.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_ppc(
        idata,
        data_pairs={"obs_T_*": "obs_T_*"},  # Matches per-temperature observed nodes
        num_pp_samples=num_pp_samples,
        ax=ax,
    )
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Posterior Predictive Check: Measured PL vs Model Fit")

    if save_path:
        fig.savefig(Path(save_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_parameter_posteriors(
    idata: az.InferenceData,
    var_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plots posterior credible intervals (HDI) and mean estimates for target physical parameters.
    """
    axes = az.plot_posterior(
        idata,
        var_names=var_names,
        hdi_prob=0.95,
        point_estimate="mean",
        textsize=10,
    )
    
    if hasattr(axes, "flat"):
        fig = axes.flat[0].get_figure()
    else:
        fig = axes[0, 0].get_figure()

    if save_path:
        fig.savefig(Path(save_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig