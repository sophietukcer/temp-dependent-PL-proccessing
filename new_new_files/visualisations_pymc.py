"""
visualisations_pymc.py
Native ArviZ and Matplotlib visualization utilities for PyMC InferenceData.

The spectral plots evaluate the shared physics in pl_physics.py with numpy, so
they can show the individual overlapping peaks, residuals and posterior draws.
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from pl_physics import (
    OPTIONAL_PARAM_NAMES,
    PARAM_NAMES,
    compute_t_ref,
    model_components,
)


def _finish(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> plt.Figure:
    if save_path:
        fig.savefig(Path(save_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------------------
# Parameter extraction helpers (pure numpy, no xarray API assumptions)
# --------------------------------------------------------------------------------------
def _flatten_draws(idata: az.InferenceData, names: Sequence[str], group: str = "posterior") -> Dict[str, np.ndarray]:
    """ {name: array of shape (n_chain * n_draw, *dims)} for each requested variable of a group. """
    dataset = getattr(idata, group)
    missing = [n for n in names if n not in dataset]
    if missing:
        raise KeyError(f"Variables {missing} not found in idata.{group}.")
    out: Dict[str, np.ndarray] = {}
    for name in names:
        values = np.asarray(dataset[name].values)
        out[name] = values.reshape((-1,) + values.shape[2:])
    return out


def _param_names_in(idata: az.InferenceData, group: str) -> List[str]:
    """ PARAM_NAMES plus any optional parameters (e.g. 'baseline') present in the group. """
    dataset = getattr(idata, group)
    return list(PARAM_NAMES) + [n for n in OPTIONAL_PARAM_NAMES if n in dataset]


def _spectrum_model(
    params: Mapping[str, Any], temperature: float, energy: np.ndarray, t_ref: float, idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """ model_components for spectrum number `idx`, picking that spectrum's baseline if the params have one. """
    p = dict(params)
    baseline = p.pop("baseline", None)
    b = 0.0 if baseline is None else float(np.asarray(baseline).reshape(-1)[idx])
    return model_components(p, temperature, energy, t_ref, baseline=b)


def posterior_mean_params(idata: az.InferenceData, group: str = "posterior") -> Dict[str, np.ndarray]:
    """ Mean of each physical parameter over all chains/draws of the given group.
    For tabulating parameters only: do NOT use it to draw a fitted spectrum (see plot_spectral_fits). """
    return {k: v.mean(axis=0) for k, v in _flatten_draws(idata, _param_names_in(idata, group), group).items()}


def sample_param_draws(
    idata: az.InferenceData,
    num_draws: int = 50,
    group: str = "posterior",
    seed: int = 0,
) -> List[Dict[str, np.ndarray]]:
    """ Random draws of the full physical parameter set from a group ('posterior' or 'prior'). """
    flat = _flatten_draws(idata, _param_names_in(idata, group), group)
    n_total = next(iter(flat.values())).shape[0]
    rng = np.random.default_rng(seed)
    picks = rng.choice(n_total, size=min(num_draws, n_total), replace=False)
    return [{k: v[i] for k, v in flat.items()} for i in picks]


def _spectrum_grid(
    n_spectra: int, ncols: int, with_residuals: bool
) -> Tuple[plt.Figure, List[Tuple[plt.Axes, Optional[plt.Axes]]]]:
    """ Figure with one panel per spectrum (plus a thin residual panel underneath if requested). """
    nrows = int(np.ceil(n_spectra / ncols))
    fig = plt.figure(figsize=(4.2 * ncols, (4.0 if with_residuals else 3.2) * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.45, wspace=0.28)
    axes: List[Tuple[plt.Axes, Optional[plt.Axes]]] = []
    for i in range(n_spectra):
        cell = outer[i // ncols, i % ncols]
        if with_residuals:
            inner = cell.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
            ax_main = fig.add_subplot(inner[0])
            ax_res = fig.add_subplot(inner[1], sharex=ax_main)
            ax_main.tick_params(labelbottom=False)
            axes.append((ax_main, ax_res))
        else:
            axes.append((fig.add_subplot(cell), None))
    return fig, axes


# --------------------------------------------------------------------------------------
# Spectral plots
# --------------------------------------------------------------------------------------
colours = ["#FFCBE1", "#B2D9FF", "#C1FFC1", "#FFF3B2", "#E0B2FF"]  

def plot_spectral_fits(
    processed_series: Sequence[Mapping[str, Any]],
    params: Optional[Mapping[str, Any]] = None,
    draws: Optional[Sequence[Mapping[str, Any]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    ncols: int = 4,
    show_components: bool = True,
) -> plt.Figure:
    """ Data, total fit, individual peaks and residuals for every temperature.
    ------
    params - point estimate (e.g. MAP) with the keys of PARAM_NAMES. Use this for MAP fits only.
    draws - list of parameter draws (see sample_param_draws). Their spectra are drawn as a faint band and,
        when `params` is None, the CENTRAL curves are the pointwise MEDIAN of the draws' spectra.
        Never plot the model at posterior-MEAN parameters: the model is nonlinear and the posterior is
        correlated/skewed (e.g. i_0 vs ln_q_ref), so the mean-parameter spectrum can be far from the data
        even when every individual draw fits well.
    """
    if params is None and not draws:
        raise ValueError("Provide either `params` (point estimate) or `draws`.")
    t_ref = compute_t_ref(processed_series)
    fig, axes = _spectrum_grid(len(processed_series), ncols, with_residuals=True)

    for idx, ((ax, ax_res), entry) in enumerate(zip(axes, processed_series)):
        temperature = float(entry["temperature"])
        energy = np.asarray(entry["energy_ev"], dtype=float)
        data = np.asarray(entry["intensity"], dtype=float)

        ax.plot(energy, data, ".", ms=2, color="0.55", label="data", zorder=1)

        draw_results = [_spectrum_model(d, temperature, energy, t_ref, idx) for d in draws] if draws else []
        for _, draw_total in draw_results:
            ax.plot(energy, draw_total, color="C0", alpha=0.08, lw=0.8, zorder=2)

        if params is not None:
            comps, total = _spectrum_model(params, temperature, energy, t_ref, idx)
        else:
            comps = np.median(np.stack([c for c, _ in draw_results]), axis=0)
            total = np.median(np.stack([t for _, t in draw_results]), axis=0)

        if show_components:
            for k in range(comps.shape[1]):
                ax.fill_between(energy, comps[:, k], lw=1.0, color=colours[k], alpha=0.5, label=f"Peak {k + 1}", zorder=3)
        ax.plot(energy, total, color="k", lw=1.4, label="total", zorder=4)
        ax.set_title(f"{temperature:g} K", fontsize=10)

        ax_res.axhline(0.0, color="k", lw=0.6)
        ax_res.plot(energy, data - total, ".", ms=2, color="C3")
        ax_res.set_xlabel("Energy (eV)")

    axes[0][0].legend(fontsize=7, loc="upper right", frameon= False)
    axes[0][0].set_ylabel("Norm. intensity")
    return _finish(fig, save_path)


def plot_prior_predictive(
    processed_series: Sequence[Mapping[str, Any]],
    idata: az.InferenceData,
    num_draws: int = 30,
    save_path: Optional[Union[str, Path]] = None,
    ncols: int = 4,
) -> plt.Figure:
    """ Prior predictive check: total model spectra from prior draws over the data.
    If these curves look nothing like spectra, fix the priors before sampling. """
    t_ref = compute_t_ref(processed_series)
    prior_draws = sample_param_draws(idata, num_draws=num_draws, group="prior")
    fig, axes = _spectrum_grid(len(processed_series), ncols, with_residuals=False)

    for idx, ((ax, _), entry) in enumerate(zip(axes, processed_series)):
        temperature = float(entry["temperature"])
        energy = np.asarray(entry["energy_ev"], dtype=float)
        ax.plot(energy, np.asarray(entry["intensity"], dtype=float), ".", ms=2, color="0.55", zorder=1)
        for draw in prior_draws:
            ax.plot(energy, _spectrum_model(draw, temperature, energy, t_ref, idx)[1],
                    color="C0", alpha=0.25, lw=0.8, zorder=2)
        ax.set_title(f"{temperature:g} K", fontsize=10)
        ax.set_xlabel("Energy (eV)")
    axes[0][0].set_ylabel("Norm. intensity")
    fig.suptitle("Prior predictive check (grey = data, blue = prior draws)", y=0.995)
    return _finish(fig, save_path)


def plot_temperature_trends(
    idata: az.InferenceData,
    processed_series: Sequence[Mapping[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    hdi_prob: float = 0.94,
) -> plt.Figure:
    """ Posterior median and credible band of peak centre, FWHM and area vs temperature, per peak. """
    temps = np.array([float(d["temperature"]) for d in processed_series])
    lo_q, hi_q = 50.0 * (1.0 - hdi_prob), 100.0 - 50.0 * (1.0 - hdi_prob)
    panels = [
        ("e_centre", "Peak centre (meV)", False),
        ("fwhm", "FWHM (meV)", False),
        ("area", "Integrated area (norm. intensity * meV)", True),
    ]
    flat = _flatten_draws(idata, [p[0] for p in panels])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (var, label, log_y) in zip(axes, panels):
        samples = flat[var]  # (n_samples, n_spectra, n_peaks)
        median = np.median(samples, axis=0)
        lo, hi = np.percentile(samples, [lo_q, hi_q], axis=0)
        for k in range(samples.shape[2]):
            ax.plot(temps, median[:, k], "-o", ms=3, color=f"C{k}", label=f"Peak {k + 1}")
            ax.fill_between(temps, lo[:, k], hi[:, k], color=f"C{k}", alpha=0.25)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(label)
        if log_y:
            ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return _finish(fig, save_path)


# --------------------------------------------------------------------------------------
# Sampler diagnostics (ArviZ)
# --------------------------------------------------------------------------------------
def plot_pymc_traces(
    idata: az.InferenceData,
    var_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    compact: bool = True,
) -> plt.Figure:
    """
    Plots MCMC chain traces and posterior distributions using native ArviZ.
    compact=True overlays the peaks of a parameter on one axis to keep the figure readable.
    """
    axes = az.plot_trace(
        idata,
        var_names=var_names,
        compact=compact,
        backend_kwargs={"layout": "constrained"},
    )
    fig = axes[0, 0].get_figure()
    return _finish(fig, save_path)


def plot_pymc_corner(
    idata: az.InferenceData,
    var_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    divergences: bool = True,
    kind: str = "scatter",
    coords: Optional[Dict[str, List[str]]] = None,
) -> plt.Figure:
    """
    Pair plot with divergent transitions highlighted. Use `coords={"peak": ["Peak 1"]}` to restrict
    per-peak parameters to one peak and keep the grid small. Look for ridges/funnels among
    gamma_ac, gamma_lo and e_lo, and between e_a and ln_q_ref.
    """
    axes = az.plot_pair(
        idata,
        var_names=var_names,
        coords=coords,
        kind=kind,
        divergences=divergences,
        marginals=True,
        textsize=10,
        backend_kwargs={"layout": "constrained"},
    )
    fig = axes.flat[0].get_figure() if hasattr(axes, "flat") else axes[0, 0].get_figure()
    return _finish(fig, save_path)


def plot_energy_diagnostics(
    idata: az.InferenceData,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plots NUTS energy transition vs. marginal energy distribution to assess sampling efficiency.
    """
    ax = az.plot_energy(idata, fill_alpha=(0.5, 0.5), fill_color=["C0", "C1"])
    fig = ax.get_figure()
    fig.suptitle("HMC Energy Diagnostic (Marginal Energy vs Energy Transition)")
    return _finish(fig, save_path)


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
    fig = axes.flat[0].get_figure() if hasattr(axes, "flat") else axes[0, 0].get_figure()
    return _finish(fig, save_path)
