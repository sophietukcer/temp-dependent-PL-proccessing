"""
NUTS_sampler.py
---------------
PyMC model for temperature-dependent PL fitting of overlapping emission peaks (Gaussian or pseudo-Voigt).

"""

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.transforms import ordered

from temp_dependent_pl_proccessing.pl_physics import (
    compute_t_ref,
    peak_physics,
    peak_profiles,
)

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed trapz

# Prior hyper-parameters. Override any of them via build_pl_pymc_model(priors={...}).
# Every entry may be a scalar or, for per-peak parameters, a list with one value per peak.
DEFAULT_PRIORS: Dict[str, Any] = {
    "e_ref_sigma": 40.0,        # meV, width of the prior on each peak centre at T_ref
    "de_dt_sigma": 0.3,         # meV/K, centre shift (zero-mean, either sign)
    "gamma_0_median": 40.0,     # meV, inhomogeneous FWHM
    "gamma_0_sigma_log": 0.5,
    "gamma_ac_median": 0.05,    # meV/K, acoustic coupling
    "gamma_ac_sigma_log": 1.0,
    "gamma_lo_median": 30.0,    # meV, optical coupling
    "gamma_lo_sigma_log": 0.7,
    "e_lo_median": 30.0,        # meV, LO phonon energy (shared by all peaks)
    "e_lo_sigma_log": 0.3,
    "i_0_median_log": 0.5,
    "i_0_sigma_log": 1.0,       # median is set from the lowest-T spectrum area
    "e_a_median": 40.0,         # meV, activation energy
    "e_a_sigma_log": 0.5,
    "ln_q_ref_sigma": 1.5,      # ln(quenched/radiative) at T_ref
    "noise_frac": 0.02,         # HalfNormal scale of the per-spectrum noise FLOOR, as a fraction of max intensity
    "noise_rel_sigma": 0.02,    # HalfNormal scale of rel_err (noise + model error as a fraction of the signal)
    "baseline_mu": 0.035,       # Normal(mu, sigma) constant offset per spectrum (normalised units)
    "baseline_sigma": 0.01,     # width of the prior on the constant offset
    "bg_amp_sigma": 0.1,        # HalfNormal scale (fraction of y_max) for per-spectrum exp background amplitude
    "bg_decay_median": 125.0,   # meV, shared exponential background decay length
    "bg_decay_sigma_log": 0.2,  # log-normal width for shared exponential background decay length
    "eta_alpha": 2.0,           # Beta(alpha, beta) prior on the Lorentzian fraction eta (pseudo-Voigt only);
    "eta_beta": 3.0,            # Beta(2, 2) = mild preference for a mixture, density -> 0 at eta = 0 and 1
}


def build_pl_pymc_model(
    processed_series: List[Dict],
    num_peaks: int,
    absorption_centers: List[float],
    priors: Optional[Mapping[str, Any]] = None,
    fit_baseline: bool = True,
    relative_noise: bool = True,
    lineshape: str = "pseudo_voigt",
) -> pm.Model:
    """ Builds a PyMC Bayesian model for temperature-dependent PL fitting.
    ------
    OUTPUT - pm.Model: Model with one Gaussian-likelihood node per spectrum ('obs_T_<T>K').
    ------
    processed_series - list[Dict]: processed spectra ('temperature' K, 'energy_ev' eV, 'intensity' normalised by
        ONE global factor, see data_loader), ordered by ascending temperature.
    num_peaks - int: number of peaks in each spectrum.
    absorption_centers - List[float]: initial peak centre estimates (eV), ascending; used as prior means.
    priors - Optional[Mapping[str, Any]]: overrides for DEFAULT_PRIORS (scalars, or one value per peak for
        per-peak parameters, e.g. {"gamma_0_median": [40, 60, 60, 40]}).
    fit_baseline - bool: add a constant offset + exponential background per spectrum.
    relative_noise - bool: heteroscedastic noise sd_i = sqrt(sigma^2 + (rel_err * y_i)^2) using the OBSERVED y_i
        (not the model), which avoids the amplitude bias of model-dependent noise. If False, one constant
        sigma per spectrum (the previous behaviour).
    lineshape - str: "pseudo_voigt" (default; per-peak Lorentzian fraction `eta`) or "gaussian".
    ------
    ValueError: if the number of centres differs from num_peaks or a spectrum contains non-finite values.
    """
    if lineshape not in ("pseudo_voigt", "gaussian"):
        raise ValueError(f"lineshape must be 'pseudo_voigt' or 'gaussian', got {lineshape!r}.")
    pri = {**DEFAULT_PRIORS, **(dict(priors) if priors else {})}

    centers_mev = np.asarray(absorption_centers, dtype=np.float64) * 1e3
    if centers_mev.shape != (num_peaks,):
        raise ValueError(f"Expected {num_peaks} absorption centres, got {centers_mev.size}.")
    if np.any(np.diff(centers_mev) <= 0):
        raise ValueError("absorption_centers must be strictly ascending (peaks are ordered by energy).")

    temps = np.array([float(d["temperature"]) for d in processed_series])
    for d in processed_series:
        if not (np.all(np.isfinite(d["energy_ev"])) and np.all(np.isfinite(d["intensity"]))):
            raise ValueError(f"Non-finite values in spectrum at T = {d['temperature']} K.")
    t_ref = compute_t_ref(processed_series)

    # Data-driven SCALES for priors and reference energy (never the likelihood):
    low = processed_series[int(np.argmin(temps))]
    area_low = float(_trapezoid(np.asarray(low["intensity"]), np.asarray(low["energy_ev"]) * 1e3))
    area_low = max(area_low, 1e-3)
    y_max = float(max(np.max(d["intensity"]) for d in processed_series))
    e_min_mev = float(min(np.min(d["energy_ev"]) for d in processed_series)) * 1e3

    coords = {
        "peak": [f"Peak {i + 1}" for i in range(num_peaks)],
        "spectrum": [f"{t:g} K" for t in temps],
    }

    with pm.Model(coords=coords) as model:
        # ---- peak centre: linear (high-T Varshni limit), anchored at T_ref -------------------------
        e_ref = pm.Normal(
            "e_ref", mu=centers_mev, sigma=pri["e_ref_sigma"], dims="peak",
            transform=ordered, initval=centers_mev,
        )
        de_dt = pm.Normal("de_dt", mu=0.0, sigma=pri["de_dt_sigma"], dims="peak")  # meV/K, blueshift > 0
        pm.Deterministic("e0_0", e_ref - de_dt * t_ref, dims="peak")  # extrapolated 0 K centre (meV)

        # ---- FWHM: inhomogeneous + acoustic + LO phonon (meV) --------------------------------------
        gamma_0 = pm.LogNormal(
            "gamma_0", mu=np.log(pri["gamma_0_median"]), sigma=pri["gamma_0_sigma_log"], dims="peak"
        )
        gamma_ac = pm.LogNormal(
            "gamma_ac", mu=np.log(pri["gamma_ac_median"]), sigma=pri["gamma_ac_sigma_log"], dims="peak"
        )
        gamma_lo = pm.LogNormal(
            "gamma_lo", mu=np.log(pri["gamma_lo_median"]), sigma=pri["gamma_lo_sigma_log"], dims="peak"
        )
        # LO phonon energy is a lattice property: ONE value shared by all peaks.
        e_lo = pm.LogNormal("e_lo", mu=np.log(pri["e_lo_median"]), sigma=pri["e_lo_sigma_log"])

        # ---- lineshape: Lorentzian fraction per peak (pseudo-Voigt only) ---------------------------
        eta = None
        if lineshape == "pseudo_voigt":
            eta = pm.Beta("eta", alpha=pri["eta_alpha"], beta=pri["eta_beta"], dims="peak")

        # ---- thermal quenching (Arrhenius), parametrised at T_ref ----------------------------------
        i_0 = pm.LogNormal(
            # "i_0", mu=np.log(area_low / num_peaks), sigma=pri["i_0_sigma_log"], dims="peak"
            "i_0", mu=pri["i_0_median_log"], sigma=pri["i_0_sigma_log"], dims="peak"
        )
        e_a = pm.LogNormal("e_a", mu=np.log(pri["e_a_median"]), sigma=pri["e_a_sigma_log"], dims="peak")
        ln_q_ref = pm.Normal("ln_q_ref", mu=0.0, sigma=pri["ln_q_ref_sigma"], dims="peak")

        # ---- noise: per-spectrum floor + a relative term (signal-proportional noise and model error) ----
        sigma = pm.HalfNormal("sigma", sigma=pri["noise_frac"] * y_max, dims="spectrum")
        if relative_noise:
            rel_err = pm.HalfNormal("rel_err", sigma=pri["noise_rel_sigma"])

        # ---- baseline: constant offset + exponential decay background ------------------------------
        if fit_baseline:
            baseline = pm.Normal(
                "baseline", mu=pri["baseline_mu"], sigma=pri["baseline_sigma"], dims="spectrum"
            )
            bg_decay = pm.LogNormal(
                "bg_decay", mu=np.log(pri["bg_decay_median"]), sigma=pri["bg_decay_sigma_log"]
            )
            bg_amp = pm.HalfNormal("bg_amp", sigma=pri["bg_amp_sigma"] * y_max, dims="spectrum")

        params: Dict[str, Any] = {
            "e_ref": e_ref, "de_dt": de_dt, "gamma_0": gamma_0, "gamma_ac": gamma_ac,
            "gamma_lo": gamma_lo, "e_lo": e_lo, "i_0": i_0, "e_a": e_a, "ln_q_ref": ln_q_ref,
        }

        # ---- forward model + likelihood ------------------------------------------------------------
        centres_t, fwhm_t, area_t_all = [], [], []
        for idx, entry in enumerate(processed_series):
            t_val = temps[idx]
            e_col = np.asarray(entry["energy_ev"], dtype=np.float64)[:, None] * 1e3  # (N, 1) in meV
            y_obs = np.asarray(entry["intensity"], dtype=np.float64)

            e_t, gamma_t, area_t = peak_physics(t_val, t_ref, params, pt.exp, pt.expm1)
            profiles = peak_profiles(e_col, e_t, gamma_t, area_t, pt.exp, eta=eta)

            spectrum_mu = pt.sum(profiles, axis=1)
            if fit_baseline:
                e_mev = e_col[:, 0]
                bg_exp = bg_amp[idx] * pt.exp(-(e_mev - e_min_mev) / bg_decay)
                spectrum_mu = spectrum_mu + baseline[idx] + bg_exp

            if relative_noise:
                # y_obs is data (a constant), so sd_i is a fixed weight per point; 1e-12 keeps sqrt differentiable.
                sd_i = pt.sqrt(sigma[idx] ** 2 + (rel_err * np.clip(y_obs, 0.0, None)) ** 2 + 1e-12)
            else:
                sd_i = sigma[idx]

            pm.Normal(
                f"obs_T_{int(round(t_val))}K",
                mu=spectrum_mu,
                sigma=sd_i,
                observed=y_obs,
            )
            centres_t.append(e_t)
            fwhm_t.append(gamma_t)
            area_t_all.append(area_t)

        # Derived quantities per (spectrum, peak) for plotting physical trends vs T.
        pm.Deterministic("e_centre", pt.stack(centres_t, axis=0), dims=("spectrum", "peak"))
        pm.Deterministic("fwhm", pt.stack(fwhm_t, axis=0), dims=("spectrum", "peak"))
        pm.Deterministic("area", pt.stack(area_t_all, axis=0), dims=("spectrum", "peak"))

    return model