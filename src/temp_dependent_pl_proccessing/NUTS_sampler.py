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
    KB_MEV,
    compute_t_ref,
    peak_physics,
    peak_profiles,
)

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed trapz

# Prior hyper-parameters. Override any of them via build_pl_pymc_model(priors={...}).
# Every entry may be a scalar or, for per-peak parameters, a list with one value per peak.
DEFAULT_PRIORS: Dict[str, Any] = {
    "e_ref_sigma": 8.0,         # meV, prior width around initial centers
    "e_ref_bound": 20.0,        # meV, hard +- bound around each initial center (prevents label switching)
    "a_te_mu": 0.0,             # meV/K, thermal expansion shift coefficient prior mean
    "a_te_sigma": 0.15,          # meV/K, thermal expansion shift coefficient prior width
    "a_ep_mu": 0.0,             # meV, electron-phonon renormalization weight prior mean
    "a_ep_sigma": 35.0,         # meV, electron-phonon renormalization weight prior width
    "gamma_0_median": 40.0,     # meV, inhomogeneous FWHM
    "gamma_0_sigma_log": 0.35,
    "gamma_ac_median": 0.015,   # meV/K, acoustic coupling
    "gamma_ac_sigma_log": 0.6,
    "gamma_lo_median": 40.0,    # meV, optical coupling
    "gamma_lo_sigma_log": 0.6,
    "e_lo_median": 35.0,        # meV, LO phonon energy (shared by all peaks)
    "e_lo_sigma_log": 0.3,
    "area_sigma_log": 0.8,      # log-normal width for free per-spectrum, per-peak areas
    "noise_frac": 0.01,         # HalfNormal scale of the per-spectrum noise FLOOR
    "noise_rel_sigma": 0.015,   # HalfNormal scale of rel_err
    "baseline_mu": 0.03,        # Normal(mu, sigma) constant offset per spectrum
    "baseline_sigma": 0.005,    # width of the prior on the constant offset
    "bg_frac_alpha": 6.0,       # Beta(alpha, beta) for bg_amp / (y_at_emin - baseline), strictly in (0, 1)
    "bg_frac_beta": 2.0,
    "bg_decay_median": 45.0,    # meV, shared exponential background decay length
    "bg_decay_sigma_log": 0.25, # log-normal width for shared exponential background decay length
    "eta_alpha": 2.0,           # Beta(alpha, beta) prior on the Lorentzian fraction eta
    "eta_beta": 2.0,
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
    spec_areas = np.array([
        max(float(_trapezoid(np.asarray(d["intensity"]), np.asarray(d["energy_ev"]) * 1e3)), 1e-3)
        for d in processed_series
    ])
    area_init_mu = np.log(spec_areas[:, None] / num_peaks)  # shape (n_spectra, 1) -> broadcasts to (spectrum, peak)

    y_max = float(max(np.max(d["intensity"]) for d in processed_series))
    e_min_mev = float(min(np.min(d["energy_ev"]) for d in processed_series)) * 1e3
    y_at_emin = np.array([float(d["intensity"][0]) for d in processed_series], dtype=np.float64)

    coords = {
        "peak": [f"Peak {i + 1}" for i in range(num_peaks)],
        "spectrum": [f"{t:g} K" for t in temps],
    }

    with pm.Model(coords=coords) as model:
        # ---- LO phonon energy (shared lattice property across shift and broadening) ----------------
        e_lo = pm.LogNormal("e_lo", mu=np.log(pri["e_lo_median"]), sigma=pri["e_lo_sigma_log"])

        # ---- peak centre: thermal expansion + Bose-Einstein phonon interaction, anchored at T_ref --
        e_ref = pm.TruncatedNormal(
                    "e_ref",
                    mu=centers_mev,
                    sigma=pri["e_ref_sigma"],
                    lower=centers_mev - pri["e_ref_bound"],
                    upper=centers_mev + pri["e_ref_bound"],
                    dims="peak",
                )
        # a_te = pm.Normal("a_te", mu=pri["a_te_mu"], sigma=pri["a_te_sigma"], dims="peak")  # meV/K
        # a_ep = pm.Normal("a_ep", mu=pri["a_ep_mu"], sigma=pri["a_ep_sigma"], dims="peak")  # meV

        # Strictly positive thermal expansion coefficient (meV/K)
        a_te = pm.HalfNormal("a_te", sigma=pri["a_te_sigma"], dims="peak")

        # Strictly negative electron-phonon renormalization weight (meV)
        a_ep_mag = pm.HalfNormal("a_ep_mag", sigma=pri["a_ep_sigma"], dims="peak")
        a_ep = pm.Deterministic("a_ep", -a_ep_mag, dims="peak")

        # Extrapolated 0 K centre: E(0) = E_ref - a_te * T_ref + a_ep * (1 - 2 / (exp(e_lo / kT_ref) - 1))
        bose_ref = 1.0 / pt.expm1(e_lo / (KB_MEV * t_ref))
        pm.Deterministic(
            "e0_0",
            e_ref - a_te * t_ref + a_ep * (1.0 - 2.0 * bose_ref),
            dims="peak",
        )

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

        # ---- lineshape: Lorentzian fraction per peak (pseudo-Voigt only) ---------------------------
        eta = None
        if lineshape == "pseudo_voigt":
            eta = pm.Beta("eta", alpha=pri["eta_alpha"], beta=pri["eta_beta"], dims="peak")

        # ---- free peak areas across temperatures ---------------------------------------------------
        area = pm.LogNormal(
            "area",
            mu=pri.get("area_median_log", area_init_mu),
            sigma=pri["area_sigma_log"],
            dims=("spectrum", "peak"),
        )

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
                    bg_frac = pm.Beta(
                        "bg_frac", alpha=pri["bg_frac_alpha"], beta=pri["bg_frac_beta"], dims="spectrum"
                    )
                    bg_amp = pm.Deterministic(
                        "bg_amp", bg_frac * pt.maximum(y_at_emin - baseline, 1e-4), dims="spectrum"
                    )

        params: Dict[str, Any] = {
            "e_ref": e_ref, "a_te": a_te, "a_ep": a_ep,
            "gamma_0": gamma_0, "gamma_ac": gamma_ac, "gamma_lo": gamma_lo, "e_lo": e_lo,
        }

        # ---- forward model + likelihood ------------------------------------------------------------
        centres_t, fwhm_t = [], []
        for idx, entry in enumerate(processed_series):
            t_val = temps[idx]
            e_col = np.asarray(entry["energy_ev"], dtype=np.float64)[:, None] * 1e3  # (N, 1) in meV
            y_obs = np.asarray(entry["intensity"], dtype=np.float64)

            e_t, gamma_t = peak_physics(t_val, t_ref, params, pt.expm1)
            area_t = area[idx]
            profiles = peak_profiles(e_col, e_t, gamma_t, area_t, pt.exp, eta=eta)

            spectrum_mu = pt.sum(profiles, axis=1)
            if fit_baseline:
                e_mev = e_col[:, 0]
                bg_exp = bg_amp[idx] * pt.exp(-(e_mev - e_min_mev) / bg_decay)
                spectrum_mu = spectrum_mu + baseline[idx] + bg_exp

            if relative_noise:
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

        # Derived quantities per (spectrum, peak) for plotting physical trends vs T.
        pm.Deterministic("e_centre", pt.stack(centres_t, axis=0), dims=("spectrum", "peak"))
        pm.Deterministic("fwhm", pt.stack(fwhm_t, axis=0), dims=("spectrum", "peak"))

    return model