"""
NUTS_sampler.py
---------------
PyMC model for temperature-dependent PL fitting of overlapping Gaussian emission peaks.

Key design choices (all aimed at removing divergences):
  * Everything is in meV and uses TRUE integrated areas (not peak heights), so
    width changes do not leak into the quenching term.
  * Peak centres use PyMC's `ordered` transform (no penalty potentials) and are
    anchored at a reference temperature T_ref to decorrelate centre and slope.
  * The Bose factor uses expm1 with a LogNormal-bounded LO energy shared by all
    peaks (no clipping, no hard cap potentials).
  * Quenching is parametrised by q_ref = (quenched/radiative) at T_ref instead of
    a prefactor A, which removes the A <-> E_a ridge.
  * All hard-bounded Uniform priors were replaced with LogNormal priors.
The physics itself lives in pl_physics.py and is shared with the plotting code.
"""

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.transforms import ordered

from pl_physics import (
    compute_t_ref,
    gaussian_profiles,
    pseudovoigt_profiles,
    peak_physics,
)

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed trapz

# Prior hyper-parameters. Override any of them via build_pl_pymc_model(priors={...}).
DEFAULT_PRIORS: Dict[str, float] = {
    "e_ref_sigma": 15.0,        # meV, width of the prior on each peak centre at T_ref
    "de_dt_sigma": 0.1,         # meV/K, centre shift (zero-mean, either sign)
    "gamma_0_median": 40.0,     # meV, inhomogeneous FWHM
    "gamma_0_sigma_log": 0.3,
    "gamma_ac_median": 0.01,    # meV/K, acoustic coupling
    "gamma_ac_sigma_log": 0.05,
    "gamma_lo_median": 10.0,    # meV, optical coupling
    "gamma_lo_sigma_log": 0.5,
    "e_lo_median": 30.0,        # meV, LO phonon energy (shared by all peaks)
    "e_lo_sigma_log": 0.3,
    "i_0_sigma_log": 1.0,       # median is set from the lowest-T spectrum area
    "e_a_median": 40.0,         # meV, activation energy
    "e_a_sigma_log": 0.5,
    "ln_q_ref_sigma": 1.5,      # ln(quenched/radiative) at T_ref
    "noise_frac": 0.05,         # HalfNormal scale of the noise, as a fraction of max intensity
}


def build_pl_pymc_model(
    processed_series: List[Dict],
    num_peaks: int,
    absorption_centers: List[float],
    priors: Optional[Mapping[str, float]] = None,
) -> pm.Model:
    """ Builds a PyMC Bayesian model for temperature-dependent PL fitting.
    ------
    OUTPUT - pm.Model: Model with one Gaussian-likelihood node per spectrum ('obs_T_<T>K').
    ------
    processed_series - list[Dict]: processed spectra ('temperature' K, 'energy_ev' eV, 'intensity' normalised by
        ONE global factor, see data_loader), ordered by ascending temperature.
    num_peaks - int: number of Gaussian peaks in each spectrum.
    absorption_centers - List[float]: initial peak centre estimates (eV), ascending; used as prior means.
    priors - Optional[Mapping[str, float]]: overrides for DEFAULT_PRIORS.
    ------
    ValueError: if the number of centres differs from num_peaks or a spectrum contains non-finite values.
    """
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

    # Data-driven SCALES for priors only (never the likelihood):
    low = processed_series[int(np.argmin(temps))]
    area_low = float(_trapezoid(np.asarray(low["intensity"]), np.asarray(low["energy_ev"]) * 1e3))
    area_low = max(area_low, 1e-3)
    y_max = float(max(np.max(d["intensity"]) for d in processed_series))

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

        # ---- thermal quenching (Arrhenius), parametrised at T_ref ----------------------------------
        i_0 = pm.LogNormal(
            "i_0", mu=np.log(area_low / num_peaks), sigma=pri["i_0_sigma_log"], dims="peak"
        )
        e_a = pm.LogNormal("e_a", mu=np.log(pri["e_a_median"]), sigma=pri["e_a_sigma_log"], dims="peak")
        ln_q_ref = pm.Normal("ln_q_ref", mu=0.0, sigma=pri["ln_q_ref_sigma"], dims="peak")

        # ---- noise (one sigma per spectrum, scaled to the normalised data) -------------------------
        sigma = pm.HalfNormal("sigma", sigma=pri["noise_frac"] * y_max, dims="spectrum")

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
            # profiles = gaussian_profiles(e_col, e_t, gamma_t, area_t, pt.exp)
            profiles=pseudovoigt_profiles(e_col, e_t, gamma_t, area_t, eta=0.5, exp=pt.exp)

            pm.Normal(
                f"obs_T_{int(round(t_val))}K",
                mu=pt.sum(profiles, axis=1),
                sigma=sigma[idx],
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
