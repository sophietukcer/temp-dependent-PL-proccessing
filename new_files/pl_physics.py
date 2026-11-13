"""
pl_physics.py
-------------
Forward physical model for temperature-dependent PL, written ONCE and shared by
the PyMC model (symbolic, via pytensor) and the plotting code (numerical, via
numpy), so the two can never drift out of sync.

Units: energies and widths in meV, temperature in K, intensity in the
normalised units produced by data_loader (areas are therefore in
normalised-intensity * meV).

Model (per peak k, at temperature T)
    centre  : E(T)   = e_ref + de_dt * (T - T_ref)                 [linear / high-T Varshni limit]
    FWHM    : G(T)   = gamma_0 + gamma_ac * T + gamma_lo / (exp(e_lo / kT) - 1)
    area    : A(T)   = i_0 / (1 + exp(ln_q_ref - e_a * (1/kT - 1/kT_ref)))
              (Arrhenius quenching, parametrised by the quenched/radiative
               ratio q_ref at T_ref to decorrelate the prefactor from e_a)
    profile : Gaussian with the given centre, FWHM and *area*
"""

from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

KB_MEV: float = 8.617333262e-2  # Boltzmann constant in meV/K

# area = height * FWHM * AREA_PER_HEIGHT_FWHM  for a Gaussian
AREA_PER_HEIGHT_FWHM: float = float(np.sqrt(np.pi / (4.0 * np.log(2.0))))

# Names of the free physical parameters, shared by the model, runner and plots.
PARAM_NAMES: Tuple[str, ...] = (
    "e_ref",     # peak centre at T_ref (meV)           [per peak]
    "de_dt",     # centre shift, meV/K (blueshift > 0)   [per peak]
    "gamma_0",   # inhomogeneous FWHM (meV)              [per peak]
    "gamma_ac",  # acoustic phonon coupling (meV/K)      [per peak]
    "gamma_lo",  # optical phonon coupling (meV)         [per peak]
    "e_lo",      # LO phonon energy (meV)                [shared scalar]
    "i_0",       # unquenched (T -> 0) area              [per peak]
    "e_a",       # activation energy (meV)               [per peak]
    "ln_q_ref",  # ln(quenched/radiative) at T_ref       [per peak]
)


def compute_t_ref(processed_series: Sequence[Mapping[str, Any]]) -> float:
    """ Reference temperature (K) used to decorrelate centre/area parameters: mean of the fitted temperatures. """
    return float(np.mean([float(d["temperature"]) for d in processed_series]))


def peak_physics(
    temperature: float,
    t_ref: float,
    p: Mapping[str, Any],
    exp: Callable,
    expm1: Callable,
) -> Tuple[Any, Any, Any]:
    """ Centre (meV), FWHM (meV) and area at one temperature.
    `exp`/`expm1` select the backend: (pt.exp, pt.expm1) for PyMC or (np.exp, np.expm1) for numpy.
    expm1 keeps the Bose factor accurate and differentiable for small e_lo/kT.
    """
    kt = KB_MEV * temperature
    kt_ref = KB_MEV * t_ref
    e_t = p["e_ref"] + p["de_dt"] * (temperature - t_ref)
    gamma_t = p["gamma_0"] + p["gamma_ac"] * temperature + p["gamma_lo"] / expm1(p["e_lo"] / kt)
    quench = exp(p["ln_q_ref"] - p["e_a"] * (1.0 / kt - 1.0 / kt_ref))
    area_t = p["i_0"] / (1.0 + quench)
    return e_t, gamma_t, area_t


def gaussian_profiles(
    energy_mev_col: Any,
    e_t: Any,
    gamma_t: Any,
    area_t: Any,
    exp: Callable,
) -> Any:
    """ Gaussian peaks of given centre, FWHM and AREA on an energy column of shape (N, 1) -> (N, n_peaks). """
    amplitude = area_t / (gamma_t * AREA_PER_HEIGHT_FWHM)
    return amplitude * exp(-4.0 * np.log(2.0) * ((energy_mev_col - e_t) / gamma_t) ** 2)

def pseudovoigt_profiles(
    energy_mev_col: Any,
    e_t: Any,
    gamma_t: Any,
    area_t: Any,
    eta: Any,
    exp: Callable,
) -> Any:
    """ Pseudo-Voigt peaks of given centre, FWHM, AREA and mixing parameter eta on an energy column of shape (N, 1) -> (N, n_peaks). """
    amplitude = area_t / (gamma_t * AREA_PER_HEIGHT_FWHM)
    gaussian = amplitude * exp(-4.0 * np.log(2.0) * ((energy_mev_col - e_t) / gamma_t) ** 2)
    lorentzian = amplitude / (1.0 + 4.0 * ((energy_mev_col - e_t) / gamma_t) ** 2)
    return eta * lorentzian + (1 - eta) * gaussian


def model_components(
    params: Mapping[str, Any],
    temperature: float,
    energy_ev: np.ndarray,
    t_ref: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Numerical model spectrum for one parameter set (e.g. MAP, posterior mean or a single draw).
    ------
    OUTPUT - (components, total): components has shape (N_energy, n_peaks); total has shape (N_energy,).
    ------
    params - mapping with the keys of PARAM_NAMES (values are numpy arrays / scalars, per-peak where applicable).
    energy_ev - energy grid in eV (converted to meV internally).
    """
    p: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
    e_col = np.asarray(energy_ev, dtype=np.float64)[:, None] * 1e3
    e_t, gamma_t, area_t = peak_physics(float(temperature), t_ref, p, np.exp, np.expm1)
    comps = gaussian_profiles(e_col, e_t, gamma_t, area_t, np.exp)
    return comps, comps.sum(axis=1)
