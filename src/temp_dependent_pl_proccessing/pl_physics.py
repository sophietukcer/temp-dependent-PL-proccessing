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
    centre  : E(T)   = e_ref + a_te * (T - T_ref)
                       + a_ep * [2 / (exp(e_lo / kT) - 1) - 2 / (exp(e_lo / kT_ref) - 1)]
    FWHM    : G(T)   = gamma_0 + gamma_ac * T + gamma_lo / (exp(e_lo / kT) - 1)
    area    : A(T)   = free per (spectrum, peak) parameter `area`
    profile : Gaussian (default) or pseudo-Voigt with the given centre, FWHM and *area*.
              Pseudo-Voigt = (1 - eta) * Gaussian + eta * Lorentzian, both with the SAME FWHM and unit area,
              so `eta` (per peak, 0 = Gaussian, 1 = Lorentzian) only changes the tails, not the width or area.
"""

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

KB_MEV: float = 8.617333262e-2  # Boltzmann constant in meV/K

# area = height * FWHM * AREA_PER_HEIGHT_FWHM  for a Gaussian
AREA_PER_HEIGHT_FWHM: float = float(np.sqrt(np.pi / (4.0 * np.log(2.0))))

# Names of the free physical parameters, shared by the model, runner and plots.
PARAM_NAMES: Tuple[str, ...] = (
    "e_ref",     # peak centre at T_ref (meV)                   [per peak]
    "a_te",      # thermal expansion shift coeff (meV/K, > 0)   [per peak]
    "a_ep",      # electron-phonon coupling weight (meV, < 0)   [per peak]
    "gamma_0",   # inhomogeneous FWHM (meV)                     [per peak]
    "gamma_ac",  # acoustic phonon coupling (meV/K)             [per peak]
    "gamma_lo",  # optical phonon coupling (meV)                [per peak]
    "e_lo",      # LO phonon energy (meV)                       [shared scalar]
    "area",      # free integrated peak area                    [per spectrum, per peak]
)

# Optional parameters, present only when the model was built with them:
#   baseline - fit_baseline=True: one value per SPECTRUM, a constant offset added to the total spectrum.
#   eta      - lineshape="pseudo_voigt": one value per PEAK, Lorentzian fraction (its presence selects the
#              pseudo-Voigt profile in model_components).
OPTIONAL_PARAM_NAMES: Tuple[str, ...] = ("baseline", "eta")


def compute_t_ref(processed_series: Sequence[Mapping[str, Any]]) -> float:
    """ Reference temperature (K) used to decorrelate centre parameters: mean of the fitted temperatures. """
    return float(np.mean([float(d["temperature"]) for d in processed_series]))


def peak_physics(
    temperature: float,
    t_ref: float,
    p: Mapping[str, Any],
    expm1: Callable,
) -> Tuple[Any, Any]:
    """ Centre (meV) and FWHM (meV) at one temperature.
    `expm1` selects the backend: `pt.expm1` for PyMC or `np.expm1` for numpy.
    expm1 keeps the Bose factor accurate and differentiable for small e_lo/kT.
    """
    kt = KB_MEV * temperature
    kt_ref = KB_MEV * t_ref

    bose_t = 1.0 / expm1(p["e_lo"] / kt)
    bose_ref = 1.0 / expm1(p["e_lo"] / kt_ref)

    e_t = (
        p["e_ref"]
        + p["a_te"] * (temperature - t_ref)
        + p["a_ep"] * 2.0 * (bose_t - bose_ref)
    )
    gamma_t = p["gamma_0"] + p["gamma_ac"] * temperature + p["gamma_lo"] * bose_t
    return e_t, gamma_t


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


def pseudo_voigt_profiles(
    energy_mev_col: Any,
    e_t: Any,
    gamma_t: Any,
    area_t: Any,
    eta: Any,
    exp: Callable,
) -> Any:
    """ Pseudo-Voigt peaks of given centre, FWHM and AREA: (1 - eta) * Gaussian + eta * Lorentzian.
    Both parts have unit area and the same FWHM, and at +-FWHM/2 both fall to half of their own maximum,
    so the mixture's FWHM is exactly gamma_t for any eta. Shape (N, 1) energies -> (N, n_peaks).
    """
    x = (energy_mev_col - e_t) / gamma_t
    gauss = (2.0 * np.sqrt(np.log(2.0) / np.pi) / gamma_t) * exp(-4.0 * np.log(2.0) * x ** 2)
    lorentz = (2.0 / (np.pi * gamma_t)) / (1.0 + 4.0 * x ** 2)
    return area_t * ((1.0 - eta) * gauss + eta * lorentz)


def peak_profiles(
    energy_mev_col: Any,
    e_t: Any,
    gamma_t: Any,
    area_t: Any,
    exp: Callable,
    eta: Any = None,
) -> Any:
    """ Gaussian peaks if `eta` is None, otherwise pseudo-Voigt peaks with per-peak Lorentzian fraction `eta`. """
    if eta is None:
        return gaussian_profiles(energy_mev_col, e_t, gamma_t, area_t, exp)
    return pseudo_voigt_profiles(energy_mev_col, e_t, gamma_t, area_t, eta, exp)


def model_components(
    params: Mapping[str, Any],
    temperature: float,
    energy_ev: np.ndarray,
    t_ref: float,
    area_t: np.ndarray,
    baseline: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Numerical model spectrum for one parameter set (e.g. MAP, posterior mean or a single draw).
    ------
    OUTPUT - (components, total): components has shape (N_energy, n_peaks); total has shape (N_energy,)
        and includes the constant `baseline` offset (the components do not).
    ------
    params - mapping with the keys of PARAM_NAMES (excluding 'area', which is passed via `area_t`).
        An 'eta' entry (per-peak Lorentzian fraction) selects the pseudo-Voigt lineshape; without it the peaks are
        Gaussian. Any 'baseline' or 'area' entries in `params` are ignored here; pass the values for THIS
        spectrum via `area_t` and `baseline`.
    energy_ev - energy grid in eV (converted to meV internally).
    area_t - 1D array of peak areas (shape (n_peaks,)) for THIS spectrum.
    baseline - constant offset for this spectrum, in normalised-intensity units.
    """
    excluded = {"baseline", "area"}
    p: Dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.float64) for k, v in params.items() if k not in excluded
    }
    e_col = np.asarray(energy_ev, dtype=np.float64)[:, None] * 1e3
    e_t, gamma_t = peak_physics(float(temperature), t_ref, p, np.expm1)
    area_arr = np.asarray(area_t, dtype=np.float64)
    comps = peak_profiles(e_col, e_t, gamma_t, area_arr, np.exp, eta=p.get("eta"))
    return comps, comps.sum(axis=1) + float(baseline)