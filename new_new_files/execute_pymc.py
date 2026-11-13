"""
execute_pymc.py
---------------
Loads and processes the spectra, builds the PyMC model, optionally runs a prior
predictive check and a MAP fit, then samples with NUTS starting from the MAP
(no random jitter by default).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
import pymc as pm

from data_loader import process_temperature_series, build_shared_triplets
from NUTS_sampler import build_pl_pymc_model
from pl_physics import OPTIONAL_PARAM_NAMES, PARAM_NAMES

logger = logging.getLogger("pl_analysis.execute_pymc")


class PLRunResult(NamedTuple):
    """ Everything downstream code needs from one run. """
    idata: Any                                   # arviz.InferenceData (posterior, posterior_predictive, prior, ...)
    processed_series: List[Dict]                 # the spectra actually fitted (globally normalised, ascending T)
    map_params: Optional[Dict[str, np.ndarray]]  # MAP point estimate of the physical parameters (None if not run)


def run_pymc(
    samples_dir: Union[str, Path],
    background_path: Union[str, Path],
    calibration_path: Union[str, Path],
    absorption_peaks: List[float],
    calibration_mode: str = "multiply",
    temperature_subset: Optional[Sequence[float]] = None,
    draws: int = 1000,
    tune: int = 1500,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 51,
    use_map_init: bool = True,
    init_jitter: bool = False,
    prior_predictive_draws: int = 200,
    priors: Optional[Mapping[str, Any]] = None,
    fit_baseline: bool = True,
    relative_noise: bool = True,
    lineshape: str = "pseudo_voigt",
) -> PLRunResult:
    """ Runs the full pipeline and returns a PLRunResult.
    ------
    temperature_subset - Optional[Sequence[float]]: fit only these temperatures (K), e.g. [100] to validate the
        peak decomposition on the sharpest spectrum first. Normalisation is still done over the FULL series,
        so intensities stay on the same scale as in the complete fit.
    use_map_init - bool: start NUTS from a MAP fit (recommended for overlapping peaks).
    init_jitter - bool: if True, add PyMC's random jitter around the starting point (gives R-hat more power
        to detect multimodality, but can start chains in poor regions).
    prior_predictive_draws - int: number of prior predictive draws to store (0 disables).
    priors - Optional[Mapping]: overrides for NUTS_sampler.DEFAULT_PRIORS, e.g.
        {"e_ref_sigma": 12.0, "gamma_0_median": 40.0, "gamma_0_sigma_log": 0.3}.
        Per-peak parameters also accept a list with one value per peak.
    fit_baseline - bool: fit a constant offset per spectrum (recommended; absorbs small background errors).
    relative_noise - bool: heteroscedastic noise sd_i = sqrt(sigma^2 + (rel_err * y_i)^2) (recommended; see
        NUTS_sampler). False = one constant sigma per spectrum.
    lineshape - str: "pseudo_voigt" (default; per-peak Lorentzian fraction eta) or "gaussian".
    """
    file_triplets = build_shared_triplets(
        samples_dir=samples_dir,
        background_path=background_path,
        calibration_path=calibration_path,
    )
    # All spectra are divided by ONE common factor so relative intensities (quenching) are preserved.
    processed_series = process_temperature_series(
        file_triplets, calibration_mode=calibration_mode, global_normalize=True
    )

    if temperature_subset is not None:
        wanted = np.asarray(temperature_subset, dtype=float)
        processed_series = [
            d for d in processed_series if np.any(np.isclose(float(d["temperature"]), wanted))
        ]
        if not processed_series:
            raise ValueError(f"No spectra match temperature_subset={list(temperature_subset)}.")
        logger.info("Fitting temperature subset (K): %s", [d["temperature"] for d in processed_series])

    model = build_pl_pymc_model(
        processed_series=processed_series,
        num_peaks=len(absorption_peaks),
        absorption_centers=absorption_peaks,  # initial centre estimates from absorption data (eV)
        priors=priors,
        fit_baseline=fit_baseline,
        relative_noise=relative_noise,
        lineshape=lineshape,
    )

    with model:
        logger.info("Individual node log-probabilities at the initial point:")
        for name, logp in model.point_logps().items():
            logger.info("  %-25s %s", name, logp)

        prior_idata = None
        if prior_predictive_draws > 0:
            prior_idata = pm.sample_prior_predictive(draws=prior_predictive_draws, random_seed=random_seed)

        start = None
        map_params: Optional[Dict[str, np.ndarray]] = None
        if use_map_init:
            try:
                map_est = pm.find_MAP(method="L-BFGS-B")
                free_names = {rv.name for rv in model.free_RVs}
                start = {k: v for k, v in map_est.items() if k in free_names}
                if all(k in map_est for k in PARAM_NAMES):
                    map_params = {
                        k: np.asarray(map_est[k])
                        for k in list(PARAM_NAMES) + list(OPTIONAL_PARAM_NAMES)
                        if k in map_est
                    }
                logger.info("MAP fit succeeded; starting NUTS from the MAP point.")
            except Exception as exc:  # MAP is a convenience, never a hard requirement
                logger.warning("MAP initialisation failed (%s); using default initial values.", exc)
                start, map_params = None, None

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            init="jitter+adapt_diag" if init_jitter else "adapt_diag",
            initvals=start,
            return_inferencedata=True,
        )

        idata.extend(pm.sample_posterior_predictive(idata, random_seed=random_seed))
        if prior_idata is not None:
            idata.extend(prior_idata)

    return PLRunResult(idata=idata, processed_series=processed_series, map_params=map_params)
