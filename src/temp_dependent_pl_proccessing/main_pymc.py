from pathlib import Path
import logging
import logging.config
import arviz as az
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from temp_dependent_pl_proccessing.execute_pymc import run_pymc
import temp_dependent_pl_proccessing.visualisations_pymc as viz

import inspect
from temp_dependent_pl_proccessing.NUTS_sampler import DEFAULT_PRIORS   



# Physical parameters reported in the convergence summary and plots.
TARGET_PARAMS = [
    "e_ref", "de_dt", "e0_0",
    "gamma_0", "gamma_ac", "gamma_lo", "e_lo",
    "i_0", "e_a", "ln_q_ref", "eta",
    "sigma", "rel_err", "baseline",
]
# Quantities that ARE identified even from a single spectrum (used for health checks of 1-temperature runs).
DERIVED_PARAMS = ["e_centre", "fwhm", "area", "sigma", "rel_err", "baseline"]
# Parameters most prone to ridges/funnels; used for the divergence-highlighted pair plot.
PAIR_PARAMS = ["gamma_0", "gamma_ac", "gamma_lo", "e_lo", "e_a", "ln_q_ref"]


def setup_logger(config_path: Union[str, Path] = "logger_config.py", log_level="INFO", log_file=None) -> logging.Logger:
    """
    Initializes logger configuration from an external config file.
    """
    config_file = Path(config_path)
    if config_file.exists():
        logging.config.fileConfig(config_file, disable_existing_loggers=False)
    else:
        # Basic fallback logger if config file is not at the exact path
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logging.warning(f"Logger config file '{config_path}' not found. Using default basicConfig.")
    logger = logging.getLogger("pl_pymc_pipeline")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if log_file is not None:                    
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(fh)

    return logger


def main(
        samples_dir: Union[str, Path],
        bckg_file: Union[str, Path],
        cal_file: Union[str, Path],
        absorption_peaks: List[float],
        output_dir: str = 'results_pymc',
        temperature_subset: Optional[Sequence[float]] = None,
        priors: Optional[Dict[str, Any]] = None,
        fit_baseline: bool = True,
        relative_noise: bool = True,
        lineshape: str = "pseudo_voigt",
        sampler_kwargs: Optional[Dict[str, Any]] = None,
         ):
    """ Runs the full PL inference pipeline.
    ------
    temperature_subset - fit only these temperatures (K), e.g. [100] to check the peak decomposition first.
    priors - overrides for NUTS_sampler.DEFAULT_PRIORS, e.g. {"e_ref_sigma": 12.0, "gamma_0_sigma_log": 0.3}.
    fit_baseline - fit a constant offset per spectrum (absorbs small background-subtraction errors).
    relative_noise - heteroscedastic noise (floor + relative term); False = one constant sigma per spectrum.
    lineshape - "pseudo_voigt" (per-peak Lorentzian fraction eta, default) or "gaussian".
    sampler_kwargs - forwarded to run_pymc (draws, tune, chains, target_accept, use_map_init, init_jitter, ...).
    """
    # Initialize Logger


    # Paths & Physical Settings
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / timestamp
    out_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("logging_config.ini", log_level="INFO", log_file=out_path / "pipeline.log")
    logger.info("Initializing PyMC Photoluminescence Sampling Pipeline")
    logger.info(f"Output directory created: {out_path.resolve()}")

    # Execute PyMC Sampling
    logger.info("Executing NUTS MCMC Sampling via run_pymc()...")
    logger.info(
    "Run config | peaks=%s | lineshape=%s | subset=%s | baseline=%s | rel_noise=%s | priors=%s | sampler=%s",
    absorption_peaks, lineshape, temperature_subset, fit_baseline, relative_noise,
    {**DEFAULT_PRIORS, **(priors or {})},
    {**{k: v.default for k, v in inspect.signature(run_pymc).parameters.items()
        if k in ("draws", "tune", "chains", "target_accept", "random_seed", "use_map_init", "init_jitter")},
     **(sampler_kwargs or {})},
    )
    try:
        result = run_pymc(
            samples_dir=samples_dir,
            background_path=bckg_file,
            calibration_path=cal_file,
            absorption_peaks=absorption_peaks,
            calibration_mode="multiply",
            temperature_subset=temperature_subset,
            priors=priors,
            fit_baseline=fit_baseline,
            relative_noise=relative_noise,
            lineshape=lineshape,
            **(sampler_kwargs or {}),
        )
        logger.info("MCMC Sampling completed successfully. ")
    except Exception as e:
        logger.error(f"Sampling failure: {str(e)}", exc_info=True)
        raise

    idata = result.idata
    series = result.processed_series

    # Export outputs (inference data + the exact spectra that were fitted)
    netcdf_path = out_path / "pl_inference_data.nc"
    idata.to_netcdf(netcdf_path)
    logger.info(f"Saved InferenceData to disk: {netcdf_path.resolve()}")

    spectra = {"temperatures": np.array([d["temperature"] for d in series], dtype=float),
               "norm_factor": float(series[0].get("norm_factor", 1.0))}
    for i, d in enumerate(series):
        spectra[f"energy_ev_{i}"] = np.asarray(d["energy_ev"])
        spectra[f"intensity_{i}"] = np.asarray(d["intensity"])
    np.savez(out_path / "fitted_spectra.npz", **spectra)

    # Convergence & HMC Diagnostic Validation
    logger.info("Running MCMC convergence & sampler health checks...")

    num_divergent = int(idata.sample_stats.diverging.sum())
    if num_divergent > 0:
        logger.warning(
            f"Detected {num_divergent} divergent transitions! Inspect pair_plot.png (divergences highlighted), "
            "especially gamma_ac vs gamma_lo vs e_lo and e_a vs ln_q_ref, and compare the prior predictive "
            "and MAP fit plots before simply raising target_accept."
        )
    else:
        logger.info("Zero divergent transitions detected (HMC geometry healthy).")

    target_params = [p for p in TARGET_PARAMS if p in idata.posterior]
    summary = az.summary(idata, var_names=target_params + [p for p in DERIVED_PARAMS[:3] if p in idata.posterior])

    if len(series) == 1:
        # One spectrum cannot identify de_dt, gamma_ac, gamma_lo, e_lo, e_a, or i_0 vs ln_q_ref separately
        # (they stay prior-driven / degenerate), so judge convergence on the quantities the data DO constrain.
        logger.info("Single-temperature run: convergence is judged on e_centre, fwhm, area, noise terms and baseline only.")
        check = summary.loc[[i for i in summary.index if i.split("[")[0] in DERIVED_PARAMS]]
    else:
        check = summary
    max_rhat = check["r_hat"].max()
    min_ess = check["ess_bulk"].min()

    logger.info(f"Convergence Summary — Max R-hat: {max_rhat:.4f} | Min ESS (bulk): {min_ess:.1f}")

    if max_rhat > 1.01:
        logger.warning(f"High R-hat detected ({max_rhat:.3f} > 1.01). Chains may not have converged.")
    if min_ess < 400:
        logger.warning(f"Low effective sample size ({min_ess:.1f} < 400). Consider increasing sample draws.")

    logger.debug(f"\nParameter Summary Table:\n{summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk']]}")
    summary.to_csv(out_path / "parameter_summary.csv")

    # Visualizations
    logger.info("Generating visualization plots...")

    # Prior predictive check: do prior draws look like plausible spectra?
    if hasattr(idata, "prior"):
        viz.plot_prior_predictive(series, idata, save_path=out_path / "prior_predictive.png")
    else:
        logger.warning("No prior predictive samples found in idata. Skipping prior predictive plot.")

    # MAP fit with components + residuals (structured residuals = model/data issue, not a sampler issue)
    if result.map_params is not None:
        viz.plot_spectral_fits(series, result.map_params, save_path=out_path / "map_fit.png")

    # Posterior fit: data, total, individual peaks, residuals, faint posterior draws
    viz.plot_spectral_fits(
        series,
        params=None,  # central curves = pointwise median of the draws' spectra (NOT the model at mean parameters)
        draws=viz.sample_param_draws(idata, num_draws=100),
        save_path=out_path / "posterior_fit.png",
    )

    # Physical trends vs temperature (centre, FWHM, area)
    viz.plot_temperature_trends(idata, series, save_path=out_path / "temperature_trends.png")

    # Trace Plot
    viz.plot_pymc_traces(idata=idata, var_names=target_params, save_path=out_path / "trace_plot.png")

    # Pair plot with divergences highlighted (restricted to Peak 1 for the per-peak parameters)
    viz.plot_pymc_corner(
        idata=idata,
        var_names=[p for p in PAIR_PARAMS if p in idata.posterior],
        coords={"peak": ["Peak 1"]},
        save_path=out_path / "pair_plot.png",
    )

    # NUTS Energy Diagnostic
    viz.plot_energy_diagnostics(idata=idata, save_path=out_path / "energy_diagnostic.png")

    # Posterior Distributions & HDIs
    viz.plot_parameter_posteriors(
        idata=idata,
        var_names=[v for v in ["e0_0", "de_dt", "gamma_0", "gamma_ac", "gamma_lo", "e_lo", "e_a", "eta"]
                   if v in idata.posterior],
        save_path=out_path / "posterior_hdi.png",
    )

    logger.info(f"Pipeline finished. All figures and outputs saved to '{out_path.resolve()}'.")
    return result


if __name__ == "__main__":
    SAMPLES_DIR = r"G:\My Drive\Data\temp dependent PL\26-10-13_ST072_FPEA_BA_n=4\26-10-14_ST072_15_BA_n=4\tests"
    BACKGROUND_FILE = Path(SAMPLES_DIR) / "bg_laser_on_0_4ns_480_766nm_50uW_294K_3rd_spot_3000gain_0.5s_100acc_2772t0_next_day.asc"   # Single shared background spectrum file
    CALIBRATION_FILE = Path(SAMPLES_DIR) / "iCCD_355_correction.csv" # Single shared response calibration spectrum file

    # Initial peak centres (eV), ascending; the number of peaks follows this list. Five components: the two
    # highest (1.934, 2.026) were stable across independent fits, the three lower ones are less well determined.
    # (The original absorption-based four-peak guess was [1.75, 1.80, 1.93, 2.02].)
    # ABSORPTION_PEAKS = [1.73, 1.84, 1.934, 2.026, 2.18]
    ABSORPTION_PEAKS = [1.730, 1.775, 1.937, 2.022]
    # ABSORPTION_PEAKS = [1.84, 1.90, 1.934, 2.026]

    # Prior overrides (None = DEFAULT_PRIORS in NUTS_sampler.py). Scalars apply to all peaks; per-peak
    # parameters also accept one value per peak (length must equal the number of peaks above).
    # Starting values derived from least-squares fits of this data set: component widths of ~65-100 meV,
    # a thermal broadening that is NOT negligible, and an activation energy near 80 meV for the total intensity.
    PRIORS = {
        "e_ref_sigma":        [8.0, 8.0, 8.0, 8.0],     # meV, ~3 sigma = +-10 nm
        "de_dt_sigma":        [0.08, 0.15, 0.08, 0.08],  # meV/K
        "gamma_0_median":     [60.0, 130.0, 78.0, 68.0], # meV (FWHM near 0 K)
        "gamma_0_sigma_log":  0.2,  
        "gamma_ac_median":    0.05,  "gamma_ac_sigma_log": 0.6,
        "gamma_lo_median":    [29, 44, 22, 10],  "gamma_lo_sigma_log": 0.6,
        "e_lo_median":        15.0,  "e_lo_sigma_log":     0.15,
        "i_0_median_log":     [2.83, 4.91, 2.83, 1.67],  # ln of areas ~ [17, 135, 17, 23, 5.3]
        "i_0_sigma_log":      [0.4, 0.25, 0.4, 0.3],
        "baseline_sigma":     0.003,
    }
    # PRIORS = {
    #     "e_ref_sigma":       [40.0, 10.0, 10.0, 10.0],   # meV, tight on the two robust upper peaks
    #     "de_dt_sigma":       0.15,                             # meV/K
    #     "gamma_0_median":    [100.0, 70.0, 70, 65.0],  # meV
    #     "gamma_0_sigma_log": 0.4,
    #     "gamma_ac_median":   0.05,                             # meV/K
    #     "gamma_ac_sigma_log": 0.8,
    #     "gamma_lo_median":   50.0,                             # meV
    #     "gamma_lo_sigma_log": 0.8,
    #     "e_a_median":        80.0,                             # meV
    #     "e_a_sigma_log":     0.5,
    # }
    LINESHAPE = "pseudo_voigt"   # or "gaussian"

    # Fit only the lowest-temperature spectrum first to validate the 4-peak decomposition,
    # e.g. TEMPERATURE_SUBSET = [100]. None = fit all temperatures.
    TEMPERATURE_SUBSET = None

    main(samples_dir=SAMPLES_DIR,
         bckg_file=BACKGROUND_FILE,
         cal_file=CALIBRATION_FILE,
         absorption_peaks=ABSORPTION_PEAKS,
         temperature_subset=TEMPERATURE_SUBSET,
         priors=PRIORS,
         fit_baseline=True,
         relative_noise=True,
         lineshape=LINESHAPE,
         )
