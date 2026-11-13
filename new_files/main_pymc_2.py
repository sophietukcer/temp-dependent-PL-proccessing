from pathlib import Path
import logging
import logging.config
import arviz as az
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from execute_pymc_2 import run_pymc
import visualisations_pymc_2 as viz


# Physical parameters reported in the convergence summary and plots.
TARGET_PARAMS = [
    "e_ref", "de_dt", "e0_0",
    "gamma_0", "gamma_ac", "gamma_lo", "e_lo",
    "i_0", "e_a", "ln_q_ref",
    "sigma",
]
# Parameters most prone to ridges/funnels; used for the divergence-highlighted pair plot.
PAIR_PARAMS = ["gamma_0", "gamma_ac", "gamma_lo", "e_lo", "e_a", "ln_q_ref"]


def setup_logger(config_path: Union[str, Path] = "logger_config.py") -> logging.Logger:
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

    return logging.getLogger("pl_pymc_pipeline")


def main(
        samples_dir: Union[str, Path],
        bckg_file: Union[str, Path],
        cal_file: Union[str, Path],
        absorption_peaks: List[float],
        output_dir: str = 'results_pymc',
        temperature_subset: Optional[Sequence[float]] = None,
        sampler_kwargs: Optional[Dict[str, Any]] = None,
         ):
    """ Runs the full PL inference pipeline.
    ------
    temperature_subset - fit only these temperatures (K), e.g. [100] to check the peak decomposition first.
    sampler_kwargs - forwarded to run_pymc (draws, tune, chains, target_accept, use_map_init, init_jitter, ...).
    """
    # 1. Initialize Logger
    logger = setup_logger("logging_config.ini")
    logger.info("Initializing PyMC Photoluminescence Sampling Pipeline")

    # 2. Paths & Physical Settings
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / timestamp
    out_path.mkdir(parents=True, exist_ok=True)

    # 3. Execute PyMC Sampling
    logger.info("Executing NUTS MCMC Sampling via run_pymc()...")
    try:
        result = run_pymc(
            samples_dir=samples_dir,
            background_path=bckg_file,
            calibration_path=cal_file,
            absorption_peaks=absorption_peaks,
            calibration_mode="multiply",
            temperature_subset=temperature_subset,
            **(sampler_kwargs or {}),
        )
        logger.info("MCMC Sampling completed successfully.")
    except Exception as e:
        logger.error(f"Sampling failure: {str(e)}", exc_info=True)
        raise

    idata = result.idata
    series = result.processed_series

    # 4. Export outputs (inference data + the exact spectra that were fitted)
    netcdf_path = out_path / "pl_inference_data.nc"
    idata.to_netcdf(netcdf_path)
    logger.info(f"Saved InferenceData to disk: {netcdf_path.resolve()}")

    spectra = {"temperatures": np.array([d["temperature"] for d in series], dtype=float),
               "norm_factor": float(series[0].get("norm_factor", 1.0))}
    for i, d in enumerate(series):
        spectra[f"energy_ev_{i}"] = np.asarray(d["energy_ev"])
        spectra[f"intensity_{i}"] = np.asarray(d["intensity"])
    np.savez(out_path / "fitted_spectra.npz", **spectra)

    # 5. Convergence & HMC Diagnostic Validation
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
    summary = az.summary(idata, var_names=target_params)

    max_rhat = summary["r_hat"].max()
    min_ess = summary["ess_bulk"].min()

    logger.info(f"Convergence Summary — Max R-hat: {max_rhat:.4f} | Min ESS (bulk): {min_ess:.1f}")

    if max_rhat > 1.01:
        logger.warning(f"High R-hat detected ({max_rhat:.3f} > 1.01). Chains may not have converged.")
    if min_ess < 400:
        logger.warning(f"Low effective sample size ({min_ess:.1f} < 400). Consider increasing sample draws.")

    logger.debug(f"\nParameter Summary Table:\n{summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk']]}")
    summary.to_csv(out_path / "parameter_summary.csv")

    # 6. Visualizations
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
        viz.posterior_mean_params(idata),
        draws=viz.sample_param_draws(idata, num_draws=40),
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
        var_names=["e0_0", "de_dt", "gamma_0", "gamma_ac", "gamma_lo", "e_lo", "e_a"],
        save_path=out_path / "posterior_hdi.png",
    )

    logger.info(f"Pipeline finished. All figures and outputs saved to '{out_path.resolve()}'.")
    return result


if __name__ == "__main__":
    SAMPLES_DIR = r"G:\My Drive\Data\temp dependent PL\26-10-13_ST072_FPEA_BA_n=4\26-10-14_ST072_15_BA_n=4\tests"
    BACKGROUND_FILE = Path(SAMPLES_DIR) / "bg_laser_on_0_4ns_480_766nm_50uW_294K_3rd_spot_3000gain_0.5s_100acc_2772t0_next_day.asc"   # Single shared background spectrum file
    CALIBRATION_FILE = Path(SAMPLES_DIR) / "iCCD_355_correction.csv" # Single shared response calibration spectrum file

    # Initial peak center estimates from absorption data (eV), ascending
    ABSORPTION_PEAKS = [1.75, 1.80, 1.93, 2.02]

    # Fit only the lowest-temperature spectrum first to validate the 4-peak decomposition,
    # e.g. TEMPERATURE_SUBSET = [100]. None = fit all temperatures.
    TEMPERATURE_SUBSET = None

    main(samples_dir=SAMPLES_DIR,
         bckg_file=BACKGROUND_FILE,
         cal_file=CALIBRATION_FILE,
         absorption_peaks=ABSORPTION_PEAKS,
         temperature_subset=TEMPERATURE_SUBSET,
         )
