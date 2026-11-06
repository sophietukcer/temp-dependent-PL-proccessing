from pathlib import Path
import logging
import logging.config
import arviz as az
from datetime import datetime
import importlib
from typing import Union, List
from pathlib import Path


from temp_dependent_pl_proccessing.execute_pymc import run_pymc 
import temp_dependent_pl_proccessing.visualisations_pymc as viz


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
        bckg_file: str,
        cal_file: str,
        absorption_peaks: List[float],
        output_dir: str = 'results_pymc'
         ):
    # 1. Initialize Logger
    logger = setup_logger("logging_config.ini")
    logger.info("Initializing PyMC Photoluminescence Sampling Pipeline")

    # 2. Paths & Physical Settings
    
    samples_dir = samples_dir
    background_path = bckg_file
    calibration_path = cal_file
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / timestamp
    out_path.mkdir(parents=True, exist_ok=True)

    # 3. Execute PyMC Sampling
    logger.info("Executing NUTS MCMC Sampling via run_pymc()...")
    try:
        idata = run_pymc(
            samples_dir=samples_dir,
            background_path=background_path,
            calibration_path=calibration_path,
            absorption_peaks=absorption_peaks,
            calibration_mode="multiply",
        )
        logger.info("MCMC Sampling completed successfully.")
    except Exception as e:
        logger.error(f"Sampling failure: {str(e)}", exc_info=True)
        raise

    # 4. Export NetCDF Output
    netcdf_path = output_dir / "pl_inference_data.nc"
    idata.to_netcdf(netcdf_path)
    logger.info(f"Saved InferenceData to disk: {netcdf_path.resolve()}")

    # 5. Convergence & HMC Diagnostic Validation
    logger.info("Running MCMC convergence & sampler health checks...")
    
    # Check Divergent Transitions
    num_divergent = int(idata.sample_stats.diverging.sum())
    if num_divergent > 0:
        logger.warning(
            f"Detected {num_divergent} divergent transitions! "
            "Consider re-running with a higher target_accept (e.g., 0.95 or 0.98)."
        )
    else:
        logger.info("Zero divergent transitions detected (HMC geometry healthy).")

    # Evaluate R-hat & Effective Sample Size
    target_params = ["e0_0", "gamma_0", "gamma_ac", "a_factor", "alpha", "sigma"]
    summary = az.summary(idata, var_names=target_params)
    
    max_rhat = summary["r_hat"].max()
    min_ess = summary["ess_bulk"].min()
    
    logger.info(f"Convergence Summary — Max R-hat: {max_rhat:.4f} | Min ESS (bulk): {min_ess:.1f}")
    
    if max_rhat > 1.01:
        logger.warning(f"High R-hat detected ({max_rhat:.3f} > 1.01). Chains may not have converged.")
    if min_ess < 400:
        logger.warning(f"Low effective sample size ({min_ess:.1f} < 400). Consider increasing sample draws.")

    # Log summary table
    logger.debug(f"\nParameter Summary Table:\n{summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk']]}")

    # 6. Generate Native ArviZ Visualizations
    logger.info("Generating native ArviZ visualization plots...")

    # Trace Plot
    viz.plot_pymc_traces(
        idata=idata,
        var_names=target_params,
        save_path=output_dir / "trace_plot.png",
    )
    
    # Corner / Pair Plot
    viz.plot_pymc_corner(
        idata=idata,
        var_names=target_params,
        save_path=output_dir / "corner_plot.png",
    )
    
    # NUTS Energy Diagnostic
    viz.plot_energy_diagnostics(
        idata=idata,
        save_path=output_dir / "energy_diagnostic.png",
    )

    # Posterior Predictive Check (PPC Fit Overlay)
    if hasattr(idata, "posterior_predictive"):
        viz.plot_posterior_predictive_fit(
            idata=idata,
            save_path=output_dir / "posterior_predictive_fit.png",
        )
    else:
        logger.warning("No posterior predictive samples found in idata. Skipping PPC plot.")

    # Posterior Distributions & HDIs
    viz.plot_parameter_posteriors(
        idata=idata,
        var_names=["e0_0", "gamma_0", "gamma_ac", "alpha"],
        save_path=output_dir / "posterior_hdi.png",
    )

    logger.info(f"Pipeline finished. All figures and outputs saved to '{output_dir.resolve()}'.")


if __name__ == "__main__":
    SAMPLES_DIR = r"G:\My Drive\Data\temp dependent PL\26-10-13_ST072_FPEA_BA_n=4\26-10-14_ST072_15_BA_n=4\tests"        
    BACKGROUND_FILE = Path(SAMPLES_DIR) / "bg_laser_on_0_4ns_480_766nm_50uW_294K_3rd_spot_3000gain_0.5s_100acc_2772t0_next_day.asc"   # Single shared background spectrum file
    CALIBRATION_FILE = Path(SAMPLES_DIR) / "iCCD_355_correction.csv" # Single shared response calibration spectrum file

    # Initial peak center estimates from absorption data (eV)
    ABSORPTION_PEAKS = [1.75, 1.80, 1.93, 2.02]

    main(samples_dir = SAMPLES_DIR,
         bckg_file = BACKGROUND_FILE,
         cal_file = CALIBRATION_FILE,
        absorption_peaks=ABSORPTION_PEAKS,
    )