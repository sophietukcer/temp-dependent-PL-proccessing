"""
main.py
-------
Master pipeline orchestrator for loading PL temperature series, running joint MCMC Bayesian fitting,
comparing model peak counts via BIC, and generating diagnostic plots.
"""

from pathlib import Path
from typing import List, Tuple, Union
import numpy as np

from bayesian_sampler import run_mcmc_sampling
from data_loader import process_temperature_series
from logger_config import setup_logger
from model_selection import evaluate_model_fit, justify_peak_addition
from visualisation import (
    plot_corner_posterior,
    plot_parameter_temperature_trends,
    plot_temperature_series_fits,
)


def run_full_analysis_pipeline(
    file_triplets: List[Tuple[Union[str, Path], Union[str, Path], Union[str, Path]]],
    absorption_peaks: List[float],
    test_additional_peak: bool = True,
    output_dir: Union[str, Path] = "results",
) -> None:
    """ Runs end-to-end PL analysis pipeline from data ingestion through MCMC sampling, BIC comparison, and visualization.
    ------
    OUTPUT - None: Generates summary logs and saves output figures to disk.
    ------
    file_triplets - List[Tuple[Union[str, Path], Union[str, Path], Union[str, Path]]]: List of (sample, background, calibration) path tuples.
    absorption_peaks - List[float]: Initial peak position energy targets in eV.
    test_additional_peak - bool: Flag to compare K vs K+1 peak model selection (default: True).
    output_dir - Union[str, Path]: Directory path to store output plots and logs (default: 'results').
    ------
    ValueError: Raised if file_triplets or absorption_peaks are empty.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="pl_analysis.main",
        log_level="INFO",
        log_file=out_path / "pipeline.log",
    )
    logger.info("Starting photoluminescence Bayesian joint analysis pipeline.")

    # 1. Data Processing
    processed_series = process_temperature_series(file_triplets)
    temperatures = np.array([d["temperature"] for d in processed_series])

    # 2. Base Model MCMC Sampling (K peaks)
    k_base = len(absorption_peaks)
    logger.info("--- Fitting Base Model (K = %d peaks) ---", k_base)
    _, flat_samples_k, flat_log_prob_k = run_mcmc_sampling(
        processed_series,
        num_peaks=k_base,
        absorption_centers=absorption_peaks,
        n_walkers=60,
        n_steps=3000,
        burn_in=1000,
    )

    eval_k = evaluate_model_fit(flat_samples_k, flat_log_prob_k, processed_series, num_peaks=k_base)
    best_params = eval_k["mle_params"]
    selected_k = k_base
    selected_samples = flat_samples_k

    # 3. Model Comparison (Optional K+1 peaks)
    if test_additional_peak:
        k_plus_1 = k_base + 1
        # Add an additional peak estimate near highest energy absorption target
        extra_target = float(absorption_peaks[-1] + 0.15)
        extended_targets = absorption_peaks + [extra_target]

        logger.info("--- Testing Extended Model (K = %d peaks) ---", k_plus_1)
        _, flat_samples_k1, flat_log_prob_k1 = run_mcmc_sampling(
            processed_series,
            num_peaks=k_plus_1,
            absorption_centers=extended_targets,
            n_walkers=70,
            n_steps=3000,
            burn_in=1000,
        )

        eval_k1 = evaluate_model_fit(flat_samples_k1, flat_log_prob_k1, processed_series, num_peaks=k_plus_1)

        if justify_peak_addition(eval_k, eval_k1, bic_threshold=10.0):
            logger.info("K = %d peaks model PREFERRED based on BIC drop.", k_plus_1)
            selected_k = k_plus_1
            best_params = eval_k1["mle_params"]
            selected_samples = flat_samples_k1
        else:
            logger.info("K = %d peaks model RETAINED; additional peak not justified.", k_base)

    # 4. Visualization & Output Generation
    logger.info("--- Generating Diagnostic Figures ---")
    plot_temperature_series_fits(
        processed_series,
        best_params,
        num_peaks=selected_k,
        save_path=out_path / "temperature_fits.png",
    )
    plot_parameter_temperature_trends(
        temperatures,
        best_params,
        num_peaks=selected_k,
        save_path=out_path / "temperature_trends.png",
    )
    plot_corner_posterior(
        selected_samples,
        num_peaks=selected_k,
        save_path=out_path / "posterior_corner.png",
    )

    logger.info("Pipeline execution completed successfully. Output saved to %s", out_path)


if __name__ == "__main__":
    # Example dataset setup
    TRIPLETS = [
        ("data/sample_100K.txt", "data/bg.txt", "data/cal.txt"),
        ("data/sample_150K.txt", "data/bg.txt", "data/cal.txt"),
        ("data/sample_200K.txt", "data/bg.txt", "data/cal.txt"),
    ]
    ABS_PEAKS = [1.85, 2.10]

    run_full_analysis_pipeline(TRIPLETS, ABS_PEAKS, test_additional_peak=True)