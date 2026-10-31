"""
main.py
-------
Master pipeline orchestrator for temperature-dependent PL analysis. Automatically pairs sample files
in a folder with a single shared background and calibration file, executes joint Bayesian MCMC fitting,
performs BIC model selection, and saves diagnostic figures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from temp_dependent_pl_proccessing.bayesian_sampler import run_mcmc_sampling
from temp_dependent_pl_proccessing.data_loader import process_temperature_series
from temp_dependent_pl_proccessing.logger_config import setup_logger
from temp_dependent_pl_proccessing.model_selection import evaluate_model_fit, justify_peak_addition
from temp_dependent_pl_proccessing.visualisation import (
    plot_corner_posterior,
    plot_parameter_temperature_trends,
    plot_temperature_series_fits,
    plot_mcmc_traceplots
)


def build_shared_triplets(
    samples_dir: Union[str, Path], 
    background_path: Union[str, Path],
    calibration_path: Union[str, Path],
    file_extension: str = ".asc",
) -> List[Tuple[Path, Path, Path]]:
    """ Discovers sample files in a directory, explicitly filtering out background and calibration files.
    ------
    OUTPUT - List[Tuple[Path, Path, Path]]: List of (sample_path, background_path, calibration_path) tuples.
    ------
    samples_dir - Union[str, Path]: Directory containing temperature-dependent sample files.
    background_path - Union[str, Path]: Shared background file path.
    calibration_path - Union[str, Path]: Shared calibration spectrum file path.
    file_extension - str: Extension matching sample files (default: '.asc').
    ------
    FileNotFoundError: Raised if directory or shared background/calibration files do not exist.
    """
    samples_folder = Path(samples_dir)
    bg_file = Path(background_path).resolve()
    cal_file = Path(calibration_path).resolve()

    if not samples_folder.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {samples_folder}")
    if not bg_file.is_file():
        raise FileNotFoundError(f"Shared background file not found: {bg_file}")
    if not cal_file.is_file():
        raise FileNotFoundError(f"Shared calibration file not found: {cal_file}")

    all_matching_files = sorted(list(samples_folder.glob(f"*{file_extension}")))
    sample_files: List[Path] = []

    for f_path in all_matching_files:
        f_resolved = f_path.resolve()
        
        # Skip if exact match to background or calibration file
        if f_resolved == bg_file or f_resolved == cal_file:
            continue

        # Skip files with background/calibration keywords in the filename
        fname_lower = f_path.name.lower()
        if any(keyword in fname_lower for keyword in ["bg_", "_bg", "background", "cal_", "_cal", "calibration"]):
            continue

        sample_files.append(f_path)

    if not sample_files:
        raise FileNotFoundError(
            f"No valid sample files found in '{samples_folder}' matching extension '{file_extension}' after filtering."
        )

    return [(s_file, bg_file, cal_file) for s_file in sample_files]


def run_full_analysis_pipeline(
    samples_dir: Union[str, Path],
    background_path: Union[str, Path],
    calibration_path: Union[str, Path],
    absorption_peaks: List[float],
    custom_peak_guesses: Optional[List[Dict[str, float]]] = None,
    calibration_mode: str = "divide",
    test_additional_peak: bool = True,
    output_dir: Union[str, Path] = "results",
    n_walkers: int = 60,
    n_steps: int = 3000,
    burn_in: int = 1000,
    use_parallel: bool = True,
) -> None:
    """ Runs end-to-end PL Bayesian joint analysis by processing samples against shared background/calibration files.
    ------
    OUTPUT - None: Writes pipeline logs and saves summary fit plots, trend curves, and posterior corner plots to disk.
    ------
    samples_dir - Union[str, Path]: Directory containing temperature series sample text files.
    background_path - Union[str, Path]: Path to single background spectrum file.
    calibration_path - Union[str, Path]: Path to single detector sensitivity calibration spectrum file.
    absorption_peaks - List[float]: Initial peak position estimates in eV (e.g., from room-T absorption).
    custom_peak_guesses - Optional[List[Dict[str, float]]]: Optional per-peak parameter initial guess dicts (e.g., e_lo, gamma_lo, e_a).
    calibration_mode - str: Calibration mode, either 'divide' or 'multiply' (default: 'divide').
    test_additional_peak - bool: Flag enabling automatic comparison of K vs K+1 peak model selection via BIC drop (default: True).
    output_dir - Union[str, Path]: Directory path to store output figures and logs (default: 'results').
    n_walkers - int: Number of ensemble MCMC walkers (default: 60).
    n_steps - int: Total sampling steps per walker (default: 3000).
    burn_in - int: Initial steps to discard as burn-in (default: 1000).
    use_parallel - bool: Enable multi-core parallel CPU execution (default: True).
    ------
    FileNotFoundError: Raised if input directory or files do not exist.
    ValueError: Raised if processing or MCMC sampling parameters are invalid.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="pl_analysis.main",
        log_level="INFO",
        log_file=out_path / "pipeline.log",
    )
    logger.info("Initializing Photoluminescence Joint Bayesian MCMC Pipeline.")

    # 1. Pairing and Ingesting Spectra
    file_triplets = build_shared_triplets(
        samples_dir=samples_dir,
        background_path=background_path,
        calibration_path=calibration_path,
    )
    logger.info("Paired %d sample spectra with shared background and calibration files.", len(file_triplets))

    processed_series = process_temperature_series(
        file_triplets, calibration_mode=calibration_mode
    )
    temperatures = np.array([d["temperature"] for d in processed_series])

    # 2. Fit Base Model (K peaks)
    k_base = len(absorption_peaks)
    logger.info("--- Fitting Base Model (K = %d peaks) ---", k_base)
    sampler_k, flat_samples_k, flat_log_prob_k = run_mcmc_sampling(
        processed_series,
        num_peaks=k_base,
        absorption_centers=absorption_peaks,
        custom_peak_guesses=custom_peak_guesses,
        n_walkers=n_walkers,
        n_steps=n_steps,
        burn_in=burn_in,
        use_parallel=use_parallel,
    )

    plot_mcmc_traceplots(
    sampler=sampler_k,
    num_peaks=k_base,
    burn_in=burn_in,
    save_path=out_path / "mcmc_traceplots.png",
    )

    print(f"Min acceptance:  {np.min(sampler_k.acceptance_fraction):.3f}" + f"Mean Acceptance Fraction: {np.mean(sampler_k.acceptance_fraction):.3f}")

    eval_k = evaluate_model_fit(flat_samples_k, flat_log_prob_k, processed_series, num_peaks=k_base)
    best_params = eval_k["mle_params"]
    selected_k = k_base
    selected_samples = flat_samples_k

    # 3. Model Selection Check (Optional K+1 peaks)
    if test_additional_peak:
        k_plus_1 = k_base + 1
        extra_target = float(absorption_peaks[-1] + 0.15)
        extended_targets = absorption_peaks + [extra_target]

        extended_guesses = None
        if custom_peak_guesses is not None:
            # Append last peak's guess dictionary for the extra candidate peak
            extended_guesses = custom_peak_guesses + [custom_peak_guesses[-1]]

        logger.info("--- Testing Extended Model (K = %d peaks) ---", k_plus_1)
        _, flat_samples_k1, flat_log_prob_k1 = run_mcmc_sampling(
            processed_series,
            num_peaks=k_plus_1,
            absorption_centers=extended_targets,
            custom_peak_guesses=extended_guesses,
            n_walkers=max(n_walkers, 2 * (9 * k_plus_1 + 1) + 10),
            n_steps=n_steps,
            burn_in=burn_in,
            use_parallel=use_parallel,
        )

        eval_k1 = evaluate_model_fit(flat_samples_k1, flat_log_prob_k1, processed_series, num_peaks=k_plus_1)

        if justify_peak_addition(eval_k, eval_k1, bic_threshold=10.0):
            logger.info("Model K = %d peaks PREFERRED based on BIC drop (ΔBIC < -10).", k_plus_1)
            selected_k = k_plus_1
            best_params = eval_k1["mle_params"]
            selected_samples = flat_samples_k1
        else:
            logger.info("Model K = %d peaks RETAINED; extra peak not statistically justified.", k_base)

    # 4. Visualization & Result Export
    logger.info("--- Rendering Diagnostic Figures ---")
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

    logger.info("Pipeline completed successfully. Figures saved to directory: %s", out_path)


if __name__ == "__main__":
    # Update directory paths to your workspace layout
    SAMPLES_DIR = r"G:\My Drive\Data\temp dependent PL\26-10-13_ST072_FPEA_BA_n=4\26-10-14_ST072_15_BA_n=4\tests"          # Folder containing 10K.txt, 50K.txt, 100K.txt, etc.
    BACKGROUND_FILE = Path(SAMPLES_DIR) / "bg_laser_on_0_4ns_480_766nm_50uW_294K_3rd_spot_3000gain_0.5s_100acc_2772t0_next_day.asc"   # Single shared background spectrum file
    CALIBRATION_FILE = Path(SAMPLES_DIR) / "iCCD_355_correction.csv" # Single shared response calibration spectrum file

    # Initial peak center estimates from absorption data (eV)
    ABSORPTION_PEAKS = [1.75, 1.92, 1.98, 2.02]

    # Optional custom initial guesses for physical parameters
    CUSTOM_GUESSES = [
        {
            "gamma_0": 0.02,    # Low-T linewidth (eV)
            "e_lo": 0.030,      # LO phonon energy (eV)
            "gamma_lo": 0.025,  # LO phonon coupling (eV)
            "e_a": 0.045,       # Quenching activation energy (eV)
        },
        {
            "gamma_0": 0.01,
            "e_lo": 0.030,
            "gamma_lo": 0.025,
            "e_a": 0.035,
        },
        {
            "gamma_0": 0.01,
            "e_lo": 0.030,
            "gamma_lo": 0.025,
            "e_a": 0.035,
        },
        {
            "gamma_0": 0.01,
            "e_lo": 0.030,
            "gamma_lo": 0.025,
            "e_a": 0.035,
        },
    ]

    run_full_analysis_pipeline(
        samples_dir=SAMPLES_DIR,
        background_path=BACKGROUND_FILE,
        calibration_path=CALIBRATION_FILE,
        absorption_peaks=ABSORPTION_PEAKS,
        custom_peak_guesses=CUSTOM_GUESSES,
        calibration_mode="multiply",
        test_additional_peak=True,
        output_dir="results",
        n_walkers=200,
        n_steps=20000,
        burn_in=6000,
        use_parallel=True,
    )