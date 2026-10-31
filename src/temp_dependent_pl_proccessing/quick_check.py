"""
quick_check.py
--------------
Visual and diagnostic validation script integrating data processing, physical peak modeling,
and logging output.
"""

from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np

from temp_dependent_pl_proccessing.data_loader import process_single_dataset
from temp_dependent_pl_proccessing.logger_config import setup_logger
from temp_dependent_pl_proccessing.physics import evaluate_composite_spectrum


def run_quick_check(
    sample_path: Union[str, Path],
    background_path: Union[str, Path],
    calibration_path: Union[str, Path],
    initial_peak_centers: List[float],
) -> None:
    """ Loads processed PL data, overlays an initial physical model evaluation, and displays diagnostic plots.
    ------
    OUTPUT - None: Renders diagnostic plot and outputs structured log messages.
    ------
    sample_path - Union[str, Path]: Path to raw sample spectrum text file.
    background_path - Union[str, Path]: Path to background spectrum text file.
    calibration_path - Union[str, Path]: Path to system calibration text file.
    initial_peak_centers - List[float]: Initial peak position estimates (eV) derived from absorption data.
    ------
    FileNotFoundError: Raised if any specified spectrum file does not exist.
    """
    logger = setup_logger(name="pl_analysis.quick_check", log_level="INFO")
    logger.info("Starting quick check diagnostic run.")

    dataset = process_single_dataset(sample_path, background_path, calibration_path)
    temp_k = float(dataset["temperature"])
    energy_ev = dataset["energy_ev"]
    intensity = dataset["intensity"]

    num_peaks = len(initial_peak_centers)
    logger.info("Setting up physical model overlay for %d peaks at T = %.2f K", num_peaks, temp_k)

    flat_params_list: List[float] = []
    for center_e in initial_peak_centers:
        flat_params_list.extend([
            0.04,     # gamma_0 (eV)
            1e-4,     # gamma_ac (eV/K)
            0.03,     # gamma_lo (eV)
            0.025,    # e_lo (eV)
            0.05,     # i_0 (integrated area)
            10.0,     # a_factor (quenching)
            0.03,     # e_a (activation energy, eV)
            center_e, # e_0_zero (center energy at 0 K, eV)
            1e-4,     # alpha_coeff (center shift, eV/K)
        ])

    flat_params = np.array(flat_params_list, dtype=np.float64)
    model_spectrum = evaluate_composite_spectrum(energy_ev, temp_k, flat_params, num_peaks)

    plt.figure(figsize=(9, 5))
    plt.plot(energy_ev, intensity, "k-", label=f"Data (T = {temp_k} K)", lw=1.5)
    plt.plot(energy_ev, model_spectrum, "r--", label="Initial Model Guess", lw=1.5)

    for i, center in enumerate(initial_peak_centers):
        label_text = f"Absorption Target {i+1} ({center:.2f} eV)" if i == 0 else ""
        plt.axvline(x=center, color="blue", linestyle=":", alpha=0.6, label=label_text)

    plt.xlabel("Energy (eV)")
    plt.ylabel("Normalized PL Intensity")
    plt.title(f"PL Spectral Diagnostic Check at T = {temp_k} K")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    logger.info("Quick check diagnostic completed successfully.")


if __name__ == "__main__":
    SAMPLE = r"G:\My Drive\Data\temp dependent PL\Sophie\2026-10-13\day2_BA_n=4\0_4ns_480_766nm_50uW_100K_3000gain_0.5s_100acc_2772t0_.asc"
    BACKGROUND = r"G:\My Drive\Data\temp dependent PL\Sophie\2026-10-13\bg_0_4ns_480_766nm_50uW_different_spot_3000gain_0.5s_100acc_2771t0_.asc"
    CALIBRATION = r"G:\My Drive\Data\temp dependent PL\iCCD_355_correction.csv"
    ABSORPTION_PEAKS = [1.75, 1.92, 2.02]

    try:
        run_quick_check(SAMPLE, BACKGROUND, CALIBRATION, ABSORPTION_PEAKS)
    except FileNotFoundError as err:
        print(f"File missing for quick check execution: {err}")