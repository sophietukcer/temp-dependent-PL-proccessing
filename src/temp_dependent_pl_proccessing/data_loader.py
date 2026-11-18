"""
data_loader.py
--------------
Module for loading, calibrating, Jacobian-transforming, and normalizing
temperature-dependent photoluminescence (PL) data files.

Normalisation
-------------
Spectra are normalised by ONE common factor (the maximum intensity found in
any spectrum of the series), not by each spectrum's own maximum. This keeps the
relative intensities between temperatures intact, which is what makes thermal
quenching (Arrhenius) information recoverable. It only has physical meaning if
all spectra were acquired under the same conditions (laser power, gain,
exposure time, accumulations, alignment); otherwise correct for those first.
"""

import logging
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger("pl_analysis.data_loader")


def extract_temperature(file_path: Union[str, Path]) -> float:
    """ Extracts the numerical temperature value in Kelvin from a given file path using regular expressions.
    ------
    OUTPUT - float: Temperature value in Kelvin parsed from the filename.
    ------
    file_path - Union[str, Path]: Path or string representing the file path to parse.
    ------
    ValueError: Raised if no numerical temperature pattern (e.g., '100K' or '100.5K') is detected in the filename.
    """
    filename = Path(file_path).name
    match = re.search(r"(\d+(?:\.\d+)?)\s*K", filename, re.IGNORECASE)
    if not match:
        logger.error("Failed to parse temperature from filename: %s", filename)
        raise ValueError(
            f"Could not extract temperature pattern (e.g., '100K') from filename: {filename}"
        )
    temp_val = float(match.group(1))
    logger.debug("Parsed temperature %.2f K from file: %s", temp_val, filename)
    return temp_val


def read_spectrum_file(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """ Reads a two-column spectroscopy text file, automatically filtering out non-numeric header rows.
    ------
    OUTPUT - Tuple[np.ndarray, np.ndarray]: Tuple containing wavelength array (nm) and intensity array (counts).
    ------
    file_path - Union[str, Path]: Path or string representing the location of the text file.
    ------
    FileNotFoundError: Raised if the specified file path does not exist.
    ValueError: Raised if the file does not contain at least two numeric data columns.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.error("Spectrum file not found: %s", path)
        raise FileNotFoundError(f"Spectrum file not found at path: {path}")

    data_rows: List[Tuple[float, float]] = []
    skipped_headers = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                skipped_headers += 1
                continue
            parts = line_str.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    wv, intensity = float(parts[0]), float(parts[1])
                    data_rows.append((wv, intensity))
                except ValueError:
                    skipped_headers += 1
                    continue

    if not data_rows:
        logger.error("No numeric spectral data rows extracted from file: %s", path)
        raise ValueError(f"No valid numeric spectral data found in file: {path}")

    logger.debug(
        "Successfully read %d points from %s (skipped %d header lines)",
        len(data_rows),
        path.name,
        skipped_headers,
    )

    
    spectral_data = np.array(data_rows, dtype=np.float64)
    return spectral_data[:, 0], spectral_data[:, 1]


def interpolate_calibration(
    target_wavelengths: np.ndarray,
    cal_wavelengths: np.ndarray,
    cal_intensities: np.ndarray,
) -> np.ndarray:
    """ Interpolates calibration factors onto target wavelength grid, clamping boundary values to prevent edge divergence.
    ------
    OUTPUT - np.ndarray: Interpolated calibration factors matching target_wavelengths shape.
    ------
    target_wavelengths - np.ndarray: Target wavelength grid (nm) to interpolate onto.
    cal_wavelengths - np.ndarray: Calibration reference wavelength grid (nm).
    cal_intensities - np.ndarray: Calibration reference sensitivity values.
    ------
    ValueError: Raised if calibration dataset has fewer than 2 points.
    """
    if len(cal_wavelengths) < 2:
        logger.error("Calibration dataset length (%d) insufficient.", len(cal_wavelengths))
        raise ValueError("Calibration dataset requires at least 2 points for linear interpolation.")

    sort_idx = np.argsort(cal_wavelengths)
    clean_cal_wv = cal_wavelengths[sort_idx]
    clean_cal_int = cal_intensities[sort_idx]

    min_cal, max_cal = clean_cal_wv[0], clean_cal_wv[-1]
    min_target, max_target = target_wavelengths.min(), target_wavelengths.max()

    if min_target < min_cal or max_target > max_cal:
        logger.warning(
            "Target wavelength range (%.2f-%.2f nm) exceeds calibration bounds (%.2f-%.2f nm). Clamping edge values.",
            min_target,
            max_target,
            min_cal,
            max_cal,
        )

    interp_func = interp1d(
        clean_cal_wv,
        clean_cal_int,
        kind="linear",
        bounds_error=False,
        fill_value=(clean_cal_int[0], clean_cal_int[-1]),
    )
    return interp_func(target_wavelengths)


def convert_wavelength_to_energy(
    wavelengths_nm: np.ndarray,
    intensities: np.ndarray,
    hc_const: float = 1239.842,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Converts spectrum from wavelength domain (nm) to energy domain (eV) via Jacobian transformation.
    ------
    OUTPUT - Tuple[np.ndarray, np.ndarray]: Tuple of ascending energy array (eV) and Jacobian-transformed intensities.
    ------
    wavelengths_nm - np.ndarray: Wavelength values in nanometers.
    intensities - np.ndarray: Spectral intensities in original wavelength domain.
    hc_const - float: Conversion constant in eV*nm (default: 1239.842).
    ------
    ValueError: Raised if any wavelength values are non-positive.
    """
    if np.any(wavelengths_nm <= 0):
        logger.error("Non-positive wavelengths encountered during Jacobian transformation.")
        raise ValueError("Wavelength values must be strictly positive for energy conversion.")

    energy_ev = hc_const / wavelengths_nm
    jacobian_factor = (wavelengths_nm**2) / hc_const
    energy_intensities = intensities * jacobian_factor

    sort_indices = np.argsort(energy_ev)
    return energy_ev[sort_indices], energy_intensities[sort_indices]


def process_single_dataset(
    sample_path: Union[str, Path],
    background_path: Union[str, Path],
    calibration_path: Union[str, Path],
    hc_const: float = 1239.842,
    calibration_mode: str = "multiply",
) -> Dict[str, Union[float, np.ndarray]]:
    """ Processes raw sample, background, and calibration files into a calibrated, background-subtracted energy spectrum.
    The returned intensity is NOT normalised; normalisation is applied across the whole series
    in `normalize_by_global_max` so that relative intensities between temperatures are preserved.
    ------
    OUTPUT - Dict[str, Union[float, np.ndarray]]: Dict with keys 'temperature', 'energy_ev', and 'intensity' (unnormalised).
    ------
    sample_path - Union[str, Path]: File path to raw sample spectrum file.
    background_path - Union[str, Path]: File path to raw background spectrum file.
    calibration_path - Union[str, Path]: File path to system response calibration spectrum file.
    hc_const - float: Conversion constant in eV*nm (default: 1239.842).
    calibration_mode - str: Calibration mode, either 'divide' or 'multiply'.
    ------
    ValueError: Raised if file reading, background subtraction, or calibration mode is invalid.
    """
    temperature = extract_temperature(sample_path)
    logger.info("Processing single dataset for T = %.2f K (%s)", temperature, Path(sample_path).name)

    sample_wv, sample_int = read_spectrum_file(sample_path)
    bg_wv, bg_int = read_spectrum_file(background_path)
    cal_wv, cal_int = read_spectrum_file(calibration_path)

    valid_mask = (sample_wv >= cal_wv.min()) & (sample_wv <= cal_wv.max())
    dropped_points = int(np.sum(~valid_mask))
    if dropped_points > 0:
        logger.warning(
            "Cropping %d points outside valid calibration range [%.2f, %.2f] nm for %s",
            dropped_points,
            cal_wv.min(),
            cal_wv.max(),
            Path(sample_path).name,
        )
    sample_wv = sample_wv[valid_mask]
    sample_int = sample_int[valid_mask]

    bg_interp = interpolate_calibration(sample_wv, bg_wv, bg_int)
    cal_interp = interpolate_calibration(sample_wv, cal_wv, cal_int)

    if calibration_mode == "divide":
        zero_or_neg = cal_interp <= 0
        if np.any(zero_or_neg):
            logger.warning(
                "Calibration curve contains %d non-positive values; clamping for safe division.",
                int(np.sum(zero_or_neg)),
            )
        cal_interp_safe = np.where(zero_or_neg, np.nan, cal_interp)
        calibrated_sample = np.nan_to_num(sample_int / cal_interp_safe, nan=0.0)
        calibrated_bg = np.nan_to_num(bg_interp / cal_interp_safe, nan=0.0)
    elif calibration_mode == "multiply":
        calibrated_sample = sample_int * cal_interp
        calibrated_bg = bg_interp * cal_interp
    else:
        logger.error("Invalid calibration_mode specified: %s", calibration_mode)
        raise ValueError("calibration_mode must be either 'divide' or 'multiply'.")

    # Background subtraction on the common sample wavelength grid, then a single
    # Jacobian conversion. 
    net_wavelength_domain = calibrated_sample - calibrated_bg
    energy_ev, energy_int = convert_wavelength_to_energy(
        sample_wv, net_wavelength_domain, hc_const=hc_const
    )

    peak = float(np.max(energy_int))
    if peak <= 0:
        logger.warning(
            "Peak intensity for %s is non-positive (%.4e) after background subtraction; check the background file.",
            Path(sample_path).name,
            peak,
        )
    logger.info("Dataset T = %.2f K processed. Raw peak intensity: %.4e", temperature, peak)

    return {
        "temperature": temperature,
        "energy_ev": energy_ev,
        "intensity": energy_int,
    }


def normalize_by_global_max(
    processed_series: List[Dict[str, Union[float, np.ndarray]]],
) -> List[Dict[str, Union[float, np.ndarray]]]:
    """ Divides every spectrum in the series by the single largest intensity found in any spectrum.
    Relative intensities between temperatures are preserved (the brightest spectrum peaks at 1,
    all others peak at <= 1), so thermal quenching remains visible in the data.
    ------
    OUTPUT - List[Dict]: New list of dicts with normalised 'intensity' and 'norm_factor' (the common divisor, in raw units).
    ------
    processed_series - List[Dict]: Output of `process_single_dataset` for each temperature (unnormalised).
    ------
    ValueError: Raised if the series is empty or the global maximum is non-positive.
    """
    if not processed_series:
        raise ValueError("Cannot normalise an empty spectral series.")

    global_max = max(float(np.max(d["intensity"])) for d in processed_series)
    if global_max <= 0:
        logger.error("Global maximum intensity is non-positive (%.4e).", global_max)
        raise ValueError("Global maximum intensity is non-positive; cannot normalise the series.")

    normalized: List[Dict[str, Union[float, np.ndarray]]] = []
    for d in processed_series:
        new_d = dict(d)
        new_d["intensity"] = np.asarray(d["intensity"], dtype=np.float64) / global_max
        new_d["norm_factor"] = global_max
        normalized.append(new_d)
        logger.info(
            "T = %.2f K: normalised peak = %.4f (global max = %.4e)",
            float(d["temperature"]),
            float(np.max(new_d["intensity"])),
            global_max,
        )
    return normalized


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


def process_temperature_series(
    file_triplets: List[Tuple[Union[str, Path], Union[str, Path], Union[str, Path]]],
    hc_const: float = 1239.842,
    calibration_mode: str = "divide",
    global_normalize: bool = True,
) -> List[Dict[str, Union[float, np.ndarray]]]:
    """ Batch-processes a series of temperature-dependent file triplets and returns them sorted by temperature.
    ------
    OUTPUT - List[Dict[str, Union[float, np.ndarray]]]: Sorted list of processed dataset dictionaries ordered by ascending temperature.
        Each dict has 'temperature', 'energy_ev', 'intensity' (divided by the single global maximum
        if global_normalize=True, otherwise raw calibrated units) and, when normalised, 'norm_factor'.
    ------
    file_triplets - List[Tuple[Union[str, Path], Union[str, Path], Union[str, Path]]]: List of (sample, background, calibration) path tuples.
    hc_const - float: Conversion constant in eV*nm (default: 1239.842).
    calibration_mode - str: Calibration mode, either 'divide' or 'multiply'.
    global_normalize - bool: If True (default), divide all spectra by one common factor (the global maximum).
    ------
    ValueError: Raised if any file triplet processing fails.
    """
    logger.info("Starting batch processing of %d temperature datasets.", len(file_triplets))
    processed_series: List[Dict[str, Union[float, np.ndarray]]] = []

    for sample_p, bg_p, cal_p in file_triplets:
        dataset = process_single_dataset(
            sample_p, bg_p, cal_p, hc_const=hc_const, calibration_mode=calibration_mode
        )
        processed_series.append(dataset)

    processed_series.sort(key=lambda d: float(d["temperature"]))

    if global_normalize:
        processed_series = normalize_by_global_max(processed_series)

    temperatures = [d["temperature"] for d in processed_series]
    logger.info("Batch processing complete. Processed temperatures (K): %s", temperatures)
    return processed_series
