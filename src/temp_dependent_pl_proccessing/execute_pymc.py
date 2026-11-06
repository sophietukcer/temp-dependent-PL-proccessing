import os
msys_bin = r"C:\msys64\ucrt64\bin"
if msys_bin not in os.environ["PATH"]:
    os.environ["PATH"] = msys_bin + os.pathsep + os.environ["PATH"]

# 2. Configure PyTensor flags (Compiler path + Clean cache location)
os.environ["PYTENSOR_FLAGS"] = (
    rf"cxx={msys_bin}\g++.exe,"
    r"compiledir=C:\pytensor_cache"  # Avoids user directory spaces/permissions issues
)

from temp_dependent_pl_proccessing.data_loader import process_temperature_series, build_shared_triplets
from temp_dependent_pl_proccessing.logger_config import setup_logger
from temp_dependent_pl_proccessing.NUTS_sampler import build_pl_pymc_model
from pathlib import Path
from typing import List, Union

import numpy as np

import pymc as pm


def run_pymc(samples_dir: Union[str, Path],
    background_path: Union[str, Path],
    calibration_path: Union[str, Path],
    absorption_peaks: List[float],
    calibration_mode: str = 'multiply'
):

    
    file_triplets = build_shared_triplets(
        samples_dir=samples_dir,
        background_path=background_path,
        calibration_path=calibration_path,
    )
    processed_series = process_temperature_series(
            file_triplets, calibration_mode=calibration_mode
    )

    temperatures = np.array([d["temperature"] for d in processed_series])

    model = build_pl_pymc_model(
    processed_series=processed_series,
    num_peaks=len(absorption_peaks),
    absorption_centers=absorption_peaks,  # From 0K empirical estimate        # From 100K empirical estimate
    )

    with model:
        print("\n--- INDIVIDUAL NODE LOG-PROBABILITIES ---")
        for name, logp in model.point_logps().items():
            print(f"{name:25s}: {logp}")
        idata = pm.sample(
        draws=100,
        tune=100,
        chains=2,
        target_accept=0.90,
        random_seed=42,
        return_inferencedata=True,
        )

        idata.extend(pm.sample_posterior_predictive(idata))

    return idata