"""
test_data_loader.py
-------------------
Unit tests for data_loader.py module.
"""

from pathlib import Path
import numpy as np
import pytest

from temp_dependent_pl_proccessing.data_loader import (
    convert_wavelength_to_energy,
    extract_temperature,
    interpolate_calibration,
    process_single_dataset,
    read_spectrum_file,
)


def test_extract_temperature():
    """ Tests extraction of temperature values from various filename conventions.
    """
    assert extract_temperature("sample_100K.txt") == 100.0
    assert extract_temperature("/path/to/data_295.5K_measurement.txt") == 295.5
    assert extract_temperature("PL_4K_series.dat") == 4.0

    with pytest.raises(ValueError):
        extract_temperature("invalid_filename_no_temp.txt")


def test_read_spectrum_file(tmp_path: Path):
    """ Verifies parsing of text files containing headers and standard numerical rows.
    """
    file_path = tmp_path / "mock_spectrum.txt"
    file_path.write_text(
        "# Header line 1\n"
        "Wavelength (nm), Intensity (counts)\n"
        "500.0, 100.0\n"
        "501.0, 105.0\n"
        "502.0, 110.0\n"
    )

    wv, intensity = read_spectrum_file(file_path)
    np.testing.assert_allclose(wv, [500.0, 501.0, 502.0])
    np.testing.assert_allclose(intensity, [100.0, 105.0, 110.0])


def test_interpolate_calibration():
    """ Tests cubic spline interpolation onto non-matching wavelength grids.
    """
    cal_wv = np.array([400.0, 500.0, 600.0, 700.0])
    cal_int = np.array([1.0, 2.0, 3.0, 4.0])
    target_wv = np.array([450.0, 550.0])

    interpolated = interpolate_calibration(target_wv, cal_wv, cal_int)
    np.testing.assert_allclose(interpolated, [1.5, 2.5], rtol=1e-3)


def test_convert_wavelength_to_energy():
    """ Verifies Jacobian transformation maths: E = hc / λ and I(E) = I(λ) * λ^2 / hc.
    """
    hc = 1239.842
    wv = np.array([620.0, 310.0])  # ~2.0 eV and ~4.0 eV
    intensity = np.array([10.0, 20.0])

    energy, energy_int = convert_wavelength_to_energy(wv, intensity, hc_const=hc)

    # Ascending energy sorting verification
    assert energy[0] < energy[1]
    np.testing.assert_allclose(energy, [hc / 620.0, hc / 310.0])

    # Jacobian scaling factor verification: I(E) = I(λ) * λ² / hc
    expected_int_310 = 20.0 * (310.0**2) / hc
    np.testing.assert_allclose(energy_int[1], expected_int_310)


def test_process_single_dataset(tmp_path: Path):
    """ End-to-end integration test verifying full dataset processing pipeline.
    """
    sample_p = tmp_path / "sample_100K.txt"
    bg_p = tmp_path / "bg.txt"
    cal_p = tmp_path / "cal.txt"

    sample_p.write_text("500.0, 1000.0\n510.0, 1200.0\n520.0, 800.0\n530.0, 500.0\n")
    bg_p.write_text("500.0, 100.0\n510.0, 100.0\n520.0, 100.0\n530.0, 100.0\n")
    cal_p.write_text("500.0, 1.0\n510.0, 1.0\n520.0, 1.0\n530.0, 1.0\n")

    res = process_single_dataset(sample_p, bg_p, cal_p)

    assert res["temperature"] == 100.0
    assert len(res["energy_ev"]) == 4
    assert res["intensity"].max() == 1.0  # Verify peak normalization
    assert np.all(res["energy_ev"][1:] > res["energy_ev"][:-1])  # Ascending energy