import os
msys_bin = r"C:\msys64\ucrt64\bin"
if msys_bin not in os.environ["PATH"]:
    os.environ["PATH"] = msys_bin + os.pathsep + os.environ["PATH"]

# 2. Configure PyTensor flags (Compiler path + Clean cache location)
os.environ["PYTENSOR_FLAGS"] = (
    rf"cxx={msys_bin}\g++.exe,"
    r"compiledir=C:\pytensor_cache"  # Avoids user directory spaces/permissions issues
)


from typing import Dict, List
import numpy as np
import arviz as az
import pymc as pm
import pytensor.tensor as pt

KB_EV: float = 8.617333262145e-5  # Boltzmann constant in eV/K


def build_pl_pymc_model(
    processed_series: List[Dict],
    num_peaks: int,
    absorption_centers: List[float],
) -> pm.Model:
    
    """ Builds a PyMC Hamiltonian Bayesian model for temperature-dependent PL fitting.
    ------
    OUTPUT - pm.Model: Model object with the bayesian likelyhood of the observed data corresponding to the model.
    ------
    processed_series - list[Dict]: Dictionary containing processed PL spectra at each temperature.
    mnum_peaks - int: number of gaussian peaks in spectra
    absorption_centres - List[float]: List of emission centres (eV) as estimated by absorption data.
    ------
    """

    num_spectra = len(processed_series)
    peak_names = [f"Peak {i+1}" for i in range(num_peaks)]

    with pm.Model(coords={"peak": peak_names}) as model:
        # 1. PEAK CENTER PRIORS & ORDERING
        # Anchored Gaussian prior for 0 K peak energy centers
        e0_0 = pm.Normal(
            "e0_0",
            mu=np.array(absorption_centers),
            sigma=0.050,
            dims="peak",
            initval=[1.75, 1.80, 1.93, 2.02],
        )

        # Enforce Peak 1 < Peak 2 < ... ordering via soft differentiable potential
        if num_peaks > 1:
            energy_diffs = e0_0[1:] - e0_0[:-1]
            pm.Potential(
                "ordering_penalty",
                pt.switch(energy_diffs < 0, -1e5 * (energy_diffs**2), 0.0),
            )

        # 2. PHYSICAL PARAMETER PRIORS (Per-Peak)
        # 0 K Inhomogeneous broadening (eV)
        gamma_0 = pm.Uniform("gamma_0", lower=0.001, upper=0.150, initval=[0.05, 0.03, 0.03, 0.03], dims="peak")

        # Acoustic phonon coupling (sampled in log-space)
        ln_gamma_ac = pm.Normal(
            "ln_gamma_ac", mu=-9, sigma=2, initval=[-9.1]*num_peaks, dims="peak"
        )
        gamma_ac = pm.Deterministic("gamma_ac", pt.exp(ln_gamma_ac))

        # Optical phonon coupling & energy (eV)
        gamma_lo = pm.TruncatedNormal("gamma_lo", mu=0.030, sigma=0.100, lower=1e-6, initval=[0.030, 0.030, 0.030, 0.030], dims="peak")
        e_lo = pm.TruncatedNormal("e_lo", mu=0.030, sigma=0.100, lower=1e-6, initval=[0.030, 0.030, 0.030, 0.030], dims="peak")

        # 0 K Integrated Intensity
        i_0 = pm.Uniform("i_0", lower=1e-4, upper=5.0, dims="peak")

        # Quenching factor (sampled in log-space)
        ln_a_factor = pm.Uniform(
            "ln_a_factor", lower=np.log(1e-3), upper=np.log(1e4), dims="peak"
        )
        a_factor = pm.Deterministic("a_factor", pt.exp(ln_a_factor))

        # Activation energy (eV)
        e_a = pm.TruncatedNormal("e_a", mu=0.030, sigma=0.020, lower=0.0, initval=[0.030, 0.030, 0.030, 0.030], dims="peak")

        s0 = 1e-5 ## crossover threshold from log-like to linear behaviour
        mu_alpha = 2e-4
        sigma_alpha = 2e-2

        # Bandgap shift coefficient (can be <0)
        z_alpha = pm.Normal("z_alpha", mu=float(np.arcsinh(mu_alpha / s0)), sigma=float(sigma_alpha / np.sqrt(sigma_alpha**2 + s0**2)), initval=[2e-4, -2e-4, -2e-4, -2e-4], dims="peak")
 
        alpha = pm.Deterministic("alpha", s0 * pt.sinh(z_alpha))

        # Residual spectral noise standard deviation
        sigma = pm.HalfNormal("sigma", sigma=0.010, shape = num_spectra)

        print("PyMC Initial Point Dict:")
        print(model.initial_point())

        # 3. FORWARD PHYSICAL MODEL & LIKELIHOOD EVALUATION
        for idx, entry in enumerate(processed_series):
            t_val = float(entry["temperature"])
            energy_grid = np.array(entry["energy_ev"], dtype=np.float64)       # Shape: (N_energy,)
            data_intensity = np.array(entry["intensity"], dtype=np.float64) # Shape: (N_energy,)

            # Temperature-dependent physical parameter expressions (Shape: num_peaks,)
            e_t = e0_0 - alpha * t_val
            exp_term = pt.exp(e_lo / (KB_EV * t_val)) - 1.0
            gamma_t = gamma_0 + gamma_ac * t_val + (gamma_lo / pt.maximum(exp_term, 1e-8))
            i_t = i_0 / (1.0 + a_factor * pt.exp(-e_a / (KB_EV * t_val)))

            # Soft differentiable penalty for maximum physical FWHM (250 meV)
            pm.Potential(
                f"fwhm_bound_T{int(t_val)}",
                pt.switch(gamma_t > 0.250, -1e4 * ((gamma_t - 0.250) ** 2), 0.0),
            )

            # Broadcast spectral components over energy grid: (N_energy, 1) vs (1, num_peaks)
            e_grid_col = energy_grid[:, None]
            peak_profiles = i_t * pt.exp(
                -4.0 * np.log(2.0) * (((e_grid_col - e_t) / gamma_t) ** 2)
            )
            total_model_spectrum = pt.sum(peak_profiles, axis=1)

            # Spectrum Gaussian likelihood
            pm.Normal(
                f"obs_T_{int(t_val)}K",
                mu=total_model_spectrum,
                sigma=sigma[idx],
                observed=data_intensity,
            )

    return model