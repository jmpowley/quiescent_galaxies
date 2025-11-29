import sys

import numpy as np
import matplotlib.pyplot as plt

from astropy import constants

# Import functions
sys.path.append("/Users/Jonah/PhD/Research/quiescent_galaxies/code/scripts/zf-uds-7329")
from loading import load_prism_data, load_grating_data, load_dispersion_data
from conversions import convert_wave_um_to_A

def return_c3k_spectral_resolution(wave_A):

    c3k_spec_sigma = []

    for w in wave_A:

        if (w >= 100) & (w < 1000):
            spec_sigma = 250
        elif (w >= 1000) & (w < 2750):
            spec_sigma = 500
        elif (w >= 2750) & (w < 9100):
            spec_sigma = 3000
        elif (w >= 9100) & (w < 24000):
            spec_sigma = 500
        elif (w >= 24000) & (w < 100000):
            spec_sigma = 50
        else:
            raise Exception("Wavelength outside of C3K range")
        
        c3k_spec_sigma.append(spec_sigma)

    return np.asarray(c3k_spec_sigma)

def convert_spectral_resolution_to_sigma(spec_res):

    # Convert dispersion from dimensionless to km/s
    c_kms = constants.c.to("km/s").value
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    sigma_kms = c_kms / spec_res / fwhm_factor

    return sigma_kms

def return_c3k_resolution_wave_limit(wave, spec_sigma_kms):

    c3k_spec_res = return_c3k_spectral_resolution(wave_A=wave)
    c3k_sigma_kms = convert_spectral_resolution_to_sigma(c3k_spec_res)

    limit = np.where(c3k_sigma_kms > spec_sigma_kms)[0][0]

    wave_limit = wave[limit]

    return wave_limit

def find_crossings(x1, y1, x2, y2):
    
    # Interpolate y2 onto x1
    y2i = np.interp(x1, x2, y2)
    d = y1 - y2i  # difference

    # Indices where sign changes
    idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]

    # Compute exact crossing by linear interpolation
    xs = x1[idx] + (x1[idx+1] - x1[idx]) * (d[idx] / (d[idx] - d[idx+1]))
    ys = y1[idx] + (y1[idx+1] - y1[idx]) * (d[idx] / (d[idx] - d[idx+1]))

    return xs, ys

# Load observations
obs_kwargs = {

    "g140m_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/g140m_f100lp",
        "data_name" : "000007_g140m_f100lp_v5.1_1D.fits",
        "data_ext" : "DATA",
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g140m_disp.fits",
        "in_wave_units" : "si",
        "out_wave_units" : "A",
        "in_flux_units" : "ujy",
        "out_flux_units" : "ujy",
        "rescale_factor" : None,
        "snr_limit" : None,
        "prefix" : "g140m",
    },

    "g235m_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/g235m_f170lp",
        "data_name" : "000007_g235m_f170lp_v5.1_1D.fits",
        "data_ext" : "DATA",
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g235m_disp.fits",
        "in_wave_units" : "si",
        "out_wave_units" : "A",
        "in_flux_units" : "ujy",
        "out_flux_units" : "ujy",
        "rescale_factor" : None,
        "snr_limit" : None,
        "prefix" : "g235m",
    },

    "g395m_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/g395m_f290lp",
            "data_name" : "000007_g395m_f290lp_v5.1_1D.fits",
            "data_ext" : "DATA",
            "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
            "disp_name" : "jwst_nirspec_g395m_disp.fits",
            "in_wave_units" : "si",
            "out_wave_units" : "A",
            "in_flux_units" : "ujy",
            "out_flux_units" : "ujy",
            "rescale_factor" : None,
            "snr_limit" : None,
            "prefix" : "g395m",
        },
}
g140m_wave, g140m_flux, g140m_err = load_grating_data(**obs_kwargs["g140m_kwargs"])
g140m_sigma = load_dispersion_data(**obs_kwargs["g140m_kwargs"])
g235m_wave, g235m_flux, g235m_err = load_grating_data(**obs_kwargs["g235m_kwargs"])
g235m_sigma = load_dispersion_data(**obs_kwargs["g235m_kwargs"])
g395m_wave, g395m_flux, g395m_err = load_grating_data(**obs_kwargs["g395m_kwargs"])
g395m_sigma = load_dispersion_data(**obs_kwargs["g395m_kwargs"])

# Calculate C3K resolution from rest-frame wavelengths
zred = 9.3133
c3k_wave = np.linspace(g140m_wave.min(), g395m_wave.max(), 1000)
c3k_rest_wave = c3k_wave / (1 + zred)

c3k_spec_res = return_c3k_spectral_resolution(c3k_rest_wave)
c3k_sigma = convert_spectral_resolution_to_sigma(c3k_spec_res)

g140m_wave_limit = find_crossings(g140m_wave, g140m_sigma, c3k_wave, c3k_sigma)
g235m_wave_limit = find_crossings(g235m_wave, g235m_sigma, c3k_wave, c3k_sigma)

print("Upper limit on G140M wavelength:", g140m_wave_limit)
print("Upper limit on G235M wavelength:", g235m_wave_limit)

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot instrument dispersion
# ax.plot(prism_wave, prism_sigma, label="Prism")
ax.plot(g140m_wave, g140m_sigma, label="G140M")
ax.plot(g235m_wave, g235m_sigma, label="G235M")
ax.plot(g395m_wave, g395m_sigma, label="G395M")
ax.plot(c3k_wave, c3k_sigma, label="C3K (obs. frame)")

# Prettify
ax.set_ylim(1e1, None)
ax.set_ylabel(r"Instrument Dispersion [${\rm km s}^-1$]", size=16)
ax.set_xlabel(r"Wavelength [Å]", size=16)
ax.set_yscale("log")
# -- add rest-frame axis
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
top_ticks_rest = convert_wave_um_to_A(np.arange(0.2, 1.4, 0.1))
top_ticks_obs = top_ticks_rest * (1 + zred)
ax_top.set_xticks(top_ticks_obs)
ax_top.set_xticklabels([f"{t.astype(int)}" for t in top_ticks_rest])

ax.legend()

plt.tight_layout()

plt.show()