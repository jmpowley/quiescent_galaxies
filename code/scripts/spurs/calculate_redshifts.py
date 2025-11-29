import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import erf
from scipy.optimize import curve_fit

from astropy.table import Table
from astropy.convolution import Gaussian1DKernel, convolve_fft

from lmfit.models import Model

# Import loading function
sys.path.append("/Users/Jonah/PhD/Research/quiescent_galaxies/code/scripts/zf-uds-7329")
from loading import load_prism_data, load_grating_data, load_dispersion_data
from conversions import convert_wave_um_to_A, convert_wave_A_to_um

def gaussian(x, flux, mu, sigma):
    return (flux / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def interpolate_spectral_resolution_at_wave(spec_res, spec_res_wave, wave):
    return np.interp(wave, spec_res_wave, spec_res)

def gaussian_shifted(x, z, flux, sigma, wave_rest):

    wave_obs = wave_rest * (1+z)
    model = gaussian(x, flux, wave_obs, sigma)

    return model

def gaussian_convolved_kernel(x, z, flux, sigma, wave_rest):

    # Define constants
    delta_lambda = x[1] - x[0]
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))

    # Calculate LSF to convolve with data
    wave_obs = wave_rest * (1+z)
    spec_res = interpolate_spectral_resolution_at_wave(spec_res=g395m_spec_res, spec_res_wave=g395m_spec_res_wave, wave=wave_obs)
    sigma_lsf_lambda = wave_obs / (spec_res * fwhm_factor)
    sigma_lsf_pix = sigma_lsf_lambda / delta_lambda

    kernel = Gaussian1DKernel(sigma_lsf_pix)
    model = gaussian_shifted(x, z, flux, sigma, wave_rest)
    model_conv = convolve_fft(model, kernel, normalize_kernel=True, nan_treatment="interpolate")

    return model_conv

def gaussian_convolved_quadrature(x, z, flux, sigma_obs, sigma_lsf, wave_rest):

    sigma_conv = np.sqrt(sigma_obs**2 + sigma_lsf**2)
    model_conv = gaussian_shifted(x, z, flux, sigma_conv, wave_rest)

    return model_conv

def constant(x, c):
    return c

def linear(x, m, c):
    return m * x + c

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def negative_exponential(x, a, b):
    return a * np.exp(-b * x)

def power_law(x, a, b, x0):
    return a * (x/x0)**b

def guess_flux_from_zguess(wave, spec, line_rest, z_guess, width_guess):

    line_obs = line_rest * (1 + z_guess)

    print(line_rest, line_obs)

    inside = np.abs(wave - line_obs) <= width_guess

    print("width:", convert_wave_um_to_A(wave[inside][-1] - wave[inside][0]))

    flux_guess = np.trapz(spec[inside], x=wave[inside])

    return flux_guess

def convert_flux(flux_in):
    flux_out = flux_in * 1e-16
    return flux_out

# Load data
obs_kwargs = {

    "g140m_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/g140m_f100lp",
        "data_name" : "000007_g140m_f100lp_v5.1_1D.fits",
        "data_ext" : "DATA",
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g140m_disp.fits",
        "return_spec_res" : True,
        "return_wave" : True,
        "in_wave_units" : "si",
        "out_wave_units" : "um",
        "in_flux_units" : "si",
        "out_flux_units" : "cgs",
        "cgs_factor" : 1e-20,
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
        "return_spec_res" : True,
        "return_wave" : True,
        "in_wave_units" : "si",
        "out_wave_units" : "um",
        "in_flux_units" : "si",
        "out_flux_units" : "cgs",
        "cgs_factor" : 1e-20,
        "rescale_factor" : None,
        "snr_limit" : None,
        "prefix" : "g295m",
    },

    "g395m_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/g395m_f290lp",
        "data_name" : "000007_g395m_f290lp_v5.1_1D.fits",
        "data_ext" : "DATA",
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g395m_disp.fits",
        "return_spec_res" : True,
        "return_wave" : True,
        "in_wave_units" : "si",
        "out_wave_units" : "um",
        "in_flux_units" : "si",
        "out_flux_units" : "cgs",
        "cgs_factor" : 1e-20,
        "rescale_factor" : None,
        "snr_limit" : None,
        "prefix" : "g395m",
    },
}
g140m_wave, g140m_spec, g140m_err = load_grating_data(**obs_kwargs["g140m_kwargs"])
g235m_wave, g235m_spec, g235m_err = load_grating_data(**obs_kwargs["g235m_kwargs"])
g395m_wave, g395m_spec, g395m_err = load_grating_data(**obs_kwargs["g395m_kwargs"])

# Load dispersion data
g140m_spec_res, g140m_spec_res_wave = load_dispersion_data(**obs_kwargs["g140m_kwargs"])
g235m_spec_res, g235m_spec_res_wave = load_dispersion_data(**obs_kwargs["g235m_kwargs"])
g395m_spec_res, g395m_spec_res_wave = load_dispersion_data(**obs_kwargs["g395m_kwargs"])

# Lines
lines_A = {
    "o2" : (3728.815 + 3726.032) / 2.0,
    "ne3_3869" : 3868.760,
    "ne3_3967" : 3967.470,
    "he" : 3970.079,
    "hg" : 4340.471,
    "o3_4363" : 4363.210,
    "o3_5007" : 5006.843,
    "o3_4959" : 4958.911,
    "hb" : 4861.333,
}
lines_um = {key: convert_wave_A_to_um(val) for key, val in lines_A.items()}

pretty_lines = {
    "o2" : "[OII]",
    "ne3_3869" : "",
    "ne3_3967" : 3967.470,
    "he" : 3970.079,
    "hg" : 4340.471,
    "o3_5007" : 5006.843,
    "o3_4959" : 4958.911,
    "hb" : 4861.333,
}

# Identify bad points: NaN or inf in flux
good = np.isfinite(g395m_spec) & np.isfinite(g395m_err)
g395m_wave_clean = g395m_wave[good]
g395m_spec_clean = g395m_spec[good]
g395m_err_clean = g395m_err[good]

# Crop data
g395m_mask = (g395m_wave > 3.8) & (g395m_wave < 5.2)
g395m_wave_crop = g395m_wave[g395m_mask]
g395m_spec_crop = g395m_spec[g395m_mask]
g395m_err_crop = g395m_err[g395m_mask]

# Build model
o2 = Model(gaussian_convolved_kernel, prefix="o2_")
ne3_3869 = Model(gaussian_convolved_kernel, prefix="ne3_3869_")
ne3_3967 = Model(gaussian_convolved_kernel, prefix="ne3_3967_")
heps = Model(gaussian_convolved_kernel, prefix="he_")
hgamma = Model(gaussian_convolved_kernel, prefix="hg_")
o3_4363 = Model(gaussian_convolved_kernel, prefix="o3_4363_")
hbeta = Model(gaussian_convolved_kernel, prefix="hb_")
o3_4959 = Model(gaussian_convolved_kernel, prefix="o3_4959_")
o3_5007 = Model(gaussian_convolved_kernel, prefix="o3_5007_")
# cont = Model(constant, prefix="cont_")
# cont = Model(linear, prefix="cont_")
# cont = Model(quadratic, prefix="cont_")
# cont = Model(negative_exponential, prefix="cont_")
cont = Model(power_law, prefix="cont_")

model = o2 + ne3_3869 + ne3_3967 + heps + hgamma + o3_4363 + hbeta + o3_4959 + o3_5007 + cont

# Define constants
ne3_ratio = 0.302

# Make guesses
z_guess = 9.31
width_guess = convert_wave_A_to_um(70)
flux_guesses = {}
for key, val in lines_A.items():
    flux_guess = guess_flux_from_zguess(g395m_wave_clean, g395m_spec_clean, lines_um[key], z_guess=z_guess, width_guess=width_guess)
    flux_guesses[key] = flux_guess
    print(f"line: {key} flux_guess: {flux_guess}")

# Define parameters
params = model.make_params()
free_names = [name for name, par in params.items() if par.vary]
# -- rest wavelengths (fixed)
for key, val in lines_A.items():
    params[f"{key}_wave_rest"].set(value=lines_um[key], vary=False)
# -- estimate flux
# params["o2_flux"].set(value=0.01, min=0.0)
# params["ne3_3869_flux"].set(value=0.02)
# # params["ne3_3967_flux"].set(value=0.02)  # left flux free
# params["ne3_3967_flux"].expr = f"ne3_3869_flux * {ne3_ratio}"  # fix flux to [Ne III] 3869
# params["he_flux"].set(value=0.02)
# params["hg_flux"].set(value=0.02)
# params["o3_4363_flux"].set(value=0.02)
# params["hb_flux"].set(value=0.01, min=0.0)
# params["o3_4959_flux"].set(value=0.02, min=0.0)
# params["o3_5007_flux"].set(value=0.05, min=0.0)
for key, val in lines_A.items():
    if key != "ne3_3967":
        params[f"{key}_flux"].set(value=flux_guesses[key], vary=True)
    elif key == "ne3_3967":
        params["ne3_3967_flux"].expr = f"ne3_3869_flux * {ne3_ratio}"
# fixed_ratio = 0.332
# params["o3_4959_flux"].expr = f"o3_5007_flux * {fixed_ratio}"
# -- tie redshift and sigma
for key, val in lines_A.items():
    if key != "o3_5007":
        params[f"{key}_z"].expr = "o3_5007_z"
        params[f"{key}_sigma"].expr = "o3_5007_sigma"
    elif key == "o3_5007":
        params["o3_5007_z"].set(value=9.3, min=8.0, max=11.0)
        params["o3_5007_sigma"].set(value=convert_wave_A_to_um(20), min=1e-6)
# -- continuum
params["cont_a"].set(value=np.median(g395m_spec_clean[(g395m_wave_clean>3.9)&(g395m_wave_clean<4.0)]), min=0, vary=True)
params["cont_b"].set(value=-2.0, min=-3, max=0.0)
params["cont_x0"].expr = f"{convert_wave_A_to_um(lines_A['o3_5007'])} * (1 + o3_5007_z)"
params["cont_x0"].vary = False

result = model.fit(g395m_spec_clean, params, x=g395m_wave_clean, weights=(1.0/g395m_err_clean), method="nelder")

print(result.fit_report())

# Set up MCMC fit
params_best = result.best_values
params_emcee = params.copy()
for key, param_emcee in params_emcee.items():
    param_emcee.set(value=params_best[key])

# Setup walkers
ndim = len(free_names)
nwalkers = max(50, 2 * ndim)
emcee_kws = dict(steps=20000, burn=500, thin=20, is_weighted=False, progress=False, nwalkers=nwalkers)

# Fit model with MCMC
# result = model.fit(g395m_spec_clean, params_emcee, x=g395m_wave_clean, weights=(1.0/g395m_err_clean), method="emcee", fit_kws=emcee_kws)
# print(result.fit_report())

# Extract best fit params
params = result.params
params_best = result.best_values
zest = params["o3_5007_z"].value
zest_err = params["o3_5007_z"].stderr
print("z_best:", zest, rf"$\pm$ {zest_err}")

# Make models based on fit
g395m_wave_grid = np.linspace(g395m_wave_clean.min(), g395m_wave_clean.max(), 10000)
model_grid = result.eval(x=g395m_wave_grid)
model_grid_unc = result.eval_uncertainty(x=g395m_wave_grid, sigma=1)
model_fit = result.eval(x=g395m_wave_clean)
residual = g395m_spec_clean - model_fit
residual_norm = residual / g395m_err_clean

# Calculate fluxes
for stri in lines_A.keys():
    print(f"Line: {stri}")

    param_flux = params[f"{stri}_flux"]
    flux = convert_flux(param_flux.value)
    flux_err = convert_flux(param_flux.stderr)
    snr = flux / flux_err

    print(f"Flux: {flux:.2g}, Error: {flux_err:.2g}, SNR: {snr:.2f}")

# Create figure
fig = plt.figure(figsize=(12, 7))
gs = GridSpec(2, 1, height_ratios=[5, 2])
ax = fig.add_subplot(gs[0])
ax_res = fig.add_subplot(gs[1], sharex=ax)

# Plot data
# -- spectra
ax.step(g395m_wave_clean, g395m_spec_clean, color="k", where="mid", label="G395M")
ax.fill_between(g395m_wave_clean, g395m_spec_clean-g395m_err_clean, g395m_spec_clean+g395m_err_clean, 
                color="k", alpha=0.3, step="mid")
# -- best fit model
ax.plot(g395m_wave_grid, model_grid, color="blue", label=r"Fit $(z_{\rm best} = $" + rf"${zest:.4f})$")
ax.fill_between(g395m_wave_grid, model_grid-model_grid_unc, model_grid+model_grid_unc, 
                color="blue", alpha=0.3)
# -- residual
ax_res.scatter(g395m_wave_clean, residual_norm, color="blue", marker=".")
# -- lines
ymax = ax.get_ylim()[1] * 0.9
print(ymax)
for stri, wv in lines_A.items():
    # dlam = offsets.get(name, 0)
    wave_obs = convert_wave_A_to_um(wv*(1+zest))
    ax.axvline(wave_obs, ls="--", color="gray", alpha=0.5)
    ax.text(wave_obs+0.01, ymax, stri, ha='left', va='top', rotation=90, size=12)


    # ax.axvspan(convert_wave_A_to_um(wv)*(1+z_guess)-width_guess, convert_wave_A_to_um(wv)*(1+z_guess)+width_guess)

# Prettify
# ax.set_xlabel(r"$\lambda_{\rm obs}~[\mu{\rm m}]$", size=16)
ax.set_ylabel(r"$f_\lambda~[10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}]$", size=16)
ax_res.set_ylabel(r"$\chi~[{\rm norm.}]$", size=16)
ax_res.set_xlabel(r"$\lambda_{\rm obs}~[\mu{\rm m}]$", size=16)
ax.get_xaxis().set_visible(False)
ax.legend()
plt.tight_layout()

plt.show()