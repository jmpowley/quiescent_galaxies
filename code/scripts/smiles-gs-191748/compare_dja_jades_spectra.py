import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

loading_path = "/Users/Jonah/PhD/Research/quiescent_galaxies/code/scripts/zf-uds-7329"
sys.path.append(loading_path)

from loading import load_grating_data

from conversions import convert_flux_si_to_jy, convert_wave_m_to_um

# 
dja_obs_kwargs = {

    "grat1_params" : {
             "grating_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/smiles-gs-191748/spectra/dja_reduction",
             "name" : "191748",
             "grating" : "g140m",
             "filter" : "f100lp",
             "version" : None,
             "nod" : None,
             "data_ext" : "DATA",
             "mask_ext" : "VALID",
             "in_wave_units" : "um",
             "out_wave_units" : "um",
             "in_flux_units" : "ujy",
             "out_flux_units" : "cgs",
             "rescale_factor" : None,
             "snr_limit" : None,
             "return_none" : False,
        },

    "grat2_params" : {
             "grating_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/smiles-gs-191748/spectra/dja_reduction",
             "name" : "191748",
             "grating" : "g235m",
             "filter" : "f170lp",
             "version" : None,
             "nod" : None,
             "data_ext" : "DATA",
             "mask_ext" : "VALID",
             "in_wave_units" : "um",
             "out_wave_units" : "um",
             "in_flux_units" : "ujy",
             "out_flux_units" : "cgs",
             "rescale_factor" : None,
             "snr_limit" : None,
             "return_none" : False,
        },
}

jades_obs_kwargs = {

    "grat1_params" : {
             "grating_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/smiles-gs-191748/spectra/jades_reduction",
             "name" : "191748",
             "grating" : "g140m",
             "filter" : "f100lp",
             "version" : "v5.1",
             "nod" : None,
            #  "nod" : "nod2",
             "data_ext" : "DIRTY_DATA",
             "mask_ext" : None,
             "in_wave_units" : "si",
             "out_wave_units" : "um",
             "in_flux_units" : "si",
             "out_flux_units" : "cgs",
             "rescale_factor" : None,
             "snr_limit" : None,
             "return_none" : False,
        },

    "grat2_params" : {
             "grating_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/smiles-gs-191748/spectra/jades_reduction",
             "name" : "191748",
             "grating" : "g235m",
             "filter" : "f170lp",
             "version" : "v5.1",
             "nod" : None,
            #  "nod" : "nod2",
             "data_ext" : "DIRTY_DATA",
             "mask_ext" : None,
             "in_wave_units" : "si",
             "out_wave_units" : "um",
             "in_flux_units" : "si",
             "out_flux_units" : "cgs",
             "rescale_factor" : None,
             "snr_limit" : None,
             "return_none" : False,
        },
}

# Load DJA data
dja_waves = []
dja_fluxes = []
dja_errs = []
dja_masks = []

for name, params in dja_obs_kwargs.items():
    wave, flux, err, mask = load_grating_data(**params)
    dja_waves.append(wave)
    dja_fluxes.append(flux)
    dja_errs.append(err)
    dja_masks.append(mask)

    print(wave.shape, flux.shape, err.shape, mask.shape)

# Load JADES data
jds_waves = []
jds_fluxes = []
jds_errs = []
jds_masks = []

for name, params in jades_obs_kwargs.items():
    wave, flux, err, mask = load_grating_data(**params)

    print(wave)

    jds_waves.append(wave)
    jds_fluxes.append(flux)
    jds_errs.append(err)
    jds_masks.append(mask)

# Load JADES data
# jades_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/smiles-gs-191748/spectra/jades_reduction"
# spec_name = "191748_g140m_f100lp_v5.1_1D.fits"
# spec_path = os.path.join(jades_dir, spec_name)
# hdul = fits.open(spec_path)
# hdul.info()
# pri_hdu = hdul['PRIMARY']
# data_hdu = hdul['DATA']
# dirty_hdu = hdul['DIRTY_DATA']
# wave_hdu = hdul['WAVELENGTH']
# print(pri_hdu.header)
# print(data_hdu.header)
# print(dirty_hdu.header)

lines_A = {  # units in Angstroms
    'MgII' : 2800.000,  # individual lines are 2795.528, 2802.705
    'CaII' : 3950.000,  # individual lines are 3933.663, 3968.469
    'Hdelta' : 4101.742,
    'Hgamma' : 4340.471,
    'Hbeta' : 4861.333,
    'OIII' : 5006.843,
    'Mgb' : 5200.000,  # individual lines are 2795.528, 2802.705
    'NaD' : 5892.500,  # individual lines are 5889.950, 5895.924
    'NII' : 6565.000,  # indiviudal lines are 6548.050, 6583.460
    'Halpha' : 6562.819,
    'SII' : 6723.00,  # individual lines are 6716.440, 6730.810
}

# Renormalise spectrum

def renormalise_flux(wave, flux):

    renorm_mask = (wave > 1.7) & (wave < 1.9)

    wave_masked = wave[renorm_mask]

    flux_masked = flux[renorm_mask]

    flux_masked_norm = flux_masked / np.nansum(flux_masked)

    return wave_masked, flux_masked_norm

# Plot DJA and JADES data on different panels
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 8.5))

dja_color = "deepskyblue"
jades_color = "red"
# -- DJA data
for i, (wave, flux, err, mask) in enumerate(zip(dja_waves, dja_fluxes, dja_errs, dja_masks)):
    # ax[i].step(wave[mask], flux[mask], color=dja_color, label="DJA")

    wave, flux = renormalise_flux(wave, flux)

    ax[i].step(wave, flux, color=dja_color, label="DJA")

    # ax[i].fill_between(wave[mask], flux[mask]-err[mask], flux[mask]+err[mask], color=dja_color, step="mid")
# -- JADES data
for i, (wave, flux, err, mask) in enumerate(zip(jds_waves, jds_fluxes, jds_errs, jds_masks)):
    # ax[i].step(wave, flux, color=jades_color, label="JADES")

    wave, flux = renormalise_flux(wave, flux)

    ax[i].step(wave, flux, color=jades_color, label="JADES")

    # ax[i].fill_between(wave, flux-err, flux+err, color=jades_color, step="mid")
    # -- prettify in loop
    ax[i].legend()
    ax[i].set_xlabel(r'Wavelength / $\mu$m', size=16)
    ax[i].set_ylabel(r'$f_\lambda~/~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$', size=16)
# -- prettify plots
# ax[0].set_ylim(0, 3.5)
# ax[1].set_ylim(0, 2.5)
ax[0].set_title('G135M-F170LP', size=16)
ax[1].set_title('G135M-F170LP', size=16)
plt.tight_layout()

# Plot all data on one panel
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))
# -- DJA data
for i, (wave, flux, err, mask) in enumerate(zip(dja_waves, dja_fluxes, dja_errs, dja_masks)):
    ax.step(wave[mask], flux[mask], color=dja_color, label="DJA" if i == 1 else None)
    # ax.fill_between(wave[mask], flux[mask]-err[mask], flux[mask]+err[mask], color=dja_color, step="mid")
# -- JADES data
for i, (wave, flux, err, mask) in enumerate(zip(jds_waves, jds_fluxes, jds_errs, jds_masks)):
    ax.step(wave, flux, color=jades_color, label="JADES" if i == 1 else None)
    # ax.fill_between(wave, flux-err, flux+err, color=jades_color, step="mid")
    # -- prettify in loop
    ax.legend()
    ax.set_xlabel(r'Wavelength / $\mu$m', size=16)
    ax.set_ylabel(r'$f_\lambda~/~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$', size=16)
# -- prettify plots
ax.set_ylim(0, 3.5)

plt.tight_layout()

plt.show()