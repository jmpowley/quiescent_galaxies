import os

import numpy as np
import matplotlib.pyplot as plt

from conversions import convert_wave_um_to_m, convert_wave_m_to_um, convert_flux_ujy_to_jy, convert_flux_jy_to_ujy, convert_flux_jy_to_cgs, convert_flux_jy_to_maggie, convert_wave_A_to_m, convert_wave_m_to_A
from loading import load_photometry_data, load_prism_data, load_grating_data

# ----------------------
# Functions to plot data
# ----------------------
def plot_spectra(spec, phot):

    # Unpack lists
    spec_waves_um, spec_fluxes_ujy, spec_fluxes_cgs, spec_fluxes_maggies, spec_errs_ujy, spec_errs_cgs, spec_errs_maggies, spec_masks, spec_names = spec

    # Create figure
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(14, 8.5))

    # Colors for plotting
    colors = ['red', 'blue', 'green', 'orange']
    # colors = ['#377eb8', '#ff7f00', '#984ea3', '#e41a1c']

    for (spec_wave_um, spec_flux_ujy, spec_flux_cgs, spec_flux_maggies, spec_err_ujy, spec_err_cgs, spec_err_maggies, spec_mask, spec_name, color) in zip(spec_waves_um, spec_fluxes_ujy, spec_fluxes_cgs, spec_fluxes_maggies, spec_errs_ujy, spec_errs_cgs, spec_errs_maggies, spec_masks, spec_names, colors):

        # Set all masked data to nan
        spec_wave_um[~spec_mask] = np.nan
        spec_flux_ujy[~spec_mask] = np.nan
        spec_err_ujy[~spec_mask] = np.nan

        # Plot spectral flux in microjanskies
        ax[0].step(spec_wave_um, spec_flux_ujy, color=color, label=spec_name)
        ax[0].fill_between(spec_wave_um, spec_flux_ujy-spec_err_ujy, spec_flux_ujy+spec_err_ujy, color=color, alpha=0.25)
        # -- prettify
        ax[0].set_xlabel(r'Observed Wavelength / $\mu$m')
        ax[0].set_ylabel(r'Flux / $\mu$Jy')
        ax[0].set_ylim(-1, 12)
        ax[0].legend()

        # Plot spectral flux in cgs unit
        ax[1].step(spec_wave_um, spec_flux_cgs, color=color, label=spec_name)
        ax[1].fill_between(spec_wave_um, spec_flux_cgs-spec_err_cgs, spec_flux_cgs+spec_err_cgs, color=color, alpha=0.25)
        # -- prettify
        ax[1].set_xlabel(r'Observed Wavelength / $\mu$m')
        ax[1].set_ylabel(r'Flux / erg s$^{-1}$ cm$^{-2}$ A$^{-1}$')
        ax[1].set_ylim(-0.2, 3)
        ax[1].legend(loc='upper left')

        # Plot spectral flux in cgs unit
        ax[2].step(spec_wave_um, spec_flux_maggies, color=color, label=spec_name)
        ax[2].fill_between(spec_wave_um, spec_flux_maggies-spec_err_maggies, spec_flux_maggies+spec_err_maggies, color=color, alpha=0.25)
        # -- prettify
        ax[2].set_xlabel(r'Observed Wavelength / $\mu$m')
        ax[2].set_ylabel(r'Flux / maggies')
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')
        ax[2].legend(loc='upper left')

    plt.tight_layout()
    # plt.savefig()

def plot_spectra_snr(waves_um, fluxes_jy, errs_jy, masks, spec_names):

    # Create figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

    # Colors for plotting
    colors = ['red', 'blue', 'green', 'orange']
    # colors = ['#377eb8', '#ff7f00', '#984ea3', '#e41a1c']

    for (wave_um, flux_jy, err_jy, mask, spec_name, color) in zip(waves_um, fluxes_jy, errs_jy, masks, spec_names, colors):

        # Set all masked data to nan
        wave_um[~mask] = np.nan
        flux_jy[~mask] = np.nan
        err_jy[~mask] = np.nan

        # Plot spectral flux in Jys
        ax.step(wave_um, flux_jy / err_jy, color=color, label=spec_name)
        # -- prettify
        ax.set_xlabel(r'Observed Wavelength / $\mu$m')
        ax.set_ylabel(r'SNR')
        ax.legend()

    plt.tight_layout()
    # plt.savefig()

def plot_four_panel_spectra_for_unit(spec, phot, flux_unit):

    # Unpack lists
    spec_waves_um, spec_fluxes_ujy, spec_fluxes_cgs, spec_fluxes_maggies, spec_errs_ujy, spec_errs_cgs, spec_errs_maggies, spec_masks, spec_names = spec
    phot_waveffs_um, phot_fluxes_ujy, phot_fluxes_cgs, phot_fluxes_maggies, phot_errs_ujy, phot_errs_cgs, phot_errs_maggies, phot_mask, phot_filters = phot

    # Create figure
    nrows, ncols = 2, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(12, 8))

    # Colors for plotting
    spec_colors = ['red', 'blue', 'green', 'orange']

    # Choose units to plot
    if flux_unit == 'cgs':
        spec_fluxes, spec_errs = spec_fluxes_cgs, spec_errs_cgs
        phot_fluxes, phot_errs = phot_fluxes_cgs, phot_errs_cgs
    elif flux_unit == 'ujy':
        spec_fluxes, spec_errs = spec_fluxes_ujy, spec_errs_ujy
        phot_fluxes, phot_errs = phot_fluxes_ujy, phot_errs_ujy
    elif flux_unit == 'maggie':
        spec_fluxes, spec_errs = spec_fluxes_maggies, spec_errs_maggies
        phot_fluxes, phot_errs = phot_fluxes_maggies, phot_errs_maggies

    # Loop over each subplot/spectrum
    for i, (wave_um, flux, err, mask, spec_name, color) in enumerate(zip(spec_waves_um, spec_fluxes, spec_errs, spec_masks, spec_names, spec_colors)):

        # Initialise row and column
        r, c = divmod(i, ncols)

        # Set all masked data to nan
        wave_um[~mask] = np.nan
        flux[~mask] = np.nan
        err[~mask] = np.nan
        phot_waveffs_um[~phot_mask] = np.nan
        phot_fluxes[~phot_mask] = np.nan
        phot_errs[~phot_mask] = np.nan

        # Plot data
        # -- spectral flux in cgs unit
        ax[r, c].step(wave_um, flux, color=color, label=spec_name)
        ax[r, c].fill_between(wave_um, flux-err, flux+err, color=color, alpha=0.25)
        # -- photometric flux in cgs unit
        ax[r, c].errorbar(phot_waveffs_um, phot_fluxes, yerr=phot_errs, ls=None, fmt='.', color='black', label='photometry')
        # -- prettify
        ax[r, c].set_xlabel(r'Observed Wavelength / $\mu$m')
        ax[r, c].set_ylabel(r'Flux / erg s$^{-1}$ cm$^{-2}$ A$^{-1}$')
        ax[r, c].set_ylim(-0.1, 2.6)
        ax[r, c].legend(loc='upper left')
    
    plt.tight_layout()
    # plt.savefig()

# -------------
# Main function
# -------------
def main():

    obs_kwargs = {

         "phot_params" : {
            "phot_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/photometry",
            "name" : "007329",
            "data_ext" : "DATA",
            "mask_ext" : "VALID",
            "in_flux_units" : "magnitude",
            "out_flux_units" : "ujy",
            "snr_limit" : 20,
            "return_none" : False,
        },

        "prism_params" : {
            "prism_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
            "name" : "007329",
            "version" : "v3.1",
            "nod" : "extr5",
            "data_ext" : "DATA",
            "mask_ext" : None,
            "in_wave_units" : "si",
            "out_wave_units" : "um",
            "in_flux_units" : "si",
            "out_flux_units" : "ujy",
            "rescale_factor" : 1.86422,
            "snr_limit" : 20,
            "return_none" : False,
        },

         "grat1_params" : {
            "grating_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
            "name" : "007329",
            "grating" : "g140m",
            "filter" : "f100lp",
            "version" : None,
            "nod" : None,
            "data_ext" : "DATA",
            "mask_ext" : "VALID",
            "in_wave_units" : "um",
            "out_wave_units" : "um",
            "in_flux_units" : "ujy",
            "out_flux_units" : "ujy",
            "rescale_factor" : 1.86422,
            "snr_limit" : 20,
            "return_none" : False,
        },

        "grat2_params" : {
            "grating_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
            "name" : "007329",
            "grating" : "g235m",
            "filter" : "f170lp",
            "version" : None,
            "nod" : None,
            "data_ext" : "DATA",
            "mask_ext" : "VALID",
            "in_wave_units" : "um",
            "out_wave_units" : "um",
            "in_flux_units" : "ujy",
            "out_flux_units" : "ujy",
            "rescale_factor" : 1.86422,
            "snr_limit" : 20,
            "return_none" : False,
        },

        "grat3_params" : {
            "grating_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
            "name" : "007329",
            "grating" : "g395m",
            "filter" : "f290lp",
            "version" : None,
            "nod" : None,
            "data_ext" : "DATA",
            "mask_ext" : "VALID",
            "in_wave_units" : "um",
            "out_wave_units" : "um",
            "in_flux_units" : "ujy",
            "out_flux_units" : "ujy",
            "rescale_factor" : 1.86422,
            "snr_limit" : 20,
            "return_none" : False,
        },
    }

    name = 'zf-uds-7329'
    grat_filts = ['g395m-f290lp', 'g235m-f170lp', 'g140m-f100lp']
    spec_names = ['prism', *grat_filts]  # combine into one list to loop over
    # spec_names = grat_filts

    # Lists to store flux information
    spec_waves_um = []
    spec_fluxes_ujy = []
    spec_fluxes_cgs = []
    spec_fluxes_maggies = []
    spec_errs_ujy = []
    spec_errs_cgs = []
    spec_errs_maggies = []
    spec_masks = []

    # Load photometry data
    
    # phot_filters, phot_fluxes_jy, phot_errs_jy, phot_mask = load_photometry_data(phot_dir, name, flux_units='jy', snr_limit=20)
    phot_filters, phot_fluxes_jy, phot_errs_jy, phot_mask = load_photometry_data(**obs_kwargs['phot_params'])
    phot_fluxes_ujy, phot_errs_ujy = convert_flux_jy_to_ujy(phot_fluxes_jy, phot_errs_jy)
    phot_waveffs_um = [f.wave_effective for f in phot_filters]
    phot_waveffs_m = convert_wave_um_to_m(phot_waveffs_um)
    phot_waveffs_A = convert_wave_m_to_A(phot_waveffs_m)
    phot_fluxes_cgs, phot_errs_cgs = convert_flux_jy_to_cgs(phot_waveffs_m, phot_fluxes_jy, phot_errs_jy, cgs_factor=1e-19)
    phot_fluxes_maggies, phot_errs_maggies = convert_flux_jy_to_maggie(phot_fluxes_jy, phot_errs_jy)

    # Loop over each spectrum
    for i, obs_key in enumerate(obs_kwargs.keys()):

        # Load data
        # -- skip photometry
        if obs_key == "phot_params":
            continue
        # -- prism
        elif obs_key == 'prism_params':
            wave_um, flux_ujy, err_ujy, mask = load_prism_data(**obs_kwargs['prism_params'])
            flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
            wave_m = convert_wave_um_to_m(wave_um)
            flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
            flux_maggie, err_maggie = convert_flux_jy_to_maggie(flux_jy, err_jy)
        # -- grating
        else:
            # Extract grating and filter
            # -- microjanskies
            wave_um, flux_ujy, err_ujy, mask = load_grating_data(**obs_kwargs[obs_key])
            # -- cgs units
            flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
            wave_m = convert_wave_um_to_m(wave_um)
            flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
            # -- maggies
            flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
            flux_maggie, err_maggie = convert_flux_jy_to_maggie(flux_jy, err_jy)

        # Append data to list
        spec_waves_um.append(wave_um)
        spec_fluxes_ujy.append(flux_ujy)
        spec_fluxes_cgs.append(flux_cgs)
        spec_fluxes_maggies.append(flux_maggie)
        spec_errs_ujy.append(err_ujy)
        spec_errs_cgs.append(err_cgs)
        spec_errs_maggies.append(err_maggie)
        spec_masks.append(mask)

    # Make lists to store variables
    spec = [spec_waves_um, spec_fluxes_ujy, spec_fluxes_cgs, spec_fluxes_maggies, spec_errs_ujy, spec_errs_cgs, spec_errs_maggies, spec_masks, spec_names]
    phot = [phot_waveffs_um, phot_fluxes_ujy, phot_fluxes_cgs, phot_fluxes_maggies, phot_errs_ujy, phot_errs_cgs, phot_errs_maggies, phot_mask, phot_filters]

    # # Make plots
    plot_spectra(spec, phot)
    plot_spectra_snr(spec_waves_um, spec_fluxes_ujy, spec_errs_ujy, spec_masks, spec_names)
    # plot_four_panel_spectra_for_unit(spec, phot, flux_unit='cgs')

    plt.show()

if __name__ == "__main__":
    main()