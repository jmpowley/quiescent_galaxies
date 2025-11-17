import os

import numpy as np
import matplotlib.pyplot as plt

from conversions import convert_wave_um_to_m, convert_wave_A_to_um, convert_flux_ujy_to_jy, convert_flux_jy_to_ujy, convert_flux_jy_to_cgs, convert_flux_jy_to_maggie, convert_wave_A_to_m, convert_wave_m_to_A
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

def plot_spectra_phot_1panel(spec, phot):

    # Unpack lists
    spec_waves_um, spec_fluxes_ujy, spec_fluxes_cgs, spec_fluxes_maggies, spec_errs_ujy, spec_errs_cgs, spec_errs_maggies, spec_masks, spec_names = spec
    phot_waveffs_um, phot_fluxes_ujy, phot_fluxes_cgs, phot_fluxes_maggies, phot_errs_ujy, phot_errs_cgs, phot_errs_maggies, phot_mask, phot_filters = phot
    
    # Create figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

    # Colors
    colors = [f"C{i}" for i in range(0, len(phot_waveffs_um))]

    # Plot spectra
    for (spec_wave_um, spec_flux_cgs, spec_err_cgs, spec_name) in zip(spec_waves_um, spec_fluxes_cgs, spec_errs_cgs, spec_names):
        ax.plot(spec_wave_um, spec_flux_cgs, color="k", label=spec_name)

    # Plot photometry
    for (phot_waveff_um, phot_flux_cgs, phot_err_cgs, color) in zip(phot_waveffs_um, phot_fluxes_cgs, phot_errs_cgs, colors):
        ax.errorbar(phot_waveff_um, phot_flux_cgs, yerr=phot_err_cgs, color=color)
        ax.scatter(phot_waveff_um, phot_flux_cgs, color=color)

    # Plot filters
    for (f, color) in zip(phot_filters, colors):
        wave = convert_wave_A_to_um(f.wavelength)
        trans = f.transmission / (2 * np.max(f.transmission))  # normalise
        name = f.nick.lstrip("jwst_").upper()  # nice name (e.g., F444W)
        ax.fill_between(wave, 0, trans, color=color, alpha=0.3)
        ax.fill_between(wave, np.nan, np.nan, color=color, label=name)
        ax.plot(wave, trans, color=color)

    # Prettify
    ax.set_xlabel(r'$\lambda_{obs}~[\mu m]$', size=18)
    ax.set_ylabel(r'$f_\lambda~[~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}]$', size=18)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", ncols=len(phot_fluxes_cgs)+len(spec_fluxes_cgs), bbox_to_anchor=[0, 1.1], framealpha=0)
    plt.tight_layout()

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

        "phot_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/photometry",
            "data_name" : "007329_nircam_photometry.fits",
            "data_ext" : "DATA",
            "in_flux_units" : "magnitude",
            "out_flux_units" : "ujy",
            "snr_limit" : 20.0,
        },

        "prism_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
            "data_name" : "007329_prism_clear_v3.1_extr5_1D.fits",
            "data_ext" : "DATA",
            "mask_dir" : None,
            "mask_name" : None,
            "mask_ext" : None,
            "in_wave_units" : "si",
            "out_wave_units" : "um",
            "in_flux_units" : "si",
            "out_flux_units" : "ujy",
            "rescale_factor" : 1.86422,
            "snr_limit" : 20.0,
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
    phot_filters, phot_fluxes_ujy, phot_errs_ujy = load_photometry_data(**obs_kwargs['phot_kwargs'])
    phot_mask = None
    phot_fluxes_jy, phot_errs_jy = convert_flux_ujy_to_jy(phot_fluxes_ujy, phot_errs_ujy)
    phot_waveffs_A = [f.wave_effective for f in phot_filters]
    phot_waveffs_um = convert_wave_A_to_um(phot_waveffs_A)
    phot_waveffs_m = convert_wave_um_to_m(phot_waveffs_um)
    phot_fluxes_cgs, phot_errs_cgs = convert_flux_jy_to_cgs(phot_waveffs_m, phot_fluxes_jy, phot_errs_jy, cgs_factor=1e-19)
    phot_fluxes_maggies, phot_errs_maggies = convert_flux_jy_to_maggie(phot_fluxes_jy, phot_errs_jy)

    # Loop over each spectrum
    for i, obs_key in enumerate(obs_kwargs.keys()):

        # Load data
        # -- skip photometry
        if obs_key == "phot_kwargs":
            continue
        # -- prism
        elif obs_key == 'prism_kwargs':
            wave_um, flux_ujy, err_ujy = load_prism_data(**obs_kwargs['prism_kwargs'])
            flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
            wave_m = convert_wave_um_to_m(wave_um)
            flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
            flux_maggie, err_maggie = convert_flux_jy_to_maggie(flux_jy, err_jy)
            mask = None
        # -- grating
        else:
            # Extract grating and filter
            # -- microjanskies
            wave_um, flux_ujy, err_ujy = load_grating_data(**obs_kwargs[obs_key])
            # -- cgs units
            flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
            wave_m = convert_wave_um_to_m(wave_um)
            flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
            # -- maggies
            flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
            flux_maggie, err_maggie = convert_flux_jy_to_maggie(flux_jy, err_jy)
            mask = None

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

    # Make plots
    plot_spectra_phot_1panel(spec, phot)
    # plot_spectra(spec, phot)
    # plot_spectra_snr(spec_waves_um, spec_fluxes_ujy, spec_errs_ujy, spec_masks, spec_names)
    # plot_four_panel_spectra_for_unit(spec, phot, flux_unit='cgs')

    plt.show()

if __name__ == "__main__":
    main()