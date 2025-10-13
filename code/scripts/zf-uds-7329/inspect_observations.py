import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits
from astropy.table import Table

from conversions import convert_wave_um_to_m, convert_wave_m_to_um, convert_flux_ujy_to_jy, convert_flux_jy_to_ujy, convert_flux_jy_to_cgs, convert_flux_si_to_jy, convert_flux_magnitude_to_maggie, convert_flux_maggie_to_jy

# ----------------------
# Functions to load data
# ----------------------
def load_photometry_data(phot_dir, name, flux_units, return_quantities=False, return_units=False):

    # Load table
    phot_name = f'{name}_nircam_photometry.fits'
    phot_path = os.path.join(phot_dir, phot_name)
    phot_tb = Table.read(phot_path)

    # Access photometry data
    filters = phot_tb['FILTER'].tolist()
    pivot_um = phot_tb['PIVOT'].data
    flux_mag = phot_tb['DATA'].data
    err_mag = phot_tb['ERR'].data
    mask = phot_tb['VALID'].data.astype(bool)

    sedpy_filters = ([f'jwst_{filt}' for filt in filters])  # change to sedpy names

    # Convert units
    if flux_units == 'maggie':
        flux_maggie, err_maggie = convert_flux_magnitude_to_maggie(flux_mag, err_mag)
        flux, err = flux_maggie, err_maggie
    elif flux_units == 'jy':
        flux_maggie, err_maggie = convert_flux_magnitude_to_maggie(flux_mag, err_mag)
        flux_jy, err_jy = convert_flux_maggie_to_jy(flux_maggie, err_maggie)
        flux, err = flux_jy, err_jy
    elif flux_units == 'cgs':
        flux_maggie, err_maggie = convert_flux_magnitude_to_maggie(flux_mag, err_mag)
        flux_jy, err_jy = convert_flux_maggie_to_jy(flux_maggie, err_maggie)
        pivot_m = convert_wave_um_to_m(pivot_um)
        flux_cgs, err_cgs = convert_flux_jy_to_cgs(pivot_m, flux_jy, err_jy, cgs_factor=1e-19)
        flux, err = flux_cgs, err_cgs
    else:
        pass

    return pivot_um, flux, err, mask, sedpy_filters


def load_prism_data(prism_dir, name, version, extra_nod, flux_units, return_quantities=False, return_units=False):

    # Load FITS file
    spec_name = f'{name}_prism_clear_v{version:.1f}_{extra_nod}_1D.fits'
    spec_path = os.path.join(prism_dir, spec_name)
    hdul = fits.open(spec_path)

    # Access spectral data
    # -- wavelength
    wave_hdu = hdul['WAVELENGTH']
    wave_m = wave_hdu.data
    # -- flux
    flux_hdu = hdul['DATA']
    flux_si = flux_hdu.data
    # -- error
    err_hdu = hdul['ERR']
    err_si = err_hdu.data
    # -- mask
    mask = np.full_like(flux_si, 1).astype(bool)  # temporary: do not mask any data

    # Convert to desired units
    if flux_units == 'si':
        flux, err = flux_si, err_si
    elif flux_units == 'jy':
        flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
        flux, err = flux_jy, err_jy
    elif flux_units == 'ujy':
        flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
        flux_ujy, err_ujy = convert_flux_jy_to_ujy(flux_jy, err_jy)
        flux, err = flux_ujy, err_ujy
    elif flux_units == 'cgs':
        flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
        flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
        flux, err = flux_cgs, err_cgs

    wave_um = convert_wave_m_to_um(wave_m)

    return wave_um, flux, err, mask

def load_grating_data(grating_dir, name, grating, filter, flux_units, return_quantities=False, return_units=False):

    # Load FITS file
    spec_name = f"{name}_nirspec_{grating.lower()}_{filter.lower()}_1D.fits"
    spec_path = os.path.join(grating_dir, spec_name)
    hdul = fits.open(spec_path)

    # Extract spectral data
    # -- wavelength
    wave_hdu = hdul['WAVELENGTH']
    wave_um = wave_hdu.data
    # -- flux
    flux_hdu = hdul['DATA']
    flux_ujy = flux_hdu.data
    # -- error
    err_hdu = hdul['ERR']
    err_ujy = err_hdu.data
    # -- mask
    mask_hdu = hdul['VALID']
    mask = mask_hdu.data.astype(bool)

    # Convert to desired units
    if flux_units == 'ujy':
        flux, err = flux_ujy, err_ujy
    elif flux_units == 'jy':
        wave_m = convert_wave_um_to_m(wave_um)
        flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
        flux, err = flux_jy, err_jy
    elif flux_units == 'cgs':
        wave_m = convert_wave_um_to_m(wave_um)
        flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
        flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
        flux, err = flux_cgs, err_cgs

    return wave_um, flux, err, mask

# ----------------------
# Functions to plot data
# ----------------------
def plot_spectra(spec, phot):

    # Unpack lists
    spec_waves_um, spec_fluxes_ujy, spec_fluxes_cgs, spec_errs_ujy, spec_errs_cgs, spec_masks, spec_names = spec

    # Create figure
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

    # Colors for plotting
    colors = ['red', 'blue', 'green', 'orange']
    # colors = ['#377eb8', '#ff7f00', '#984ea3', '#e41a1c']

    for (spec_wave_um, spec_flux_ujy, spec_flux_cgs, spec_err_ujy, spec_err_cgs, spec_mask, spec_name, color) in zip(spec_waves_um, spec_fluxes_ujy, spec_fluxes_cgs, spec_errs_ujy, spec_errs_cgs, spec_masks, spec_names, colors):

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
        ax[0].set_ylim(-1, 8)
        ax[0].legend()

        # Plot spectral flux in cgs unit
        ax[1].step(spec_wave_um, spec_flux_cgs, color=color, label=spec_name)
        ax[1].fill_between(spec_wave_um, spec_flux_cgs-spec_err_cgs, spec_flux_cgs+spec_err_cgs, color=color, alpha=0.25)
        # -- prettify
        ax[1].set_xlabel(r'Observed Wavelength / $\mu$m')
        ax[1].set_ylabel(r'Flux / erg s$^{-1}$ cm$^{-2}$ A$^{-1}$')
        ax[1].set_ylim(-0.2, 2)
        ax[1].legend(loc='upper left')

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
    spec_waves_um, spec_fluxes_ujy, spec_fluxes_cgs, spec_errs_ujy, spec_errs_cgs, spec_masks, spec_names = spec
    phot_pivots_um, phot_fluxes_ujy, phot_fluxes_cgs, phot_errs_ujy, phot_errs_cgs, phot_mask, phot_filters = phot

    # Create figure
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Colors for plotting
    spec_colors = ['red', 'blue', 'green', 'orange']

    # Choose units to plot
    if flux_unit == 'cgs':
        spec_fluxes, spec_errs = spec_fluxes_cgs, spec_errs_cgs
        phot_fluxes, phot_errs = phot_fluxes_cgs, phot_errs_cgs
    elif flux_unit == 'ujy':
        spec_fluxes, spec_errs = spec_fluxes_ujy, spec_errs_ujy
        phot_fluxes, phot_errs = phot_fluxes_ujy, phot_errs_ujy

    # Loop over each subplot/spectrum
    for i, (wave_um, flux, err, mask, spec_name, color) in enumerate(zip(spec_waves_um, spec_fluxes, spec_errs, spec_masks, spec_names, spec_colors)):

        # Initialise row and column
        r = i // 2
        c = i % 2

        # Multiply flux by scaling factor
        factor = 1.86422
        flux = flux * factor
        err = err * factor

        # Set all masked data to nan
        wave_um[~mask] = np.nan
        flux[~mask] = np.nan
        err[~mask] = np.nan
        phot_pivots_um[~phot_mask] = np.nan
        phot_fluxes[~phot_mask] = np.nan
        phot_errs[~phot_mask] = np.nan

        # Plot data
        # -- spectral flux in cgs unit
        ax[r, c].step(wave_um, flux, color=color, label=spec_name)
        ax[r, c].fill_between(wave_um, flux-err, flux+err, color=color, alpha=0.25)
        # -- photometric flux in cgs unit
        ax[r, c].errorbar(phot_pivots_um, phot_fluxes, yerr=phot_errs, ls=None, fmt='.', color='black', label='photometry')
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

    data_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329"
    spec_dir = os.path.join(data_dir, 'spectra')
    phot_dir = os.path.join(data_dir, 'photometry')

    name = 'zf-uds-7329'
    grat_filts = ['g395m-f290lp', 'g235m-f170lp', 'g140m-f100lp']
    spec_names = ['prism', *grat_filts]  # combine into one list to loop over
    # spec_names = grat_filts

    # Lists to store flux information
    spec_waves_um = []
    spec_fluxes_ujy = []
    spec_fluxes_cgs = []
    spec_errs_ujy = []
    spec_errs_cgs = []
    spec_masks = []

    # Load photometry data
    phot_pivot_um, phot_fluxes_jy, phot_errs_jy, phot_mask, phot_filters = load_photometry_data(phot_dir, name, flux_units='jy')
    phot_fluxes_ujy, phot_errs_ujy = convert_flux_jy_to_ujy(phot_fluxes_jy, phot_errs_jy)
    phot_pivot_m = convert_wave_um_to_m(phot_pivot_um)
    phot_fluxes_cgs, phot_errs_cgs = convert_flux_jy_to_cgs(phot_pivot_m, phot_fluxes_jy, phot_errs_jy, cgs_factor=1e-19)
    
    # Loop over each spectrum
    for i, spec_name in enumerate(spec_names):

        # Load data
        # -- prism
        if spec_name == 'prism':
            wave_um, flux_ujy, err_ujy, mask = load_prism_data(spec_dir, name, version=3.1, extra_nod='extr5', flux_units='ujy', return_quantities=False, return_units=False)
            flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
            wave_m = convert_wave_um_to_m(wave_um)
            flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
        # -- grating
        else:
            # Extract grating and filter
            grat, filt = spec_name.split('-')

            # Load in data with different units
            # -- microjanskies
            wave_um, flux_ujy, err_ujy, mask = load_grating_data(spec_dir, name, grating=grat, filter=filt, flux_units='ujy')
            # -- cgs units (and janskies)
            flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
            wave_m = convert_wave_um_to_m(wave_um)
            flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)

        # Append data to list
        spec_waves_um.append(wave_um)
        spec_fluxes_ujy.append(flux_ujy)
        spec_fluxes_cgs.append(flux_cgs)
        spec_errs_ujy.append(err_ujy)
        spec_errs_cgs.append(err_cgs)
        spec_masks.append(mask)

    # Make lists to store variables
    spec = [spec_waves_um, spec_fluxes_ujy, spec_fluxes_cgs, spec_errs_ujy, spec_errs_cgs, spec_masks, spec_names]
    phot = [phot_pivot_um, phot_fluxes_ujy, phot_fluxes_cgs, phot_errs_ujy, phot_errs_cgs, phot_mask, phot_filters]

    # Make plots
    plot_spectra(spec, phot)
    plot_spectra_snr(spec_waves_um, spec_fluxes_ujy, spec_errs_ujy, spec_masks, spec_names)
    plot_four_panel_spectra_for_unit(spec, phot, flux_unit='cgs')

    plt.show()

if __name__ == "__main__":
    main()