import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits
from astropy.table import Table

from conversions import convert_wave_um_to_m, convert_wave_m_to_um, convert_flux_ujy_to_jy, convert_flux_jy_to_ujy, convert_flux_jy_to_cgs, convert_flux_si_to_jy, convert_magnitude_to_maggie

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
    jwst_filters = ([f'jwst_{filt}' for filt in filters])  # change to sedpy names
    flux_mag = phot_tb['DATA'].data
    err_mag = phot_tb['ERR'].data

    # Convert units
    if flux_units == 'maggie':
        convert_magnitude_to_maggie(flux_mag, err_mag)

    # # Convert units
    # if units == 'original':
    #     pass
    # elif units == 'maggie':
    #     phot, err = convert_magnitude_to_maggie(phot_mag, err_mag)
    # elif units == 'jy':
    #     flux_maggie, err_maggie = convert_magnitude_to_maggie(phot_mag, err_mag)
    #     phot = convert_maggie_to_janksy(flux_maggie)
    #     err = convert_maggie_to_janksy(err_maggie)
    # elif units == 'cgs':
    #     # TODO: add conversion for cgs units    
    #     pass

    # if not return_none:
    #     if return_quantity:
    #         return jwst_filters, phot, err
    #     else:
    #         return jwst_filters, phot.value, err.value
    # else:
    #     return None, None, None


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
    if flux_units == 'jy':
        flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
        flux, err = flux_jy, err_jy
    if flux_units == 'ujy':
        flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
        flux_ujy, err_ujy = convert_flux_jy_to_ujy(flux_jy, err_jy)
        flux, err = flux_ujy, err_ujy
    if flux_units == 'cgs':
        flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
        flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-20)
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
        wave_m = convert_wave_m_to_um(wave_um)
        flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
        flux, err = flux_jy, err_jy
    elif flux_units == 'cgs':
        wave_m = convert_wave_m_to_um(wave_um)
        flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
        flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, wave_um, cgs_factor=1e-20)
        flux, err = flux_cgs, err_cgs

    return wave_um, flux, err, mask

# ----------------------
# Functions to plot data
# ----------------------
def plot_spectra(waves_um, fluxes_jy, fluxes_cgs, errs_jy, errs_cgs, masks, spec_names):

    # Create figure
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

    # Colors for plotting
    colors = ['red', 'blue', 'green', 'orange']
    # colors = ['#377eb8', '#ff7f00', '#984ea3', '#e41a1c']

    for (wave_um, flux_jy, flux_cgs, err_jy, err_cgs, mask, spec_name, color) in zip(waves_um, fluxes_jy, fluxes_cgs, errs_jy, errs_cgs, masks, spec_names, colors):

        # Set all masked data to nan
        wave_um[~mask] = np.nan
        flux_jy[~mask] = np.nan
        err_jy[~mask] = np.nan

        # Plot spectral flux in Jys
        ax[0].step(wave_um, flux_jy, color=color, label=spec_name)
        ax[0].fill_between(wave_um, flux_jy-err_jy, flux_jy+err_jy, color=color, alpha=0.25)
        # -- prettify
        ax[0].set_xlabel(r'Observed Wavelength / $\mu$m')
        ax[0].set_ylabel(r'Flux / $\mu$Jy')
        ax[0].set_ylim(-2, 10)
        ax[0].legend()

        # Plot spectral flux in cgs unit
        ax[1].step(wave_um, flux_cgs, color=color, label=spec_name)
        ax[1].fill_between(wave_um, flux_cgs-err_cgs, flux_cgs+err_cgs, color=color, alpha=0.25)
        # -- prettify
        ax[1].set_xlabel(r'Observed Wavelength / $\mu$m')
        ax[1].set_ylabel(r'Flux / erg s$^{-1}$ cm$^{-2}$ A$^{-1}$')
        ax[1].set_ylim(-2, 20)
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

def plot_four_panel_spectra_for_unit(waves_um, fluxes_jy, fluxes_cgs, errs_jy, errs_cgs, masks, spec_names, flux_unit):

    # Create figure
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Colors for plotting
    colors = ['red', 'blue', 'green', 'orange']

    if flux_unit == 'cgs':
        fluxes, errs = fluxes_cgs, errs_cgs
    elif flux_unit == 'ujy':
        fluxes, errs = fluxes_jy, errs_jy

    for i, (wave_um, flux, err, mask, spec_name, color) in enumerate(zip(waves_um, fluxes, errs, masks, spec_names, colors)):

        # Set all masked data to nan
        wave_um[~mask] = np.nan
        flux[~mask] = np.nan
        err[~mask] = np.nan

        # Initialise row and column
        r = i // 2
        c = i % 2

        # Plot spectral flux in cgs unit
        ax[r, c].step(wave_um, flux, color=color, label=spec_name)
        ax[r, c].fill_between(wave_um, flux-err, flux+err, color=color, alpha=0.25)
        # -- prettify
        ax[r, c].set_xlabel(r'Observed Wavelength / $\mu$m')
        ax[r, c].set_ylabel(r'Flux / erg s$^{-1}$ cm$^{-2}$ A$^{-1}$')
        ax[r, c].set_ylim(-2, 20)
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
    phot_names = []
    waves_um = []
    fluxes_ujy = []
    fluxes_cgs = []
    errs_ujy = []
    errs_cgs = []
    masks = []

    load_photometry_data(phot_dir, name, flux_units='maggie')
    
    # Loop over each spectrum
    for i, spec_name in enumerate(spec_names):

        # Load data
        # -- prism
        if spec_name == 'prism':
            wave_um, flux_ujy, err_ujy, mask = load_prism_data(spec_dir, name, version=3.1, extra_nod='extr5', flux_units='ujy', return_quantities=False, return_units=False)
            flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
            wave_m = convert_wave_um_to_m(wave_um)
            flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-20)
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
            flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-20)

        # Append data to list
        waves_um.append(wave_um)
        fluxes_ujy.append(flux_ujy)
        fluxes_cgs.append(flux_cgs)
        errs_ujy.append(err_ujy)
        errs_cgs.append(err_cgs)
        masks.append(mask)

    # Make plots
    plot_spectra(waves_um, fluxes_ujy, fluxes_cgs, errs_ujy, errs_cgs, masks, spec_names)
    plot_spectra_snr(waves_um, fluxes_ujy, errs_ujy, masks, spec_names)
    plot_four_panel_spectra_for_unit(waves_um, fluxes_ujy, fluxes_cgs, errs_ujy, errs_cgs, masks, spec_names, flux_unit='cgs')

    plt.show()

if __name__ == "__main__":
    main()