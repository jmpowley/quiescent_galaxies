import os

import numpy as np

from astropy.io import fits
from astropy.table import Table

from sedpy.observate import load_filters

from conversions import convert_wave_um_to_m, convert_wave_m_to_um, convert_wave_um_to_A, convert_wave_m_to_A, convert_flux_ujy_to_jy, convert_flux_jy_to_ujy, convert_flux_jy_to_cgs, convert_flux_si_to_jy, convert_flux_magnitude_to_maggie, convert_flux_maggie_to_jy, convert_flux_jy_to_maggie

# ----------------------
# Functions to load data
# ----------------------
def load_photometry_data(phot_dir, name, wave_units, flux_units, return_none=False, return_quantities=False, return_units=False):

    # Load table
    phot_name = f"{name}_nircam_photometry.fits"
    phot_path = os.path.join(phot_dir, phot_name)
    phot_tb = Table.read(phot_path)

    # Access photometry data
    filters = phot_tb["FILTER"].tolist()
    pivot_um = phot_tb["PIVOT"].data
    flux_mag = phot_tb["DATA"].data
    err_mag = phot_tb["ERR"].data
    mask = phot_tb["VALID"].data.astype(bool)

    jwst_filters = ([f"jwst_{filt}" for filt in filters])
    sedpy_filters = load_filters(jwst_filters)

    # Convert units
    # -- flux
    if flux_units == "maggie":
        flux_maggie, err_maggie = convert_flux_magnitude_to_maggie(flux_mag, err_mag)
        flux, err = flux_maggie, err_maggie
    elif flux_units == "jy":
        flux_maggie, err_maggie = convert_flux_magnitude_to_maggie(flux_mag, err_mag)
        flux_jy, err_jy = convert_flux_maggie_to_jy(flux_maggie, err_maggie)
        flux, err = flux_jy, err_jy
    elif flux_units == "cgs":
        flux_maggie, err_maggie = convert_flux_magnitude_to_maggie(flux_mag, err_mag)
        flux_jy, err_jy = convert_flux_maggie_to_jy(flux_maggie, err_maggie)
        pivot_m = convert_wave_um_to_m(pivot_um)
        flux_cgs, err_cgs = convert_flux_jy_to_cgs(pivot_m, flux_jy, err_jy, cgs_factor=1e-19)
        flux, err = flux_cgs, err_cgs
    else:
        pass
    # -- wavelength
    if wave_units == 'um':
        pivot = pivot_um
    if wave_units == 'A':
        pivot_A = convert_wave_um_to_A(pivot_um)
        pivot = pivot_A
    else:
        pass

    if return_none:
        return None, None, None, None
    else:
        return sedpy_filters, flux, err, mask

def load_prism_data(prism_dir, name, version, extra_nod, wave_units, flux_units, return_none=False, return_quantities=False, return_units=False):

    # Load FITS file
    spec_name = f"{name}_prism_clear_v{version:.1f}_{extra_nod}_1D.fits"
    spec_path = os.path.join(prism_dir, spec_name)
    hdul = fits.open(spec_path)

    # Access spectral data
    # -- wavelength
    wave_hdu = hdul["WAVELENGTH"]
    wave_m = wave_hdu.data
    # -- flux
    flux_hdu = hdul["DATA"]
    flux_si = flux_hdu.data
    # -- error
    err_hdu = hdul["ERR"]
    err_si = err_hdu.data
    # -- mask
    mask = np.full_like(flux_si, 1).astype(bool)  # temporary: do not mask any data

    # Convert to desired units
    # -- flux
    if flux_units == "si":
        flux, err = flux_si, err_si
    elif flux_units == "jy":
        flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
        flux, err = flux_jy, err_jy
    elif flux_units == "ujy":
        flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
        flux_ujy, err_ujy = convert_flux_jy_to_ujy(flux_jy, err_jy)
        flux, err = flux_ujy, err_ujy
    elif flux_units == "cgs":
        flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
        flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
        flux, err = flux_cgs, err_cgs
    elif flux_units == "maggie":
        flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
        flux_maggie, err_maggie = convert_flux_jy_to_maggie(flux_jy, err_jy)
        flux, err = flux_maggie, err_maggie
    # -- wavelength
    if wave_units == 'um':
        wave_um = convert_wave_m_to_um(wave_m)
        wave = wave_um
    if wave_units == 'A':
        wave_A = convert_wave_m_to_A(wave_m)
        wave = wave_A
    else:
        pass

    if return_none:
        return None, None, None, None
    else:
        return wave, flux, err, mask

def load_grating_data(grating_dir, name, grating, filter, wave_units, flux_units, return_none=False, return_quantities=False, return_units=False):

    # Load FITS file
    spec_name = f"{name}_nirspec_{grating.lower()}_{filter.lower()}_1D.fits"
    spec_path = os.path.join(grating_dir, spec_name)
    hdul = fits.open(spec_path)

    # Extract spectral data
    # -- wavelength
    wave_hdu = hdul["WAVELENGTH"]
    wave_um = wave_hdu.data
    # -- flux
    flux_hdu = hdul["DATA"]
    flux_ujy = flux_hdu.data
    # -- error
    err_hdu = hdul["ERR"]
    err_ujy = err_hdu.data
    # -- mask
    mask_hdu = hdul["VALID"]
    mask = mask_hdu.data.astype(bool)

    # Convert to desired units
    if flux_units == "ujy":
        flux, err = flux_ujy, err_ujy
    elif flux_units == "jy":
        wave_m = convert_wave_um_to_m(wave_um)
        flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
        flux, err = flux_jy, err_jy
    elif flux_units == "cgs":
        wave_m = convert_wave_um_to_m(wave_um)
        flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
        flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
        flux, err = flux_cgs, err_cgs
    # -- wavelength
    if wave_units == 'um':
        wave = wave_um
    if wave_units == 'A':
        wave_A = convert_wave_um_to_A(wave_um)
        wave = wave_A
    else:
        pass

    if return_none:
        return None, None, None, None
    else:
        return wave, flux, err, mask