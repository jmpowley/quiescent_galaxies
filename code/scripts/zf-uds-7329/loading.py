import os

import numpy as np

from astropy.io import fits
from astropy.table import Table

from sedpy.observate import load_filters

from conversions import convert_wave_um_to_m, convert_wave_m_to_um, convert_wave_um_to_A, convert_wave_m_to_A, convert_flux_ujy_to_jy, convert_flux_jy_to_ujy, convert_flux_jy_to_cgs, convert_flux_si_to_jy, convert_flux_magnitude_to_maggie, convert_flux_maggie_to_jy, convert_flux_si_to_cgs, convert_flux_si_to_jy, convert_flux_maggie_to_cgs
from preprocessing import apply_snr_limit, apply_rescaling_factor

# ----------------------
# Functions to load data
# ----------------------
def load_photometry_data(phot_dir, name, data_ext, mask_ext, in_flux_units, out_flux_units, snr_limit, return_none=False, return_quantities=False, return_units=False, **extras):

    if return_none:
        return None, None, None, None

    # Load table
    phot_name = f"{name}_nircam_photometry.fits"
    phot_path = os.path.join(phot_dir, phot_name)
    phot_tb = Table.read(phot_path)

    # Access photometry data
    filters = phot_tb["FILTER"].tolist()
    wave_in = phot_tb["WAVEFF"].data
    flux_in = phot_tb[data_ext].data
    err_in = phot_tb["ERR"].data
    mask = phot_tb[mask_ext].data.astype(bool)

    # Load JWST filters
    jwst_filters = ([f"jwst_{filt}" for filt in filters])
    sedpy_filters = load_filters(jwst_filters)

    # Convert flux units
    # -- from magnitudes
    if in_flux_units == "magnitude":
        if out_flux_units == "maggie":
            flux_out, err_out = convert_flux_magnitude_to_maggie(flux_in, err_in)
        elif out_flux_units == "ujy":
            flux_maggie, err_maggie = convert_flux_magnitude_to_maggie(flux_in, err_in)
            flux_jy, err_jy = convert_flux_maggie_to_jy(flux_maggie, err_maggie)
            flux_out, err_out = convert_flux_jy_to_ujy(flux_jy, err_jy)
        elif out_flux_units == "cgs":
            flux_maggie, err_maggie = convert_flux_magnitude_to_maggie(flux_in, err_in)
            wave_m = convert_wave_um_to_m(wave_in)  # assumes wavelength given in microns
            flux_out, err_out = convert_flux_maggie_to_cgs(flux_maggie, err_maggie, wave_m, cgs_factor=1e-19)
        else:
            raise ValueError(f'Output flux unit ({out_flux_units}) not recognised')
    else:
        flux_out, err_out = None, None

    # print(flux_out)

    # Apply noise floor
    if snr_limit is not None:
        flux_out, err_out = apply_snr_limit(flux_out, err_out, snr_limit)

    return sedpy_filters, flux_out, err_out, mask

def load_prism_data(prism_dir, name, version, nod, data_ext, mask_ext, in_wave_units, out_wave_units, in_flux_units, out_flux_units, rescale_factor, snr_limit, return_none=False, return_quantities=False, return_units=False, **extras):

    if return_none:
        return None, None, None, None

    # Customise
    # -- version of spectra/reduction
    if version is not None:
        version_str = f"_{version}"
    else:
        version_str = ""
    # -- nod
    if nod is not None:
        nod_str = f"_{nod}"
    else:
        nod_str = ""
    # -- spectrum dimensions
    dim_str = "_1D"
    # -- combine
    extra_str = version_str + nod_str + dim_str

    # Load FITS file
    spec_name = f"{name}_prism_clear{version_str}{nod_str}{dim_str}.fits"
    spec_path = os.path.join(prism_dir, spec_name)
    hdul = fits.open(spec_path)

    # Extract spectral data
    # -- wavelength
    wave_hdu = hdul["WAVELENGTH"]
    wave_in = wave_hdu.data
    # -- flux
    if data_ext is not None:
        flux_hdu = hdul[data_ext]
    else:    
        flux_hdu = hdul["DATA"]
    flux_in = flux_hdu.data
    # -- error
    err_hdu = hdul["ERR"]
    err_in = err_hdu.data
    # -- mask
    if mask_ext is not None:
        mask_hdu = hdul[mask_ext]
        mask = mask_hdu.data.astype(bool)
    else:
        mask = None
        mask = np.full(flux_in.shape, True)  # TODO: add NaN (or like) mask

    # Convert wavelength units
    # -- from SI
    if in_wave_units == "si":
        if out_wave_units == "um":
            wave_out = convert_wave_m_to_um(wave_in)
        elif out_wave_units == "A":
            wave_out = convert_wave_m_to_A(wave_in)
        else:
            raise ValueError(f'Output wave unit ({out_wave_units}) not recognised')
    # -- from microns
    elif in_wave_units == "um":
        if out_wave_units == "A":
            wave_out = convert_wave_um_to_A(wave_in)
        elif out_wave_units == "um":
            wave_out = wave_in
        else:
            raise ValueError(f'Output wave unit ({out_wave_units}) not recognised')
    else:
        raise ValueError(f'Input wave unit ({in_wave_units}) not recognised')

    # Convert flux units
    # -- from SI
    if in_flux_units == "si":
        if out_flux_units == "cgs":
            flux_out, err_out = convert_flux_si_to_cgs(flux_in, err_in, cgs_factor=1e-19)
        if out_flux_units == "ujy":
            wave_m = wave_in
            flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_in, err_in)
            flux_out, err_out = convert_flux_jy_to_ujy(flux_jy, err_jy)
    # -- from uJy
    elif in_flux_units == "ujy":
        if out_flux_units == "cgs":
            if in_wave_units == "um":
                wave_m = convert_wave_um_to_m(wave_in)
            flux_jy, err_jy = convert_flux_ujy_to_jy(flux_in, err_in)
            flux_out, err_out = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
        else:
            raise ValueError(f'Output flux unit ({out_flux_units}) not recognised')
    else:
        raise ValueError(f'Input flux unit ({in_flux_units}) not recognised')

    # Apply rescaling factor
    if rescale_factor is not None:
        flux_out, err_out = apply_rescaling_factor(flux_out, err_out, rescale_factor)

    # Apply noise floor
    if snr_limit is not None:
        flux_out, err_out = apply_snr_limit(flux_out, err_out, snr_limit)

    # Convert to desired units
    # -- flux
    # if in_flux_units == "si":
    #     flux, err = flux_si, err_si
    # elif flux_units == "jy":
    #     flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
    #     flux, err = flux_jy, err_jy
    # elif flux_units == "ujy":
    #     flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
    #     flux_ujy, err_ujy = convert_flux_jy_to_ujy(flux_jy, err_jy)
    #     flux, err = flux_ujy, err_ujy
    # elif flux_units == "cgs":
    #     flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
    #     flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
    #     flux, err = flux_cgs, err_cgs
    # elif flux_units == "maggie":
    #     flux_jy, err_jy = convert_flux_si_to_jy(wave_m, flux_si, err_si)
    #     flux_maggie, err_maggie = convert_flux_jy_to_maggie(flux_jy, err_jy)
    #     flux, err = flux_maggie, err_maggie
    # # -- wavelength
    # if wave_units == 'um':
    #     wave_um = convert_wave_m_to_um(wave_m)
    #     wave = wave_um
    # elif wave_units == 'A':
    #     wave_A = convert_wave_m_to_A(wave_m)
    #     wave = wave_A
    # else:
    #     pass

    return wave_out, flux_out, err_out, mask

def load_grating_data(grating_dir, name, grating, filter, version, nod, data_ext, mask_ext, in_wave_units, out_wave_units, in_flux_units, out_flux_units, rescale_factor, snr_limit, return_none=False, return_quantities=False, return_units=False, **extras):

    if return_none:
        return None, None, None, None
    
    # Customise
    # -- version of spectra/reduction
    if version is not None:
        version_str = f"_{version}"
    else:
        version_str = ""
    # -- nod
    if nod is not None:
        nod_str = f"_{nod}"
    else:
        nod_str = ""
    # -- spectrum dimensions
    dim_str = "_1D"
    # -- combine
    extra_str = version_str + nod_str + dim_str

    # Load FITS file
    # spec_name = name + grating + filter + version_str + nod_str + dim_str
    spec_name = f"{name}_{grating}_{filter}{version_str}{nod_str}{dim_str}.fits"
    spec_path = os.path.join(grating_dir, spec_name)
    hdul = fits.open(spec_path)

    # Extract spectral data
    # -- wavelength
    wave_hdu = hdul["WAVELENGTH"]
    wave_in = wave_hdu.data
    # -- flux
    if data_ext is not None:
        flux_hdu = hdul[data_ext]
    else:    
        flux_hdu = hdul["DATA"]
    flux_in = flux_hdu.data
    # -- error
    err_hdu = hdul["ERR"]
    err_in = err_hdu.data
    # -- mask
    if mask_ext is not None:
        mask_hdu = hdul[mask_ext]
        mask = mask_hdu.data.astype(bool)
    else:
        mask = None
        mask = np.full(flux_in.shape, True)  # TODO: add NaN (or like) mask

    # Convert wavelength units
    # -- from SI
    if in_wave_units == "si":
        if out_wave_units == "um":
            wave_out = convert_wave_m_to_um(wave_in)
        elif out_wave_units == "A":
            wave_out = convert_wave_m_to_A(wave_in)
        else:
            raise ValueError(f'Output wave unit ({out_wave_units}) not recognised')
    # -- from microns
    elif in_wave_units == "um":
        if out_wave_units == "A":
            wave_out = convert_wave_um_to_A(wave_in)
        elif out_wave_units == "um":
            wave_out = wave_in
        else:
            raise ValueError(f'Output wave unit ({out_wave_units}) not recognised')
    else:
        raise ValueError(f'Input wave unit ({out_wave_units}) not recognised')
    
    # Convert flux units
    # -- from SI
    if in_flux_units == "si":
        if out_flux_units == "cgs":
            flux_out, err_out = convert_flux_si_to_cgs(flux_in, err_in, cgs_factor=1e-19)
        else:
            raise ValueError(f'Output flux unit ({out_flux_units}) not recognised')
    # -- from uJy
    elif in_flux_units == "ujy":
        if out_flux_units == "ujy":
            flux_out, err_out = flux_in, err_in
        elif out_flux_units == "cgs":
            if in_wave_units == "um":
                wave_m = convert_wave_um_to_m(wave_in)
            flux_jy, err_jy = convert_flux_ujy_to_jy(flux_in, err_in)
            flux_out, err_out = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
        else:
            raise ValueError(f'Output flux unit ({out_flux_units}) not recognised')
    else:
        raise ValueError(f'Input flux unit ({in_flux_units}) not recognised')

    # Apply rescaling factor
    if rescale_factor is not None:
        flux_out, err_out = apply_rescaling_factor(flux_out, err_out, rescale_factor)

    # Apply noise floor
    if snr_limit is not None:
        flux_out, err_out = apply_snr_limit(flux_out, err_out, snr_limit)

    return wave_out, flux_out, err_out, mask