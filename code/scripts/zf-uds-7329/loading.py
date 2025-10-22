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
    """Function to load JWST NIRCam photometry from a FITS table

    Path to the FITS file is built as the following:
        /path/to/directory/[name]_nircam_photometry.fits

    Parameters
    ----------
    phot_dir : str
        Directory containing the photometry FITS file.
    name : str
        Name of the object at the start of the file name.
    data_ext : str
        Column name in the photometry table that contains the flux values.
    mask_ext : str
        Column name in the photometry table that contains the mask values. Set to Boolean array of True values if None.
    in_flux_units : str
        Shorthand units of the input flux data (e.g. 'magnitude').
    out_flux_units : str
        Shorthand units of the desired output flux and error data (e.g. 'ujy', 'maggie', 'cgs').
    snr_limit : float
        Maximum signal-to-noise ratio in the output flux data. Performed by calculating the new error as flux / snr_limit. Applies basic check for NaN values and positive error values.
    return_none : bool
        Returns None for all outputs if True. Useful if you want to toggle whether an observation is to be fit in Prospector.
    return_quantities : bool
        *CURRENTLY NOT IMPLEMENTED* Returns flux outputs as `astropy.units.Quantity` objects.
    return_units : bool
        *CURRENTLY NOT IMPLEMENTED* Returns numerical flux outputs plus associated astropy units as extra outputs.
    **extras
        Additional keyword arguments are accepted but ignored.

    Returns
    -------
    sedpy_filters : list
        List of sedpy filter objects loaded via `sedpy.observate.load_filters`. JWST filter names are constructed as `jwst_<FILTER>` from the table FILTER column.
    flux_out : np.ndarray
        Output flux values converted to `out_flux_units`. Returned as a NumPy array.
    err_out : np.ndarray
        Output flux errors converted to `out_flux_units`. Returned as a NumPy array.
    mask_out : np.ndarray
        Boolean mask for each photometric point. Returned as a NumPy boolean array.
    """

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
    mask_in = phot_tb[mask_ext].data.astype(bool)

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

    # Apply noise floor
    if snr_limit is not None:
        flux_out, err_out = apply_snr_limit(flux_out, err_out, snr_limit)

    # TODO: Apply custom masking routine
    mask_out = mask_in

    # Explicitly make NumPy arrays
    flux_out = np.asarray(flux_out)
    err_out = np.asarray(err_out)

    return sedpy_filters, flux_out, err_out, mask_out

def load_prism_data(prism_dir, name, version, nod, data_ext, mask_ext, in_wave_units, out_wave_units, in_flux_units, out_flux_units, rescale_factor, snr_limit, return_none=False, return_quantities=False, return_units=False, **extras):
    """Function to load JWST NIRSpec/Prism data from a FITS file

    Path to the FITS file is built as the following:
        /path/to/directory/[name]_prism_clear[_version][_nod][_dim].fits

    Parameters
    ----------
    prism_dir : str
        Directory of prism spectrum FITS files.
    name : str
        Name of the object at the start of the file name.
    version : str
        Version of the spectrum information in the file name. Can be set to None if no version.
    nod : str
        Nodding pattern information in the file name. Can be set to None if no nodding.
    data_ext : str
        Name of the FITS extension that contains the flux data. Set to 'DATA' if None.
    mask_ext : str
        Name of the FITS extension that contains the mask data. Set to a Boolean array of True values if None.
    in_wave_units : str
        Shorthand units of the input wavelength data ('si' for metres, 'um' for microns).
    out_wave_units : str
        Shorthand units of the output wavelength data ('um' or 'A').
    in_flux_units : str
        Shorthand units of the input flux and error data ('si', 'ujy', ...).
    out_flux_units : str
        Shorthand units of the output flux and error data ('cgs', 'ujy', ...).
    rescale_factor : float
        Factor applied to uniformly rescale the spectrum and error. If None no rescaling is performed.
    snr_limit : float
        Maximum signal-to-noise ratio in the output flux data. Performed by calculating the new error as flux / snr_limit. Applies basic checks for NaN values and positive error values.
    return_none : bool
        Returns None for all outputs if True. Useful if you want to toggle whether an observation is to be fit in Prospector.
    return_quantities : bool
        *CURRENTLY NOT IMPLEMENTED* Returns wavelength/flux outputs as `astropy.units.Quantity` objects.
    return_units : bool
        *CURRENTLY NOT IMPLEMENTED* Returns numerical outputs plus associated astropy units as extra outputs.
    **extras
        Additional keyword arguments are accepted but ignored.

    Returns
    -------
    wave_out : np.ndarray
        Output wavelength data of the prism spectrum in the chosen units. Returned as a NumPy array.
    flux_out : np.ndarray
        Output flux data of the prism spectrum in the chosen units. Returned as a NumPy array.
    err_out : np.ndarray
        Output error data of the prism spectrum in the chosen units. Returned as a NumPy array.
    mask_out : np.ndarray
        Output mask data of the prism spectrum as a Boolean array. Returned as a NumPy boolean array.
    """

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
        mask_in = mask_hdu.data.astype(bool)
    else:
        mask_in = None
        mask_in = np.full(flux_in.shape, True)  # TODO: add NaN (or like) mask

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

    # TODO: Apply custom masking routine
    mask_out = mask_in

    # Explicitly make NumPy arrays
    wave_out = np.asarray(wave_out)
    flux_out = np.asarray(flux_out)
    err_out = np.asarray(err_out)
    mask_out = np.asarray(mask_out)

    return wave_out, flux_out, err_out, mask_out

def load_grating_data(grating_dir, name, grating, filter, version, nod, data_ext, mask_ext, in_wave_units, out_wave_units, in_flux_units, out_flux_units, rescale_factor, snr_limit, return_none=False, return_quantities=False, return_units=False, **extras):
    """Function to load JWST NIRSpec grating data from a FITS file

    Path to the FITS file is built as the following:
        /path/to/directory/[name]_[grating]_[filter][_version][_nod][_dim].fits

    Parameters
    ----------
    grating_dir : str
        Directory of the grating spectrum FITS files.
    name : str
        Name of the object at the start of the file name.
    grating : str
        Grating identifier (e.g. 'g140m', 'g235h').
    filter : str
        Filter identifier associated with the grating.
    version : str
        Version of the spectrum/reduction in the file name. Can be set to None if no version.
    nod : str
        Nodding pattern information in the file name. Can be set to None if no nodding.
    data_ext : str
        Name of the FITS extension that contains the flux data. Set to 'DATA' if None.
    mask_ext : str
        Name of the FITS extension that contains the mask data. Set to Boolean array of True values if None.
    in_wave_units : str
        Shorthand units of the input wavelength data ('si' for metres, 'um' for microns).
    out_wave_units : str
        Shorthand units of the output wavelength data ('um' or 'A').
    in_flux_units : str
        Shorthand units of the input flux and error data ('si', 'ujy', ...).
    out_flux_units : str
        Shorthand units of the output flux and error data ('cgs', 'ujy', ...).
    rescale_factor : float
        Factor applied to uniformly rescale the spectrum and error.
    snr_limit : float
        Maximum signal-to-noise ratio in the output flux data. Performed by calculating the new error as flux / snr_limit. Applies basic check for NaN values and positive error values.
    return_none : bool
        Returns None for all outputs if True. Useful if you want to toggle whether an observation is to be fit in Prospector.
    return_quantities : bool
        *CURRENTLY NOT IMPLEMENTED* Returns wavelength/flux outputs as `astropy.units.Quantity` objects.
    return_units : bool
        *CURRENTLY NOT IMPLEMENTED* Returns numerical outputs with associated astropy units as extra outputs.
    **extras
        Additional keyword arguments are accepted but ignored.

    Returns
    -------
    wave_out : np.ndarray
        Output wavelength data of the grating spectrum in the chosen units. Returned as a NumPy array.
    flux_out : np.ndarray
        Output flux data of the grating spectrum in the chosen units. Returned as a NumPy array.
    err_out : np.ndarray
        Output error data of the grating spectrum in the chosen units. Returned as a NumPy array.
    mask_out : np.ndarray
        Output mask data of the grating spectrum as a Boolean array. Returned as a NumPy boolean array.
    """

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
        mask_in = mask_hdu.data.astype(bool)
    else:
        mask_in = None
        mask_in = np.full(flux_in.shape, True)  # TODO: add NaN (or like) mask

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

    # TODO: Apply custom masking routine
    mask_out = mask_in

    # Explicitly make NumPy arrays
    wave_out = np.asarray(wave_out)
    flux_out = np.asarray(flux_out)
    err_out = np.asarray(err_out)
    mask_out = np.asarray(mask_out)

    return wave_out, flux_out, err_out, mask_out