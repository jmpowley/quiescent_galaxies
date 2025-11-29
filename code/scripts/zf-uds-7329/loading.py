from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from astropy.io import fits
from astropy.table import Table
from astropy import constants

from sedpy.observate import load_filters

from conversions import convert_wave, convert_flux
from preprocessing import apply_snr_limit, apply_rescaling_factor

# ----------------------
# Functions to load data
# ----------------------

def _resolve_path(file_dir: str, file_name: str) -> Path:
    """
    Construct a `pathlib.Path` from a directory and file name.

    Parameters
    ----------
    file_dir : str
        Directory containing the file. If `None`, a ValueError is raised.
    file_name : str
        File name (may include subdirectories or suffix).

    Returns
    -------
    pathlib.Path
        Path object pointing to `file_dir / file_name`.

    Raises
    ------
    ValueError
        If `file_dir` is `None`.
    """
    if file_dir is None:
        raise ValueError("Either path or data_dir must be provided")
    return Path(file_dir) / file_name


def load_photometry_data(data_dir : str, data_name : str, data_ext : str, in_flux_units : str, out_flux_units : str, snr_limit : float, 
                         return_none : bool = False, return_quantities : bool = False, return_units : bool = False, **extras):
    """
    Load JWST NIRCam photometry from a FITS table and convert flux units.

    The FITS file path is constructed as ``Path(data_dir) / data_name`` and the table
    is read with :class:`astropy.table.Table`.

    Parameters
    ----------
    data_dir : str or None
        Directory containing the photometry FITS file.
    data_name : str or None
        FITS file name (including any suffix). If `None` a ValueError will be raised by
        `_resolve_path`.
    data_ext : str
        Column name in the photometry table that contains the flux values. If the
        provided name is `None` the code falls back to the "DATA" column.
    in_flux_units : str
        Input flux units shorthand (e.g. ``'magnitude'``).
    out_flux_units : str
        Desired output flux units shorthand (e.g. ``'ujy'``, ``'maggie'``, ``'cgs'``).
    snr_limit : float
        Maximum allowed signal-to-noise; if provided the function will call
        :func:`apply_snr_limit` to enforce a noise floor.
    return_none : bool, optional
        If True, return ``(None, None, None)`` immediately. Default is False.
    return_quantities : bool, optional
        *Currently not implemented.* If True, would return :class:`astropy.units.Quantity`
        arrays. Default is False.
    return_units : bool, optional
        *Currently not implemented.* If True, would return numerical arrays plus units.
        Default is False.
    **extras
        Additional keyword arguments are accepted but ignored.

    Returns
    -------
    sedpy_filters : list
        List of sedpy filter objects loaded via ``sedpy.observate.load_filters``. JWST
        filter names are constructed as ``"jwst_<FILTER>"`` from the table FILTER column.
    flux_out : numpy.ndarray
        Flux values converted to ``out_flux_units``. Shape (N,) for N photometric points.
    err_out : numpy.ndarray
        Flux errors converted to ``out_flux_units``. Shape (N,).

    Raises
    ------
    ValueError
        If `out_flux_units` or `in_flux_units` are not recognised.
    OSError / FileNotFoundError
        If the FITS file does not exist or cannot be read by Astropy.
    """
    if return_none:
        return None, None, None

    # Load table
    path = _resolve_path(data_dir, data_name)
    tb = Table.read(path)

    # Access photometry data
    filters = tb["FILTER"].tolist()
    wave_in = tb["WAVEFF"].data
    if data_ext is not None:
        flux_in = tb[data_ext].data
    else:
        flux_in = tb["DATA"].data
    err_in = tb["ERR"].data

    # Load JWST filters
    jwst_filters = ([f"jwst_{filt}" for filt in filters])
    sedpy_filters = load_filters(jwst_filters)

    # Convert units
    flux_out, err_out = convert_flux(flux_in, err_in, in_flux_units, out_flux_units, wave_in, in_wave_units="um")

    # Apply noise floor
    if snr_limit is not None:
        flux_out, err_out = apply_snr_limit(flux_out, err_out, snr_limit)

    # Explicitly make NumPy arrays
    flux_out = np.asarray(flux_out)
    err_out = np.asarray(err_out)

    return sedpy_filters, flux_out, err_out

def load_prism_data(data_dir : str, data_name : str, data_ext : str, in_wave_units : str, out_wave_units : str, in_flux_units : str, out_flux_units : str, rescale_factor : float, snr_limit : float, cgs_factor : float = None,
                    return_none : bool = False, return_quantities : bool = False, return_units : bool = False, **extras):
    """
    Load a JWST NIRSpec/Prism 1D spectrum from a FITS file and convert units.

    The FITS path is built as ``Path(data_dir) / data_name``. The function expects the
    FITS file to contain HDUs named "WAVELENGTH", "DATA" (or a custom `data_ext`), and "ERR".

    Parameters
    ----------
    data_dir : str
        Directory containing the prism FITS file.
    data_name : str
        FITS file name (including suffix).
    data_ext : str
        Name of the FITS extension that contains the flux data. If `None`, "DATA" is used.
    in_wave_units : str
        Shorthand for input wavelength units: ``'si'`` (metres) or ``'um'`` (microns).
    out_wave_units : str
        Desired output wavelength units: e.g. ``'um'`` or ``'A'``.
    in_flux_units : str
        Shorthand for input flux units (e.g. ``'si'``, ``'ujy'``).
    out_flux_units : str
        Desired output flux units (e.g. ``'cgs'``, ``'maggie'``, ``'ujy'``).
    rescale_factor : float
        Multiplicative factor applied to flux and error (if not `None`).
    snr_limit : float
        Maximum allowed signal-to-noise; if provided the function will call
        :func:`apply_snr_limit`.
    return_none : bool, optional
        If True, return ``(None, None, None)`` immediately. Default is False.
    return_quantities : bool, optional
        *Currently not implemented.* Default is False.
    return_units : bool, optional
        *Currently not implemented.* Default is False.
    **extras
        Additional keyword arguments are accepted but ignored.

    Returns
    -------
    wave_out : numpy.ndarray
        Wavelength array converted to `out_wave_units`. Shape (M,) for M spectral pixels.
    flux_out : numpy.ndarray
        Flux array converted to `out_flux_units`. Shape (M,).
    err_out : numpy.ndarray
        Error array converted to `out_flux_units`. Shape (M,).

    Raises
    ------
    ValueError
        If `in_wave_units`, `in_flux_units`, or output units are not recognised.
    OSError / FileNotFoundError
        If the FITS file cannot be opened.
    """
    if return_none:
        return None, None, None

    # Load FITS file    
    path = _resolve_path(data_dir, data_name)
    hdul = fits.open(path)

    # Extract spectral data
    # -- wavelength
    wave_in = hdul["WAVELENGTH"].data
    # -- flux
    if data_ext is not None:
        flux_in = hdul[data_ext].data
    else:    
        flux_in = hdul["DATA"].data
    # -- error
    err_in = hdul["ERR"].data

    # Convert units
    wave_out = convert_wave(wave_in, in_wave_units, out_wave_units)
    flux_out, err_out = convert_flux(flux_in, err_in, in_flux_units, out_flux_units, wave_in, in_wave_units, cgs_factor=cgs_factor)

    # Apply rescaling factor
    if rescale_factor is not None:
        flux_out, err_out = apply_rescaling_factor(flux_out, err_out, rescale_factor)

    # Apply noise floor
    if snr_limit is not None:
        flux_out, err_out = apply_snr_limit(flux_out, err_out, snr_limit)

    # Explicitly make NumPy arrays
    wave_out = np.asarray(wave_out)
    flux_out = np.asarray(flux_out)
    err_out = np.asarray(err_out)

    return wave_out, flux_out, err_out

def load_grating_data(data_dir : str, data_name : str, data_ext : str, in_wave_units : str, out_wave_units : str, in_flux_units : str, out_flux_units : str, rescale_factor : float, snr_limit : float, cgs_factor : float = None,
                      return_none : bool = False, return_quantities : bool = False, return_units : bool = False, **extras):
    """
    Load a JWST NIRSpec grating 1D spectrum from a FITS file and convert units.

    The FITS path is constructed as ``Path(data_dir) / data_name`` and the function
    expects HDUs named "WAVELENGTH", "DATA" (or custom `data_ext`) and "ERR".

    Parameters
    ----------
    data_dir : str
        Directory containing the grating FITS file.
    data_name : str
        FITS file name (including suffix).
    data_ext : str
        Name of the FITS extension that contains the flux data. If `None`, "DATA" is used.
    in_wave_units : str
        Shorthand for input wavelength units (``'si'`` or ``'um'``).
    out_wave_units : str
        Desired output wavelength units (e.g. ``'A'`` or ``'um'``).
    in_flux_units : str
        Shorthand for input flux units (e.g. ``'si'``, ``'ujy'``).
    out_flux_units : str
        Desired output flux units (e.g. ``'cgs'``, ``'maggie'``, ``'ujy'``).
    rescale_factor : float
        Multiplicative factor applied to flux and error (if not `None`).
    snr_limit : float
        Maximum allowed signal-to-noise; see :func:`apply_snr_limit`.
    return_none : bool, optional
        If True, return ``(None, None, None)`` immediately. Default is False.
    return_quantities : bool, optional
        *Currently not implemented.* Default is False.
    return_units : bool, optional
        *Currently not implemented.* Default is False.
    **extras
        Additional keyword arguments are accepted but ignored.

    Returns
    -------
    wave_out : numpy.ndarray
        Wavelength array converted to `out_wave_units`. Shape (M,).
    flux_out : numpy.ndarray
        Flux array converted to `out_flux_units`. Shape (M,).
    err_out : numpy.ndarray
        Error array converted to `out_flux_units`. Shape (M,).

    Raises
    ------
    ValueError
        If unknown input or output units are provided.
    OSError / FileNotFoundError
        If the FITS file cannot be opened.
    """
    if return_none:
        return None, None, None

    # Load FITS file    
    path = _resolve_path(data_dir, data_name)
    hdul = fits.open(path)

    # Extract spectral data
    # -- wavelength
    wave_in = hdul["WAVELENGTH"].data
    # -- flux
    if data_ext is not None:
        flux_in = hdul[data_ext].data
    else:    
        flux_in = hdul["DATA"].data
    # -- error
    err_in = hdul["ERR"].data

    # Convert units
    wave_out = convert_wave(wave_in, in_wave_units, out_wave_units)
    flux_out, err_out = convert_flux(flux_in, err_in, in_flux_units, out_flux_units, wave_in, in_wave_units, cgs_factor=cgs_factor)

    # Apply rescaling factor
    if rescale_factor is not None:
        flux_out, err_out = apply_rescaling_factor(flux_out, err_out, rescale_factor)

    # Apply noise floor
    if snr_limit is not None:
        flux_out, err_out = apply_snr_limit(flux_out, err_out, snr_limit)

    # Explicitly make NumPy arrays
    wave_out = np.asarray(wave_out)
    flux_out = np.asarray(flux_out)
    err_out = np.asarray(err_out)

    return wave_out, flux_out, err_out

def load_dispersion_data(disp_dir : str, disp_name : str, data_dir : str, data_name : str, in_wave_units : str, return_spec_res : bool = False, return_wave : bool = False, **extras):
    """
    Load a spectral resolution table and interpolate resolution onto data wavelengths.

    The dispersion table is read from ``Path(disp_dir) / disp_name`` and is expected to
    contain columns "R" and "WAVELENGTH" (in microns). The data wavelengths are read
    from the supplied spectrum FITS file (``Path(data_dir) / data_name``) from the
    "WAVELENGTH" HDU.

    Parameters
    ----------
    disp_dir : str
        Directory containing the dispersion (resolution) table.
    disp_name : str
        Dispersion table FITS file name.
    data_dir : str
        Directory containing the spectral FITS file whose wavelengths will be used.
    data_name : str
        Spectral FITS file name
    in_wave_units : str
        Input wavelength units for the spectral FITS file: ``'si'`` (metres) or ``'um'`` (microns).
    **extras
        Additional keyword arguments are accepted but ignored.

    Returns
    -------
    sigma_interp_kms : numpy.ndarray
        Interpolated velocity dispersion (sigma) in km/s at the data wavelengths. Shape (M,).

    Raises
    ------
    ValueError
        If `in_wave_units` is not recognised.
    OSError / FileNotFoundError
        If either the dispersion table or data FITS file cannot be opened.
    """
    # Load dispersion table
    disp_path = _resolve_path(disp_dir, disp_name)
    tb = Table.read(disp_path)
    spec_res = tb["R"].data
    disp_wave_um = tb["WAVELENGTH"].data

    # Optionally return spectral resolution
    if return_spec_res and return_wave:
        return spec_res, disp_wave_um
    if return_spec_res:
        return spec_res

    # Load wavelength data
    wave_path = _resolve_path(data_dir, data_name)  # use same directory as load_x_data
    hdul = fits.open(wave_path)
    data_wave = hdul["WAVELENGTH"].data

    # Convert dispersion wavelengths to angstroms
    disp_wave_A = convert_wave(disp_wave_um, in_wave_units="um", out_wave_units="A")
    data_wave_A = convert_wave(data_wave, in_wave_units, out_wave_units="A")

    # Convert dispersion from dimensionless to km/s
    c_kms = constants.c.to("km/s").value
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    sigma_kms = c_kms / spec_res / fwhm_factor

    # Interpolate for data wavelengths using dispersion curve
    interp = interp1d(disp_wave_A, sigma_kms, bounds_error=False, fill_value='extrapolate')
    sigma_interp_kms = interp(data_wave_A)

    return sigma_interp_kms

def load_mask_data(mask_dir : str, mask_name : str, mask_ext : str, convert_to_bool : bool = True, **extras):
    """
    Load a mask HDU from a FITS file and optionally convert to boolean.

    Parameters
    ----------
    mask_dir : str
        Directory containing the mask FITS file.
    mask_name : str
        Mask FITS file name (including suffix).
    mask_ext : str
        Name of the HDU/extension that contains the mask array.
    convert_to_bool : bool, optional
        If True convert the returned mask to boolean with ``mask.astype(bool)``.
        Default is True.

    Returns
    -------
    mask : numpy.ndarray
        The mask array read from the FITS file. If `convert_to_bool` is True the
        returned array has dtype ``bool``.

    Raises
    ------
    OSError / FileNotFoundError
        If the FITS file cannot be opened.
    KeyError
        If `mask_ext` is not present in the FITS file.
    """
    # Load FITS file
    path = _resolve_path(mask_dir, mask_name)
    hdul = fits.open(path)

    # Load mask data
    mask = hdul[mask_ext].data
    mask = np.asarray(mask)

    # Optionally convert to Boolean array
    if convert_to_bool:
        mask = mask.astype(bool)

    return mask