import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from pysersic.results import plot_image
from pysersic import check_input_data

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

def _rebuild_wave_from_header(header):

    try:
        wave = np.arange(header["CRVAL3"], header["CRVAL3"] + (header["NAXIS3"] - 1) * header["CDELT3"], header["CDELT3"])
    except Exception as e:
        return e
    
    return wave

def load_cutout_data(data_dir, data_name, data_ext, centre, width, height, psf_dir, psf_name, psf_ext, snr_limit, **extras):
    """
    Loads NIRCam cutouts
    """

    # Load FITS file
    path = _resolve_path(data_dir, data_name)
    hdul = fits.open(path)

    # Extract image data
    image_in = hdul[data_ext].data
    wht_in = hdul["WHT"].data
    # mask_in = np.zeros(image_in.shape)  # TODO: Change to mask_ext etc
    
    # Create mask
    mask_in = np.zeros_like(image_in)
    subzero = image_in <= 0
    mask_in[subzero] = 1  # mask all negative flux pixels
    # -- apply mask conditions to image
    image_in[subzero] = 1e-5  # small nonzero value

    # Impose SNR limit
    #sig = 0.01/np.sqrt(np.abs(wht)) + 0.1*np.sqrt(np.abs(im))
    sig = (1 / snr_limit) * np.sqrt(np.abs(image_in))
    
    # Extract subregion
    # -- find lower/upper bounds
    ycen, xcen = centre
    halfw = width // 2
    halfh = height // 2
    x0 = xcen - halfw
    x1 = x0 + width
    y0 = ycen - halfh
    y1 = y0 + height
    # -- apply to images
    image_out  = image_in[y0:y1, x0:x1]
    wht_out = wht_in[y0:y1, x0:x1]
    sig_out = sig[y0:y1, x0:x1]
    mask_out= mask_in[y0:y1, x0:x1]

    # load the PSF data
    psf_path = _resolve_path(psf_dir, psf_name)
    psf_hdul = fits.open(psf_path)
    psf_in = fits.getdata(psf_path)
    # -- extract subregion (smaller than data)
    pcen = int(0.5*psf_in.shape[0])
    n_pad = 1
    px0 = pcen - halfw + n_pad
    px1 = px0 + (width - 2*n_pad)
    py0 = pcen - halfh + n_pad
    py1 = py0 + (height - 2*n_pad)
    psf_out = psf_in[py0:py1, px0:px1]
    # psf_crop = psf_in[cen-18:cen+19, cen-18:cen+19]
    # -- normalise
    psf_out /= np.sum(psf_out)  # normalise
    psf_out = psf_out.astype(float)

    return image_out, mask_out, sig_out, psf_out

def load_cube_data(data_dir, data_name, data_ext, wave_from_hdr, in_wave_units, out_wave_units, centre, width, height, psf_dir, psf_name, psf_ext, wave_min, wave_max):

    # TODO: Add in PSF for IFS data?

    # Load FITS file
    path = _resolve_path(data_dir, data_name)
    hdul = fits.open(path)

    # Extract spectral data
    # -- wavelength
    if wave_from_hdr:
        wave_in = _rebuild_wave_from_header(hdul[data_ext].header)
    # -- flux
    if data_ext is not None:
        cube_in = hdul[data_ext].data
    else:
        cube_in = hdul["DATA"].data
    # -- error
    err_in = hdul["ERR"].data

    # Convert wavelength units
    # TODO: Apply conversions

    wave_out = wave_in
    cube_out = cube_in
    err_out = err_in
    nlam, ny, nx = cube_in.shape

    # Extract subregion
    if all(x is not None for x in (centre, height, width)):
        # -- find lower/upper bounds
        ycen, xcen = centre
        halfw = width // 2
        halfh = height // 2
        x0 = xcen - halfw
        x1 = x0 + width
        y0 = ycen - halfh
        y1 = y0 + height
        # -- apply to images
        cube_out = cube_out[:, y0:y1, x0:x1]
        err_out = err_out[:, y0:y1, x0:x1]
        psf_out = err_out = err_out[:, y0:y1, x0:x1]
        nlam, ny, nx = cube_out.shape  # update shape

    # Load PSF data
    if all(x is not None for x in (psf_dir, psf_name, psf_ext)):
        psf_path = _resolve_path(psf_dir, psf_name)
        psf_hdul = fits.open(psf_path)
        psf_in = psf_hdul[psf_ext].data
        # -- extract subregion (smaller than data)
        pcen = int(0.5*psf_in.shape[0])
        n_pad = 1
        px0 = pcen - halfw + n_pad
        px1 = px0 + (width - 2*n_pad)
        py0 = pcen - halfh + n_pad
        py1 = py0 + (height - 2*n_pad)
        psf_out = psf_in[py0:py1, px0:px1]
        # -- normalise along each spectral axis
        psf_norm = psf_out.copy()
        for n in range(nlam):
            psf_norm[n, :, :] = psf_out[n, :, :] / np.sum(psf_out[n, :, :])
        psf_out = psf_norm.astype(float)

    # Crop wavelength axis
    if wave_min is not None and wave_max is not None:
        wave_mask = (wave_out > wave_min) & (wave_out < wave_max)
        wave_out = wave_out[wave_mask]
        cube_out = cube_out[wave_mask, :, :]
        err_out  = err_out[wave_mask, :, :]
        # -- crop PSF (assumes same shape as cube)
        if all(x is not None for x in (psf_dir, psf_name, psf_ext)):
            psf_out = psf_out[wave_mask, :, :]
        nlam, ny, nx = cube_out.shape  # update shape

    # Return depending if PSF supplied or not
    if all(x is not None for x in (psf_dir, psf_name, psf_ext)):
        return wave_out, cube_out, err_out, psf_out
    else:
        return wave_out, cube_out, err_out