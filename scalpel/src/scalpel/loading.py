import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from pysersic.results import plot_image
from pysersic import check_input_data
from pysersic.priors import SourceProperties

def load_cutout_data(data_dir, data_name, data_ext, centre, xrange, yrange, psf_dir, psf_name, psf_ext, snr_limit, plot=False, **extras):
    """
    Loads NIRCam cutouts
    """

    # Load FITS file
    path = os.path.join(data_dir, data_name)  # TODO: Change to _resolve_path
    hdul = fits.open(path)

    # Extract image data
    im_in = hdul[data_ext].data
    wht_in = hdul["WHT"].data
    mask_in = np.zeros(im_in.shape)  # TODO: Change to mask_ext etc

    # Impose SNR limit
    #sig = 0.01/np.sqrt(np.abs(wht)) + 0.1*np.sqrt(np.abs(im))
    sig = (1 / snr_limit) * np.sqrt(np.abs(im_in))
    
    # load the PSF data
    psf_path = os.path.join(psf_dir, psf_name)  # TODO: Change to/add in _resolve_path function
    psf_hdul = fits.open(psf_path)
    psf_in = fits.getdata(psf_path)
    # -- extract subregion (smaller than data)
    cen = int(0.5*psf_in.shape[0])
    psf_crop = psf_in[cen-18:cen+19, cen-18:cen+19]
    # -- normalise
    psf_crop /= np.sum(psf_crop)  # normalise
    psf_crop = psf_crop.astype(float)

    # Extract subregion (use ycen, xcen to match numpy [row, col])
    ycen, xcen = centre
    im_crop  = im_in[ycen-21:ycen+22, xcen-21:xcen+22]
    wht_crop = wht_in[ycen-21:ycen+22, xcen-21:xcen+22]
    sig_crop = sig[ycen-21:ycen+22, xcen-21:xcen+22]
    mask_crop = mask_in[ycen-21:ycen+22, xcen-21:xcen+22]

    # Plot data
    if plot:
        fig, ax = plot_image(im_crop, mask_crop, sig_crop, psf_crop)
        plt.show()

    # Check data
    if check_input_data(data=im_crop, rms=sig_crop, psf=psf_crop, mask=mask_crop):
        print("Data looks good!")
    else:
        print("Data looks bad!")

    return im_crop, mask_crop, sig_crop, psf_crop

def set_priors(im, mask, profile_type, sky_type, prior_dict):

    # Generate priors from image
    props = SourceProperties(im, mask=mask) # Optional mask
    prior = props.generate_prior(profile_type=profile_type, sky_type=sky_type)

    # Set uniform priors from dict
    for key, val in prior_dict.items():
        lo, hi = val
        prior.set_uniform_prior(key, lo, hi)

    return prior