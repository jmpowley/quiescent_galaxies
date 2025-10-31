import numpy as np

def apply_snr_limit(flux, err, snr_limit):

    # Calculate SNR
    valid = (err > 0) & np.isfinite(err) & np.isfinite(flux)
    snr = np.zeros_like(flux)
    snr[valid] = flux[valid] / err[valid]
    
    # Identify data with SNR above limit
    above_limit = valid & (snr > snr_limit)

    # Impose noise floor
    err_min = np.abs(flux) / float(snr_limit)  # minimum error based on SNR
    err_new  = err.copy()
    err_new[above_limit] = err_min[above_limit]

    return flux, err_new

def apply_rescaling_factor(flux, err, rescale_factor):
    """Apply a factor to rescale to spectrum to match the observed photometry
    """
    
    # Apply uniform rescaling factor
    flux_scale = flux * rescale_factor
    err_scale = err * rescale_factor

    return flux_scale, err_scale

def crop_bad_spectral_resolution(wavelength, flux, uncertainty, mask, resolution, zred, wave_lo=None, wave_hi=None, good_res=False, lsf_smooth=False):
    
    #  Define wavelength limits where the template resolution is too low for the model
    if good_res:
        wave_lo = 2000 #2500
        wave_hi = 12000 #7000
    elif lsf_smooth:
        wave_lo = 2000 #2400
        wave_hi = 12000 #7000
    elif wave_lo is not None and wave_hi is not None:
        wave_lo = wave_lo
        wave_hi = wave_hi
    else:
        raise ValueError("Error. No valid limits chosen.")
    
    print("wave_lo:", wave_lo)
    print("wave_hi:", wave_hi)

    # Cut into good range by 100 Angstroms each side (plus extra) to account for padding in prospector
    wave_pad = 100
    extra = 10  # just in case
    wave_lo += (wave_pad + extra)
    wave_hi -= (wave_pad + extra)

    # Create good spectral resolution mask based on rest-frame wavelength limits
    good_spec_res = (
        (wavelength / (1+zred) > wave_lo) &
        (wavelength / (1+zred) < wave_hi)
        )
    
    # Create new arrays
    new_wavelength = wavelength[good_spec_res]
    new_flux = flux[good_spec_res]
    new_uncertainty = uncertainty[good_spec_res]
    new_mask = mask[good_spec_res]
    new_resolution = resolution[good_spec_res]
    
    return new_wavelength, new_flux, new_uncertainty, new_mask, new_resolution