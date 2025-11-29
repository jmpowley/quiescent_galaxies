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

def crop_bad_spectral_resolution(wave_obs, flux, uncertainty, mask, sigma, zred, wave_rest_lo=None, wave_rest_hi=None, good_res=False, lsf_smooth=False):
    
    #  Define wavelength limits where the template resolution is too low for the model
    if good_res:
        wave_rest_lo = 2000 #2500
        wave_rest_hi = 12000 #7000
    elif lsf_smooth:
        wave_rest_lo = 2000 #2400
        wave_rest_hi = 12000 #7000
    elif wave_rest_lo is not None and wave_rest_hi is not None:
        wave_rest_lo = wave_rest_lo
        wave_rest_hi = wave_rest_hi
    else:
        raise ValueError("Error. No valid limits chosen.")

    # Cut into good range by 100 Angstroms each side (plus extra) to account for padding in prospector
    wave_pad = 100
    extra = 10  # just in case
    wave_rest_lo += (wave_pad + extra)
    wave_rest_hi -= (wave_pad + extra)

    # Create good spectral resolution mask based on rest-frame wavelength limits
    wave_rest = wave_obs / (1 + zred)
    good_spec_res = (
        (wave_rest > wave_rest_lo) &
        (wave_rest < wave_rest_hi)
        )
    
    # Create new arrays
    wave_good = wave_obs[good_spec_res]
    flux_good = flux[good_spec_res]
    unc_good = uncertainty[good_spec_res]
    mask_good = mask[good_spec_res]
    sigma_good = sigma[good_spec_res]
    
    return wave_good, flux_good, unc_good, mask_good, sigma_good