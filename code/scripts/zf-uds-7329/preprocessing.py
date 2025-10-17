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