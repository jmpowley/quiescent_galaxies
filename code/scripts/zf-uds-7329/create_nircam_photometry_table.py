import numpy as np
from astropy.table import Table
import astropy.units as u
import os

# Load filter names
jwst_filter_names = np.array(['f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w'])

# Photometry data used by Turner et al. (2025)
phot = np.array([25.567, 24.299, 22.826, 22.154, 21.827, 21.616, 21.546])

# Photometry error used (both errors and noise floor)
noise_floor = True
systematic_floor = False  # use 5% noise floor as greater than systematic error
if noise_floor:
    # -- impose noise floor
    noise_frac = 0.05  # 5% floor from Johnson et al. (2021)
    noise_err_mag = 2.5 * np.log10(1 + noise_frac)  # convert fraction to magnitude
    phot_err = np.full_like(phot, noise_err_mag)
elif systematic_floor:
    # -- use systematic uncertainties
    phot_err = np.array([0.034, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])  # 0.02 systematic error floor from Glazebrook et al. (2024)
else:
    # -- use direct uncertainty
    phot_err = np.array([0.034, 0.009, 0.002, 0.001, 0.001, 0.001, 0.001])

# Build the table
tb = Table()
tb['FILTER'] = jwst_filter_names
tb['DATA'] = phot * u.mag  # assign magnitude units
tb['ERR'] = phot_err * u.mag

# Save as an astropy Table
out_dir = '/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/photometry'
tb_name = 'zf-uds-7329_nircam_photometry.fits'
out_path = os.path.join(out_dir, tb_name)
tb.write(out_path, overwrite=True)