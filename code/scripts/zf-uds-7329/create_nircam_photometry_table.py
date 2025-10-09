import numpy as np
from astropy.table import Table, QTable
import astropy.units as u
import os

# Load filter names
jwst_filter_names = np.array(['f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w'])

# Pivot wavelengths in microns
jwst_filter_pivots = np.array([1.154, 1.501, 1.988, 2.776, 3.565, 4.083, 4.402])

# Photometry data used by Turner et al. (2025)
flux = np.array([25.567, 24.299, 22.826, 22.154, 21.827, 21.616, 21.546])

# Photometry error used (both errors and noise floor)
noise_floor = True
systematic_floor = False  # use 5% noise floor as greater than systematic error
if noise_floor:
    # -- impose noise floor
    noise_frac = 0.05  # 5% floor from Johnson et al. (2021)
    noise_err_mag = 2.5 * np.log10(1 + noise_frac)  # convert fraction to magnitude
    err = np.full_like(flux, np.round(noise_err_mag, decimals=3))  # limit to same decimals places in Glazebrook et al. (2024)
elif systematic_floor:
    # -- use systematic uncertainties
    err = np.array([0.034, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])  # 0.02 systematic error floor from Glazebrook et al. (2024)
else:
    # -- use direct uncertainty
    err = np.array([0.034, 0.009, 0.002, 0.001, 0.001, 0.001, 0.001])

# Mask for photometric data
valid = np.full_like(flux, 1).astype(np.uint8)  # 1 is true

# Build the table
tb = QTable()
tb['FILTER'] = jwst_filter_names
tb['PIVOT'] = jwst_filter_pivots * u.um
tb['DATA'] = flux * u.mag
tb['ERR'] = err * u.mag
tb['VALID'] = valid

# Save as an astropy Table
out_dir = '/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/photometry'
tb_name = 'zf-uds-7329_nircam_photometry.fits'
out_path = os.path.join(out_dir, tb_name)
tb.write(out_path, overwrite=True)