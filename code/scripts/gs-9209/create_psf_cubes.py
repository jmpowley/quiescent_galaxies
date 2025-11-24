import os

import stpsf

from scalpel.loading import load_cube_data

# Load NIRSpec object
nrs = stpsf.NIRSpec()
nrs.mode = "IFU"
nrs.disperser = "G235H"
nrs.filter = "F170LP"
print("Band is", nrs.band)

# Create PSF cube
nrs_waves = nrs.get_IFU_wavelengths()

# Load wavelengths from cube itself
cube_kwargs = {
    "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/gs-9209/cubes",
    "data_name" : "GS9209-4.67_jw3659_px0.05_dr_ODfde95.0_VSC_AMRC_mMSA_m2ff_xy96_CTX1236_v1.15.1_g235h-f170lp_cgs_s3d.fits",
    "data_ext" : "SCI",
    "wave_from_hdr" : True,
    "in_wave_units" : None,
    "out_wave_units" : None,
    "centre" : (44, 40),
    "width" : 20,
    "height" : 20,
    "psf_dir" : None,
    "psf_name" : None,
    "psf_ext" : None,
    "wave_min" : None,
    "wave_max" : None,
}
cube_waves,_, _ = load_cube_data(**cube_kwargs)

# Calculate PSF data cube from set of wavelengths
# psf_cube = nrs.calc_datacube(cube_waves)
psf_cube = nrs.calc_datacube(nrs_waves)

# Save cube
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/cube_psfs"
out_name = "stpsf_nrs_ifu_g235h_f170lp_mpsf.fits"
out_path = os.path.join(out_dir, out_name)
psf_cube.writeto(out_path, overwrite=True)
print(f"PSF cube written to {out_path}")