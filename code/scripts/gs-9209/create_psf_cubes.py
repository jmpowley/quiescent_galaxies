import os

import stpsf

# Load NIRSpec object
nrs = stpsf.NIRSpec()
nrs.mode = "IFU"
nrs.disperser = "G235H"
nrs.filter = "F170LP"
print("Band is", nrs.band)

# Create PSF cube
waves = nrs.get_IFU_wavelengths()
cube = nrs.calc_datacube(waves)

# Save cube
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/cube_psfs"
out_name = "stpsf_nrs_ifu_g235h_f170lp_mpsf.fits"
out_path = os.path.join(out_dir, out_name)
cube.writeto(out_path, overwrite=True)
print(f"PSF cube written to {out_path}")