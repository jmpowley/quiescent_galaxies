import numpy as np
import matplotlib.pyplot as plt

from loading import load_prism_data, load_dispersion_data
from preprocessing import crop_bad_spectral_resolution

from conversions import convert_wave_um_to_A

obs_kwargs = {

    "prism_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "data_name" : "007329_prism_clear_v3.1_extr5_1D.fits",
        "data_ext" : "DATA",
        "mask_dir" : None,
        "mask_name" : None,
        "mask_ext" : None,
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        # "disp_name" : "uds7329_nirspec_prism_disp.fits",
        "disp_name" : "jwst_nirspec_prism_disp.fits",
        "in_wave_units" : "si",
        "out_wave_units" : "A",
        "in_flux_units" : "si",
        "out_flux_units" : "cgs",
        "rescale_factor" : 1.86422,
        "snr_limit" : 20.0,
        "prefix" : "prism",
        "add_jitter" : True,
        "include_outliers" : True,
        "fit_obs" : True,
    },
}

# Load spectral data
prism_wave, prism_flux, prism_err = load_prism_data(**obs_kwargs["prism_kwargs"])

# Load resolution data
prism_disp = load_dispersion_data(**obs_kwargs["prism_kwargs"])

# Redshift information
zred = 3.2

# NOTE: Added in to fix library resolution issue. Considered (at least in part) temporary
# -- define prism mask for crop function
prism_mask = np.ones_like(prism_wave).astype(bool)
# -- apply crop to prism
prism_wave_crop, prism_flux_crop, prism_err_crop, prism_mask_crop, prism_res_crop = crop_bad_spectral_resolution(prism_wave, 
                                                                                                                prism_flux, 
                                                                                                                prism_err,
                                                                                                                prism_mask,
                                                                                                                prism_disp,
                                                                                                                zred=zred,
                                                                                                                wave_rest_lo=2000,
                                                                                                                wave_rest_hi=7000,
                                                                                                                )

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# -- plot original prism data
ax.plot(prism_wave, prism_flux, color="black", label="Prism")
# ax.fill_between(prism_wave, prism_flux-prism_err, prism_flux+prism_err, color="C0", alpha=0.3)
# -- plot cropped data
ax.plot(prism_wave_crop, prism_flux_crop, color="orange", label="Cropped")
# ax.fill_between(prism_wave_crop, prism_flux_crop-prism_err_crop, prism_flux_crop+prism_err_crop, color="orange", alpha=0.3)
# -- prettify
ax.set_xlabel(r"Wavelength [Å]", size=18)
ax.set_ylabel(r'$f_\lambda~/~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$', size=18)
ax.set_xticks(convert_wave_um_to_A(np.arange(1, 6)))
if zred is not None:
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    top_ticks_rest = convert_wave_um_to_A(np.arange(0.2, 1.4, 0.2))
    top_ticks_obs = top_ticks_rest * (1 + zred)  # map to bottom scale
    ax_top.set_xticks(top_ticks_obs)
    ax_top.set_xticklabels([f"{t:.1f}" for t in top_ticks_rest])

ax.legend()

plt.show()