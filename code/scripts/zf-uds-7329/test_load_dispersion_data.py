import numpy as np
import matplotlib.pyplot as plt

from loading import load_prism_data, load_grating_data, load_dispersion_data

obs_kwargs = {

    "phot_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/photometry",
        "data_name" : "007329_nircam_photometry.fits",
        "data_ext" : "DATA",
        "in_flux_units" : "magnitude",
        "out_flux_units" : "maggie",
        "snr_limit" : 20.0,
        "prefix" : "phot",
        "add_jitter" : True,
        "include_outliers" : True,
        "fit_obs" : True,
    },

    "prism_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "data_name" : "007329_prism_clear_v3.1_extr5_1D.fits",
        "data_ext" : "DATA",
        "mask_dir" : None,
        "mask_name" : None,
        "mask_ext" : None,
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "uds7329_nirspec_prism_disp.fits",
        # "disp_name" : "jwst_nirspec_prism_disp.fits",
        "in_wave_units" : "si",
        "out_wave_units" : "A",
        "in_flux_units" : "si",
        "out_flux_units" : "maggie",
        "rescale_factor" : 1.86422,
        "snr_limit" : 20.0,
        "prefix" : "prism",
        "add_jitter" : True,
        "include_outliers" : True,
        "fit_obs" : True,
    },

    "grat1_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "data_name" : "007329_g140m_f100lp_1D.fits",
        "data_ext" : "DATA",
        "mask_dir" : None,
        "mask_name" : None,
        "mask_ext" : None,
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g140m_disp.fits",
        "in_wave_units" : "um",
        "out_wave_units" : "A",
        "in_flux_units" : "ujy",
        "out_flux_units" : "maggie",
        "rescale_factor" : 1.86422,
        "snr_limit" : 20.0,
        "prefix" : "g140m",
        "add_jitter" : True,
        "include_outliers" : True,
        "fit_obs" : False,
    },

    "grat2_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "data_name" : "007329_g235m_f170lp_1D.fits",
        "data_ext" : "DATA",
        "mask_dir" : None,
        "mask_name" : None,
        "mask_ext" : None,
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g235m_disp.fits",
        "in_wave_units" : "um",
        "out_wave_units" : "A",
        "in_flux_units" : "ujy",
        "out_flux_units" : "maggie",
        "rescale_factor" : 1.86422,
        "snr_limit" : 20,
        "return_none" : False,
        "prefix" : "g235m",
        "add_jitter" : True,
        "include_outliers" : True,
        "fit_obs" : False,
    },

    "grat3_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "data_name" : "007329_g395m_f290lp_1D.fits",
        "data_ext" : "DATA",
        "mask_dir" : None,
        "mask_name" : None,
        "mask_ext" : None,
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g140m_disp.fits",
        "in_wave_units" : "um",
        "out_wave_units" : "A",
        "in_flux_units" : "ujy",
        "out_flux_units" : "maggie",
        "rescale_factor" : 1.86422,
        "snr_limit" : 20.0,
        "prefix" : "g395m",
        "add_jitter" : True,
        "include_outliers" : True,
        "fit_obs" : False,
    },
}

# Load spectral data
prism_wave, prism_flux, prism_err = load_prism_data(**obs_kwargs["prism_kwargs"])
grat1_wave, grat1_flux, grat1_err = load_grating_data(**obs_kwargs["grat1_kwargs"])
grat2_wave, grat2_flux, grat2_err = load_grating_data(**obs_kwargs["grat2_kwargs"])
grat3_wave, grat3_flux, grat3_err = load_grating_data(**obs_kwargs["grat3_kwargs"])

# Load resolution data
prism_disp = load_dispersion_data(**obs_kwargs["prism_kwargs"])
grat1_disp = load_dispersion_data(**obs_kwargs["grat1_kwargs"])
grat2_disp = load_dispersion_data(**obs_kwargs["grat2_kwargs"])
grat3_disp = load_dispersion_data(**obs_kwargs["grat3_kwargs"])

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# -- plot dispersion curves
ax.plot(prism_wave, prism_disp, label="Prism")
ax.plot(grat1_wave, grat1_disp, label="G140M")
ax.plot(grat2_wave, grat2_disp, label="G235M")
ax.plot(grat3_wave, grat3_disp, label="G395M")
# -- prettify
ax.set_xlabel(r"Wavelength [Å]", size=18)
ax.set_ylabel(r"Instrument Dispersion [km s$^{-1}$]", size=18)
# ax.set_yscale('log')
ax.legend()

plt.show()
