import os

import numpy as np
import matplotlib.pyplot as plt

import prospect.io.read_results as reader

from postprocessing import load_build_model_from_string
from conversions import convert_wave_A_to_um, convert_flux_maggie_to_cgs, convert_wave_A_to_m
from loading import load_photometry_data, load_prism_data
from mocking import build_mock_obs, predict_mock_obs

def plot_mock_prediction(obs, pred, lines=None, z=None, fig_dir=None, fig_name=None, savefig=False):

    # Load wavelength data from observations
    phot_obs, prism_obs = obs
    # -- photometry
    phot_wave_A = phot_obs.wavelength
    phot_wave_m = convert_wave_A_to_m(phot_wave_A)
    phot_wave_um = convert_wave_A_to_um(phot_wave_A)
    # -- prism
    prism_wave_A = prism_obs.wavelength
    prism_wave_m = convert_wave_A_to_m(prism_wave_A)
    prism_wave_um = convert_wave_A_to_um(prism_wave_A)

    # Load predicted flux data
    phot_pred_flux_maggie, prism_pred_flux_maggie = pred

    # Convert predicted data to plot
    # -- photometry
    phot_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(phot_pred_flux_maggie, err_maggie=np.nan, wave_m=phot_wave_m, cgs_factor=1e-19)
    # -- prism
    prism_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(prism_pred_flux_maggie, err_maggie=np.nan, wave_m=prism_wave_m, cgs_factor=1e-19)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot data and model predictions
    # -- photometry
    ax.scatter(phot_wave_um, phot_pred_flux_cgs, 
               color='orange', marker='o', label='Mock Photometry')
    # -- prism
    ax.step(prism_wave_um, prism_pred_flux_cgs, 
            color='blue', label='Mock Prism', where='mid')

    # Plot emission lines
    if lines is not None:
        for (stri, wl) in lines.items():
            ax.axvline(wl, color='gray', ls="--")
            ax.text(wl+0.01, 0.1, stri, color='gray', ha='left')

    # Prettify
    # -- limits
    ax.set_xlim(0.9, 5.1)
    ax.set_ylim(0, 2.5)
    # ax_res.set_ylim(-0.51, 0.51)
    # -- axis ticks
    ax.set_xticks(np.arange(1, 6))
    if z is not None:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        top_ticks_rest = np.arange(0.2, 1.4, 0.2)
        top_ticks_obs = top_ticks_rest * (1 + z)  # map to bottom scale
        ax_top.set_xticks(top_ticks_obs)
        ax_top.set_xticklabels([f"{t:.1f}" for t in top_ticks_rest])
    # -- labels
    ax.set_xlabel(r'Observed Wavelength [$\mu$m]', size=18)
    ax.set_ylabel(r'$f_\lambda~[~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]', size=18)
    ax.legend(loc="upper right", ncols=1, bbox_to_anchor=[0.97, 0.97], framealpha=0)
    plt.tight_layout()

    # Save figure
    if savefig:
        fig.savefig(os.path.join(fig_dir, fig_name), dpi=400)

    return fig

def plot_many_mock_predictions(obs, preds, theta_labels, vary_theta, new_thetas, pred_orig=None, theta_best=None, z=None, fig_dir=None, fig_name=None, savefig=False):
    
    # Load wavelength data from observations
    phot_obs, prism_obs = obs
    # -- photometry
    phot_wave_A = phot_obs.wavelength
    phot_wave_m = convert_wave_A_to_m(phot_wave_A)
    phot_wave_um = convert_wave_A_to_um(phot_wave_A)
    # -- prism
    prism_wave_A = prism_obs.wavelength
    prism_wave_m = convert_wave_A_to_m(prism_wave_A)
    prism_wave_um = convert_wave_A_to_um(prism_wave_A)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    colors = ["blue", "green", "red", "orange", "purple"]

    # Find 
    theta_idx = theta_labels.index(vary_theta)

    # Plot original model
    if pred_orig is not None and theta_best is not None:

        # Load predicted flux data
        phot_pred_flux_maggie, prism_pred_flux_maggie = pred_orig

        # Convert predicted data to plot
        # -- photometry
        phot_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(phot_pred_flux_maggie, err_maggie=np.nan, wave_m=phot_wave_m, cgs_factor=1e-19)
        # -- prism
        prism_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(prism_pred_flux_maggie, err_maggie=np.nan, wave_m=prism_wave_m, cgs_factor=1e-19)

        # Plot mock observations
        # -- photometry
        ax.scatter(phot_wave_um, phot_pred_flux_cgs, color="k", marker='o')
        # -- prism
        ax.step(prism_wave_um, prism_pred_flux_cgs, color="k", where='mid')
        # -- legend
        ax.step(np.nan, np.nan, color="k", label=f'[Z/H] = {theta_best[theta_idx]:.2f} (Best)', where='mid')

    # Loop over each prediction
    for i, (pred, color) in enumerate(zip(preds, colors)):

        # Load predicted flux data
        phot_pred_flux_maggie, prism_pred_flux_maggie = pred

        # Convert predicted data to plot
        # -- photometry
        phot_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(phot_pred_flux_maggie, err_maggie=np.nan, wave_m=phot_wave_m, cgs_factor=1e-19)
        # -- prism
        prism_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(prism_pred_flux_maggie, err_maggie=np.nan, wave_m=prism_wave_m, cgs_factor=1e-19)

        # Plot mock observations
        # -- photometry
        ax.scatter(phot_wave_um, phot_pred_flux_cgs, color=color, marker='o')
        # -- prism
        ax.step(prism_wave_um, prism_pred_flux_cgs, color=color, where='mid')
        # -- legend
        ax.step(np.nan, np.nan, color=color, label=f'[Z/H] = {new_thetas[vary_theta][i]}', where='mid')

    # Prettify
    # -- limits
    ax.set_xlim(0.9, 5.1)
    # ax.set_ylim(0, 2.5)
    # -- axis ticks
    ax.set_xticks(np.arange(1, 6))
    if z is not None:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        top_ticks_rest = np.arange(0.2, 1.4, 0.2)
        top_ticks_obs = top_ticks_rest * (1 + z)  # map to bottom scale
        ax_top.set_xticks(top_ticks_obs)
        ax_top.set_xticklabels([f"{t:.1f}" for t in top_ticks_rest])
    # -- labels
    ax.set_xlabel(r'Observed Wavelength [$\mu$m]', size=18)
    ax.set_ylabel(r'$f_\lambda~[~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]', size=18)
    ax.legend(loc="upper right", ncols=1, bbox_to_anchor=[0.97, 0.97], framealpha=0)
    plt.tight_layout()

    # Save figure
    if savefig:
        fig.savefig(os.path.join(fig_dir, fig_name), dpi=400)

    return fig

obs_kwargs = {

    "phot_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/photometry",
        "name" : "007329",
        "data_ext" : "DATA",
        "mask_ext" : "VALID",
        "in_flux_units" : "magnitude",
        "out_flux_units" : "maggie",
        "snr_limit" : 20,
        "return_none" : False,
        "prefix" : "phot",
        "add_jitter" : True,
        "include_outliers" : True,
    },

    "prism_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "name" : "007329",
        "version" : "v3.1",
        "nod" : "extr5",
        "data_ext" : "DATA",
        "mask_ext" : None,
        "in_wave_units" : "si",
        "out_wave_units" : "A",
        "in_flux_units" : "si",
        "out_flux_units" : "maggie",
        "rescale_factor" : 1.86422,
        "snr_limit" : 20,
        "return_none" : False,
        "prefix" : "prism",
        "add_jitter" : True,
        "include_outliers" : True,
    },

    "grat2_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "name" : "007329",
        "grating" : "g235m",
        "filter" : "f170lp",
        "version" : None,
        "nod" : None,
        "data_ext" : "DATA",
        "mask_ext" : "VALID",
        "in_wave_units" : "um",
        "out_wave_units" : "A",
        "in_flux_units" : "ujy",
        "out_flux_units" : "maggie",
        "rescale_factor" : 1.86422,
        "snr_limit" : 20,
        "return_none" : False,
        "prefix" : "grat2",
        "add_jitter" : True,
        "include_outliers" : True,
    },
}

# Load results from output file
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/prospector_outputs"
out_file = "zf-uds-7329_flat_model_nautlius_phot_prism_nebF_smoothT.h5"
out_path = os.path.join(out_dir, out_file)
results_type = "nautlius"
results, obs, _ = reader.results_from(out_path.format(results_type), dangerous=True)

# Redshift information
zred = 3.2

# Get SPS object
sps = reader.get_sps(results)
# -- PythonFSPS object
sp = sps.ssp
sp_params = sp.params

# Extract variables from results
paramfile_text = results["paramfile_text"]
model_params = results['model_params']
lnprob = results['lnprobability']
numeric_chain = results['unstructured_chain']
weights = results["weights"]
samples = numeric_chain.T  # reshape for allcorner

# Rebuild model from parameter file text
build_model, namespace = load_build_model_from_string(paramfile_text)
# -- call build_model
run_params = results['run_params']
model_kwargs = run_params['model_kwargs']
model = build_model(model_kwargs, obs_kwargs=obs_kwargs)

# Extract parameter names
parnames = model.free_params
theta_labels = model.theta_labels()  # includes labels for all logsfr_ratios

# Extract truths
imax = np.argmax(lnprob)
theta_best = numeric_chain[imax, :].copy()  # max of log probability
theta_med = np.nanmedian(numeric_chain, axis=0)  # median of posteriors

# Load wavelength data
phot_filters, _, _, _ = load_photometry_data(**obs_kwargs["phot_kwargs"])
prism_wave, _, _, _ = load_prism_data(**obs_kwargs["prism_kwargs"])

# Build mock observations
mock_obs = build_mock_obs(filterset=phot_filters, wavelength=prism_wave)

# New model parameters
new_thetas = {
    "logzsol" : [-1.00, -0.30, 0.00, 0.18],
}

# Predict observations using original model parameters
mock_pred_orig, mfrac_orig = predict_mock_obs(model=model,
                                    theta=theta_best,
                                    observations=obs,
                                    mock_observations=mock_obs,
                                    sps=sps,
                                    add_noise=False,
                                    **model_params
                                    )

# Store different model predictions
mock_preds = []
mfracs = []

# Loop over each parameter
for key, values in new_thetas.items():
    theta_idx = theta_labels.index(key)

    # Loop over each parameter value
    for val in values:
        # -- update model parameters
        new_theta = theta_best.copy()
        new_theta[theta_idx] = val  # set parameter to new value
        # -- predict new observations
        mock_pred, mfrac = predict_mock_obs(model=model,
                                    theta=new_theta,
                                    observations=obs,
                                    mock_observations=mock_obs,
                                    sps=sps,
                                    add_noise=False,
                                    **model_params,
                                    )
        mock_preds.append(mock_pred)

# Plot model predictions
# -- original model
plot_mock_prediction(obs, pred=mock_pred_orig, z=zred)
# -- original model changing solar metallicity
plot_many_mock_predictions(obs, mock_preds, 
                           theta_labels=theta_labels, 
                           vary_theta="logzsol",
                           new_thetas=new_thetas,
                           pred_orig=mock_pred_orig, 
                           theta_best=theta_best,
                           z=zred,
                           )

plt.show()