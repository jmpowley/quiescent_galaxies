import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import prospect.io.read_results as reader

from plotting import call_subcorner
from postprocessing import return_sfh, return_sfh_chain, return_sfh_for_one_sigma_quantiles
from conversions import convert_flux_maggie_to_jy, convert_flux_jy_to_cgs, convert_wave_A_to_um, convert_wave_A_to_m, convert_flux_maggie_to_cgs

import importlib.util

def plot_sfh(age_bins, sfh_best, sfh_chain, weights, logscale=False):

        logages = np.array(age_bins).ravel()
        ages = 10**(logages-9) # ages in Gyr

        sfh_16, sfh_50, sfh_84 = return_sfh_for_one_sigma_quantiles(sfh_chain, weights)

        # Create figure
        fig, ax = plt.subplots()

        # Plot SFHs
        # -- best fit
        ax.plot(ages, [val for val in sfh_best for _ in (0,1)], color='blue', label=u'MAP')
        # -- median
        ax.plot(ages, [val for val in sfh_50 for _ in (0,1)], color='red', label='Median')
        # -- 16th and 84th percentiles
        ax.fill_between(ages, 
                        [val for val in sfh_16 for _ in (0,1)],
                        [val for val in sfh_84 for _ in (0,1)], 
                        color="red", alpha=0.2, linewidth=0.)
        
        ax.set_xlabel(r'$t_{\mathrm{obs}} - t$ (Gyr)')
        ax.set_ylabel('SFR '+ u'(M$_\u2609$/yr)')
        if logscale:
            ax.set_yscale('log')
            plt.legend(loc='upper left')
        #     plt.savefig(f'{arbname}_SFR_log.png', bbox_inches='tight', pad_inches=0.1, dpi=800)
        else:
            plt.legend(loc='upper left')
        #     plt.savefig(f'{arbname}_SFR.png', bbox_inches='tight', pad_inches=0.1, dpi=800)

        return fig

def plot_obs_model_comparison(obs, pred):

    phot_obs, prism_obs = obs
    phot_pred_flux_maggie, prism_pred_flux_maggie = pred  # arrays of predicted flux in units of maggies

    # Extract observational data
    # -- photometry
    phot_filters = phot_obs.filters
    phot_wave_A = phot_obs.wavelength
    phot_wave_m = convert_wave_A_to_m(phot_wave_A)
    phot_wave_um = convert_wave_A_to_um(phot_wave_A)
    # phot_filter_wave_A = [f.wavelength for f in phot_filters]
    # phot_filter_trans = [f.transmission for f in phot_filters]
    phot_obs_flux_maggie, phot_obs_err_maggie = phot_obs.flux, phot_obs.uncertainty
    phot_obs_flux_cgs, phot_obs_err_cgs = convert_flux_maggie_to_cgs(phot_obs_flux_maggie, phot_obs_err_maggie, phot_wave_m, cgs_factor=1e-19)
    # -- prism
    prism_wave_A = prism_obs.wavelength
    prism_wave_m = convert_wave_A_to_m(prism_wave_A)
    prism_wave_um = convert_wave_A_to_um(prism_wave_A)
    prism_obs_flux_maggie, prism_obs_err_maggie = prism_obs.flux, prism_obs.uncertainty
    prism_obs_flux_cgs, prism_obs_err_cgs = convert_flux_maggie_to_cgs(prism_obs_flux_maggie, prism_obs_err_maggie, prism_wave_m, cgs_factor=1e-19)

    # Convert predicted data to plot
    # -- phot
    phot_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(phot_pred_flux_maggie, err_maggie=np.nan, wave_m=phot_wave_m, cgs_factor=1e-19)
    # -- prism
    prism_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(prism_pred_flux_maggie, err_maggie=np.nan, wave_m=prism_wave_m, cgs_factor=1e-19)

    # Create figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    # Plot data and model predictions
    # -- photometry
    ax.errorbar(phot_wave_um, phot_obs_flux_cgs, yerr=phot_obs_err_cgs, 
                color='red', marker='o', markerfacecolor='none', ls='none', label='Photometry')
    ax.scatter(phot_wave_um, phot_pred_flux_cgs, 
               color='orange', marker='o', label='Photometry fit')
    # -- prism
    ax.plot(prism_wave_um, prism_obs_flux_cgs, color='black', label='Prism')
    ax.fill_between(prism_wave_um, prism_obs_flux_cgs-prism_obs_err_cgs, prism_obs_flux_cgs+prism_obs_err_cgs, 
                    color='black', alpha=0.25, label='Prism error')
    ax.plot(prism_wave_um, prism_pred_flux_cgs, color='blue', label='Prism fit')

    # -- prettify
    ax.set_xlabel(r'Observed Wavelength / $\mu$m')
    ax.set_ylabel(r'Flux / $1~\times~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ A$^{-1}$')
    ax.legend()
    plt.tight_layout()

# Load results from output file
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329"
# out_file = "zf-uds-7329_flat_model_nautilus.h5"
# out_file = "zf-uds-7329_flat_model_nautilus_2.h5"
out_file = "zf-uds-7329_flat_model_nautilus_phot_prism.h5"
out_path = os.path.join(out_dir, out_file)
results_type = "nautlius"
results, obs, _ = reader.results_from(out_path.format(results_type), dangerous=True)

# Build model from parameter file
# -- load parameter file
paramfile_path = "/Users/Jonah/PhD/Research/quiescent_galaxies/code/scripts/zf-uds-7329/zf-uds-7329_flat_model.py"
spec = importlib.util.spec_from_file_location("flat_model", paramfile_path)
paramfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paramfile)
# -- run build_model
model_kwargs = {
        "zred" : 3.19,
        "add_nebular" : True,
        }
model = paramfile.build_model(model_kwargs)

# Get SPS object
sps = reader.get_sps(results)

# Extract variables from results
model_params = results['model_params']
lnprob = results['lnprobability']
numeric_chain = results['unstructured_chain']
weights = results["weights"]

# Extract truths
imax = np.argmax(lnprob)
theta_best = numeric_chain[imax, :].copy()  # max of log probability
theta_med = np.nanmedian(numeric_chain, axis=0)  # median of posteriors

# Parameters to show in corner plot
showpars=['zred', 'logmass', 'logzsol', 'logsfr_ratios', 'gas_logz', 'gas_logu', 'eline_sigma']

# Extract SFHs
age_bins = np.asarray(model_params['agebins'])
sfh_best, logmass_best = return_sfh(results, theta_best)
sfh_chain = return_sfh_chain(results)  # chain of SFR vectors (solar masses / year)

# Predict model based on theta parameters
pred, mass_frac = model.predict(theta=theta_best, observations=obs, sps=sps)

plot_obs_model_comparison(obs, pred)

# Make plots
fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/zf-uds-7329"
# -- corner plot
fig_name = "zf-uds-7329_prospector_cornerplot.png"
# call_subcorner(results, showpars, truths=theta_best, color="purple", fig_dir=fig_dir, fig_name=fig_name, savefig=True)
# -- SFH plot
# plot_sfh(age_bins, sfh_best, sfh_chain, weights, logscale=False)

show_plots = True
if show_plots:
     plt.show()