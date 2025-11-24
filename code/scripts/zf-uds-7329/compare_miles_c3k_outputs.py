import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import prospect.io.read_results as reader

from loading import load_prism_data
from conversions import convert_wave_A_to_um, convert_wave_A_to_m, convert_flux_maggie_to_cgs
from postprocessing import return_sfh, return_sfh_chain, return_sfh_for_one_sigma_quantiles, return_assembly_time, convert_lookback_at_redshift_to_z, load_build_model_from_string, return_assembly_time_for_one_sigma_quantities


def plot_obs_model_comparison(obs, pred, lines=None, z=None, fig_dir=None, fig_name=None, savefig=False):

    phot_obs, prism_obs = obs
    phot_pred_flux_maggie, prism_pred_flux_maggie = pred  # arrays of predicted flux in units of maggies

    # Extract observational data
    # -- photometry
    phot_filters = phot_obs.filters
    phot_wave_A = phot_obs.wavelength
    phot_wave_m = convert_wave_A_to_m(phot_wave_A)
    phot_wave_um = convert_wave_A_to_um(phot_wave_A)
    phot_filter_waves_um = [convert_wave_A_to_um(f.wavelength) for f in phot_filters]
    phot_filter_trans = [f.transmission / f.transmission.max() for f in phot_filters]  # normalised transmission
    phot_obs_flux_maggie, phot_obs_err_maggie = phot_obs.flux, phot_obs.uncertainty
    phot_obs_flux_cgs, phot_obs_err_cgs = convert_flux_maggie_to_cgs(phot_obs_flux_maggie, phot_obs_err_maggie, phot_wave_m, cgs_factor=1e-19)
    # -- prism
    prism_wave_A = prism_obs.wavelength
    prism_wave_m = convert_wave_A_to_m(prism_wave_A)
    prism_wave_um = convert_wave_A_to_um(prism_wave_A)
    prism_obs_flux_maggie, prism_obs_err_maggie = prism_obs.flux, prism_obs.uncertainty
    prism_obs_flux_cgs, prism_obs_err_cgs = convert_flux_maggie_to_cgs(prism_obs_flux_maggie, prism_obs_err_maggie, prism_wave_m, cgs_factor=1e-19)

    # Convert predicted data to plot
    # -- photometry
    phot_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(phot_pred_flux_maggie, err_maggie=np.nan, wave_m=phot_wave_m, cgs_factor=1e-19)
    # -- prism
    prism_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(prism_pred_flux_maggie, err_maggie=np.nan, wave_m=prism_wave_m, cgs_factor=1e-19)

    # Calculate reduced/chisq
    # phot_chisq = calculate_chisq(data=phot_obs_flux_cgs, model=phot_pred_flux_cgs, error=phot_obs_err_cgs)
    # prism_chisq = calculate_chisq(data=prism_obs_flux_cgs, model=prism_pred_flux_cgs, error=prism_obs_err_cgs)

    print("prism obs size:", np.size(prism_obs_flux_cgs))
    print("prism pred size:", np.size(prism_pred_flux_cgs))

    # Calculate overall chisq
    # prism_chisq = np.size()
    # phot_chisq

    # Create figure
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 1, height_ratios=[5, 2])
    ax = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax)

    # Plot data and model predictions
    # -- photometry
    ax.errorbar(phot_wave_um, phot_obs_flux_cgs, yerr=phot_obs_err_cgs, 
                color='red', marker='o', markerfacecolor='none', ls='none', label='Photometry')
    ax.scatter(phot_wave_um, phot_pred_flux_cgs, 
               color='orange', marker='o', label='Photometry fit')
    # -- prism
    ax.step(prism_wave_um, prism_obs_flux_cgs, 
            color='black', label='Prism', where='mid')
    ax.fill_between(prism_wave_um, prism_obs_flux_cgs-prism_obs_err_cgs, prism_obs_flux_cgs+prism_obs_err_cgs, 
                    step='mid', color='black', alpha=0.25)
    # ax.plot(prism_wave_um, prism_pred_flux_cgs, color='blue', label='Prism fit')
    ax.step(prism_wave_um, prism_pred_flux_cgs, color='blue', label='Prism fit', where='mid')
    # -- residuals
    ax_res.scatter(prism_wave_um, 
                #    (prism_obs_flux_cgs-prism_pred_flux_cgs),  # unnormalised
                   ((prism_obs_flux_cgs-prism_pred_flux_cgs) / prism_obs_err_cgs),  # normalised
                   color='blue', marker='.')
    ax_res.scatter(phot_wave_um, 
                #    phot_obs_flux_cgs-phot_pred_flux_cgs,  # unnormalised
                   ((phot_obs_flux_cgs-phot_pred_flux_cgs) / phot_obs_err_cgs),  # normalised
                   color="orange", marker="o")
    ax_res.axhline(0, color="gray", ls="--")
    # Plot emission lines
    if lines is not None:
        for (stri, wl) in lines.items():
            ax.axvline(wl, color='gray', ls="--")
            ax.text(wl+0.01, 0.1, stri, color='gray', ha='left')

    # Prettify
    # -- limits
    ax.set_xlim(0.9, 5.1)
    ax.set_ylim(0, None)
    # ax_res.set_ylim(-0.51, 0.51)
    # -- axis ticks
    ax.set_xticks(np.arange(1, 6))
    ax.tick_params(labelbottom=False)
    # ax_res.set_xticks(np.arange(1, 6))
    if z is not None:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        top_ticks_rest = np.arange(0.2, 1.4, 0.2)
        top_ticks_obs = top_ticks_rest * (1 + z)  # map to bottom scale
        ax_top.set_xticks(top_ticks_obs)
        ax_top.set_xticklabels([f"{t:.1f}" for t in top_ticks_rest])
    # -- labels
    ax_res.set_xlabel(r'Observed Wavelength [$\mu$m]', size=18)
    ax.set_ylabel(r'$f_\lambda~[10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}]$', size=18)
    ax_res.set_ylabel(r'$\chi$ norm.', size=18)
    # -- legend (draw in explicit order)
    handles, labels = ax.get_legend_handles_labels()
    label2handle = dict(zip(labels, handles))
    desired = ['Photometry', 'Prism', 'Photometry fit', 'Prism fit']
    ordered_labels = [lab for lab in desired if lab in label2handle]
    ordered_handles = [label2handle[lab] for lab in ordered_labels]
    ax.legend(ordered_handles, ordered_labels, loc="upper left", ncols=4,
          bbox_to_anchor=[0, 1.15], framealpha=0)

    ax.legend(loc="upper right", ncols=1, bbox_to_anchor=[0.97, 0.97], framealpha=0)
    plt.tight_layout()

    # Save figure
    if savefig:
        fig.savefig(os.path.join(fig_dir, fig_name), dpi=400)

    return fig

# Load results from output file
miles_out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/prospector_outputs/zf-uds-7329_flat_model_mistmiles"
c3k_out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/prospector_outputs/zf-uds-7329_flat_model_mistc3ka"
miles_out_file = "zf-uds-7329_flat_model_nautlius_phot_prism_nebF_smoothT_nuisF.h5"
c3k_out_file = "zf-uds-7329_flat_model_mistc3ka_nautilus_phot_prism_nebF_smoothT_nuisF.h5"
results_type = "nautlius"
# -- MILES
miles_out_path = os.path.join(miles_out_dir, miles_out_file)
miles_results, miles_obs, _ = reader.results_from(miles_out_path.format(results_type), dangerous=True)
# -- C3K
c3k_out_path = os.path.join(c3k_out_dir, c3k_out_file)
c3k_results, c3k_obs, _ = reader.results_from(c3k_out_path.format(results_type), dangerous=True)

# Get SPS objects
# -- MILES
miles_sps = reader.get_sps(miles_results)
miles_sp = miles_sps.ssp
miles_sp_params = miles_sp.params
# -- C3K
c3k_sps = reader.get_sps(c3k_results)
c3k_sp = c3k_sps.ssp
c3k_sp_params = c3k_sp.params

# Rebuild model from parameter file text assuming model for miles/C3K does not change
miles_paramfile_text = miles_results["paramfile_text"]
build_model, namespace = load_build_model_from_string(miles_paramfile_text)
# -- call build_model
miles_run_params = miles_results['run_params']
model_kwargs = miles_run_params['model_kwargs']
obs_kwargs = miles_run_params['obs_kwargs']
model = build_model(model_kwargs, obs_kwargs=obs_kwargs)
# -- extract model info
nfree = len(model.free_params)
parnames = model.free_params
parnames_full = model.theta_labels()  # includes labels for all logsfr_ratios

# Extract spectral response
for ob in miles_obs:
    if ob.kind == "spectrum":
        response = ob.response

# Extract variables from results
# -- MILES
miles_model_params = miles_results['model_params']
miles_lnprob = miles_results['lnprobability']
miles_numeric_chain = miles_results['unstructured_chain']
miles_weights = miles_results["weights"]
miles_samples = miles_numeric_chain.T  # reshape for allcorner
# -- C3K
c3k_model_params = c3k_results['model_params']
c3k_lnprob = c3k_results['lnprobability']
c3k_numeric_chain = c3k_results['unstructured_chain']
c3k_weights = c3k_results["weights"]
c3k_samples = c3k_numeric_chain.T  # reshape for allcorner

# Extract truths
# -- MILES
miles_imax = np.argmax(miles_lnprob)
miles_theta_best = miles_numeric_chain[miles_imax, :].copy()  # max of log probability
miles_theta_med = np.nanmedian(miles_numeric_chain, axis=0)  # median of posteriors
# -- C3K
c3k_imax = np.argmax(c3k_lnprob)
c3k_theta_best = c3k_numeric_chain[c3k_imax, :].copy()  # max of log probability
c3k_theta_med = np.nanmedian(c3k_numeric_chain, axis=0)  # median of posteriors

# Get redshift
z_idx = parnames_full.index('zred')
miles_z = miles_theta_best[z_idx]
c3k_z = c3k_theta_best[z_idx]

# Extract SFHs
# -- MILES
age_bins = np.asarray(miles_model_params['agebins'])
sfh_best, logmass_best = return_sfh(miles_results, miles_theta_best)
sfh_chain = return_sfh_chain(miles_results)  # chain of SFR vectors (solar masses / year)
sfh_16, sfh_50, sfh_84 = return_sfh_for_one_sigma_quantiles(sfh_chain, miles_weights)
# -- C3K
c3k_age_bins = np.asarray(c3k_model_params['agebins'])
c3k_sfh_best, c3k_logmass_best = return_sfh(c3k_results, c3k_theta_best)
c3k_sfh_chain = return_sfh_chain(c3k_results)  # chain of SFR vectors (solar masses / year)
c3k_sfh_16, c3k_sfh_50, c3k_sfh_84 = return_sfh_for_one_sigma_quantiles(c3k_sfh_chain, c3k_weights)

# Predict model based on theta parameters
# -- MILES
miles_pred, miles_mass_frac = model.predict(theta=miles_theta_best, observations=miles_obs, sps=miles_sps)
miles_logmass_best_surv = np.log10(10**logmass_best * miles_mass_frac)
# -- C3K
c3k_pred, c3k_mass_frac = model.predict(theta=c3k_theta_best, observations=c3k_obs, sps=c3k_sps)
c3k_logmass_best_surv = np.log10(10**logmass_best * c3k_mass_frac)

print(miles_pred)

print(c3k_pred)

plot_obs_model_comparison(miles_obs, miles_pred)

plot_obs_model_comparison(c3k_obs, c3k_pred)

plt.show()