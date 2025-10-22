import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.stats import gaussian_kde
from astropy.stats import sigma_clip

import prospect.io.read_results as reader
from prospect.plotting.utils import sample_prior, sample_posterior, get_simple_prior

from plotting import call_subcorner, call_allcorner
from postprocessing import return_sfh, return_sfh_chain, return_sfh_for_one_sigma_quantiles, load_build_model_from_string, extract_model_kwargs
from conversions import convert_wave_A_to_um, convert_wave_A_to_m, convert_flux_maggie_to_cgs

def plot_sfh(age_bins, sfh_best, sfh_chain, weights, logscale=False, fig_dir=None, fig_name=None, savefig=False):

        log_ages = np.array(age_bins).ravel()
        ages = 10**(log_ages-9) # ages in Gyr

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

        # Save figure
        if savefig:
            fig.savefig(os.path.join(fig_dir, fig_name), dpi=400)

        return fig

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
    ax_res.set_xlabel(r'Observed Wavelength / $\mu$m', size=18)
    ax.set_ylabel(r'$f_\lambda~/~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$', size=18)
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

def plot_prior_posterior_comparison(prior_samples, post_samples, labels, fig_dir=None, fig_name=None, savefig=False):

    ndim = np.shape(post_samples)[1]

    # Create figure
    fig, ax = plt.subplots(nrows=ndim, ncols=1, figsize=(8, 4*ndim))
    if ndim == 1:
        ax = [ax]  # make iterable for single dimension

    for i, label in enumerate(labels):

        # Clip priors
        prior_clipped = sigma_clip(prior_samples[:, i], sigma=3, maxiters=None)
        post_clipped = sigma_clip(post_samples[:, i], sigma=3, maxiters=None)

        # Plot KDE
        kde = False
        if kde:
            kde_prior = gaussian_kde(prior_clipped)
            kde_post = gaussian_kde(post_clipped)
            xs = np.linspace(
                    min(prior_clipped.min(), post_clipped.min()),
                    max(prior_clipped.max(), post_clipped.max()),
                    300
                )

            ax[i].plot(xs, kde_prior(xs), 'r--', label='Prior PDF')
            ax[i].plot(xs, kde_post(xs), 'b-', label='Posterior PDF')

        # Plot histograms
        hist = True
        if hist:
            # normalized PDFs
            # ax[i].hist(prior_samples[i], bins=40, color='red', alpha=0.4,
            #         histtype='stepfilled', density=True, label='Prior')
            # ax[i].hist(post_samples[i], bins=40, color='blue', alpha=0.4,
            #         histtype='stepfilled', density=True, label='Posterior')

            # weighted histograms
            ax[i].hist(prior_clipped, bins=40, color='red', alpha=0.4, weights= np.full(np.size(prior_clipped), 1/np.size(prior_clipped)),
                    histtype='stepfilled', label='Prior')
            ax[i].hist(post_clipped, bins=40, color='blue', alpha=0.4, weights= np.full(np.size(post_clipped), 1/np.size(post_clipped)),
                    histtype='stepfilled', label='Posterior')
    
        # Prettify
        ax[i].set_title(label)
        ax[i].legend(loc="upper right")

    fig.tight_layout()

    # Save figure
    if savefig:
        fig.savefig(os.path.join(fig_dir, fig_name), dpi=400)

    return fig

# Load results from output file
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329"
# out_file = "zf-uds-7329_flat_model_nautilus.h5"
# out_file = "zf-uds-7329_flat_model_nautilus_2.h5"
# out_file = "zf-uds-7329_flat_model_nautilus_phot_prism.h5"
out_file = "zf-uds-7329_flat_model_nautlius_phot_prism_nebF_snr20.h5"
out_path = os.path.join(out_dir, out_file)
results_type = "nautlius"
results, obs, _ = reader.results_from(out_path.format(results_type), dangerous=True)

# Get SPS object
sps = reader.get_sps(results)

# Build model from parameter file text
paramfile_text = results["paramfile_text"]
build_model, namespace = load_build_model_from_string(paramfile_text)
model = build_model({'zred': 3.19, 'add_nebular': False})

# Extract variables from results
model_params = results['model_params']
lnprob = results['lnprobability']
numeric_chain = results['unstructured_chain']
weights = results["weights"]
samples = numeric_chain.T  # reshape for allcorner

# Extract truths
imax = np.argmax(lnprob)
theta_best = numeric_chain[imax, :].copy()  # max of log probability
theta_med = np.nanmedian(numeric_chain, axis=0)  # median of posteriors

# Extract parameter names
# parnames = getattr(results['chain'].dtype, "names", None)
# parnames = list(parnames)
parnames = model.free_params
# -- replace logsfr_ratios with label for each logsfr bin
# nbins_sfh = model_params['nbins_sfh'][0]
# logsfr_ratios_idx = parnames.index('logsfr_ratios')
# logsfr_ratios_labels = [f'logsfr_ratios_{i}' for i in np.arange(1, nbins_sfh)]
# parnames_full = parnames.copy()
# parnames_full[logsfr_ratios_idx:logsfr_ratios_idx+1] = logsfr_ratios_labels
parnames_full = model.theta_labels()

# Extract SFHs
age_bins = np.asarray(model_params['agebins'])
sfh_best, logmass_best = return_sfh(results, theta_best)
sfh_chain = return_sfh_chain(results)  # chain of SFR vectors (solar masses / year)

# Limit posteriors shown in corner plot
# showpars = ['zred', 'logmass', 'logzsol', 'logsfr_ratios', 'gas_logz', 'gas_logu', 'eline_sigma', 'dust_index', 'f_outlier_spec']
showpars = parnames  # show all free parameters
# TODO: apply any changes to samples, weights etc

# Predict model based on theta parameters
pred, mass_frac = model.predict(theta=theta_best, observations=obs, sps=sps)

# Sample prior and posterior distributions
nsample = 1e4
prior_samples, labels = sample_prior(model, nsample=int(nsample))
post_samples = sample_posterior(numeric_chain, weights=weights, nsample=int(nsample))
print("priors:", prior_samples.shape)
print("posteriors:", post_samples.shape)
print("samples:", samples.shape)

# Emission lines to show
lines_A = {  # units in Angstroms
    'MgII' : 2800.000,  # individual lines are 2795.528, 2802.705
    'CaII' : 3950.000,  # individual lines are 3933.663, 3968.469
    'Hdelta' : 4101.742,
    'Hgamma' : 4340.471,
    'Hbeta' : 4861.333,
    'Mgb' : 5200.00,  # individual lines are 2795.528, 2802.705
    'NIII4511?' : 4510.910,
    'FeII4556?' : 4555.893,
    # 'FeII5198?' : 5197.577,
    'FeIII5270?' : 5270.400,
    'FeII8892?' : 8891.910,
    'NaD' : 5892.500,  # individual lines are 5889.950, 5895.924
    'Halpha' : 6562.819,
    'TiO' : 7200.000,  # approx.
    'TiO/ZrO/CN' : 9300.000,  # approx.
    'Unlisted' : 10000.00,  # unlisted line?
    'CN' : 11000.000,  # approx.
}
lines_um = {
    stri : convert_wave_A_to_um(wl) for stri, wl in lines_A.items()
}
# Redshift lines
# z = 3.2
# z = theta_best[0]
z_idx = parnames_full.index('zred')
z = theta_best[z_idx]
zlines_um = {
    stri : wl*(1.+z) for stri, wl in lines_um.items()
    }

# Print results info
# print("results.keys():", results.keys())
# print("len:", len(results.keys()))
# print("parnames:", parnames)
# print("len:", len(parnames))
# print("model_params:", model_params)
# print("len:", len(model_params.keys()))
# print("theta_best:", theta_best)
# print("len:", len(theta_best))

# Make plots
fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/zf-uds-7329"
# -- corner plot
fig_name = "zf-uds-7329_flat_model_corner.png"
# call_subcorner(results, showpars, truths=theta_best, color="purple", fig_dir=fig_dir, fig_name=fig_name, savefig=True)
call_allcorner(samples, parnames_full, weights, fig_dir, fig_name, savefig=True, color="purple")
# -- SFH plot
fig_name = "zf-uds-7329_flat_model_sfh.png"
plot_sfh(age_bins, sfh_best, sfh_chain, weights, logscale=False, fig_dir=fig_dir, fig_name=fig_name, savefig=True)
# -- data-model comparison
fig_name = "zf-uds-7329_flat_model_obs_pred_comparison.png"
plot_obs_model_comparison(obs, pred, lines=zlines_um, z=z, fig_dir=fig_dir, fig_name=fig_name, savefig=True)
# -- prior-posterior comparison
fig_name = "zf-uds-7329_priors_posteriors_comparison.png"
plot_prior_posterior_comparison(prior_samples, post_samples, parnames_full, fig_dir=fig_dir, fig_name=fig_name, savefig=True)

show_plots = True
if show_plots:
     plt.show()