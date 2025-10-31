import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.stats import gaussian_kde
from astropy.stats import sigma_clip
import astropy.units as u
import astropy.cosmology as cosmology

import prospect.io.read_results as reader
from prospect.plotting.utils import sample_prior, sample_posterior, get_simple_prior

from plotting import call_allcorner
from postprocessing import return_sfh, return_sfh_chain, return_sfh_for_one_sigma_quantiles, return_assembly_time, convert_lookback_at_redshift_to_z, load_build_model_from_string, return_assembly_time_for_one_sigma_quantities
from conversions import convert_wave_A_to_um, convert_wave_A_to_m, convert_flux_maggie_to_cgs
from statistics import calculate_chisq, calculate_reduced_chisq

def plot_sfh(age_bins, sfh_best, sfh_chain, weights, logscale=False, show_mass=True, logmass=None, logmass_surv=None, fig_dir=None, fig_name=None, savefig=False):

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
        # -- add mass information
        if show_mass and logmass is not None:
            ax.text(0.02, 0.84, rf"$\log_{{10}}(M_\ast/M_\odot) = {logmass:.2f}$", transform=ax.transAxes, ha="left", va="top")
            if logmass_surv is not None:
                ax.text(0.02, 0.79, rf"$\log_{{10}}(M_{{surv}}/M_\odot) = {logmass_surv:.2f}$", transform=ax.transAxes, ha="left", va="top")
        # -- prettify
        ax.set_xlabel(r'$t_{\mathrm{obs}} - t$ (Gyr)')
        ax.set_ylabel('SFR '+ u'(M$_\u2609$/yr)')
        if logscale:
            ax.set_yscale('log')
            plt.legend(loc='upper left')
        else:
            plt.legend(loc='upper left')

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

    # Calculate reduced/chisq
    phot_chisq = calculate_chisq(data=phot_obs_flux_cgs, model=phot_pred_flux_cgs, error=phot_obs_err_cgs)
    prism_chisq = calculate_chisq(data=prism_obs_flux_cgs, model=prism_pred_flux_cgs, error=prism_obs_err_cgs)

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

def plot_spec_obs_model_comparison(obs, pred, spec_to_show, obs_order, lines=None, z=None, fig_dir=None, fig_name=None, savefig=False):

    # Load observations
    # -- find indexes in observations
    phot_idx = obs_order.index('phot')  # assume photometry is 'phot'
    spec_idx = obs_order.index(spec_to_show)
    # -- photometry
    phot_obs = obs[phot_idx]
    phot_pred_flux_maggie = pred[phot_idx]
    # -- spectra
    spec_obs = obs[spec_idx]
    spec_pred_flux_maggie = pred[spec_idx]  # arrays of predicted flux in units of maggies

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
    spec_wave_A = spec_obs.wavelength
    spec_wave_m = convert_wave_A_to_m(spec_wave_A)
    spec_wave_um = convert_wave_A_to_um(spec_wave_A)
    spec_obs_flux_maggie, spec_obs_err_maggie = spec_obs.flux, spec_obs.uncertainty
    spec_obs_flux_cgs, spec_obs_err_cgs = convert_flux_maggie_to_cgs(spec_obs_flux_maggie, spec_obs_err_maggie, spec_wave_m, cgs_factor=1e-19)

    # Convert predicted data to plot
    # -- photometry
    phot_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(phot_pred_flux_maggie, err_maggie=np.nan, wave_m=phot_wave_m, cgs_factor=1e-19)
    # -- prism
    spec_pred_flux_cgs, _ = convert_flux_maggie_to_cgs(spec_pred_flux_maggie, err_maggie=np.nan, wave_m=spec_wave_m, cgs_factor=1e-19)

    # Calculate reduced/chisq
    phot_chisq = calculate_chisq(data=phot_obs_flux_cgs, model=phot_pred_flux_cgs, error=phot_obs_err_cgs)
    spec_chisq = calculate_chisq(data=spec_obs_flux_cgs, model=spec_pred_flux_cgs, error=spec_obs_err_cgs)

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
    ax.step(spec_wave_um, spec_obs_flux_cgs, 
            color='black', label=spec_to_show, where='mid')
    ax.fill_between(spec_wave_um, spec_obs_flux_cgs-spec_obs_err_cgs, spec_obs_flux_cgs+spec_obs_err_cgs, 
                    step='mid', color='black', alpha=0.25)
    # ax.plot(prism_wave_um, prism_pred_flux_cgs, color='blue', label='Prism fit')
    ax.step(spec_wave_um, spec_pred_flux_cgs, color='blue', label=f'{spec_to_show} fit', where='mid')
    # -- residuals
    ax_res.scatter(spec_wave_um, 
                #    (prism_obs_flux_cgs-prism_pred_flux_cgs),  # unnormalised
                   ((spec_obs_flux_cgs-spec_pred_flux_cgs) / spec_obs_err_cgs),  # normalised
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
    desired = ['Photometry', f'{spec_to_show}', 'Photometry fit', f'{spec_to_show} fit']
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

def plot_formation_timescale_posterior(t_forms, t_form_best, t_form_16, t_form_50, t_form_84, fig_dir=None, fig_name=None, savefig=False):

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Convert to redshifts
    cosmo = cosmology.FlatLambdaCDM(H0=67.4, Om0=0.315, Tcmb0=2.726)
    z_form_best = convert_lookback_at_redshift_to_z(t_form_best, z_start=z, cosmo=cosmo)
    z_form_16 = convert_lookback_at_redshift_to_z(t_form_16,  z_start=z, cosmo=cosmo)
    z_form_50 = convert_lookback_at_redshift_to_z(t_form_50,  z_start=z, cosmo=cosmo)
    z_form_84 = convert_lookback_at_redshift_to_z(t_form_84, z_start=z, cosmo=cosmo)
    print("z_form_best:", z_form_best)

    # Calculate absolute errors as differences from MAP
    t_form_err_16 = np.abs(t_form_best - t_form_16)
    t_form_err_84 = np.abs(t_form_best - t_form_84)
    z_form_err_16 = np.abs(z_form_best - z_form_16)
    z_form_err_84 = np.abs(z_form_best - z_form_84)

    # Plot KDE
    kde = True
    if kde:
        t_forms_clean = t_forms[~np.isnan(t_forms)]  # remove NaNs for KDE
        xs = np.linspace(t_forms_clean.min(), t_forms_clean.max(), 300)
        kde_t_form = gaussian_kde(t_forms_clean)

        kde_t_form_norm = kde_t_form(xs) / np.nansum(kde_t_form(xs))

        ax.fill_between(xs, kde_t_form_norm, color="purple", alpha=0.3)
        ax.plot(xs, kde_t_form_norm, color="purple", ls="-")

    # Plot histogram
    hist = False
    if hist:
        ax.hist(t_forms, color="purple", density=True)

    # Plot estimates
    ax.axvline(t_form_best, color="C0", ls="-", label="MAP")
    ax.axvline(t_form_50, color="black", ls="--", label=f"Median ({t_form_50:.2f} Gyr; z = {z_form_50:.2f})")
    ax.axvline(t_form_16, color="black", ls=":", label="16th/84th percentile")
    ax.axvline(t_form_84, color="black", ls=":")
    
    # Prettify
    ax.text(0.02, 0.83, rf"$t_{{form}} = {t_form_best:.2f}^{{+{t_form_err_84:.2f}}}_{{-{t_form_err_16:.2f}}}$ Gyr", transform=ax.transAxes, ha="left", va="top")
    ax.text(0.02, 0.79, rf"$z_{{form}} = {z_form_best:.2f}^{{+{z_form_err_84:.2f}}}_{{-{z_form_err_16:.2f}}}$", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel(r'$t - t_{obs}$ [Gyr]', size=16)
    ax.set_ylabel(r'PDF [Norm.]', size=16)
    ax.legend(loc="upper left")

    # Save figure
    if savefig:
        fig.savefig(os.path.join(fig_dir, fig_name), dpi=400)

    return fig

def plot_quenching_timescale_posterior(t_quenchs, t_quench_best, t_quench_16, t_quench_50, t_quench_84, fig_dir=None, fig_name=None, savefig=False):

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Convert to redshifts
    cosmo = cosmology.FlatLambdaCDM(H0=67.4, Om0=0.315, Tcmb0=2.726)
    z_quench_best = convert_lookback_at_redshift_to_z(t_quench_best, z_start=z, cosmo=cosmo)
    z_quench_16 = convert_lookback_at_redshift_to_z(t_quench_16,  z_start=z, cosmo=cosmo)
    z_quench_50 = convert_lookback_at_redshift_to_z(t_quench_50,  z_start=z, cosmo=cosmo)
    z_quench_84 = convert_lookback_at_redshift_to_z(t_quench_84, z_start=z, cosmo=cosmo)
    print("z_quench_best:", z_quench_best)

    # Calculate absolute errors as differences from MAP
    t_quench_err_16 = np.abs(t_quench_best - t_quench_16)
    t_quench_err_84 = np.abs(t_quench_best - t_quench_84)
    z_quench_err_16 = np.abs(z_quench_best - z_quench_16)
    z_quench_err_84 = np.abs(z_quench_best - z_quench_84)

    # Plot KDE
    kde = True
    if kde:
        t_quenchs_clean = t_quenchs[~np.isnan(t_quenchs)]  # remove NaNs for KDE
        xs = np.linspace(t_quenchs_clean.min(), t_quenchs_clean.max(), 300)
        kde_t_quench = gaussian_kde(t_quenchs_clean)
        kde_t_quench_norm = kde_t_quench(xs) / np.nansum(kde_t_quench(xs))  # normalise
        ax.fill_between(xs, kde_t_quench_norm, color="green", alpha=0.3)
        ax.plot(xs, kde_t_quench_norm, color="green", ls="-")

    # Plot histogram
    hist = False
    if hist:
        ax.hist(t_quenchs, color="green", density=True)

    # Plot estimates
    ax.axvline(t_quench_best, color="C0", ls="-", label="MAP")
    ax.axvline(t_quench_50, color="black", ls="--", label=f"Median ({t_quench_50:.2f} Gyr; z = {z_quench_50:.2f})")
    ax.axvline(t_quench_16, color="black", ls=":", label="16th/84th percentile")
    ax.axvline(t_quench_84, color="black", ls=":")
    
    # Prettify
    ax.text(0.02, 0.83, rf"$t_{{quench}} = {t_quench_best:.2f}^{{+{t_quench_err_84:.2f}}}_{{-{t_quench_err_16:.2f}}}$ Gyr", transform=ax.transAxes, ha="left", va="top")
    ax.text(0.02, 0.79, rf"$z_{{quench}} = {z_quench_best:.2f}^{{+{z_quench_err_84:.2f}}}_{{-{z_quench_err_16:.2f}}}$", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel(r'$t - t_{obs}$ [Gyr]', size=16)
    ax.set_ylabel(r'PDF [Norm.]', size=16)
    ax.legend(loc="upper left")

    # Save figure
    if savefig:
        fig.savefig(os.path.join(fig_dir, fig_name), dpi=400)

    return fig

# Load results from output file
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/prospector_outputs"
# out_file = "zf-uds-7329_flat_model_nautilus.h5"
# out_file = "zf-uds-7329_flat_model_nautilus_2.h5"
# out_file = "zf-uds-7329_flat_model_nautilus_phot_prism.h5"
# out_file = "zf-uds-7329_flat_model_nautlius_phot_prism_nebF_snr20.h5"
# out_file = "zf-uds-7329_flat_model_nautlius_phot_prism_nebF_smoothT.h5"
# out_file = "zf-uds-7329_flat_model_nautlius_phot_prism_g235m_nebF.h5"
out_file = "zf-uds-7329_flat_model_nautlius_phot_prism_g235m_nebF_smoothT.h5"
# out_file = "zf-uds-7329_flat_model_nautlius_phot_prism_g235m_nebT.h5"
out_path = os.path.join(out_dir, out_file)
results_type = "nautlius"
results, obs, _ = reader.results_from(out_path.format(results_type), dangerous=True)
print("Observations:\n", obs)

# Get SPS object
sps = reader.get_sps(results)
# -- PythonFSPS object
sp = sps.ssp
sp_params = sp.params

# Rebuild model from parameter file text
paramfile_text = results["paramfile_text"]
build_model, namespace = load_build_model_from_string(paramfile_text)
# -- call build_model
run_params = results['run_params']
model_kwargs = run_params['model_kwargs']
obs_kwargs = run_params['obs_kwargs']
model = build_model(model_kwargs, obs_kwargs=obs_kwargs)

# Extract variables from results
model_params = results['model_params']
lnprob = results['lnprobability']
numeric_chain = results['unstructured_chain']
weights = results["weights"]
samples = numeric_chain.T  # reshape for allcorner
nfree = len(model.free_params)

# Extract truths
imax = np.argmax(lnprob)
theta_best = numeric_chain[imax, :].copy()  # max of log probability
theta_med = np.nanmedian(numeric_chain, axis=0)  # median of posteriors

# Extract parameter names
parnames = model.free_params
parnames_full = model.theta_labels()  # includes labels for all logsfr_ratios

# Get best redshift
# z = 3.2
# z = theta_best[0]
z_idx = parnames_full.index('zred')
z = theta_best[z_idx]

# Order observations fit
obs_order = [obs.rstrip('_kwargs') for obs in list(obs_kwargs.keys()) if obs_kwargs[obs]['fit_obs']]
# print(obs_order)
# print(obs_order.index('prism'))
# print(obs_order.index('grat2'))

# Extract SFHs
age_bins = np.asarray(model_params['agebins'])
sfh_best, logmass_best = return_sfh(results, theta_best)
sfh_chain = return_sfh_chain(results)  # chain of SFR vectors (solar masses / year)
sfh_16, sfh_50, sfh_84 = return_sfh_for_one_sigma_quantiles(sfh_chain, weights)

# Calculate formation and quenching times
t_form_best = return_assembly_time(q=0.5, sfh=sfh_best, age_bins=age_bins)
t_quench_best = return_assembly_time(q=0.1, sfh=sfh_best, age_bins=age_bins)  # 0.1 not 0.9 as age bins start from most recent
t_form_16, t_form_50, t_form_84, t_forms = return_assembly_time_for_one_sigma_quantities(q=0.5, sfh_chain=sfh_chain, age_bins=age_bins, weights=weights, return_distribution=True)
t_quench_16, t_quench_50, t_quench_84, t_quenchs = return_assembly_time_for_one_sigma_quantities(q=0.1, sfh_chain=sfh_chain, age_bins=age_bins, weights=weights, return_distribution=True)

# Predict model based on theta parameters
pred, mass_frac = model.predict(theta=theta_best, observations=obs, sps=sps)
logmass_best_surv = np.log10(10**logmass_best * mass_frac)
print("best mass:", logmass_best)
print("best surviving mass:", logmass_best_surv)

# Sample prior and posterior distributions
nsample = 1e4
prior_samples, labels = sample_prior(model, nsample=int(nsample))
post_samples = sample_posterior(numeric_chain, weights=weights, nsample=int(nsample))

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
zlines_um = {
    stri : wl*(1.+z) for stri, wl in lines_um.items()
    }

# Limit posteriors shown in corner plot
# showpars = ['zred', 'logmass', 'logzsol', 'logsfr_ratios', 'gas_logz', 'gas_logu', 'eline_sigma', 'dust_index', 'f_outlier_spec']
showpars = parnames  # show all free parameters
# TODO: apply any changes to samples, weights etc

# Plot preparation
# -- create directory for plots
fig_base = out_file.rstrip(".h5")
fig_dir = f"/Users/Jonah/PhD/Research/quiescent_galaxies/figures/zf-uds-7329/{fig_base}"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# Make plots
show_plots = True
save_figs = True
# -- corner plot
fig_name = f"{fig_base}_corner.png"
# call_subcorner(results, showpars, truths=theta_best, color="purple", fig_dir=fig_dir, fig_name=fig_name, savefig=True)
# allcorner_kwargs = dict(color="purple", show_titles=True, qcolor='k')
# call_allcorner(samples, parnames_full, weights, fig_dir, fig_name, savefig=save_figs, show_all_ticklabels=True, **allcorner_kwargs)
# -- SFH plot
fig_name = f"{fig_base}_sfh.png"
sfh_kwargs = dict(show_mass=True, logmass=logmass_best, logmass_surv=logmass_best_surv, fig_dir=fig_dir, fig_name=fig_name, savefig=save_figs)
plot_sfh(age_bins, sfh_best, sfh_chain, weights, **sfh_kwargs)
# -- data-model comparison
# fig_name = f"{fig_base}_obs_pred_comparison.png"
# plot_obs_model_comparison(obs, pred, lines=zlines_um, z=z, fig_dir=fig_dir, fig_name=fig_name, savefig=True)
# -- compare specific spectrum
# TODO: Change to extract desired spectrum
fig_name1 = f"{fig_base}_spec_obs_pred_comparison1.png"
fig_name2 = f"{fig_base}_spec_obs_pred_comparison2.png"
plot_spec_obs_model_comparison(obs, pred, spec_to_show='grat2', obs_order=obs_order, lines=zlines_um, z=z, fig_dir=fig_dir, fig_name=fig_name1, savefig=save_figs)
plot_spec_obs_model_comparison(obs, pred, spec_to_show='prism', obs_order=obs_order, lines=zlines_um, z=z, fig_dir=fig_dir, fig_name=fig_name2, savefig=save_figs)
# -- prior-posterior comparison
fig_name = f"{fig_base}_priors_posteriors_comparison.png"
plot_prior_posterior_comparison(prior_samples, post_samples, parnames_full, fig_dir=fig_dir, fig_name=fig_name, savefig=save_figs)
# -- formation timescale posterior
fig_name = f"{fig_base}_tzform_posterior.png"
plot_formation_timescale_posterior(t_forms, t_form_best, t_form_16, t_form_50, t_form_84, fig_dir=fig_dir, fig_name=fig_name, savefig=save_figs)
# -- quenching timescale posterior
fig_name = f"{fig_base}_tzquench_posterior.png"
plot_quenching_timescale_posterior(t_quenchs, t_quench_best, t_quench_16, t_quench_50, t_quench_84, fig_dir=fig_dir, fig_name=fig_name, savefig=save_figs)

if show_plots:
     plt.show()
else:
    plt.close()