import os

import numpy as np
from sedpy import observate

import jax
import jax.numpy as jnp
from jax.random import (
    PRNGKey,  # Need to use a seed to start jax's random number generation
)

from pysersic.priors import SourceProperties
from pysersic import FitSingle
from pysersic.multiband import FitMultiBandPoly

from .loading import load_cutout_data, set_priors
from .plotting import call_plot_residual


def fit_band(im, mask, sig, psf, prior, loss_func, seed, verbose):

    # Setup fitter
    fitter = FitSingle(data=im, rms=sig ,mask=mask, psf=psf, prior=prior, loss_func=loss_func)

    # Sampling
    fitter.sample(rkey=PRNGKey(seed))
    if verbose:
        print(fitter.sampling_results.retrieve_param_quantiles(return_dataframe=True))

    return fitter

def fit_independent_bands(cutout_kwargs, prior_dict, prior_type, profile_type, sky_type, loss_func, seed, verbose, out_dir, fig_dir, **extras):

    for filter_kwargs in cutout_kwargs.values():

        # Obtain effective wavelengths
        filter = filter_kwargs["filter"]

        # Load data
        im, mask, sig, psf = load_cutout_data(**filter_kwargs)

        # Set priors
        # if prior_type == "single":
        #     prior = set_single_component_prior(im, mask)
        # elif prior_type == "double":
        #     prior = set_double_component_prior(im, mask)
        prior = set_priors(im, mask, profile_type, sky_type, prior_dict)

        # Fit band
        fitter = fit_band(im, mask, sig, psf, prior, loss_func, seed, verbose)

        # Estimate posterior
        # rkey,_ = jax.random.split(rkey, 2) # use different random number key for each run
        # result = fitter.estimate_posterior(rkey=rkey)

        # Save results
        out_name = f"{prior_type}_fit_{filter}.asdf"
        out_path = os.path.join(out_dir, out_name)
        fitter.sampling_results.save_result(out_path)

        # Make plots
        # -- residual
        fig = call_plot_residual(fitter, im, psf, profile_type)
        fig_name = f"{prior_type}_{filter}_residual.pdf"
        fig.savefig(os.path.join(fig_dir, fig_name))
        # -- corner
        fig = fitter.sampling_results.corner(color='C0') 
        fig_name = f"{prior_type}_{filter}_corner.pdf"
        fig.savefig(os.path.join(fig_dir, fig_name))

    # return result

def fit_simultaneous_bands(cutout_kwargs, prior_dict, prior_type, profile_type, sky_type, loss_func, seed, verbose, out_dir, **extras):
    
    filters = []
    waveffs = []

    im_dict = {}
    mask_dict = {}
    rms_dict = {}
    psf_dict = {}
    prior_dict = {}

    fitter_dict = {}
    ind_result_dict = {}

    # Loop over each set of kwargs
    for filter_kwargs in cutout_kwargs.values():

        # Obtain effective wavelengths
        filter = filter_kwargs["filter"]
        sedpy_filter = observate.load_filters(["jwst_" + filter])

        # Load data
        im, sig, mask, psf = load_cutout_data(**filter_kwargs)

        # Set priors
        # if prior_type == "single":
        #     prior = set_single_component_prior(im, mask)
        # elif prior_type == "double":
        #     prior = set_double_component_prior(im, mask)
        prior = set_priors(im, mask, profile_type, sky_type, prior_dict)

        # Fit band
        fitter = fit_band(im, mask, sig, psf, prior, loss_func, seed, verbose, save_plots=None, fig_dir=None)

        # Estimate posterior
        rkey,_ = jax.random.split(rkey, 2) # use different random number key for each run
        ind_result = fitter.estimate_posterior(rkey=rkey)

        # Save results
        out_name = f"{prior_type}_fit_{filter}.asdf"
        out_path = os.path.join(out_dir, out_name)
        fitter.sampling_results.save_result(out_path)

        # Save to lists/dicts
        filters.append(filter)
        waveffs.append(sedpy_filter.wave_effective / 1e4)  # TODO: Change to conversions

        im_dict[filter] = im
        mask_dict[filter] = mask
        rms_dict[filter] = sig
        psf_dict[filter] = psf
        prior_dict[filter] = prior
        
        ind_result_dict[filter] = ind_result.retrieve_med_std()
        fitter_dict[filter] = fitter

    # Create fine grid of wavelengths
    wv_to_save = np.linspace(min(waveffs),max(waveffs), num=50)

    # Create MultiFitter object
    # -- single component
    if prior_type == "single":
        MultiFitter = FitMultiBandPoly(fitter_list=[fitter_dict[b] for b in filters],
                                    wavelengths=waveffs,
                                    band_names=filters,
                                    linked_params=['n','ellip','r_eff'],
                                    const_params=['xc','yc','theta'],
                                    wv_to_save=wv_to_save,
                                    poly_order=2
                                    )
    # -- double component
    elif prior_type == "double":
        MultiFitter = FitMultiBandPoly(fitter_list=[fitter_dict[b] for b in filters],
                                    wavelengths=waveffs,
                                    band_names=filters,
                                    linked_params=['f_1'],
                                    const_params=['xc','yc','n','theta','r_eff_1','ellip_1','r_eff_2','ellip_2'],
                                    wv_to_save= wv_to_save,
                                    poly_order = 2
                                    )
    
    # Estimate posterior
    rkey,_ = jax.random.split(rkey,2)
    multi_result = MultiFitter.estimate_posterior(method='svi-flow', rkey=rkey)

    return multi_result