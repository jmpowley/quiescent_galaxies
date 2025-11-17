import os

import numpy as np
from sedpy.observate import load_filters

import jax
import jax.numpy as jnp
from jax.random import (
    PRNGKey,  # Need to use a seed to start jax's random number generation
)

from pysersic import FitSingle
from pysersic.multiband import FitMultiBandPoly, FitMultiBandBSpline
from pysersic.priors import autoprior, SourceProperties
from pysersic import loss

from .loading import load_cutout_data, load_cube_data
from .plotting import call_plot_residual

def _set_priors(image, mask, profile_type, sky_type, prior_dict):
    """Sets priors for model in PySersic"""

    # Generate priors from image
    # props = SourceProperties(im, mask=mask) # Optional mask
    # prior = props.generate_prior(profile_type=profile_type, sky_type=sky_type)
    prior = autoprior(image=image, profile_type=profile_type, mask=mask, sky_type=sky_type)

    # Set uniform priors from dict
    for prior_type, prior_type_dict in prior_dict.items():
        if prior_type == "uniform":
            for param, range in prior_type_dict.items():
                lo, hi = range
                prior.set_uniform_prior(param, lo, hi)

    return prior

def fit_band(image, mask, sig, psf, prior, loss_func, method, rkey, verbose):

    if verbose:
        print("Starting fit...")

    loss_map = {
        "student_t": loss.student_t_loss,
        "gaussian": loss.gaussian_loss,
        "cash": loss.cash_loss,
    }
    loss_func = loss_map[loss_func]

    # Setup fitter
    fitter = FitSingle(data=image, rms=sig ,mask=mask, psf=psf, prior=prior, loss_func=loss_func)

    # Estimate posterior
    rkey, rkey_est = jax.random.split(rkey, 2) # use different random number key for each run
    # -- MCMC sampling
    if method == "mcmc":
        fitter.sample(rkey=rkey_est)
        result = fitter.sampling_results
    # -- SVI
    elif method == "svi-flow":
        result = fitter.estimate_posterior(rkey=rkey_est)
    if verbose:
        print("Results from fit:")
        if method == "mcmc":
            print(fitter.sampling_results.retrieve_param_quantiles(return_dataframe=True))

    return fitter, result

def fit_bands_independent(cutout_kwargs, prior_dict, profile_type, sky_type, loss_func, method, seed, verbose, out_dir, fig_dir, **extras):

    rkey = PRNGKey(seed)

    # Loop over each set fo kwargs
    for filter_kwargs in cutout_kwargs.values():

        # Obtain effective wavelengths
        filter = filter_kwargs["filter"]
        if verbose:
            print("------------------------")
            print(f"Independent fit of {filter.upper()}")

        # Load data
        image, mask, sig, psf = load_cutout_data(**filter_kwargs)

        # Set priors
        prior = _set_priors(image=image, mask=mask, profile_type=profile_type, sky_type=sky_type, prior_dict=prior_dict)

        # Fit band
        rkey, rkey_fit = jax.random.split(rkey)
        fitter, result = fit_band(image=image, mask=mask, sig=sig, psf=psf, prior=prior, loss_func=loss_func, method=method, rkey=rkey_fit, verbose=verbose)

        # Save results
        out_name = f"{profile_type}_fit_{filter}.asdf"
        out_path = os.path.join(out_dir, out_name)
        fitter.sampling_results.save_result(out_path)

        # Make plots
        # -- residual
        fig = call_plot_residual(fitter, image, mask, psf, profile_type)
        fig_name = f"{profile_type}_{filter}_residual.pdf"
        fig.savefig(os.path.join(fig_dir, fig_name))
        # -- corner
        fig = fitter.sampling_results.corner(color='C0') 
        fig_name = f"{profile_type}_{filter}_corner.pdf"
        fig.savefig(os.path.join(fig_dir, fig_name))

    return fitter, result

def fit_bands_simultaneous(cutout_kwargs, cube_kwargs, in_prior_dict, linked_params, const_params, profile_type, sky_type, loss_func, method, multifitter, use_cube_wave, invert_wave, seed, verbose, out_dir, fig_dir, **extras):
    
    rkey = PRNGKey(seed)

    filters = []
    waveffs = []

    im_dict = {}
    mask_dict = {}
    rms_dict = {}
    psf_dict = {}
    prior_dict = {}

    fitter_dict = {}
    result_dict = {}

    # Loop over each set of kwargs
    for filter_kwargs in cutout_kwargs.values():

        # Obtain effective wavelengths
        filter = filter_kwargs["filter"]
        print("filter:", filter.upper())
        sedpy_filter = load_filters(["jwst_" + filter])[0]

        # Load data
        image, mask, sig, psf = load_cutout_data(**filter_kwargs)

        # Create PySersic priors
        prior = _set_priors(image=image, mask=mask, profile_type=profile_type, sky_type=sky_type, prior_dict=in_prior_dict)

        # Fit band
        rkey, rkey_fit = jax.random.split(rkey)
        fitter, result = fit_band(image=image, 
                                  mask=mask, 
                                  sig=sig, 
                                  psf=psf, 
                                  prior=prior, 
                                  loss_func=loss_func, 
                                  method=method, 
                                  rkey=rkey_fit, 
                                  verbose=verbose,
                                  )

        # Save results
        out_name = f"{profile_type}_{method}_fit_{filter}.asdf"
        out_path = os.path.join(out_dir, out_name)
        result.save_result(out_path)

        # Make plots
        # -- residual
        fig = call_plot_residual(fitter, image, mask, psf, profile_type)
        fig_name = f"{profile_type}_{method}_{filter}_residual.pdf"
        fig.savefig(os.path.join(fig_dir, fig_name))
        # -- corner
        fig = fitter.sampling_results.corner(color='C0')
        fig_name = f"{profile_type}_{method}_{filter}_corner.pdf"
        fig.savefig(os.path.join(fig_dir, fig_name))

        # Save to lists/dicts
        filters.append(filter)
        waveffs.append(sedpy_filter.wave_effective / 1e4)  # TODO: Change to conversions

        im_dict[filter] = image
        mask_dict[filter] = mask
        rms_dict[filter] = sig
        psf_dict[filter] = psf
        prior_dict[filter] = prior
        
        result_dict[filter] = result.retrieve_med_std()
        fitter_dict[filter] = fitter

    # Create fine grid of wavelengths
    if use_cube_wave and cube_kwargs is not None:
        wave, _, _ = load_cube_data(**cube_kwargs)
        wv_to_save = wave
    else:
        waveffs = np.asarray(waveffs)
        wv_to_save = np.linspace(min(waveffs), max(waveffs), num=50)
    # -- invert wavelengths
    if invert_wave:
        wv_to_save = 1 / wv_to_save

    # Create MultiFitter object
    # -- polynomial
    if multifitter == "poly":
        MultiFitter = FitMultiBandPoly(fitter_list=[fitter_dict[f] for f in filters],
                                    wavelengths=wv_to_save,
                                    band_names=filters,
                                    linked_params=linked_params,
                                    const_params=const_params,
                                    wv_to_save=wv_to_save,
                                    poly_order=2,
                                    )
    # -- b-spline
    elif multifitter == "bspline":
        MultiFitter = FitMultiBandBSpline(fitter_list=[fitter_dict[f] for f in filters],
                                    # wavelengths=inv_waveffs,
                                    wavelengths=wv_to_save,
                                    band_names=filters,
                                    linked_params=linked_params,
                                    const_params=const_params,
                                    wv_to_save=wv_to_save,
                                    N_knots=4,
                                    spline_k=2,
                                    pad_knots=True,
                                    )
    
    # Estimate posterior
    rkey, rkey_multifit = jax.random.split(rkey, 2)
    # -- MCMC sampling
    if method == "mcmc":
        MultiFitter.sample(rkey=rkey_multifit)
        results = MultiFitter.sampling_results
    # -- SVI
    elif method == "svi-flow":
        results = MultiFitter.estimate_posterior(method=method, rkey=rkey_multifit)
    results_dict = results.retrieve_med_std()

    return results_dict

def return_linked_param(results_dict, wv_to_save, param, invert_wave : bool, return_std : bool):

    # Load median and standard deviation
    param_at_wv, std_at_wav = results_dict[f'{param}_at_wv']

    # Return in inverted or normal wavelengths
    if invert_wave:
        wv_out = 1 / wv_to_save
    else:
        wv_out = wv_to_save

    if return_std:
        return wv_out, param_at_wv, std_at_wav
    else:
        return wv_out, param_at_wv

def return_const_param(results_dict, param, return_std : bool):

    # Load median and standard deviation
    param, std = results_dict[f'{param}']

    if return_std:
        return param, std
    else:
        return param