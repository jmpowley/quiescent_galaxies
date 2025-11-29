import os

import numpy as np
from sedpy.observate import load_filters

import matplotlib.pyplot as plt

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
from .plotting import call_plot_residual, call_plot_image, make_plots
from .helpers import return_linked_param_at_wv, return_const_param

def _set_priors(image, mask, profile_type, sky_type, prior_dict):
    """Sets priors for model in PySersic"""

    # Generate priors from image
    # props = SourceProperties(image) # Optional mask
    # prior = props.generate_prior(profile_type=profile_type, sky_type=sky_type)
    prior = autoprior(image=image, profile_type=profile_type, mask=mask, sky_type=sky_type)

    # Set priors from config dict
    for prior_type, prior_type_dict in prior_dict.items():
        # -- uniform priors
        if prior_type == "uniform":
            for param, range in prior_type_dict.items():
                lo, hi = range
                prior.set_uniform_prior(param, lo, hi)
        # -- gaussian priors
        if prior_type == "gaussian":
            for param, gauss in prior_type_dict.items():
                loc, std = gauss
                prior.set_gaussian_prior(param, loc, std)

    return prior

def fit_band(data, mask, rms, psf, prior, loss_func, method, rkey, verbose):
    """Fits individual band"""

    if verbose:
        print("Starting fit...")

    loss_map = {
        "student_t": loss.student_t_loss,
        "gaussian": loss.gaussian_loss,
        "cash": loss.cash_loss,
    }
    loss_func = loss_map[loss_func]

    # Setup fitter
    fitter = FitSingle(data=data, rms=rms, mask=mask, psf=psf, prior=prior, loss_func=loss_func)

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
        print("Result from fit:")
        if method == "mcmc":
            print(result.retrieve_param_quantiles(return_dataframe=True))

    fitter_map = fitter.find_MAP()
    print("MAP of fit:\n", fitter_map)

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

        # Load cutout data
        image, mask, sig, psf = load_cutout_data(**filter_kwargs)

        # Set priors
        prior = _set_priors(image=image, mask=mask, profile_type=profile_type, sky_type=sky_type, prior_dict=prior_dict)

        # Fit band
        rkey, rkey_fit = jax.random.split(rkey)
        fitter, result = fit_band(data=image, mask=mask, rms=sig, psf=psf, prior=prior, loss_func=loss_func, method=method, rkey=rkey_fit, verbose=verbose)

        # Save results
        out_name = f"{profile_type}_{method}_fit_{filter}.asdf"
        out_path = os.path.join(out_dir, out_name)
        result.save_result(out_path)

        # Make plots
        # -- data
        fig = call_plot_image(image, mask, sig, psf)
        fig_name = f"{profile_type}_{filter}_data.pdf"
        fig.savefig(os.path.join(fig_dir, fig_name))
        # -- residual
        fig = call_plot_residual(fitter, image, mask, psf, profile_type)
        fig_name = f"{profile_type}_{filter}_residual.pdf"
        fig.savefig(os.path.join(fig_dir, fig_name))
        # -- corner
        fig = fitter.sampling_results.corner(color='C0') 
        fig_name = f"{profile_type}_{filter}_corner.pdf"
        fig.savefig(os.path.join(fig_dir, fig_name))

        plt.show()

    return fitter, result

def fit_bands_simultaneous(cutout_kwargs, cube_kwargs, in_prior_dict, linked_params, const_params, profile_type, sky_type, loss_func, method, multifitter, multifit_kwargs, use_cube_wave, invert_wave, seed, verbose, out_dir, fig_dir, **extras):
    
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

        # Load cutout data
        image, mask, sig, psf = load_cutout_data(**filter_kwargs)

        # Create PySersic priors
        prior = _set_priors(image=image, mask=mask, profile_type=profile_type, sky_type=sky_type, prior_dict=in_prior_dict)
        print(prior)

        # Fit band
        rkey, rkey_fit = jax.random.split(rkey)
        fitter, result = fit_band(data=image, 
                                  mask=mask, 
                                  rms=sig, 
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
        make_plots(fitter, image, mask, sig, psf, profile_type=profile_type, method=method, filter=filter, fig_dir=fig_dir)

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
    waveffs = np.asarray(waveffs)
    if use_cube_wave and cube_kwargs is not None:
        wave, _, _ = load_cube_data(**cube_kwargs)
        wv_to_save = wave
    else:
        wv_to_save = np.linspace(min(waveffs), max(waveffs), num=50)
    # -- invert wavelengths
    if invert_wave:
        waveffs = 1 / waveffs
        wv_to_save = 1 / wv_to_save

    # Create MultiFitter object
    # -- polynomial
    if multifitter == "poly":
        MultiFitter = FitMultiBandPoly(fitter_list=[fitter_dict[f] for f in filters],
                                    wavelengths=waveffs,
                                    band_names=filters,
                                    linked_params=linked_params,
                                    const_params=const_params,
                                    wv_to_save=wv_to_save,
                                    **multifit_kwargs,
                                    )
    # -- b-spline
    elif multifitter == "bspline":
        MultiFitter = FitMultiBandBSpline(fitter_list=[fitter_dict[f] for f in filters],
                                    wavelengths=waveffs,
                                    band_names=filters,
                                    linked_params=linked_params,
                                    const_params=const_params,
                                    wv_to_save=wv_to_save,
                                    **multifit_kwargs,
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

def fit_cube(cube_kwargs, results_dict, linked_params, const_params, profile_type, sky_type, loss_func, method, seed, verbose, **extras):

    return

    rkey = PRNGKey(seed)

    # Load cube data
    wave, cube, cube_err = load_cube_data(**cube_kwargs)
    nlam, ny, nx = cube.shape

    # Load values for parameters
    linked_params_dict = {p: return_linked_param_at_wv(results_dict, param=p, return_std=True) for p in linked_params}
    const_params_dict = {p: return_const_param(results_dict, param=p, return_std=True) for p in const_params}

    # 
    bulge_fraction = []

    # Loop over each wavelength slice
    for i in range(nlam):
        slice = cube[i, :, :]
        slice_err = cube_err[i, :, :]

        # Build Gaussian priors for this slice using medians and stds
        # -- add linked parameters
        slice_gaussian_priors = {}
        for param, (meds, stds) in linked_params_dict.items():
            # -- do not limit f_1
            if profile_type == "sersic_exp" and param != "f_1":
                slice_gaussian_priors[param] = (meds[i], stds[i])
        # -- add constant parameters
        slice_const_priors = {}
        for param, (meds, stds) in const_params_dict.items():
            slice_const_priors[param] = (meds, stds)
        # -- combine into a dict
        slice_prior_dict = {
            "uniform" : {"f_1" : (0.0, 1.0)},
            "gaussian": {**slice_const_priors, **slice_gaussian_priors},
        }
        # -- set priors
        slice_prior = _set_priors(image=slice, mask=mask, profile_type=profile_type, sky_type=sky_type, prior_dict=slice_prior_dict)

        # Fit individual slice
        rkey, rkey_fit = jax.random.split(rkey)
        fitter, result = fit_band(data=slice, 
                                  mask=mask, 
                                  rms=slice_err, 
                                  psf=psf, 
                                  prior=slice_prior, 
                                  loss_func=loss_func, 
                                  method=method, 
                                  rkey=rkey_fit, 
                                  verbose=verbose, 
                                  )
        
        # Add bulge fraction information to list
        
    return fitter, result