import numpy as np

from sedpy.observate import load_filters

from pysersic.rendering import HybridRenderer

from .loading import load_cube_data

def return_filters(cutout_kwargs):
    """Returns filters fit by pipeline"""

    filters = []
    for filter_kwargs in cutout_kwargs.values():
        filter = filter_kwargs['filter']
        filters.append(filter)

    return filters

def return_wv_to_save(cutout_kwargs, cube_kwargs, fit_kwargs):
    """Returns the wavelength vector used to produce a PySersic multi-band fit"""

    use_cube_wave = fit_kwargs["use_cube_wave"]
    invert_wave = fit_kwargs["invert_wave"]

    # -- use cube wavelength range
    if use_cube_wave:
        wave, _, _ = load_cube_data(**cube_kwargs)
        wv_to_save = wave
    # -- use grid of value in range of effective wavelengths
    else:
        filters = return_filters(cutout_kwargs)
        sedpy_filters = [load_filters(["jwst_" + filter])[0] for filter in filters]
        waveffs = [filter.wave_effective / 1e4 for filter in sedpy_filters] # TODO: Change to conversions
        waveffs = np.asarray(waveffs)
        wv_to_save = np.linspace(min(waveffs), max(waveffs), num=50)
    
    # Optionally, invert wavelengths
    if invert_wave:
        wv_to_save = 1 / wv_to_save

    return wv_to_save

def return_linked_param_at_wv(results_dict,  param, return_std : bool):

    # Load median and standard deviation
    param_at_wv, std_at_wav = results_dict[f'{param}_at_wv']

    if return_std:
        return param_at_wv, std_at_wav
    else:
        return param_at_wv
    
def return_linked_param_at_filter(results_dict,  param, filter, return_std : bool):

    # Load median and standard deviation
    median_at_filter, std_at_filter = results_dict[f'{param}_{filter}']

    if return_std:
        return median_at_filter, std_at_filter
    else:
        return median_at_filter

def return_const_param(results_dict, param, return_std : bool):

    # Load median and standard deviation
    param, std = results_dict[f'{param}']

    if return_std:
        return param, std
    else:
        return param
    
def return_individual_fit_posteriors(tree, params, return_medians : bool, return_uncs : bool):

    posterior = tree['posterior']
    param_posts = {}
    param_meds = []
    param_uncs = []

    for p in params:
        param_post = posterior[p][1]  # use second output...
        param_med = np.nanpercentile(param_post, q=50)
        param_16 = np.nanpercentile(param_post, q=16)
        param_84 = np.nanpercentile(param_post, q=84)

        param_posts[p] = param_post
        param_meds.append(param_med)
        param_uncs.append([param_med-param_16, param_84-param_med])

    # Optionally return image and residual
    if return_medians:
        if return_uncs:
            return param_posts, param_meds, param_uncs
        else:
            return param_posts, param_meds
    else:
        if return_uncs:
            return param_posts, param_uncs
        else:
            return param_posts
        
def return_joint_fit_medians_at_filter(joint_tree, params, filter, return_list : bool, return_unc : bool = False):

    # Load variables from tree
    joint_results_dict = joint_tree["results_dict"]
    fit_kwargs = joint_tree["fit_kwargs"]
    linked_params = fit_kwargs["linked_params"]
    const_params = fit_kwargs["const_params"]

    # Load parameter medians from results dict
    param_medians = {}
    param_uncs = {}
    for param in params:
        if param in const_params:
            param_median, param_unc = return_const_param(joint_results_dict, param=param, return_std=True)
        elif param in linked_params:
            param_median, param_unc = return_linked_param_at_filter(joint_results_dict, param=param, filter=filter, return_std=True)
        else:
            raise Exception(f"Parameter {param} not in list {linked_params + const_params}")
        param_medians[param] = param_median
        param_uncs[param] = param_unc

    # Optionally return uncertainty as well or convert return to lists
    if return_unc:
        if return_list:
            return list(param_medians.values()), list(param_uncs.values())
        else:
            return param_medians, param_uncs
    else:
        if return_list:
            return list(param_medians.values())
        else:
            return param_medians
    
def return_median_model_from_individual_tree(tree, profile_type, return_image : bool, return_residual : bool):

    # Load data
    input_data = tree["input_data"]
    image = np.asarray(input_data["image"])
    psf = np.asarray(input_data["psf"]).astype(np.float32)  # recast as numpy
    
    # Calculate medians from posterior samples
    posterior = tree["posterior"]
    param_medians = {key : np.nanmedian(val) for key, val in posterior.items()}

    # Render model and residual
    renderer = HybridRenderer(im_shape=image.shape, pixel_PSF=psf)
    median_model = np.asarray(renderer.render_source(param_medians, profile_type=profile_type))
    residual = image - median_model

    # Optionally return image and residual
    if return_image:
        if return_residual:
            return median_model, image, residual
        else:
            return median_model, image
    else:
        if return_residual:
            return median_model, residual
        else:
            return median_model

def return_median_model_from_joint_tree(joint_tree, individual_tree, profile_type, filter, return_image, return_residual):

    # Extract joint fit variables
    joint_results_dict = joint_tree["results_dict"]
    fit_kwargs = joint_tree["fit_kwargs"]
    linked_params = fit_kwargs["linked_params"]
    const_params = fit_kwargs["const_params"]
    all_params = linked_params + const_params

    # Load parameter medians from joint fits
    param_medians = {}
    for param in all_params:
        if param in const_params:
            param_median = return_const_param(joint_results_dict, param=param, return_std=False)
        if param in linked_params:
            param_median = return_linked_param_at_filter(joint_results_dict, param=param, filter=filter, return_std=False)
        param_medians[param] = param_median

    # Load data
    input_data = individual_tree["input_data"]
    image = np.asarray(input_data["image"])
    psf = np.asarray(input_data["psf"]).astype(np.float32)  # recast as numpy

    # Render model and residual
    renderer = HybridRenderer(im_shape=image.shape, pixel_PSF=psf)
    median_model = np.asarray(renderer.render_source(param_medians, profile_type=profile_type))
    residual = image - median_model

    # Optionally return image and residual
    if return_image:
        if return_residual:
            return median_model, image, residual
        else:
            return median_model, image
    else:
        if return_residual:
            return median_model, residual
        else:
            return median_model
