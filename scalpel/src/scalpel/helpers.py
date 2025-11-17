import numpy as np

from sedpy.observate import load_filters

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