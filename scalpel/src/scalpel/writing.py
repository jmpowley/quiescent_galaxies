import os

import asdf

from .helpers import return_filters, return_wv_to_save

def save_simultaneous_fit_results_to_asdf(results_dict, prior_dict, cutout_kwargs, cube_kwargs, fit_kwargs):
    """Write results and other information to an asdf file"""

    # Return filters
    filters = return_filters(cutout_kwargs)
    wv_to_save = return_wv_to_save(cutout_kwargs, cube_kwargs, fit_kwargs)

    # Create data tree
    tree = {
        "results_dict" : results_dict,
        "prior_dict" : prior_dict,
        "filters" : filters,
        "wv_to_save" : wv_to_save,
        "cutout_kwargs" : cutout_kwargs,
        "cube_kwargs" : cube_kwargs,
        "fit_kwargs" : fit_kwargs,
    }

    # Build output string
    out_dir = fit_kwargs["out_dir"]
    fit_type = fit_kwargs["fit_type"]
    profile_str = fit_kwargs["profile_type"]
    method_str = fit_kwargs["method"]
    multifit_str = fit_kwargs["multifitter"]
    multifit_kwargs = fit_kwargs["multifit_kwargs"]
    if multifit_str == "poly":
        multifit_str = multifit_str + str(multifit_kwargs["poly_order"])
    elif multifit_str == "bspline":
        multifit_str = multifit_str + str(multifit_kwargs["N_knots"])
    # -- compile
    out_name = f"{profile_str}_{method_str}_simfit_{multifit_str}_results.asdf"

    print(f"Saved simultaneous fit result to: {out_name}")

    # Save results
    af = asdf.AsdfFile(tree)
    out_path = os.path.join(out_dir, out_name)
    af.write_to(out_path)