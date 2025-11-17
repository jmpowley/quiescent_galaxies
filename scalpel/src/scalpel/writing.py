import os

import asdf

from .helpers import return_filters, return_wv_to_save

def save_photometry_fit_results_to_asdf(results_dict, prior_dict, cutout_kwargs, cube_kwargs, fit_kwargs):
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
    profile_str = fit_kwargs["profile_type"]
    method_str = fit_kwargs["method"]
    multi_str = fit_kwargs["multifitter"]
    fit_str = f"sim_{method_str}_{multi_str}" if fit_kwargs["fit_type"] == "simultaneous" else "ind_{method_str}"
    # -- compile
    out_name = f"{profile_str}_{fit_str}_fit_results.asdf"

    # Save results
    af = asdf.AsdfFile(tree)
    out_path = os.path.join(out_dir, out_name)
    af.write_to(out_path)