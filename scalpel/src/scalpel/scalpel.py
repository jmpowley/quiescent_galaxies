from .fitting import fit_bands_independent, fit_bands_simultaneous, fit_cube
from .writing import save_simultaneous_fit_results_to_asdf

class Scalpel:
    """Performs bulge-disc decomposition of IFU cubes"""

    def __init__(self, config):
        print("Scalpel initialised")

        self.config = config

    def fit_bands(self, cutout_kwargs, cube_kwargs, prior_dict, fit_kwargs):
        """Fit photometric bands to obtain structural parameters"""
        
        # Fit photometric bands depending on type
        fit_type = fit_kwargs["fit_type"]
        # -- independently
        if fit_type == "independent":
            fit_bands_independent(cutout_kwargs=cutout_kwargs, prior_dict=prior_dict, **fit_kwargs)
        # -- simultaneously
        elif fit_type == "simultaneous":
            results_dict = fit_bands_simultaneous(cutout_kwargs=cutout_kwargs, cube_kwargs=cube_kwargs, in_prior_dict=prior_dict, **fit_kwargs)
        else:
            raise ValueError("Options for fitting are 'independent' or 'simultaneous'")

        return results_dict

    def fit_cube(self, cube_kwargs, results_dict, fit_kwargs):
        """Fit cube slices using structural parameters from fit"""

        fit_cube(cube_kwargs=cube_kwargs, results_dict=results_dict, **fit_kwargs)

    def dissect(self):
        """Wrapper for scalpel. Prepares data for fitting and then runs dissection procedure"""

        # Load config info
        cutout_kwargs = self.config["cutout_kwargs"]
        cube_kwargs = self.config["cube_kwargs"]
        prior_dict = self.config["prior_dict"]
        fit_kwargs = self.config["fit_kwargs"]

        # Fit photometric bands to obtain structural parameters
        results_dict = self.fit_bands(cutout_kwargs=cutout_kwargs, cube_kwargs=cube_kwargs, prior_dict=prior_dict, fit_kwargs=fit_kwargs)
        save_simultaneous_fit_results_to_asdf(results_dict, prior_dict, cutout_kwargs, cube_kwargs, fit_kwargs)

        # Extract structural components from results
        self.fit_cube(cube_kwargs=cube_kwargs, results_dict=results_dict, fit_kwargs=fit_kwargs)