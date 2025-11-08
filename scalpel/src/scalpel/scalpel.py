from .fitting import fit_bands_independent, fit_bands_simultaneous

class Scalpel:
    """Performs bulge-disc decomposition of IFU cubes"""

    def __init__(self, config):
        print("Scalpel initialised")

        self.config = config

    def fit_bands(self, cutout_kwargs, prior_dict, fit_kwargs):
        """Fit photometric bands to obtain structural parameters"""
        
        # Fit photometric bands depending on type
        fit_type = fit_kwargs["fit_type"]
        # -- independently
        if fit_type == "independent":
            fit_bands_independent(cutout_kwargs=cutout_kwargs, prior_dict=prior_dict, **fit_kwargs)
        # -- simultaneously
        elif fit_type == "simultaneous":
            fit_bands_simultaneous(cutout_kwargs=cutout_kwargs, prior_dict=prior_dict, **fit_kwargs)
        else:
            raise ValueError("Options for fitting are 'independent' or 'simultaneous'")

    def dissect(self):
        """Wrapper for scalpel. Prepares data for fitting and then runs dissection procedure"""

        # Load config info
        cutout_kwargs = self.config["cutout_kwargs"]
        prior_dict = self.config["prior_dict"]
        fit_kwargs = self.config["fit_kwargs"]

        # Step 1: fit photometric bands to obtain structural parameters
        self.fit_bands(cutout_kwargs, prior_dict, fit_kwargs)

        # Step 2: extract structural components from results