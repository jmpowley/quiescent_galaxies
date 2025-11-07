from .fitting import fit_independent_bands, fit_simultaneous_bands

class Scalpel:
    """Performs bulge-disc decomposition of IFU cubes"""

    def __init__(self, config):
        print("Scalpel initialised")

        self.config = config

    def dissect(self):

        # Load config info
        cutout_kwargs = self.config["cutout_kwargs"]
        prior_dict = self.config["prior_dict"]
        fit_kwargs = self.config["fit_kwargs"]

        fit_type = fit_kwargs["fit_type"]

        # Fit photometric bands depending on type
        # -- individual
        if fit_type == "individual":
            fit_independent_bands(cutout_kwargs=cutout_kwargs, prior_dict=prior_dict, **fit_kwargs)
        # -- multiple
        if fit_type == "multiple":
            fit_simultaneous_bands(cutout_kwargs=cutout_kwargs, prior_dict=prior_dict, **fit_kwargs)

    def run_scalpel(self):
        """Wrapper for scalpel. Prepares data for fitting and then runs dissection procedure"""

        # Load data
        self.dissect()