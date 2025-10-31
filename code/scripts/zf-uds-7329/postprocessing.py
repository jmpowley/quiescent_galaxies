import numpy as np

from dynesty.utils import quantile as weighted_quantile

import astropy.units as u
from astropy.cosmology import z_at_value

import ast
import importlib
from typing import Optional, Tuple, Dict, Any

# def extract_model_kwargs(source: str) -> Optional[str]:
#     """
#     Parse a Python source string and return model_kwargs as dict or None
#     """
#     tree = ast.parse(source)

#     # Find all Assign nodes that target a Name 'model_kwargs'
#     model_nodes = []
#     for node in ast.walk(tree):
#         if isinstance(node, ast.Assign):
#             for t in node.targets:
#                 if isinstance(t, ast.Name) and t.id == "model_kwargs":
#                     model_nodes.append(node)

#     # Try and evaluate model nodes to extract model_kwargs 
#     model_kwargs = None
#     if model_nodes:
#         node = model_nodes[-1]   # take last assignment if multiple
#         try:
#             model_kwargs = ast.literal_eval(node.value)
#         except Exception:
#             model_kwargs = None

#     return model_kwargs

def find_globals_in_function(source, func_name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_node = node
            break
    else:
        raise KeyError(f"{func_name} not found")
    names = set()
    for n in ast.walk(func_node):
        if isinstance(n, ast.Name):
            names.add(n.id)
    param_names = {arg.arg for arg in func_node.args.args}
    builtins = {'True','False','None'}
    return sorted(names - param_names - builtins)

# a mapping for common names to import paths (expand as needed)
known_imports = {
    'np': ('numpy', None),  # (module_name, attribute) -> import module, take attribute if not None
    'priors': ('prospect.models.priors', None),
    'SpecModel': ('prospect.models.sedmodel', 'SpecModel'),
    'TemplateLibrary': ('prospect.models.templates', 'TemplateLibrary'),
    'convert_zred_to_agebins': ('prospect.models.transforms', 'convert_zred_to_agebins'),
    'convert_logmass_to_masses': ('prospect.models.transforms', 'convert_logmass_to_masses'),
    'convert_to_dust1': ('prospect.models.transforms', 'convert_to_dust1'),
    # add more mappings for names you commonly use
}

def try_import(name):
    """Try to import a name using known_imports; return object or raise ImportError."""
    if name not in known_imports:
        raise ImportError(f"No known import mapping for {name}")
    module_name, attr = known_imports[name]
    mod = importlib.import_module(module_name)
    return getattr(mod, attr) if attr else mod

def load_build_model_from_string(source, func_name='build_model'):
    ns = {}  # namespace to exec into
    exec(source, ns)               # run whole file (top-level definitions placed into ns)
    if func_name not in ns:
        raise KeyError(f"{func_name} not defined after exec")

    # wrapper that forwards any args/kwargs to the loaded function
    def call_with_kwargs(*args, **kwargs):
        try:
            return ns[func_name](*args, **kwargs)
        except NameError as e:
            # parse the function to find referenced globals
            missing_names = find_globals_in_function(source, func_name)
            # find which of these are missing from the namespace
            to_inject = [n for n in missing_names if n not in ns]
            injected = {}
            for name in to_inject:
                if name in known_imports:
                    try:
                        obj = try_import(name)
                        ns[name] = obj
                        injected[name] = obj
                    except Exception as imp_e:
                        # fallback behavior for TemplateLibrary (you can change to raising an error)
                        if name == 'TemplateLibrary':
                            ns[name] = {}
                            injected[name] = ns[name]
                        else:
                            raise ImportError(f"Could not import dependency '{name}': {imp_e}") from imp_e
                else:
                    # no mapping — you may want to raise or log
                    raise ImportError(f"No import mapping for dependency '{name}'; add it to known_imports")

            if injected:
                print("Injected dependencies into namespace:", list(injected.keys()))  # optional debug

            # Recall function with required globals
            return ns[func_name](*args, **kwargs)

    return call_with_kwargs, ns

def return_sfh(results, theta):
        """Returns the star formation history, or the  mass formed in each age bin, from the inputted theta values"""

        # Extract variables from results
        # -- parameters in chain
        chain_names = getattr(results['chain'].dtype, "names", None)
        # -- all model parameters
        model_params = results['model_params']
        agebins = model_params['agebins']
        nbins = np.squeeze(model_params['nbins_sfh'])

        # Create bins
        firstindex = chain_names.index('logsfr_ratios')
        sfr_bins = []  # Normalised SFR bins (so s_0 = 1)
        ratio_bins = 10**theta[firstindex:firstindex+nbins-1]  # bins of SFR ratios
        
        # Calculate age bin widths (yr):
        delta_t = np.array([10**(age_bin[1]) - 10**age_bin[0] for age_bin in agebins])
        for i in range(0, nbins):
            sfr_bins.append((1. / np.prod(ratio_bins[:i])))
        sfr_bins = np.array(sfr_bins)

        # Calculate mass formed in each bins
        log_mass = theta[chain_names.index('logmass')]
        # zred = theta[chain_names.index('zred')]
        M_bin = sfr_bins * 10**log_mass / np.sum(delta_t * sfr_bins)
        M_bin = np.squeeze(M_bin).tolist()
        
        return M_bin, log_mass

def return_sfh_for_one_sigma_quantiles(sfh_chain, weights):
    """Returns the 16th, 50th and 84th quantile of a chain of star formation histories using the weights of the results"""
    
    # Get the weighted quantiles of the SFH chain
    sfh_16, sfh_50, sfh_84 = np.squeeze(np.array([
        [weighted_quantile(sfh, q=quantile, weights=weights) for sfh in sfh_chain.T] for quantile in (.16, .50, .84)]
        ))
    
    return sfh_16, sfh_50, sfh_84

def return_sfh_chain(results):
    """ Return the chain of star formation histories from the `prospector` numeric/unstructured chain of model results"""

    # Extract variables from results
    numeric_chain = results["unstructured_chain"]

    # Call return_sfh at each point on the chain
    sfh_chain = np.array([return_sfh(results, theta)[0] for theta in numeric_chain])

    return sfh_chain

def return_assembly_time(q, sfh, age_bins, log_mass=None, age_units=u.Gyr, return_quantity=False):
    """Returns the time that specified fraction, q, of mass formed given a galaxies star formation history"""

    # Define widths of each age bin (yr):
    delta_t = np.array([10**(abin[1]) - 10**abin[0] for abin in age_bins])

    # Calculate total mass formed
    if log_mass is not None:
        mass_tot = 10**log_mass
    else:
        mass_tot = np.sum(sfh*delta_t)
    target = mass_tot * q  # target percentile of mass

    # Calculate cumulative mass formed in each bin
    mass_per_bin = sfh * delta_t
    mass_cum = np.cumsum(mass_per_bin)

    # Calculate index of percentile target
    qindex = int(np.searchsorted(mass_cum, target, side='left'))
    
    # Subtract remaining time from target percentile bin's upper bound
    excess = mass_cum[qindex] - mass_tot * q
    q_time = (10**age_bins[qindex][1] - excess / sfh[qindex]) * u.yr

    # Convert to age units specified by user 
    if age_units is not None:
        try:
            q_time = q_time.to(age_units)
        except Exception as e:
            print(f"Error converting to units {age_units}: {e}")
            print("Returning assembly time in years")

    if return_quantity:
        return q_time
    else:
        return q_time.value

def return_assembly_time_for_one_sigma_quantities(q, sfh_chain, age_bins, weights, age_units=u.Gyr, return_distribution=False, return_quantity=False):

    # Loop over each sfh chain
    q_times = []
    for sfh in sfh_chain:
        # Convert assembly to value in the specified units
        q_time = return_assembly_time(q=q, sfh=sfh, age_bins=age_bins, return_quantity=True)
        if hasattr(q_time, 'to'):
            q_val = q_time.to(age_units).value  # astropy Quantity
        else:
            q_val = q_time
        q_times.append(q_val)
    q_times = np.asarray(q_times)  # convert to array

    # Compute weighted fractions
    q_times_weighted = None  # TODO: Calculate weighted quantile

    # Compute the weighted quantiles of the SFH chain
    q_time_16 = np.squeeze(weighted_quantile(q_times, q=0.16, weights=weights))
    q_time_50 = np.squeeze(weighted_quantile(q_times, q=0.50, weights=weights))
    q_time_84 = np.squeeze(weighted_quantile(q_times, q=0.84, weights=weights))
    
    if return_distribution:
        if return_quantity:
            return q_time_16  * age_units, q_time_50 * age_units, q_time_84 * age_units, q_times * age_units
        else:
            return q_time_16, q_time_50, q_time_84, q_times
    else:
        return q_time_16, q_time_50, q_time_84

def convert_lookback_at_redshift_to_z(delta_t, z_start, cosmo, time_unit=u.Gyr, return_quantity=False):

    # Ensure delta_t is a quantity
    if not hasattr(delta_t, 'unit'):
        delta_t = delta_t * time_unit

    # Make total loookback time
    t_start = cosmo.lookback_time(z_start)
    total_lookback = t_start + delta_t

    print("total_lookback:", total_lookback)

    # Convert to redshift
    f = lambda z: cosmo.lookback_time(z)
    z_targ = z_at_value(f, total_lookback)  # invert

    if return_quantity:
        return z_targ
    else:
        return z_targ.value