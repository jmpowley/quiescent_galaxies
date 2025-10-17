import numpy as np

import astropy.units as u

# ----------------------
# Wavelength conversions
# ----------------------
def convert_wave_m_to_um(wave_m, return_quantity=False, return_unit=False):
    """Convert wavelength values from meters to microns
    """

    # Assign units
    if not isinstance(wave_m, u.Quantity):
        wave_m = wave_m * u.m

    # Convert wavelength to microns
    wave_um = wave_m.to(u.um)
    wave_unit = u.um

    # Optionally return unit and quantity 
    if return_unit:
        if return_quantity:
            return wave_um, wave_unit
        else:
            return wave_um.value, wave_unit
    else:
        return wave_um if return_quantity else wave_um.value
    
def convert_wave_um_to_m(wave_um, return_quantity=False, return_unit=False):
    """Convert wavelength values from microns to metres
    """

    # Assign units
    if not isinstance(wave_um, u.Quantity):
        wave_um = wave_um * u.um

    # Convert wavelength to metres
    wave_m = wave_um.to(u.m)
    wave_unit = u.m

    # Optionally return unit and quantity 
    if return_unit:
        if return_quantity:
            return wave_m, wave_unit
        else:
            return wave_m.value, wave_unit
    else:
        return wave_m if return_quantity else wave_m.value

def convert_wave_A_to_um(wave_A, return_quantity=False, return_unit=False):

    # Assign units
    if not isinstance(wave_A, u.Quantity):
        wave_A = wave_A * u.AA

    # Convert wavelength to microns
    wave_um = wave_A.to(u.um)
    wave_unit = u.um

    # Optionally return unit and quantity 
    if return_unit:
        if return_quantity:
            return wave_um, wave_unit
        else:
            return wave_um.value, wave_unit
    else:
        return wave_um if return_quantity else wave_um.value
    
def convert_wave_um_to_A(wave_um, return_quantity=False, return_unit=False):

    # Assign units
    if not isinstance(wave_um, u.Quantity):
        wave_um = wave_um * u.um

    # Convert wavelength to metres
    wave_A = wave_um.to(u.AA)
    wave_unit = u.AA

    # Optionally return unit and quantity 
    if return_unit:
        if return_quantity:
            return wave_A, wave_unit
        else:
            return wave_A.value, wave_unit
    else:
        return wave_A if return_quantity else wave_A.value
    
def convert_wave_m_to_A(wave_m, return_quantity=False, return_unit=False):

    # Assign units
    if not isinstance(wave_m, u.Quantity):
        wave_m = wave_m * u.m

    # Convert wavelength to metres
    wave_A = wave_m.to(u.AA)
    wave_unit = u.AA

    # Optionally return unit and quantity 
    if return_unit:
        if return_quantity:
            return wave_A, wave_unit
        else:
            return wave_A.value, wave_unit
    else:
        return wave_A if return_quantity else wave_A.value
    
def convert_wave_A_to_m(wave_A, return_quantity=False, return_unit=False):

    # Assign units
    if not isinstance(wave_A, u.Quantity):
        wave_A = wave_A * u.AA

    # Convert wavelength to metres
    wave_m = wave_A.to(u.m)
    wave_unit = u.m

    # Optionally return unit and quantity 
    if return_unit:
        if return_quantity:
            return wave_m, wave_unit
        else:
            return wave_m.value, wave_unit
    else:
        return wave_m if return_quantity else wave_m.value

# -----------------------------
# Flux/flux density conversions
# -----------------------------
def convert_flux_jy_to_ujy(flux_jy, err_jy, return_quantity=False, return_unit=False):

    # Assign units
    if not isinstance(flux_jy, u.Quantity):
        flux_jy = flux_jy * u.Jy
    if not isinstance(err_jy, u.Quantity):
        err_jy = err_jy * u.Jy

    # convert to microjanskies
    flux_ujy = flux_jy.to(u.uJy)
    err_ujy = err_jy.to(u.uJy)
    flux_unit = u.uJy

    # Optionally return quantity and unit
    if return_unit:
        if return_quantity:
            return flux_ujy, err_ujy, flux_unit
        else:
            return flux_ujy.value, err_ujy.value, flux_unit
    else:
        if return_quantity:
            return flux_ujy, err_ujy
        else:
            return flux_ujy.value, err_ujy.value
    
def convert_flux_ujy_to_jy(flux_ujy, err_ujy, return_quantity=False, return_unit=False):

    # Assign units
    if not isinstance(flux_ujy, u.Quantity):
        flux_ujy = flux_ujy * u.uJy
    if not isinstance(err_ujy, u.Quantity):
        err_ujy = err_ujy * u.uJy

    # convert to microjanskies
    flux_jy = flux_ujy.to(u.Jy)
    err_jy = err_ujy.to(u.Jy)
    flux_unit = u.Jy

    # Optionally return quantity and unit
    if return_unit:
        if return_quantity:
            return flux_jy, err_jy, flux_unit
        else:
            return flux_jy.value, err_jy.value, flux_unit
    else:
        if return_quantity:
            return flux_jy, err_jy
        else:
            return flux_jy.value, err_jy.value

def convert_flux_si_to_jy(wave_m, flux_si, err_si, return_quantity=False, return_unit=False):

    # Assign units
    if not isinstance(wave_m, u.Quantity):
        wave_m = wave_m * u.m
    if not isinstance(flux_si, u.Quantity):
        flux_si = flux_si * u.Unit('W / m3')
    if not isinstance(err_si, u.Quantity):
        err_si = err_si * u.Unit('W / m3')

    # convert to janksies
    flux_unit = u.Jy
    flux_jy = flux_si.to(flux_unit, equivalencies=u.spectral_density(wave_m))
    err_jy = err_si.to(flux_unit, equivalencies=u.spectral_density(wave_m))

    # Optionally return quantity and unit
    if return_unit:
        if return_quantity:
            return flux_jy, err_jy, flux_unit
        else:
            return flux_jy.value, err_jy.value, flux_unit
    else:
        if return_quantity:
            return flux_jy, err_jy
        else:
            return flux_jy.value, err_jy.value

def convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1.0, return_quantity=False, return_unit=False):
    """Convert flux values from janksies to cgs units

    Optionally, supply `cgs_factor` to scale orders of magnitude or return result as `astropy.units.quantity.Quantity` object
    """

    # Assign units
    if not isinstance(wave_m, u.Quantity):
        wave_m = wave_m * u.m
    if not isinstance(flux_jy, u.Quantity):
        flux_jy = flux_jy * u.Jy
    if not isinstance(err_jy, u.Quantity):
        err_jy = err_jy * u.Jy

    # convert to cgs units
    flux_unit = u.erg/(u.s * u.cm**2 * u.AA) * cgs_factor
    flux_cgs = flux_jy.to(flux_unit, equivalencies=u.spectral_density(wave_m))
    err_cgs = err_jy.to(flux_unit, equivalencies=u.spectral_density(wave_m))

    # # Optionally return quantity and unit
    # if return_unit:
    #     if return_quantity:
    #         return flux_cgs, err_cgs, flux_unit
    #     else:
    #         return flux_cgs.value, err_cgs.value, flux_unit
    # else:
    #     return flux_cgs, err_cgs if return_quantity else flux_cgs.value, err_cgs.value
    
    # Optionally return quantity and unit
    if return_unit:
        if return_quantity:
            return flux_cgs, err_cgs, flux_unit
        else:
            return flux_cgs.value, err_cgs.value, flux_unit
    else:
        if return_quantity:
            return flux_cgs, err_cgs
        else:
            return flux_cgs.value, err_cgs.value

def convert_flux_magnitude_to_maggie(flux_mag, err_mag, return_quantity=False, return_unit=False):
    """ Convert flux from magnitudes to maggies
    """

    # Assign units
    if not isinstance(flux_mag, u.Quantity):
        flux_mag = flux_mag * u.mag
    if not isinstance(err_mag, u.Quantity):
        err_mag = err_mag * u.mag

    # convert to maggies
    flux_unit = u.dimensionless_unscaled
    flux_maggie = (10.0 ** (-0.4 * flux_mag.value)) * flux_unit
    factor = 0.4 * np.log(10.0)
    err_maggie = np.abs(flux_maggie.value * factor * err_mag.value) * flux_unit

    # Optionally return quantity and unit
    if return_unit:
        if return_quantity:
            return flux_maggie, err_maggie, flux_unit
        else:
            return flux_maggie.value, err_maggie.value, flux_unit
    else:
        if return_quantity:
            return flux_maggie, err_maggie
        else:
            return flux_maggie.value, err_maggie.value
        
def convert_flux_maggie_to_jy(flux_maggie, err_maggie, return_quantity=False, return_unit=False):

    # Assign units
    if not isinstance(flux_maggie, u.Quantity):
        flux_maggie = flux_maggie * u.dimensionless_unscaled
    if not isinstance(err_maggie, u.Quantity):
        err_maggie = err_maggie * u.dimensionless_unscaled

    # Convert to janskies
    flux_unit =  u.Jy
    flux_jy = flux_maggie * 3631 * flux_unit
    err_jy = err_maggie * 3631 * flux_unit
    
    # Optionally return quantity and unit
    if return_unit:
        if return_quantity:
            return flux_jy, err_jy, flux_unit
        else:
            return flux_jy.value, err_jy.value, flux_unit
    else:
        if return_quantity:
            return flux_jy, err_jy
        else:
            return flux_jy.value, err_jy.value

def convert_flux_jy_to_maggie(flux_jy, err_jy, return_quantity=False, return_unit=False):

    # Assign units
    if not isinstance(flux_jy, u.Quantity):
        flux_jy = flux_jy * u.Jy
    if not isinstance(err_jy, u.Quantity):
        err_jy = err_jy * u.Jy

    # Convert to janskies
    flux_unit =  u.dimensionless_unscaled
    flux_maggie = flux_jy / 3631 * flux_unit
    err_maggie = err_jy / 3631 * flux_unit
    
    # Optionally return quantity and unit
    if return_unit:
        if return_quantity:
            return flux_maggie, err_maggie, flux_unit
        else:
            return flux_maggie.value, err_maggie.value, flux_unit
    else:
        if return_quantity:
            return flux_maggie, err_maggie
        else:
            return flux_maggie.value, err_maggie.value
        
def convert_flux_maggie_to_cgs(flux_maggie, err_maggie, wave_m, cgs_factor, return_quantity=False, return_unit=False):

    # Assign units
    if not isinstance(flux_maggie, u.Quantity):
        flux_maggie = flux_maggie * u.dimensionless_unscaled
    if not isinstance(err_maggie, u.Quantity):
        err_maggie = err_maggie * u.dimensionless_unscaled

    # Convert to janskies
    flux_unit =  u.Jy
    flux_jy = flux_maggie * 3631 * flux_unit
    err_jy = err_maggie * 3631 * flux_unit

    # Convert to cgs units
    flux_unit = u.erg/(u.s * u.cm**2 * u.AA) * cgs_factor
    flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor, return_quantity=True)
    
    # Optionally return quantity and unit
    if return_unit:
        if return_quantity:
            return flux_cgs, err_cgs, flux_unit
        else:
            return flux_cgs.value, err_cgs.value, flux_unit
    else:
        if return_quantity:
            return flux_cgs, err_cgs
        else:
            return flux_cgs.value, err_cgs.value