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

    # Convert wavelength to microns
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

# -----------------------------
# Flux/flux density conversions
# -----------------------------
def convert_flux_jy_to_ujy(flux_jy, err_jy, return_quantity=False, return_unit=False):

    # Assign units
    if not isinstance(flux_jy, u.Quantity):
        flux_jy = flux_jy * u.Jy
    if not isinstance(err_jy, u.Quantity):
        err_jy = err_jy * u.Jy

    # Convert flux to microjansky
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

    # Convert flux to microjansky
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

    # Convert flux to janksies
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

def convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor, return_quantity=False, return_unit=False):
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

    # Convert flux to cgs units
    # if isinstance(cgs_factor, None):
    #     flux_unit = u.erg/(u.s * u.cm**2 * u.AA)
    # else:    
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