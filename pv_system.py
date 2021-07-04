import inspect
import sys
import os
import io
import re
from pvlib import temperature
from pvlib.inverter import sandia
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.spa import solar_position
import requests
# from openpyxl import load_workbook
# import pathlib
from IPython.display import display

import time
import datetime
import dateutil
import pandas as pd
import math as m
import numpy as np

# import sklearn
import pvlib

# Writing format
import pyarrow

################
# PV FUNCTIONS #
################


def list_pv_modules():
    """
    This function lists pv modules in an interactive selction system.
    """
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    for module in sandia_modules:
        print('Module: {}'.format(module))
        print('Voltage, V: {:f}'.format(module['Vmpo']))
        print('Current, A: {:f}'.format(module['Impo']))
        print('Power, W: {:f}'.format(module['Vmpo'] * module['Impo']))


def list_inverters():
    """
    This function lists inverters in an interactive selction system.
    """
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    for inverter in sapm_inverters:
        print('Inverter: {}'.format(inverter))
        print('Voltage, V: {:f}'.format(inverter['Vdcmax']))
        print('Current, A: {:f}'.format(inverter['Idcmax']))
        print('Power, W: {:f}'.format(inverter['Vdcmax'] * inverter['Idcmax']))


def get_module_parameters(module):
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    return sandia_modules[module]


def get_inverter_parameters(inverter):
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    inverter_parameters = cec_inverters[inverter]

    if isinstance(inverter_parameters, pd.Series) or isinstance(inverter_parameters, pd.DataFrame):

        inverter_parameters = inverter_parameters.transpose()
        inverter_dict = inverter_parameters.to_dict()
        inverter_dict = {'Model':inverter,
                         'Vac':inverter_dict['Vac'],
                         'Pso':inverter_dict['Pso'],
                         'Paco':inverter_dict['Paco'],
                         'Pdco':inverter_dict['Pdco'],
                         'Vdco':inverter_dict['Vdco'],
                         'C0':inverter_dict['C0'],
                         'C1':inverter_dict['C1'],
                         'C2':inverter_dict['C2'],
                         'C3':inverter_dict['C3'],
                         'Pnt':inverter_dict['Pnt'],
                         'Vdcmax':inverter_dict['Vdcmax'],
                         'Idcmax':inverter_dict['Idcmax'],
                         'Mppt_low':inverter_dict['Mppt_low'],
                         'Mppt_high':inverter_dict['Mppt_high'],
                         'CEC_Date':inverter_dict['CEC_Date'],
                         'CEC_Type':inverter_dict['CEC_Type']}
    
    return inverter_dict

def size_pv_modules(design_load, module, oversize_factor=1):
    module_parameters = get_module_parameters(module)

    # Nominal power rating
    v_mpo = module_parameters['Vmpo']  # Volts
    i_mpo = module_parameters['Impo']  # Ampere
    power_rating = v_mpo * i_mpo  # Watts

    # Open circuit power
    v_oco = module_parameters['Voco']  # V
    i_sco = module_parameters['Isco']  # A
    surge_power = v_oco * i_sco

    # Minimum number of modules
    minimum_number_of_modules = m.ceil(design_load / power_rating)
    minimum_number_of_modules = m.ceil(
        minimum_number_of_modules * oversize_factor)
    return minimum_number_of_modules


def calculate_power_rating(module, number_of_modules):
    module_parameters = get_module_parameters(module)
    # Nominal power rating
    v_mpo = module_parameters['Vmpo']  # Volts
    i_mpo = module_parameters['Impo']  # Ampere
    power_rating = v_mpo * i_mpo  # Watts

    return power_rating * number_of_modules


def size_inverter(design_load, units='W', grid_tied=True):
    # Check units of design load
    if units == 'kW':
        design_load = design_load * 1000
    elif units == 'W':
        pass
    else:
        design_load = input('Input the maximum AC load in W')

    # max_AC_load is the maximum power draw
    if grid_tied is True:
        # Design according to PV rating
        sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

        df = sapm_inverters.transpose().reset_index().rename(
            columns={'index': 'inverter'})

        # Filter for inverters that fit the design load
        df = df[(df['CEC_Type'] == 'Utility Interactive') &
                (df['Pdco']) < design_load]

        df['power_difference'] = np.abs(df['Pdco'] - design_load)

        # Select only Utility Interactive Inverters
        chosen_inverter = df[(df['power_difference'] ==
                              df['power_difference'].min())]['inverter']


        if (len(chosen_inverter.index) > 1) or isinstance(chosen_inverter, pd.Series):
            chosen_inverter.reset_index(inplace=True, drop=True)
            return chosen_inverter[0]

    else:
        # Design for building load
        pass

    return chosen_inverter


def size_pv_array(number_of_modules,
                  module,
                  inverter,
                  module_parameters=None,
                  inverter_parameters=None):

    if module_parameters is None:
        module_parameters = get_module_parameters(module)

    if inverter_parameters is None:
        inverter_parameters = get_inverter_parameters(inverter)

    modules_per_string = m.ceil(inverter_parameters['Vdcmax'] / module_parameters['Vmpo'])

    strings = m.ceil(number_of_modules / modules_per_string)

    # Check
    array_size = modules_per_string * strings
    array_power_rating = calculate_power_rating(module, array_size)

    check = array_power_rating > inverter_parameters['Pdco']

    if check is False:
        raise Exception('Your PV Array is Too Small')
    else:
        return modules_per_string, strings


def design_PVSystem(design_load,
                    module,
                    surface_azimuth=180,
                    surface_tilt=0,
                    oversize_factor=1,
                    name=''):
    # Design Load should be in W
    # Get minimum module size
    number_of_modules = size_pv_modules(design_load, module, oversize_factor)
    pv_power_rating = calculate_power_rating(module, number_of_modules)
    module_parameters = get_module_parameters(module)

    # Size Inverter
    inverter = size_inverter(design_load=pv_power_rating)
    inverter_parameters = get_inverter_parameters(inverter)

    # Size Array
    modules_per_string, strings_per_inverter = size_pv_array(number_of_modules,
                                                             module,
                                                             inverter,
                                                             module_parameters,
                                                             inverter_parameters)

    PVSystem_ = pvlib.pvsystem.PVSystem(name=name,
                                        module=module, module_parameters=module_parameters,
                                        inverter=inverter, inverter_parameters=inverter_parameters,
                                        modules_per_string=modules_per_string, strings_per_inverter=strings_per_inverter,
                                        surface_azimuth=surface_azimuth, surface_tilt=surface_tilt,
                                        module_type='glass_polymer', racking_model='open_rack')

    return PVSystem_


def select_PVSystem(
        module=None,
        inverter=None,
        surface_azimuth=None,
        name=None):
    r'''DEPRACATED'''

    r"""This function allows you to select a module and inverter for your system."""
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    if module is None:
        see_modules = input('Do you want to see a list of modules?: y/n')
        if see_modules == 'y':
            module_dict = list(sandia_modules.keys())
            for key in module_dict:
                print(key)
            # selected_module = input('Select module: ')
            module = sandia_modules[input('Select module: ')]
    else:
        module = module
        module_parameters = sandia_modules[module]

    if inverter is None:
        see_inverters = input('Do you want to see a list of inverters?: y/n')
        if see_inverters == 'y':
            inverter_dict = list(sapm_inverters.keys())
            for key in inverter_dict:
                print(key)
            # selected_inverter = input('Select inverter: ')
            inverter = sapm_inverters[input('Select inverter: ')]
    else:
        inverter = inverter
        inverter_parameters = sapm_inverters[inverter].squeeze()

    if surface_azimuth is None:
        surface_azimuth = float(
            input('Enter surface azimuth (degrees from N): '))

    PVSystem_ = pvlib.pvsystem.PVSystem(module=module, module_parameters=module_parameters,
                                        inverter=inverter, inverter_parameters=inverter_parameters,
                                        surface_azimuth=surface_azimuth, name=name,
                                        module_type='glass_polymer',
                                        racking_model='open_rack')

    return PVSystem_

def calculate_clipped_energy(v_dc, p_dc, inverter):
    r'''
    Calculate the AC power clipped using Sandia's
    Grid-Connected PV Inverter model.

    Parameters
    ----------
    v_dc : numeric
        DC voltage input to the inverter. [V]

    p_dc : numeric
        DC power input to the inverter. [W]

    inverter : dict-like
        Defines parameters for the inverter model in [1]_.

    Returns
    -------
    clipped_power : pandas dataframe
        AC power output. [W]

     Notes
    -----

    Determines the AC power output of an inverter given the DC voltage and DC
    power. Output AC power is bounded above by the parameter ``Paco``, to
    represent inverter "clipping".  When `power_ac` would be less than
    parameter ``Pso`` (startup power required), then `power_ac` is set to
    ``-Pnt``, representing self-consumption. `power_ac` is not adjusted for
    maximum power point tracking (MPPT) voltage windows or maximum current
    limits of the inverter.

    Required model parameters are:

    ======   ============================================================
    Column   Description
    ======   ============================================================
    Paco     AC power rating of the inverter. [W]
    Pdco     DC power input that results in Paco output at reference
             voltage Vdco. [W]
    Vdco     DC voltage at which the AC power rating is achieved
             with Pdco power input. [V]
    Pso      DC power required to start the inversion process, or
             self-consumption by inverter, strongly influences inverter
             efficiency at low power levels. [W]
    C0       Parameter defining the curvature (parabolic) of the
             relationship between AC power and DC power at the reference
             operating condition. [1/W]
    C1       Empirical coefficient allowing ``Pdco`` to vary linearly
             with DC voltage input. [1/V]
    C2       Empirical coefficient allowing ``Pso`` to vary linearly with
             DC voltage input. [1/V]
    C3       Empirical coefficient allowing ``C0`` to vary linearly with
             DC voltage input. [1/V]
    Pnt      AC power consumed by the inverter at night (night tare). [W]
    ======   ============================================================

    A copy of the parameter database from the System Advisor Model (SAM) [2]_
    is provided with pvlib and may be read  using
    :py:func:`pvlib.pvsystem.retrieve_sam`.

    References
    ----------
    .. [1] D. King, S. Gonzalez, G. Galbraith, W. Boyson, "Performance Model
       for Grid-Connected Photovoltaic Inverters", SAND2007-5036, Sandia
       National Laboratories.

    .. [2] System Advisor Model web page. https://sam.nrel.gov.

    See also
    --------
    pvlib.pvsystem.retrieve_sam
    '''
    Paco = inverter['Paco']
    Pnt = inverter['Pnt']
    Pso = inverter['Pso']

    # _sandia_eff calculates the inverter AC power without clipping
    p_ac = pvlib.inverter._sandia_eff(v_dc, p_dc, inverter)
    # _sandia_limits applies the minimum and maximum power limits to 'power_ac)
    p_ac_limit = pvlib.inverter._sandia_limits(
        p_ac, p_dc, Paco, Pnt, Pso)

    clipped_p_ac = p_ac - p_ac_limit

    inverter_efficiency = p_ac / p_dc

    clipped_p_dc = clipped_p_ac / inverter_efficiency

    clipped_power = pd.concat([p_ac, clipped_p_ac, p_dc, clipped_p_dc, inverter_efficiency],
                              axis=1)

    clipped_power.rename(columns={0:'p_ac', 
                                  1:'clipped_p_ac',
                                  'p_mp':'p_dc',
                                  2:'clipped_p_dc',
                                  3:'inverter_efficiency'},
                         inplace=True)

    # Clean Dataset
    clipped_power['clipped_p_ac'] = np.where(clipped_power.p_dc <= 0, 0., clipped_p_ac)
    clipped_power['inverter_efficiency'].replace(np.inf, 0, inplace=True)

    return clipped_power


def pv_simulation(PVSystem_, City_):
    '''
    To Do
    - Modify pv simulation to match your model
    - Calculate poa global
    -
    '''

    '''
    This function runs the simulation for the energy produced from a PV system.

    1) SET UP PV SYSTEM
    ===================
    The following functions set up the PV system. The functions take a PVSystem_ which contains the module
    parameters, inverter parameters, and the surface azimuth.
    Other parameters (e.g., albedo and surface type) are not currently functional.
    '''
    print('Running PV Simulation for {}'.format(City_.name.upper()))

    # the PVSystem_ contains data on the module, inverter, and azimuth.
    if PVSystem_.surface_tilt is None:
        PVSystem_.surface_tilt = City_.latitude  # Tilt angle of array

    location = Location(latitude=City_.latitude, longitude=City_.longitude,
                        tz=City_.tz, altitude=City_.altitude, name=City_.name)

    weather_data = City_.tmy_data

    # The surface_type_list is for a future iteration.
    # It will be used to calculate the ground albedo and subsequent reflected
    # radiation.
    surface_type_list = ['urban',
                         'grass',
                         'fresh grass',
                         'snow',
                         'fresh snow',
                         'asphalt',
                         'concrete',
                         'aluminum',
                         'copper',
                         'fresh steel',
                         'dirty steel',
                         'sea']

    # system['albedo'] = input('Input albedo (default is 0.25; typically 0.1-0.4 for surfaces on Earth)')
    # system['surface_type'] = input('To overwrite albedo, input surface type from {}'.format(surface_type_list))

    """
    2) IRRADIANCE CALCULATIONS
    ====================
    The following functions calculate the energy output of the PV system. The simulation incorporates the efficiency losses
    from temperature increase, PV module efficiency, and the inverter efficiency (dc-ac conversion)
    """


    # Calculate Solar Position
    solar_pos = location.get_solarposition(
        times=weather_data.index,
        pressure=weather_data.Pressure,
        temperature=weather_data.DryBulb)

    # For some reason,
    weather_data.index = solar_pos.index

    # Calculate Airmass
    airmass = location.get_airmass(
        times=weather_data.index,
        solar_position=solar_pos,
        model='pickering2002')

    # Calculate the AOI
    aoi = pvlib.irradiance.aoi(PVSystem_.surface_tilt,
                               PVSystem_.surface_azimuth,
                               solar_pos['apparent_zenith'],
                               solar_pos['azimuth'])

    # Calculate the POA Sky Diffuse
    # Using isotropic model, since all other models output only NAN. Appears to be an issue with the
    # surface tilt calculation. Possibly an index mismatch.
    sky_diffuse = pvlib.irradiance.get_sky_diffuse(surface_tilt=PVSystem_.surface_tilt,
                                                   surface_azimuth=PVSystem_.surface_azimuth,
                                                   solar_zenith=solar_pos['apparent_zenith'],
                                                   solar_azimuth=solar_pos['azimuth'],
                                                   dni=weather_data.DNI,
                                                   ghi=weather_data.GHI,
                                                   dhi=weather_data.DHI,
                                                   dni_extra=weather_data.ETRN,
                                                   airmass=airmass,
                                                   model='king')

    # Calculate POA Ground Diffuse
    # Set albedo if data exists. Else albedo is none.
    PVSystem_.albedo = weather_data['Alb']

    ground_diffuse = pvlib.irradiance.get_ground_diffuse(surface_tilt=PVSystem_.surface_tilt,
                                                         ghi=weather_data['GHI'],
                                                         albedo=PVSystem_.albedo)

    # POA Components needs AOI, DNI, POA SKY DIFFUSE, POA GROUND DIFFUSE
    poa_irradiance = pvlib.irradiance.poa_components(aoi=aoi,
                                                     dni=weather_data.DNI,
                                                     poa_sky_diffuse=sky_diffuse,
                                                     poa_ground_diffuse=ground_diffuse)

    """
    3) ENERGY SIMULATION
    ====================
    The following functions calculate the energy output of the PV system. The simulation incorporates the efficiency losses
    from temperature increase, PV module efficiency, and the inverter efficiency (dc-ac conversion)
    """

    # Calculate the PV cell and module temperature
    pvtemps = PVSystem_.sapm_celltemp(poa_global=poa_irradiance['poa_global'],
                                      wind_speed=weather_data.Wspd,
                                      temp_air=weather_data.DryBulb)

    # DC power generation
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_irradiance.poa_direct,
                                                                    poa_irradiance.poa_diffuse,
                                                                    airmass.airmass_absolute,
                                                                    aoi,
                                                                    PVSystem_.module_parameters)

    # SAPM = Sandia PV Array Performance Model, generates a dataframe with short-circuit current,
    # current at the maximum-power point, open-circuit voltage, maximum-power
    # point voltage and power
    dc_out = pvlib.pvsystem.sapm(
        effective_irradiance,
        pvtemps,
        PVSystem_.module_parameters)  # This will calculate the DC power output for a module

    ac_out = pd.DataFrame()

    array_v_mp = dc_out.v_mp * PVSystem_.strings_per_inverter
    array_p_dc = dc_out.p_mp * (PVSystem_.strings_per_inverter * PVSystem_.modules_per_string)
    # ac_out['p_ac'] is the AC power output in W from the DC power input.
    ac_out['p_ac'] = pvlib.inverter.sandia(
        array_v_mp, array_p_dc, PVSystem_.inverter_parameters)

    # p_ac/sqm is the AC power generated per square meter of module (W/m^2)

    energy_output = pd.DataFrame(index=ac_out.index)
    energy_output['v_dc'] = array_v_mp
    energy_output['p_dc'] = array_p_dc
    energy_output['p_ac'] = ac_out['p_ac']

    # Calculate the clipped energy
    clipped_energy = calculate_clipped_energy(v_dc=array_v_mp, # energy_output['v_dc'],
                                              p_dc=array_p_dc, # energy_output['p_dc'],
                                              inverter=PVSystem_.inverter_parameters)

    energy_output['clipped_p_dc'] = clipped_energy['clipped_p_dc']
    energy_output['clipped_p_ac'] = clipped_energy['clipped_p_ac']
    energy_output['inverter_efficiency'] = clipped_energy['inverter_efficiency']

    energy_output.to_csv(r'model_outputs\testing\pv_output.csv')

    return energy_output


# Pending
def pv_system_costs(pv_system_power_rating=0, building_type='commercial'):
    # Solar PV Prices from:
    # NREL (National Renewable Energy Laboratory). 2020. 2020 Annual Technology
    # Baseline. Golden, CO: National Renewable Energy Laboratory.
    # https://atb.nrel.gov/
    if Building_.building_type == 'single_family_residential':
        # CAPEX includes construction and overnight capital cost in $/kW
        CAPEX = 3054
        # OM Costs
        fixed_om_cost = 22  # $/kW/yr
        variable_om_cost = 0  # $/MWh
    else:
        # CAPEX includes construction and overnight capital cost in $/kW
        CAPEX = 2052
        # OM Costs
        fixed_om_cost = 18  # $/kW/yr
        variable_om_cost = 0  # $/MWh

    # Capital Cost in $
    capital_cost_PV = CAPEX * pv_system_power_rating / 1000
    # O&M cost in $/yr
    fixed_cost = fixed_om_cost * pv_system_power_rating

    return capital_cost_PV, fixed_cost, variable_om_cost



def calculate_surplus_dc_power():
    pass


def run_pv_output():
    pass
#####################
# TESTING FUNCTIONS #
#####################


def test():
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    sandia_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

    cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

    pvsystem = PVSystem(surface_tilt=20, surface_azimuth=200,
                        module=sandia_module,
                        inverter=cec_inverter)

    nominal_P_out = pvsystem.module.Vmpo * pvsystem.module.Impo
    peak_electricity = 1000

    number_of_modules = m.ceil(peak_electricity / nominal_P_out)

    modules_per_string = m.ceil(pvsystem.inverter.Vdcmax
                                / pvsystem.module.Vmpo)

    strings = m.ceil(number_of_modules / modules_per_string)

    pvsystem.modules_per_string = modules_per_string
    pvsystem.strings_per_inverter = strings

    print(pvsystem.modules_per_string, pvsystem.strings_per_inverter)


####################################
# DISCONTINUED, FOR REFERENCE ONLY #
####################################
