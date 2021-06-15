import inspect
import sys
import os
import io
import re
from pvlib import temperature
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


def select_PVSystem(
        module=None,
        inverter=None,
        surface_azimuth=None,
        name=None):
    """This function allows you to select a module and inverter for your system."""
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
        module = sandia_modules[module]

    if inverter is None:
        see_inverters = input('Do you want to see a list of inverters?: y/n')
        if see_inverters == 'y':
            inverter_dict = list(sapm_inverters.keys())
            for key in inverter_dict:
                print(key)
            # selected_inverter = input('Select inverter: ')
            inverter = sapm_inverters[input('Select inverter: ')]
    else:
        inverter = sapm_inverters[inverter]

    if surface_azimuth is None:
        surface_azimuth = float(
            input('Enter surface azimuth (degrees from N): '))

    module_parameters = {'a': module['A'],
                                    'b': module['B'],
                                    'deltaT': module['DTC']}
    
    PVSystem_ = pvlib.pvsystem.PVSystem(module=module, inverter=inverter,
                                        surface_azimuth=surface_azimuth, name=name,
                                        module_type='glass_polymer',
                                        racking_model='open_rack')

    return PVSystem_


def size_pv(PVSystem_,
            peak_electricity=0,
            roof_area=0,
            percent_roof_cover=100,
            method='peak'):

    # Nominal Power Output in W
    nominal_P_out = PVSystem_.module.Vmpo * PVSystem_.module.Impo

    if method == 'peak':
        num_modules = m.ceil(peak_electricity / nominal_P_out)

    if method == 'roof':
        module_area = PVSystem_.module['Area']
        # print('Module area: {}'.format(module_area))
        covered_roof_area = Building_.roof_area * percent_roof_cover / 100

        num_modules = covered_roof_area // module_area

    modules_per_string = m.ceil(PVSystem_.inverter.Vdcmax
                                / PVSystem_.module.Vmpo)

    strings = m.ceil(num_modules / modules_per_string)

    PVSystem_.modules_per_string = modules_per_string
    PVSystem_.strings_per_inverter = strings

    return PVSystem_


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
    pvtemps = PVSystem_.sapm_celltemp(poa_global = poa_irradiance['poa_global'],
                                    wind_speed=weather_data.Wspd,
                                    temp_air=weather_data.DryBulb)

    # DC power generation
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_irradiance.poa_direct,
                                                               poa_irradiance.poa_diffuse,
                                                               airmass.airmass_absolute,
                                                               aoi,
                                                               PVSystem_.module)

    # SAPM = Sandia PV Array Performance Model, generates a dataframe with short-circuit current,
    # current at the maximum-power point, open-circuit voltage, maximum-power
    # point voltage and power
    dc_out = pvlib.pvsystem.sapm(
        effective_irradiance,
        pvtemps,
        PVSystem_.module)  # This will calculate the DC power output for a module

    ac_out = pd.DataFrame()
    # ac_out['p_ac'] is the AC power output in W from the DC power input.
    ac_out['p_ac'] = pvlib.pvsystem.snlinverter(
        dc_out.v_mp, dc_out.p_mp, PVSystem_.inverter)

    # p_ac/sqm is the AC power generated per square meter of module (W/m^2)
    # ac_out['p_ac/sqm'] = ac_out.p_ac.apply(lambda x: x / PVSystem_.module.Area)

    energy_output = pd.DataFrame(index=ac_out.index)
    energy_output['voltage_dc'] = dc_out['v_mp']
    energy_output['power_dc'] = dc_out['p_mp']
    energy_output['power_ac'] = ac_out['p_ac']

    print('PV simulation completed for {}'.format(City_.name.upper()))

    return energy_output


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

    num_modules = m.ceil(peak_electricity / nominal_P_out)

    modules_per_string = m.ceil(pvsystem.inverter.Vdcmax
                                / pvsystem.module.Vmpo)

    strings = m.ceil(num_modules / modules_per_string)

    pvsystem.modules_per_string = modules_per_string
    pvsystem.strings_per_inverter = strings

    print(pvsystem.modules_per_string, pvsystem.strings_per_inverter)


####################################
# DISCONTINUED, FOR REFERENCE ONLY #
####################################
