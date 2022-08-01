####################################
# LIBRARIES NEEDED TO RUN THE TOOL #
####################################
import abc
from cmath import nan
from logging import error
from msilib.schema import Error
from tkinter import Label
from openpyxl import load_workbook
import math as m

# Scientific python add-ons
import pandas as pd     # To install: pip install pandas
import numpy as np
from py import process      # To install: pip install numpy

# Data Storage and Reading
from pyarrow import feather  # To install: pip install pyarrow

# Plotting modules
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import seaborn as sns  # To install: pip install seaborn

# PV library and classes for Object-oriented
import pvlib  # [1] To install: pip install pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem

# Thermofluids modules
from iapws import iapws97
from iapws import humidAir
from sqlalchemy import column
from sympy import LC, QQ_gmpy

from sysClasses import *

##########################################################################

# TO DO
# Function to calculate lifetime water consumption
#   Must include CHP variety for ABCs when CHP is included
############################
climate_city_dict = {'1A': 'miami',
                     '2A': 'houston',
                     '2B': 'phoenix',
                     '3A': 'atlanta',
                     '3B': 'las_vegas',
                     '3B-CA': 'los_angeles',
                     '3C': 'san_francisco',
                     '4A': 'baltimore',
                     '4B': 'albuquerque',
                     '4C': 'seattle',
                     '5A': 'chicago',
                     '5B': 'denver',
                     '6A': 'minneapolis',
                     '6B': 'helena',
                     '7': 'duluth',
                     '8': 'fairbanks'}


'''
Functions to interact with ESS program
'''
def generate_EES_inputs():
    save_path = r'model_outputs\AbsorptionChillers\cooling_demand'
    for city in city_list:
        for building in building_type_list:
            # building_datapath = city_building_dictionary[city][building]
            # climate_datapath = processed_tmy3_dictionary[city]
            City_ = City(city)
            City_._infer_tmy_data()

            Building_ = Building(
                name=building,
                building_type=building,
                City_=City_)

            cooling_demand = Building_.cooling_demand

            ees_df = organize_EES_inputs(cooling_demand, City_.tmy_data)

            cols = [
                'hour',
                'CoolingDemand_kW',
                'DryBulb_C',
                'Pressure_kPa',
                'RHum']
            ees_df = ees_df[cols]

            save_file = F'{save_path}\\{city}_{building}_CoolDem.csv'
            ees_df.to_csv(save_file)
            print(F'Saved {city} {building}')


def organize_EES_inputs(building_cooling_demand, climate_data):
    ees_df = climate_data[['DryBulb', 'RHum', 'Pressure']].copy()
    ees_df['Pressure_kPa'] = ees_df['Pressure'] / 10
    ees_df['RHum'] = ees_df['RHum'] / 100
    ees_df['CoolingDemand_kW'] = building_cooling_demand
    ees_df.rename(columns={'DryBulb': 'DryBulb_C'}, inplace=True)
    ees_df.drop(columns=['Pressure'], inplace=True)

    ees_df['hour'] = np.arange(1, 8761, 1)

    return ees_df


def organize_EES_outputs(city_name, chiller_type='AbsorptionChiller'):

    if chiller_type == 'AbsorptionChiller':
        filepath = r'model_outputs\AbsorptionChillers\cooling_supply\AbsorptionChiller'
        filename = F'{filepath}\\{city_name}.csv'

        building_path = r'model_outputs\AbsorptionChillers\cooling_demand'
        building_file = F'{building_path}\\{city_name}_hospital_CoolDem.csv'

        building_df = pd.read_csv(building_file, index_col='datetime')

        cols = [
            'Qe_nom',
            'T_db',
            'Patm',
            'RH',
            'Qheat_kW',
            'Welec_kW',
            'makeup_water_kg_per_s',
            'percent_makeup']
        df = pd.read_csv(filename, names=cols)
    elif chiller_type == 'WaterCooledChiller':
        filepath = r'model_outputs\AbsorptionChillers\cooling_supply\WaterCooled_Chiller'
        filename = F'{filepath}\\{city_name}.csv'

        building_path = r'model_outputs\AbsorptionChillers\cooling_demand'
        building_file = F'{building_path}\\{city_name}_hospital_CoolDem.csv'

        building_df = pd.read_csv(building_file, index_col='datetime')

        cols = [
            'Qe_nom',
            'T_db',
            'Patm',
            'RH',
            'Wcomp_kW',
            'Wct_kW',
            'Welec_kW',
            'makeup_water_kg_per_s',
            'percent_makeup']
        df = pd.read_csv(filename, names=cols)

    try:
        df['datetime'] = building_df.index
        df.set_index('datetime', inplace=True, drop=True)
    except ValueError:
        print(F'{city_name} has mismatched indices')
    return df


def clean_EES_outputs(chiller_type='AbsorptionChiller'):
    cities = city_list

    try:
        cities.remove('fairbanks')
    except ValueError:
        pass

    savepath = r'model_outputs\AbsorptionChillers\cooling_supply'

    for city in cities:
        # Reads the dataframe for each city
        print(chiller_type, city)
        df = organize_EES_outputs(city, chiller_type)
        df.to_csv(F'{savepath}\\{chiller_type}\\{city}_supply.csv')


def calculate_EES_building_output(
        city_df, building_df, chiller_type='AbsorptionChiller', district_cooling_loss=0.1):
    dcl_factor = (1 - district_cooling_loss)

    city_df['datetime'] = building_df['datetime']
    city_df.set_index('datetime', inplace=True, drop=True)

    supply_df = building_df.copy()
    supply_df.set_index('datetime', inplace=True, drop=True)

    if chiller_type == 'AbsorptionChiller':
        supply_df['Qfrac'] = (
            supply_df.CoolingDemand_kW / dcl_factor) / city_df.Qe_nom
        supply_df['Cooling_HeatDemand_kW'] = city_df.Qheat_kW * supply_df.Qfrac
        supply_df['Cooling_ElecDemand_kW'] = city_df.Welec_kW * supply_df.Qfrac
        supply_df['MakeupWater_kph'] = city_df.makeup_water_kg_per_s * \
            supply_df.Qfrac * 3600
        supply_df.drop(columns=['hour'], inplace=True)
    elif chiller_type == 'WaterCooledChiller':
        supply_df['Qfrac'] = (
            supply_df.CoolingDemand_kW / dcl_factor) / city_df.Qe_nom
        supply_df['Cooling_HeatDemand_kW'] = 0
        supply_df['Cooling_ElecDemand_kW'] = city_df.Welec_kW * supply_df.Qfrac
        supply_df['MakeupWater_kph'] = city_df.makeup_water_kg_per_s * \
            supply_df.Qfrac * 3600
        supply_df.drop(columns=['hour'], inplace=True)

    return supply_df

'''
Functions to calculate w4r values. 
'''
# Gather data from datafiles

def retrieve_water_consumption_intensities(how='NERC'):
    '''
    how: the scope of the water intensities (NERC, fuel_type)
    '''
    if how == 'NERC':
        # Peer 2019 water consumption by eGRID region (L / kWh)
        water_consumption_df = pd.read_csv(
            r'data\Tech_specs\WaterConsInt_NERC_LperkWh_delivered.csv',
            index_col='eGRID_region')
    if how == 'fuel_type':
        # Grubert 2018 water consumption by fuel type (L / kWh)
        water_consumption_df = pd.read_csv(
            r'data\Tech_specs\WaterConsInt_byFuel_LperkWh_delivered.csv',
            index_col='fuel_type')
    return water_consumption_df

def generate_climate_grid_tuples():
    df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\climate_eGRIDsubregion.csv')

    df['city'] = df['climate_zone'].apply(lambda x: climate_city_dict[x])

    ls = list(df.itertuples(index=False, name=None))

    return ls


def annual_building_sim(chiller_type='AbsorptionChiller',
                        district_cooling_loss=0):
    demand_path = r'model_outputs\AbsorptionChillers\cooling_demand'

    if chiller_type == 'AbsorptionChiller':
        supply_path = r'model_outputs\AbsorptionChillers\cooling_supply\AbsorptionChiller'
        save_path = r'model_outputs\AbsorptionChillers\building_supply_sim\AbsorptionChiller'

    elif chiller_type == 'WaterCooledChiller':
        supply_path = r'model_outputs\AbsorptionChillers\cooling_supply\WaterCooled_Chiller'
        save_path = r'model_outputs\AbsorptionChillers\building_supply_sim\WaterCooledChiller'

    cities = city_list

    try:
        cities.remove('fairbanks')
    except ValueError:
        pass

    for city in cities:
        for building in building_type_list:
            print('City: {} | Building: {} Time: {}'.format(
                city, building, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())), end='\r')

            city_df = pd.read_csv(F'{supply_path}\\{city}_supply.csv')
            demand_df = pd.read_csv(
                F'{demand_path}\\{city}_{building}_CoolDem.csv')
            supply_df = calculate_EES_building_output(
                city_df, demand_df, chiller_type, district_cooling_loss)
            supply_df['building_type'] = building
            supply_df.reset_index(inplace=True, drop=False)
            supply_df.to_feather(
                F'{save_path}\\{city}_{building}_simulation.feather')


def calculate_peak_demand_reductions(district_cooling_loss=0.1):
    wcc_supply_path = r'model_outputs\AbsorptionChillers\building_supply_sim\WaterCooledChiller'
    abc_supply_path = r'model_outputs\AbsorptionChillers\building_supply_sim\AbsorptionChiller'

    cities = city_list

    try:
        cities.remove('fairbanks')
    except ValueError:
        pass

    city_dataframes = []
    for city in cities:
        eGRID_subregion = nerc_region_dictionary[city]
        climate_zone = climate_zone_dictionary[city]

        building_dataframes = []

        for building in building_type_list:
            print('City: {} | Building: {} Time: {}'.format(
                city, building, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())), end='\r')

            # Read Water-cooled Chiller file
            wcc_df = pd.read_feather(
                F'{wcc_supply_path}\\{city}_{building}_simulation.feather')
            wcc_df.set_index('datetime', inplace=True, drop=True)
            wcc_df.index = pd.to_datetime(wcc_df.index)

            # Read Absorption Chiller file
            abc_df = pd.read_feather(
                F'{abc_supply_path}\\{city}_{building}_simulation.feather')
            abc_df.set_index('datetime', inplace=True, drop=True)
            abc_df.index = pd.to_datetime(abc_df.index)

            df = pd.DataFrame(index=wcc_df.index)

            df['city'] = city
            df['building'] = building

            df['CoolingDemand_kW'] = wcc_df['CoolingDemand_kW']
            df['Electricity_ACC_kW'] = df['CoolingDemand_kW'] / \
                3.5  # 3.5 is the COP of air cooled chillers
            df['Electricity_WCC_kW'] = wcc_df['Cooling_ElecDemand_kW']
            df['Electricity_ABC_kW'] = abc_df['Cooling_ElecDemand_kW']
            df['Heating_ABC_kW'] = abc_df['Cooling_HeatDemand_kW']

            # Get the peak demands for each day of the year
            agg_df = df.groupby(['city', 'building']).resample('D').agg({
                'CoolingDemand_kW': ['max'],
                'Electricity_ACC_kW': ['max'],
                'Electricity_WCC_kW': ['max'],
                'Electricity_ABC_kW': ['max'],
                'Heating_ABC_kW': ['max']
            })

            agg_df.columns = agg_df.columns.map('_'.join)
            # agg_df.columns = agg_df.columns.droplevel(1)
            agg_df.rename(columns={'CoolingDemand_kW_max': 'CoolingDemand_kWh',
                                   'Electricity_ACC_kW_max': 'Electricity_ACC_kW',
                                   'Electricity_WCC_kW_max': 'Electricity_WCC_kW',
                                   'Electricity_ABC_kW_max': 'Electricity_ABC_kW',
                                   'Heating_ABC_kW_max': 'Heating_ABC_kW'
                                   }, inplace=True)

            agg_df.reset_index(inplace=True, drop=False)
            agg_df['datetime'] = agg_df['datetime'].dt.tz_localize(None)

            agg_df['floor_area_m^2'] = floor_area_dictionary[building]

            building_dataframes.append(agg_df)

        city_df = pd.concat(building_dataframes, axis=0).reset_index(drop=True)

        city_df['climate_zone'] = climate_zone
        city_df['eGRID_subregion'] = eGRID_subregion

        city_dataframes.append(city_df)

    peak_df = pd.concat(city_dataframes, axis=0).reset_index(drop=True)

    for label in ['Electricity_ACC_kW',
                  'Electricity_WCC_kW', 'Electricity_ABC_kW']:
        peak_df[F'{label}_m^-2'] = peak_df[label] / peak_df['floor_area_m^2']

    peak_df['WCC_peak_reduction_%'] = (
        peak_df['Electricity_WCC_kW'] - peak_df['Electricity_ACC_kW']) / peak_df['Electricity_ACC_kW']
    peak_df['ABC_peak_reduction_%'] = (
        peak_df['Electricity_ABC_kW'] - peak_df['Electricity_ACC_kW']) / peak_df['Electricity_ACC_kW']

    peak_df.fillna(0, inplace=True)

    # The above calculation gives the fractional difference. Multiply by -100
    # to get percent reduced
    peak_df['WCC_peak_reduction_%'] = peak_df['WCC_peak_reduction_%'] * -100
    peak_df['ABC_peak_reduction_%'] = peak_df['ABC_peak_reduction_%'] * -100

    file_path = r'model_outputs\AbsorptionChillers'
    file_name = r'\\peak_electricity_reduction.csv'
    peak_df.to_csv(F'{file_path}\\{file_name}')


def calculate_PoG_water_consumption(
        chiller_type='AbsorptionChiller', how='NERC'):
    cities = city_list

    try:
        cities.remove('fairbanks')
    except ValueError:
        pass

    climate_grid_ls = generate_climate_grid_tuples()

    city_dataframes = []
    for i in climate_grid_ls:
        climate_zone = i[0]
        eGRID_subregion = i[1]
        city = i[2]

        building_dataframes = []

        for building in building_type_list:
            floor_area = floor_area_dictionary[building]

            filepath = r'model_outputs\AbsorptionChillers\building_supply_sim'
            filename = F'\\{chiller_type}\\{city}_{building}_simulation.feather'

            df = pd.read_feather(F'{filepath}\\{filename}')
            df.set_index('datetime', inplace=True, drop=True)
            df.index = pd.to_datetime(df.index)

            df['city'] = city
            df['building'] = building

            agg_df = df.groupby(['city', 'building']).resample('A').agg({
                'CoolingDemand_kW': ['sum'],
                'Cooling_HeatDemand_kW': ['sum'],
                'Cooling_ElecDemand_kW': ['sum'],
                'MakeupWater_kph': ['sum']
            })

            agg_df.columns = agg_df.columns.map('_'.join)
            # agg_df.columns = agg_df.columns.droplevel(1)
            agg_df.rename(columns={'CoolingDemand_kW_sum': 'CoolingDemand_kWh',
                                   'Cooling_HeatDemand_kW_sum': 'Cooling_HeatDemand_kWh',
                                   'Cooling_ElecDemand_kW_sum': 'Cooling_ElecDemand_kWh',
                                   'MakeupWater_kph_sum': 'MakeupWater_kg'}, inplace=True)

            agg_df['floor_area_m^2'] = floor_area
            agg_df['climate_zone'] = climate_zone
            agg_df['eGRID_subregion'] = eGRID_subregion

            agg_df.reset_index(inplace=True)
            agg_df.drop(columns=['datetime'], inplace=True)

            building_dataframes.append(agg_df)

        city_df = pd.concat(building_dataframes, axis=0).reset_index(drop=True)

        # Correct electricity consumption to reflect grid_losses
        if chiller_type == 'AbsorptionChiller':
            # Absorption Chiller gets electricity from CHP, so no distribution
            # loss
            city_df['grid_loss_factor'] = 0
        else:
            grid_loss_df = retrieve_water_consumption_intensities('NERC')[
                'grid_gross_loss']
            try:
                city_df['grid_loss_factor'] = city_df[eGRID_subregion].apply(
                    lambda x: grid_loss_df[x])
            except KeyError:
                city_df['grid_loss_factor'] = 0.053
            city_df['Cooling_ElecDemand_kWh'] = city_df['Cooling_ElecDemand_kWh'] / \
                (1 - city_df['grid_loss_factor'])

        city_dataframes.append(city_df)

    annual_df = pd.concat(city_dataframes, axis=0).reset_index(drop=True)
    annual_df['chiller_type'] = chiller_type

    annual_df['CoolingDemand_intensity_kWh/sqm'] = annual_df['CoolingDemand_kWh'] / \
        annual_df['floor_area_m^2']

    water_consumption_intensity = retrieve_water_consumption_intensities(how)
    PoG_water_consumption = water_consumption_intensity['PoG']

    if how == 'NERC':
        df = annual_df.copy()

        if chiller_type == 'AbsorptionChiller':
            df['w4e_int_factor_(L/kWhe)'] = 0
        else:
            df['w4e_int_factor_(L/kWhe)'] = df['eGRID_subregion'].apply(
                lambda x: PoG_water_consumption[x])

        # PoG
        df['PoG_annual_water_consumption_L'] = df.MakeupWater_kg + \
            df.Cooling_ElecDemand_kWh * \
            df['w4e_int_factor_(L/kWhe)']  # 1 kg = 1 L
        df['percent_evaporation'] = df.MakeupWater_kg / \
            df['PoG_annual_water_consumption_L']

        df['PoG_WaterConsumption_intensity_L/kWh'] = df['PoG_annual_water_consumption_L'] / \
            df['CoolingDemand_kWh']
        df['PoG_WaterConsumption_intensity_L/kWh_sqm'] = df['PoG_WaterConsumption_intensity_L/kWh'] / \
            df['floor_area_m^2']

        w4e_df = df.copy()

    elif how == 'fuel_type':
        fuel_dataframes = []

        for fuel in PoG_water_consumption.index:
            df = annual_df.copy()

            df['fuel_type'] = fuel

            w4e_factor = PoG_water_consumption[fuel]
            df['w4e_int_factor_(L/kWhe)'] = w4e_factor

            # PoG
            df['PoG_annual_water_consumption_L'] = df.MakeupWater_kg \
                + df.Cooling_ElecDemand_kWh * \
                df['w4e_int_factor_(L/kWhe)']  # 1 kg = 1 L
            df['percent_evaporation'] = df.MakeupWater_kg / \
                df['PoG_annual_water_consumption_L']

            df['PoG_WaterConsumption_intensity_L/kWh'] = df['PoG_annual_water_consumption_L'] / \
                df['CoolingDemand_kWh']
            df['PoG_WaterConsumption_intensity_L/kWh_sqm'] = df['PoG_WaterConsumption_intensity_L/kWh'] / \
                df['floor_area_m^2']

            fuel_dataframes.append(df)

        w4e_df = pd.concat(fuel_dataframes, axis=0).reset_index(drop=True)

    file_path = r'model_outputs\AbsorptionChillers\water_consumption'
    file_name = F'{chiller_type}\\PoG_water_for_cooling_{how}.csv'
    w4e_df.to_csv(F'{file_path}\\{file_name}')


def baseline_PoG_water_consumption(how='NERC'):
    filepath = r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller'
    filename = F'PoG_water_for_cooling_{how}.csv'
    df = pd.read_csv(F'{filepath}\\{filename}')

    if how == 'NERC':
        df = df[['city', 'building', 'CoolingDemand_kWh',
                 'floor_area_m^2', 'climate_zone', 'eGRID_subregion',
                 'grid_loss_factor',
                 'w4e_int_factor_(L/kWhe)',
                 'CoolingDemand_intensity_kWh/sqm']].copy()

    if how == 'fuel_type':
        df = df[['city', 'building', 'CoolingDemand_kWh',
                 'floor_area_m^2', 'climate_zone', 'eGRID_subregion', 'fuel_type',
                 'grid_loss_factor',
                 'w4e_int_factor_(L/kWhe)',
                 'CoolingDemand_intensity_kWh/sqm']].copy()

    df['chiller_type'] = 'AirCooledChiller'
    df['Cooling_ElecDemand_kWh'] = df['CoolingDemand_kWh'] / (3.5 * (1 - df['grid_loss_factor']))
    df['Cooling_HeatDemand_kWh'] = 0
    df['CoolingDemand_intensity_kWh/sqm'] = df['CoolingDemand_kWh'] / \
        df['floor_area_m^2']

    # Conversion Stage
    df['PoG_annual_water_consumption_L'] = df['Cooling_ElecDemand_kWh'] * \
        df['w4e_int_factor_(L/kWhe)']
    df['PoG_WaterConsumption_intensity_L/kWh'] = df['PoG_annual_water_consumption_L'] / \
        df['CoolingDemand_kWh']
    df['PoG_WaterConsumption_intensity_L/kWh_sqm'] = df['PoG_WaterConsumption_intensity_L/kWh'] / \
        df['floor_area_m^2']

    df['MakeupWater_kg'] = 0
    df['percent_evaporation'] = 0

    file_path = r'model_outputs\AbsorptionChillers\water_consumption'
    file_name = F'\\PoG_water_for_cooling_baseline_{how}.csv'
    df.to_csv(F'{file_path}\\{file_name}')


def calculate_LC_water_consumption(
        chiller_type='AbsorptionChiller', how='NERC'):
    cities = city_list

    try:
        cities.remove('fairbanks')
    except ValueError:
        pass

    climate_grid_ls = generate_climate_grid_tuples()

    city_dataframes = []
    for i in climate_grid_ls:
        climate_zone = i[0]
        eGRID_subregion = i[1]
        city = i[2]

        building_dataframes = []

        for building in building_type_list:
            floor_area = floor_area_dictionary[building]

            filepath = r'model_outputs\AbsorptionChillers\building_supply_sim'
            filename = F'\\{chiller_type}\\{city}_{building}_simulation.feather'

            df = pd.read_feather(F'{filepath}\\{filename}')
            df.set_index('datetime', inplace=True, drop=True)
            df.index = pd.to_datetime(df.index)

            df['city'] = city
            df['building'] = building

            agg_df = df.groupby(['city', 'building']).resample('A').agg({
                'CoolingDemand_kW': ['sum'],
                'Cooling_HeatDemand_kW': ['sum'],
                'Cooling_ElecDemand_kW': ['sum'],
                'MakeupWater_kph': ['sum']
            })

            agg_df.columns = agg_df.columns.map('_'.join)
            # agg_df.columns = agg_df.columns.droplevel(1)
            agg_df.rename(columns={'CoolingDemand_kW_sum': 'CoolingDemand_kWh',
                                   'Cooling_HeatDemand_kW_sum': 'Cooling_HeatDemand_kWh',
                                   'Cooling_ElecDemand_kW_sum': 'Cooling_ElecDemand_kWh',
                                   'MakeupWater_kph_sum': 'MakeupWater_kg'}, inplace=True)

            agg_df['floor_area_m^2'] = floor_area
            agg_df['climate_zone'] = climate_zone
            agg_df['eGRID_subregion'] = eGRID_subregion

            agg_df.reset_index(inplace=True)
            agg_df.drop(columns=['datetime'], inplace=True)

            building_dataframes.append(agg_df)

        city_df = pd.concat(building_dataframes, axis=0).reset_index(drop=True)

        # Correct electricity consumption to reflect grid_losses
        if chiller_type == 'AbsorptionChiller':
            # Absorption Chiller gets electricity from CHP, so no distribution
            # loss
            city_df['grid_loss_factor'] = 0
        else:
            grid_loss_df = retrieve_water_consumption_intensities('NERC')[
                'grid_gross_loss']
            try:
                city_df['grid_loss_factor'] = city_df[eGRID_subregion].apply(
                    lambda x: grid_loss_df[x])
            except KeyError:
                city_df['grid_loss_factor'] = 0.053
            city_df['Cooling_ElecDemand_kWh'] = city_df['Cooling_ElecDemand_kWh'] / \
                (1 - city_df['grid_loss_factor'])

        city_dataframes.append(city_df)

    annual_df = pd.concat(city_dataframes, axis=0).reset_index(drop=True)
    annual_df['chiller_type'] = chiller_type

    annual_df['CoolingDemand_intensity_kWh/sqm'] = annual_df['CoolingDemand_kWh'] / \
        annual_df['floor_area_m^2']

    water_consumption_intensity = retrieve_water_consumption_intensities(how)
    LC_water_consumption = water_consumption_intensity['total']

    if how == 'NERC':
        df1 = annual_df.copy()

        if chiller_type == 'AbsorptionChiller':
            # L/kWh of NG for conventional natural gas, 0.0727 for
            # unconventional NG
            w4e_factor_LC = 0.0279
            df1['w4e_int_factor_(L/kWhe)'] = w4e_factor_LC

            chp_df = retrieve_PrimeMover_specs()
            chp_df.drop(columns=['ST1', 'ST2', 'ST3'], inplace=True)

            chp_sim_dataframes = []

            for CHP in chp_df.columns.unique():
                df = df1.copy()
                df['chp_id'] = CHP
                chp = chp_df[CHP]

                # CHP efficiency at lower heating value
                eta_CHP = max(chp['chp_EFF_LHV'], chp['chp_EFF_HHV'])
                hpr_CHP = 1 / chp['phr']    # Heat to power ratio of the CHP

                chp_heat_demand = df['Cooling_HeatDemand_kWh']

                chp_fuel_demand = (chp_heat_demand *
                                   (1 + 1 / hpr_CHP)) / eta_CHP

                df['Total_annual_water_consumption_L'] = df.MakeupWater_kg \
                    + chp_fuel_demand * w4e_factor_LC  # 1 kg = 1 L

                df['percent_evaporation'] = df.MakeupWater_kg / \
                    df['Total_annual_water_consumption_L']

                chp_sim_dataframes.append(df)

            df = pd.concat(chp_sim_dataframes, axis=0).reset_index(drop=True)

        else:
            df = annual_df.copy()
            df['w4e_int_factor_(L/kWhe)'] = df['eGRID_subregion'].apply(
                lambda x: LC_water_consumption[x])
            df['Total_annual_water_consumption_L'] = df.MakeupWater_kg \
                + df.Cooling_ElecDemand_kWh * \
                df['w4e_int_factor_(L/kWhe)']  # 1 kg = 1 L

            df['percent_evaporation'] = df.MakeupWater_kg / \
                df['Total_annual_water_consumption_L']

        # Total Lifecycle
        df['Total_WaterConsumption_intensity_L/kWh'] = df['Total_annual_water_consumption_L'] / \
            df['CoolingDemand_kWh']
        df['Total_WaterConsumption_intensity_L/kWh_sqm'] = df['Total_WaterConsumption_intensity_L/kWh'] / df['floor_area_m^2']

        w4e_df = df.copy()

    elif how == 'fuel_type':
        fuel_dataframes = []

        for fuel in LC_water_consumption.index:
            df1 = annual_df.copy()
            df1['fuel_type'] = fuel

            if chiller_type == 'AbsorptionChiller':
                # L/kWh of NG for conventional natural gas, 0.0727 for
                # unconventional NG
                w4e_factor_LC = 0.0279

                df1['w4e_int_factor_(L/kWhe)'] = w4e_factor_LC

                chp_df = retrieve_PrimeMover_specs()
                chp_df.drop(columns=['ST1', 'ST2', 'ST3'], inplace=True)

                chp_sim_dataframes = []

                for CHP in chp_df.columns.unique():
                    df = df1.copy()
                    df['chp_id'] = CHP
                    chp = chp_df[CHP]
                    # CHP efficiency at lower heating value
                    eta_CHP = max(chp['chp_EFF_LHV'], chp['chp_EFF_HHV'])
                    # Heat to power ratio of the CHP
                    hpr_CHP = 1 / chp['phr']

                    chp_heat_demand = df['Cooling_HeatDemand_kWh']

                    chp_fuel_demand = (chp_heat_demand *
                                       (1 + 1 / hpr_CHP)) / eta_CHP

                    df['Total_annual_water_consumption_L'] = df.MakeupWater_kg \
                        + chp_fuel_demand * w4e_factor_LC  # 1 kg = 1 L

                    df['percent_evaporation'] = df.MakeupWater_kg / \
                        df['Total_annual_water_consumption_L']

                    chp_sim_dataframes.append(df)

                df = pd.concat(
                    chp_sim_dataframes,
                    axis=0).reset_index(
                    drop=True)

            else:
                df = annual_df.copy()
                df['fuel_type'] = fuel

                w4e_factor_LC = LC_water_consumption[fuel]
                df['w4e_int_factor_(L/kWhe)'] = w4e_factor_LC

                # Total Lifecycle
                df['Total_annual_water_consumption_L'] = df.MakeupWater_kg \
                    + df.Cooling_ElecDemand_kWh * \
                    df['w4e_int_factor_(L/kWhe)']  # 1 kg = 1 L

                df['percent_evaporation'] = df.MakeupWater_kg / \
                    df['Total_annual_water_consumption_L']

            df['Total_WaterConsumption_intensity_L/kWh'] = df['Total_annual_water_consumption_L'] / \
                df['CoolingDemand_kWh']
            df['Total_WaterConsumption_intensity_L/kWh_sqm'] = df['Total_WaterConsumption_intensity_L/kWh'] / df['floor_area_m^2']

            fuel_dataframes.append(df)

        w4e_df = pd.concat(fuel_dataframes, axis=0).reset_index(drop=True)

    file_path = r'model_outputs\AbsorptionChillers\water_consumption'
    file_name = F'{chiller_type}\\LC_water_for_cooling_{how}.csv'
    w4e_df.to_csv(F'{file_path}\\{file_name}')


def baseline_LC_water_consumption(how='NERC'):
    filepath = r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller'
    filename = F'LC_water_for_cooling_{how}.csv'
    df = pd.read_csv(F'{filepath}\\{filename}')

    if how == 'NERC':
        df = df[['city', 'building', 'CoolingDemand_kWh',
                 'floor_area_m^2', 'climate_zone', 'eGRID_subregion',
                 'grid_loss_factor',
                 'w4e_int_factor_(L/kWhe)',
                 'CoolingDemand_intensity_kWh/sqm']].copy()

    if how == 'fuel_type':
        df = df[['city', 'building', 'CoolingDemand_kWh',
                 'floor_area_m^2', 'climate_zone', 'eGRID_subregion', 'fuel_type',
                 'grid_loss_factor',
                 'w4e_int_factor_(L/kWhe)',
                 'CoolingDemand_intensity_kWh/sqm']].copy()

    df['chiller_type'] = 'AirCooledChiller'
    df['Cooling_ElecDemand_kWh'] = df['CoolingDemand_kWh'] / (3.5 * (1 - df['grid_loss_factor']))
    df['Cooling_HeatDemand_kWh'] = 0
    df['CoolingDemand_intensity_kWh/sqm'] = df['CoolingDemand_kWh'] / \
        df['floor_area_m^2']

    # Lifecycle Stage
    df['Total_annual_water_consumption_L'] = df['Cooling_ElecDemand_kWh'] * \
        df['w4e_int_factor_(L/kWhe)']
    df['Total_WaterConsumption_intensity_L/kWh'] = df['Total_annual_water_consumption_L'] / \
        df['CoolingDemand_kWh']
    df['Total_WaterConsumption_intensity_L/kWh_sqm'] = df['Total_WaterConsumption_intensity_L/kWh'] / df['floor_area_m^2']

    df['MakeupWater_kg'] = 0
    df['percent_evaporation'] = 0

    file_path = r'model_outputs\AbsorptionChillers\water_consumption'
    file_name = F'\\LC_water_for_cooling_baseline_{how}.csv'
    df.to_csv(F'{file_path}\\{file_name}')


def concatenate_dataframes(how='NERC', scope=2):
    '''
    Inputs:
        how: "NERC" or "fuel_type"
        scope: 2 or 3
    '''

    # Read simulation files
    ACC_filepath = r'model_outputs\AbsorptionChillers\water_consumption'
    WCC_filepath = r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller'
    ABC_filepath = r'model_outputs\AbsorptionChillers\water_consumption\AbsorptionChiller'

    if scope == 2:
        prefix = 'PoG'
    elif scope == 3:
        prefix = 'LC'

    baseline_filename = F'{prefix}_water_for_cooling_baseline_{how}.csv'
    chiller_filename = F'{prefix}_water_for_cooling_{how}.csv'

    ACC_df = pd.read_csv(F'{ACC_filepath}\\{baseline_filename}', index_col=0)
    WCC_df = pd.read_csv(F'{WCC_filepath}\\{chiller_filename}', index_col=0)
    ABC_df = pd.read_csv(F'{ABC_filepath}\\{chiller_filename}', index_col=0)

    dataframes = [ACC_df, WCC_df, ABC_df]
    #short-term fix for missing values
    # for df in dataframes:
    #     # print(df['chiller_type'].unique())
    #     if 'AirCooledChiller' in df['chiller_type'].unique():
    #         df['MakeupWater_kg'] = 0
    #         df['percent_evaporation'] = 0

    # Concatenate data
    df = pd.concat(dataframes, axis=0)
    df.reset_index(inplace=True, drop=True)
    
    # print(df.isna().any())
    # df.fillna(0, inplace=True)

    return df


def concatenate_all_data(how='NERC', save=True):

    df_S2 = concatenate_dataframes(how, scope=2)
    df_S3 = concatenate_dataframes(how, scope=3)

    df_S2['scope'] = 'PoG'
    df_S3['scope'] = 'LC'

    # Rename columns
    df_S2.rename(columns={'PoG_w4e_intensity_factor_(L/kWh)': 'w4e_int_factor_(L/kWhe)',
                          'PoG_annual_water_consumption_L': 'annual_water_consumption_L',
                          'PoG_WaterConsumption_intensity_L/kWh': 'WaterConsumption_int_(L/kWhr)',
                          'PoG_WaterConsumption_intensity_L/kWh_sqm': 'WaterConsumption_int_(L/kWhr_sqm)'},
                 inplace=True)

    df_S3.rename(columns={'Total_w4e_intensity_factor_(L/kWh)': 'w4e_int_factor_(L/kWhe)',
                          'Total_annual_water_consumption_L': 'annual_water_consumption_L',
                          'Total_WaterConsumption_intensity_L/kWh': 'WaterConsumption_int_(L/kWhr)',
                          'Total_WaterConsumption_intensity_L/kWh_sqm': 'WaterConsumption_int_(L/kWhr_sqm)'},
                 inplace=True)
    
    df_S2 = calculate_percent_diff(df_S2)

    df_S3 = calculate_percent_diff(df_S3)

    df = pd.concat([df_S2, df_S3], axis=0).reset_index(drop=True)

    df['chp_id'].fillna(value='NotApplicable', inplace=True)

    convert_dict = {'city':'object', 
                    'building':'object', 
                    'CoolingDemand_kWh':'float', 
                    'floor_area_m^2':'float',
                    'climate_zone':'object', 
                    'eGRID_subregion':'object', 
                    'w4e_int_factor_(L/kWhe)':'float',       
                    'CoolingDemand_intensity_kWh/sqm':'float', 
                    'chiller_type':'object',       
                    'Cooling_ElecDemand_kWh':'float', 
                    'Cooling_HeatDemand_kWh':'float',       
                    'annual_water_consumption_L':'float', 
                    'WaterConsumption_int_(L/kWhr)':'float',       
                    'WaterConsumption_int_(L/kWhr_sqm)':'float', 
                    'MakeupWater_kg':'float',       
                    'grid_loss_factor':'float', 
                    'percent_evaporation':'float', 
                    'scope':'object', 
                    'percent_diff':'float',
                    'chp_id':'object',
                    }
                
    df = df.astype(convert_dict)

    save_path = r'model_outputs\AbsorptionChillers\water_consumption'
    file_name = F'annual_w4r_{how}.feather'

    df.to_feather(F'{save_path}\{file_name}')

    return df



#CONCATENATE TEST
# concatenate_all_data(how='fuel_type')

# Other Calcs
def averages_for_water_cons(how='NERC'):

    data = concatenate_all_data(how)

    PoG_subset = data[data['scope'] == 'PoG']
    LC_subset = data[data['scope'] == 'LC']

    PoG_subset = agg_avg_water_cons_NERC(PoG_subset)
    LC_subset = agg_avg_water_cons_NERC(LC_subset)

    df = pd.concat([PoG_subset, LC_subset], axis=0).reset_index(drop=True)

    file_path = r'model_outputs\AbsorptionChillers\water_consumption'
    df.to_csv(F'{file_path}\\NERC_w4r_summary.csv')


def agg_avg_water_cons_NERC(df):
    df = calculate_percent_diff(df)

    df['WaterConsumption_int_(L/MWhr_sqm)'] = df['WaterConsumption_int_(L/kWhr_sqm)'] * 1000

    df = df.groupby(['eGRID_subregion', 'climate_zone', 'chiller_type', 'scope']).agg({'WaterConsumption_int_(L/MWhr_sqm)': ['mean', 'std'],
                                                                                       'percent_diff': ['mean', 'std']})

    df.columns = df.columns.map('-'.join)
    df.rename(columns={'WaterConsumption_int_(L/MWhr_sqm)-mean': 'WC_mean',
                       'WaterConsumption_int_(L/MWhr_sqm)-std': 'WC_std',
                       'perc_diff-mean': '%_mean',
                       'perc_diff-std': '%_std'},
              inplace=True)

    df.reset_index(inplace=True)

    return df


def calculate_percent_diff(df):
    ACC_df = df[df['chiller_type'] == 'AirCooledChiller'].copy()
    WCC_df = df[df['chiller_type'] == 'WaterCooledChiller'].copy()
    ABC_df = df[df['chiller_type'] == 'AbsorptionChiller'].copy()

    ACC_df.set_index(['eGRID_subregion', 'climate_zone',
                     'city', 'building'], inplace=True, drop=True)
    ACC_series = ACC_df['WaterConsumption_int_(L/kWhr)'].copy()
    WCC_df.set_index(['eGRID_subregion', 'climate_zone',
                     'city', 'building'], inplace=True, drop=True)
    WCC_series = WCC_df['WaterConsumption_int_(L/kWhr)'].copy()
    ABC_df.set_index(['eGRID_subregion', 'climate_zone',
                     'city', 'building'], inplace=True, drop=True)

    ACC_df['percent_diff'] = 0
    WCC_df['percent_diff'] = (WCC_series - ACC_series) / ACC_series * 100

    try:
        subsets = []
        for chp in ABC_df['chp_id'].unique():
            subset = ABC_df[ABC_df['chp_id'] == chp].copy()
            ABC_series = subset['WaterConsumption_int_(L/kWhr)'].copy()
            subset['percent_diff'] = (
                ABC_series - ACC_series) / ACC_series * 100
            subsets.append(subset)
        ABC_df = pd.concat(subsets, axis=0)
    except KeyError:
        ABC_series = ABC_df['WaterConsumption_int_(L/kWhr_sqm)'].copy()
        ABC_df['percent_diff'] = (ABC_series - ACC_series) / ACC_series * 100

    ACC_df['chiller_type'] = 'AirCooledChiller'
    WCC_df['chiller_type'] = 'WaterCooledChiller'
    ABC_df['chiller_type'] = 'AbsorptionChiller'

    ACC_df.reset_index(inplace=True)
    WCC_df.reset_index(inplace=True)
    ABC_df.reset_index(inplace=True)

    processed_df = pd.concat([ACC_df, WCC_df, ABC_df], axis=0)
    processed_df.reset_index(inplace=True, drop=True)

    return processed_df

##################################
# Functions to normalize results #
##################################
def reference_val(df, how, scope):
    w4e_factors = retrieve_water_consumption_intensities(how)

    # Get w4e and grid loss factors
    if scope == 'PoG':
        w4e_US = w4e_factors['PoG']['United States']
    elif scope == 'LC':
        w4e_US = w4e_factors['total']['United States']
    # US Grid Loss Factor = 5.3%
    grid_loss_factor = 0.053

    # Refine subset to ACC and the desired scope
    ACC_df = df[df['chiller_type'] == 'AirCooledChiller'].copy()
    ACC_df = ACC_df[ACC_df['scope'] == scope].copy()

    ACC_df['grid_loss_factor'] = grid_loss_factor
    ACC_df['w4e_int_factor_(L/kWhe)'] = w4e_US
    
    ACC_df['Cooling_ElecDemand_kWh'] = ACC_df['CoolingDemand_kWh'] / (3.5 * (1 - df['grid_loss_factor']))
    
    ACC_df['annual_water_consumption_L'] = ACC_df['Cooling_ElecDemand_kWh'] * \
        ACC_df['w4e_int_factor_(L/kWhe)']
    ACC_df['WaterConsumption_int_(L/kWhr)'] = ACC_df['annual_water_consumption_L'] / \
        ACC_df['CoolingDemand_kWh']
    ACC_df['WaterConsumption_int_(L/kWhr_sqm)'] = ACC_df['WaterConsumption_int_(L/kWhr)'] / ACC_df['floor_area_m^2']

    reference_w4r = ACC_df['WaterConsumption_int_(L/kWhr_sqm)'].mean()

    return reference_w4r

def normalized_w4r(how='NERC', df=None, by_climate=False):
    
    if df is None:
        file_path = r'model_outputs\AbsorptionChillers\water_consumption'
        file_name = F'annual_w4r_{how}.feather'
        df = pd.read_feather(F'{file_path}\{file_name}')

    PoG_ref_val = reference_val(df, how, 'PoG')
    LC_ref_val = reference_val(df, how, 'LC')

    # CONTINUE HERE - FROM THE DATAFRAME, EXTRACT ONLY WCC AND ABC AND DIVIDE BY THE VALUE
    NERC_columns = ['city', 'building', 'climate_zone', 'eGRID_subregion', 
       'chiller_type', 
       'annual_water_consumption_L', 'MakeupWater_kg', 
       'WaterConsumption_int_(L/kWhr)',
       'WaterConsumption_int_(L/kWhr_sqm)', 
       'percent_evaporation', 'scope', 'chp_id']
    fuel_columns = ['city', 'building', 'climate_zone', 'eGRID_subregion', 
       'chiller_type', 'fuel_type',
       'annual_water_consumption_L', 'MakeupWater_kg', 
       'WaterConsumption_int_(L/kWhr)',
       'WaterConsumption_int_(L/kWhr_sqm)', 
       'percent_evaporation', 'scope', 'chp_id']
    try:
        normalized_df = df[NERC_columns].copy()
    except KeyError:
        normalized_df = df[fuel_columns].copy()
    normalized_df = df[(df['chiller_type'] == 'WaterCooledChiller') | (df['chiller_type'] == 'AbsorptionChiller')].copy()

    normalized_df['reference_w4r_(L/kWhr_sqm)'] = normalized_df['scope'].apply(lambda x: PoG_ref_val \
                                                                            if x == 'PoG' \
                                                                            else LC_ref_val)
    
    normalized_df['normalized_w4r'] = normalized_df['WaterConsumption_int_(L/kWhr_sqm)'] / normalized_df['reference_w4r_(L/kWhr_sqm)']

    normalized_df.reset_index(inplace=True, drop=True)
    file_path = r'model_outputs\AbsorptionChillers\water_consumption'
    file_name = F'\\normalized_w4r_{how}.feather'
    normalized_df.to_feather(F'{file_path}\\{file_name}')


def normalized_w4r_aggregate(how='NERC'):
    file_path = r'model_outputs\AbsorptionChillers\water_consumption'
    file_name = F'normalized_w4r_{how}.feather'
    data = pd.read_feather(F'{file_path}\{file_name}')

    data = data[['city', 'building', 'climate_zone', 'eGRID_subregion', 'chiller_type', 
       'WaterConsumption_int_(L/kWhr_sqm)',
       'percent_evaporation', 'scope', 'reference_w4r_(L/kWhr_sqm)',
       'normalized_w4r', 'percent_diff']].copy()

    PoG_subset = data[data['scope'] == 'PoG']
    LC_subset = data[data['scope'] == 'LC']

    PoG_subset = normalized_mean_std(PoG_subset)
    LC_subset = normalized_mean_std(LC_subset)

    df = pd.concat([PoG_subset, LC_subset], axis=0).reset_index(drop=True)

    print(df)
    file_path = r'model_outputs\AbsorptionChillers\water_consumption'
    df.to_feather(F'{file_path}\\NERC_w4r_summary.feather')

def normalized_mean_std(df):
    df = df.copy()
    df['WaterConsumption_int_(L/MWhr_sqm)'] = df['WaterConsumption_int_(L/kWhr_sqm)'] * 1000
    df['reference_w4r_(L/MWhr_sqm)'] = df['reference_w4r_(L/kWhr_sqm)'] * 1000
    df = df.groupby(['eGRID_subregion', 'climate_zone', 'chiller_type', 'scope']).agg({'WaterConsumption_int_(L/MWhr_sqm)': ['mean', 'std'],
                                                                                       'percent_evaporation': ['mean', 'std'],
                                                                                       'reference_w4r_(L/MWhr_sqm)':['mean'],
                                                                                       'normalized_w4r':['mean','std'],
                                                                                       'percent_diff':['mean', 'std']
                                                                                       })

    df.columns = df.columns.map('-'.join)
    df.rename(columns={'WaterConsumption_int_(L/MWhr_sqm)-mean': 'WC_mean',
                       'WaterConsumption_int_(L/MWhr_sqm)-std': 'WC_std',
                       'percent_evaporation-mean': '%_evap_mean',
                       'percent_evaporation-std': '%_evap_std',
                       'reference_w4r_(L/MWhr_sqm)-mean':'reference_w4r_(L/MWhr_sqm)',
                       'normalized_w4r-mean':'normalized_w4r_mean',
                       'normalized_w4r-std':'normalized_w4r_std',
                       },
              inplace=True)

    df.reset_index(inplace=True)

    return df


def WC_int_mean_std():
    df = pd.read_feather(r'model_outputs\AbsorptionChillers\water_consumption\annual_w4r_NERC.feather')

    df['WaterConsumption_int_(L/MWhr_sqm)'] = df['WaterConsumption_int_(L/kWhr_sqm)'] * 1000
    df['WaterConsumption_int_(L/sqm)'] = df['WaterConsumption_int_(L/kWhr_sqm)'] * df['CoolingDemand_kWh']
    df['MakeupWater_m^3'] = df['MakeupWater_kg'] / 1000
    df['MakeupWater_int_m^3/kWhr'] = df['MakeupWater_m^3'] /  df['CoolingDemand_kWh']

    df = df.groupby(['eGRID_subregion', 'climate_zone', 'chiller_type', 'scope']).agg({'WaterConsumption_int_(L/MWhr_sqm)': ['mean', 'std'],
                                                                                        'WaterConsumption_int_(L/sqm)':['mean', 'std'],
                                                                                       'WaterConsumption_int_(L/kWhr)':['mean', 'std'],
                                                                                       'MakeupWater_int_m^3/kWhr':['mean', 'std'],
                                                                                       'percent_evaporation': ['mean', 'std'],
                                                                                       })

    df.columns = df.columns.map('-'.join)
    df.rename(columns={'WaterConsumption_int_(L/MWhr_sqm)-mean': 'WC_mean',
                       'WaterConsumption_int_(L/MWhr_sqm)-std': 'WC_std',
                       'WaterConsumption_int_(L/sqm)-mean':'WC_2_mean',
                       'WaterConsumption_int_(L/sqm)-std':'WC_2_std',
                       'WaterConsumption_int_(L/kWhr)-mean':'W4C_mean',
                       'WaterConsumption_int_(L/kWhr)-std':'W4C_std',
                       'MakeupWater_int_m^3/kWhr-mean':'MuW_mean',
                       'MakeupWater_int_m^3/kWhr-std':'MuW_std',
                       'percent_evaporation-mean': '%_evap_mean',
                       'percent_evaporation-std': '%_evap_std',
                       },
              inplace=True)

    df.reset_index(inplace=True)

    save_path = r'model_outputs\AbsorptionChillers\water_consumption'
    save_file = r'total_w4r_NERC.feather'
    df.to_feather(F'{save_path}\{save_file}')

    return

def get_min_max():
    df = pd.read_feather(r'model_outputs\AbsorptionChillers\water_consumption\annual_w4r_NERC.feather')

    for scope in df['scope'].unique():
        data = df[df['scope'] == scope]
        print(scope)
        for chiller in df['chiller_type'].unique():
            subset = data[data['chiller_type'] == chiller]
            w4c_min = subset["WaterConsumption_int_(L/kWhr)"].min()
            w4c_max = subset["WaterConsumption_int_(L/kWhr)"].max()
            print('\n' + F'{chiller}')
            print(F'Min: {w4c_min}, Max: {w4c_max}')
            print(F'{scope}_{chiller}_min')
            print(subset[subset['WaterConsumption_int_(L/kWhr)'] == w4c_min])
            print(F'{scope}_{chiller}_max')
            print(subset[subset['WaterConsumption_int_(L/kWhr)'] == w4c_max])

# WC_int_mean_std()
# get_min_max()
# datafile = r'model_outputs\AbsorptionChillers\water_consumption\total_w4r_NERC.feather'
# df = pd.read_feather(datafile)
# df.to_csv(r'model_outputs\AbsorptionChillers\water_consumption\total_w4r_NERC.csv')
# # print(df)
# print(df[df['chiller_type'] == 'AirCooledChiller'].scope.unique())

def calculate_makeup_water_stats(chiller=None):
    annual_data = r'model_outputs\AbsorptionChillers\water_consumption\annual_w4r.feather'
    annual_df = pd.read_feather(annual_data)
    climate_zones = ['2B', '3B']#, '4B', '5B']# , '6B']
    
    if chiller is None:
        annual_df = annual_df[(annual_df['chiller_type'] != 'AirCooledChiller') & (annual_df['climate_zone'].isin(climate_zones))
                          ].copy()
    else:
        annual_df = annual_df[(annual_df['chiller_type'] == chiller)# & (annual_df['climate_zone'].isin(climate_zones))
                          ].copy()

    annual_df['MuW_kg/kWhr'] = annual_df['MakeupWater_kg'] / annual_df['CoolingDemand_kWh']
    
    print(chiller)
    print('mean')
    print(annual_df['MuW_kg/kWhr'].mean())
    print('median')
    print(annual_df['MuW_kg/kWhr'].median())
    print('min, max')
    print(annual_df['MuW_kg/kWhr'].min(), annual_df['MuW_kg/kWhr'].max())
    print('std')
    print(annual_df['MuW_kg/kWhr'].std())

# calculate_makeup_water_stats('WaterCooledChiller')
# calculate_makeup_water_stats('AbsorptionChiller')


############
# Continue #
############
# Notes
# Normalized ACC values should be the values for the average US PoG and lifecycle water consumption
# Make a function to calculate the reference value OR add it into your simulation runs.

# def calculate_normalized_values(df):
#     ACC_subset = df[df['chiller_type'] == 'AirCooledChiller'].copy()
#     reference_value = ACC_subset['WaterConsumption_int_(L/kWhr_sqm)'].mean()

#     normalized_df = df[(df['chiller_type'] == 'WaterCooledChiller') | (df['chiller_type'] == 'AbsorptionChiller')].copy()

#     normalized_df['WC_normalized'] = normalized_df['WaterConsumption_int_(L/kWhr_sqm)'] / reference_value

#     normalized_df.reset_index(inplace=True, drop=True)

#     return normalized_df

# def normalize_w4r(how='NERC'):
#     data = concatenate_all_data(how)

#     PoG_subset = data[data['scope'] == 'PoG']
#     LC_subset = data[data['scope'] == 'LC']

#     PoG_subset = calculate_normalized_values(PoG_subset)
#     LC_subset = calculate_normalized_values(LC_subset)

#     df = pd.concat([PoG_subset, LC_subset], axis=0).reset_index(drop=True)

#     file_path = r'model_outputs\AbsorptionChillers\water_consumption'
#     df.to_csv(F'{file_path}\\NERC_w4r_normalized.csv')

###################
# DATA PROCESSING #
###################
# 1. Clean the EES outputs, label columns for each chiller
# clean_EES_outputs('WaterCooledChiller')
# clean_EES_outputs('AbsorptionChiller')

# 2. Calculate the cooling, electricity for cooling, and makeup water per building
# annual_building_sim(chiller_type='AbsorptionChiller', district_cooling_loss=0.1)
# annual_building_sim(chiller_type='WaterCooledChiller', district_cooling_loss=0.1)

# 3. Calculate the water-for-cooling consumption
# print('PoG Impacts:')
# print('Calculating "NERC" water consumption')
# print('Absorption chiller')
# calculate_PoG_water_consumption(chiller_type='AbsorptionChiller', how='NERC')
# print('Water-cooled chiller')
# calculate_PoG_water_consumption(chiller_type='WaterCooledChiller', how='NERC')
# print('Air-cooled chiller')
# baseline_PoG_water_consumption(how='NERC')

# print('Calculating "fuel type" water consumption')
# print('Absorption chiller')
# calculate_PoG_water_consumption(chiller_type='AbsorptionChiller', how='fuel_type')
# print('Water-cooled chiller')
# calculate_PoG_water_consumption(chiller_type='WaterCooledChiller', how='fuel_type')
# print('Air-cooled chiller')
# baseline_PoG_water_consumption(how='fuel_type')

# print('LC Impacts:')
# print('Calculating "NERC" water consumption')
# print('Absorption chiller')
# calculate_LC_water_consumption(chiller_type='AbsorptionChiller', how='NERC')
# print('Water-cooled chiller')
# calculate_LC_water_consumption(chiller_type='WaterCooledChiller', how='NERC')
# print('Air-cooled chiller')
# baseline_LC_water_consumption(how='NERC')

# print('Calculating "fuel type" water consumption')
# print('Absorption chiller')
# calculate_LC_water_consumption(chiller_type='AbsorptionChiller', how='fuel_type')
# print('Water-cooled chiller')
# calculate_LC_water_consumption(chiller_type='WaterCooledChiller', how='fuel_type')
# print('Air-cooled chiller')
# baseline_LC_water_consumption(how='fuel_type')


# Calculate peak electricity demand reductions
# calculate_peak_demand_reductions()

# Calculate average water consumption
# averages_for_water_cons()

# Normalize Data
# concatenate_all_data('NERC')
# normalized_w4r('NERC')
# normalized_w4r('NERC')
# normalized_w4r_aggregate()