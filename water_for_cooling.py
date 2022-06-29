####################################
# LIBRARIES NEEDED TO RUN THE TOOL #
####################################
from logging import error
from msilib.schema import Error
from tkinter import Label
from openpyxl import load_workbook
import math as m

# Scientific python add-ons
import pandas as pd     # To install: pip install pandas
import numpy as np      # To install: pip install numpy

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
from sympy import QQ_gmpy

from sysClasses import *

##########################################################################

# TO DO
# Function to calculate lifetime water consumption 
#   Must include CHP variety for ABCs when CHP is included
############################


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


electric_chiller_COP = {'rooftop_air_conditioner': 3.4,
                        'air_cooled_reciprocating_chiller': 3.5}


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


def calculate_PoG_water_consumption( chiller_type='AbsorptionChiller', how='NERC'):
    cities = city_list

    try:
        cities.remove('fairbanks')
    except ValueError:
        pass

    city_dataframes = []
    for city in city_list:
        eGRID_subregion = nerc_region_dictionary[city]
        climate_zone = climate_zone_dictionary[city]

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
            agg_df.reset_index(inplace=True)
            agg_df.drop(columns=['datetime'], inplace=True)

            building_dataframes.append(agg_df)

        city_df = pd.concat(building_dataframes, axis=0).reset_index(drop=True)
        city_df['climate_zone'] = climate_zone
        city_df['eGRID_subregion'] = eGRID_subregion

        # Correct electricity consumption to reflect grid_losses
        if chiller_type == 'AbsorptionChiller':
            # Absorption Chiller gets electricity from CHP, so no distribution loss
            grid_loss_factor = 0
        else:
            grid_loss_df = retrieve_water_consumption_intensities('NERC')['grid_gross_loss']
            grid_loss_factor = grid_loss_df[eGRID_subregion]
            city_df['Cooling_ElecDemand_kWh'] = city_df['Cooling_ElecDemand_kWh'] / (1 - grid_loss_factor)
        
        city_dataframes.append(city_df)

    annual_df = pd.concat(city_dataframes, axis=0).reset_index(drop=True)
    annual_df['chiller_type'] = chiller_type

    annual_df['CoolingDemand_intensity_kWh/sqm'] = annual_df['CoolingDemand_kWh'] / floor_area

    water_consumption_intensity = retrieve_water_consumption_intensities(how)
    PoG_water_consumption = water_consumption_intensity['PoG']

    if how == 'NERC':
        if chiller_type == 'AbsorptionChiller':
            w4e_factor = 0
            # w4e_factor_LC = 
        else: 
            w4e_factor = PoG_water_consumption[eGRID_subregion]

        df = annual_df.copy()

        df['PoG_w4e_intensity (L/kWh)'] = w4e_factor

        # PoG
        df['annual_water_consumption_L'] = df.MakeupWater_kg + df.Cooling_ElecDemand_kWh * w4e_factor  # 1 kg = 1 L

        df['WaterConsumption_intensity_L/kWh'] = df['annual_water_consumption_L'] / \
            df['CoolingDemand_kWh']
        df['WaterConsumption_intensity L_per_kWh_sqm'] = df['WaterConsumption_intensity_L/kWh'] / floor_area

        w4e_df = df.copy()

    elif how == 'fuel_type':
        fuel_dataframes = []

        for fuel in PoG_water_consumption.index:
            df = annual_df.copy()

            df['fuel_type'] = fuel

            w4e_factor = PoG_water_consumption[fuel]
            df['PoG_w4e_intensity (L/kWh)'] = w4e_factor

            # PoG
            df['annual_water_consumption_L'] = df.MakeupWater_kg \
                + df.Cooling_ElecDemand_kWh * w4e_factor  # 1 kg = 1 L

            df['WaterConsumption_intensity_L/kWh'] = df['annual_water_consumption_L'] / \
                df['CoolingDemand_kWh']
            df['WaterConsumption_intensity L_per_kWh_sqm'] = df['WaterConsumption_intensity_L/kWh'] / floor_area

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
                 'PoG_w4e_intensity (L/kWh)', 'Total_w4e_intensity (L/kWh)',
                 'CoolingDemand_intensity_kWh/sqm']].copy()

    if how == 'fuel_type':
        df = df[['city', 'building', 'CoolingDemand_kWh',
                 'floor_area_m^2', 'climate_zone', 'eGRID_subregion', 'fuel_type',
                 'PoG_w4e_intensity (L/kWh)',
                 'CoolingDemand_intensity_kWh/sqm']].copy()

    df['chiller_type'] = 'AirCooledChiller'
    df['Cooling_ElecDemand_kWh'] = df['CoolingDemand_kWh'] / 3.4
    df['Cooling_HeatDemand_kWh'] = 0
    df['CoolingDemand_intensity_kWh/sqm'] = df['CoolingDemand_kWh'] / \
        df['floor_area_m^2']

    # Conversion Stage
    df['PoG_annual_water_consumption_L'] = df['Cooling_ElecDemand_kWh'] * \
        df['PoG_w4e_intensity (L/kWh)']
    df['PoG_WaterConsumption_intensity_L/kWh'] = df['PoG_annual_water_consumption_L'] / \
        df['CoolingDemand_kWh']
    df['PoG_WaterConsumption_intensity L_per_kWh_sqm'] = df['PoG_WaterConsumption_intensity_L/kWh'] / df['floor_area_m^2']

    file_path = r'model_outputs\AbsorptionChillers\water_consumption'
    file_name = F'\\PoG_water_for_cooling_baseline_{how}.csv'
    df.to_csv(F'{file_path}\\{file_name}')

def calculate_LC_water_consumption(
        chiller_type='AbsorptionChiller', how='NERC', stage='PoG'):
    cities = city_list

    try:
        cities.remove('fairbanks')
    except ValueError:
        pass

    city_dataframes = []
    for city in city_list:
        eGRID_subregion = nerc_region_dictionary[city]
        climate_zone = climate_zone_dictionary[city]

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
            agg_df.reset_index(inplace=True)
            agg_df.drop(columns=['datetime'], inplace=True)

            building_dataframes.append(agg_df)

        city_df = pd.concat(building_dataframes, axis=0).reset_index(drop=True)
        city_df['climate_zone'] = climate_zone
        city_df['eGRID_subregion'] = eGRID_subregion

        # Correct electricity consumption to reflect grid_losses
        if chiller_type == 'AbsorptionChiller':
            # Absorption Chiller gets electricity from CHP, so no distribution loss
            grid_loss_factor = 0
        else:
            grid_loss_df = retrieve_water_consumption_intensities('NERC')['grid_gross_loss']
            grid_loss_factor = grid_loss_df[eGRID_subregion]
            city_df['Cooling_ElecDemand_kWh'] = city_df['Cooling_ElecDemand_kWh'] / (1 - grid_loss_factor)
        
        city_dataframes.append(city_df)

    annual_df = pd.concat(city_dataframes, axis=0).reset_index(drop=True)
    annual_df['chiller_type'] = chiller_type

    annual_df['CoolingDemand_intensity_kWh/sqm'] = annual_df['CoolingDemand_kWh'] / floor_area

    water_consumption_intensity = retrieve_water_consumption_intensities(how)
    LC_water_consumption = water_consumption_intensity['total']

    if how == 'NERC':
        df = annual_df.copy()
        
        if chiller_type == 'AbsorptionChiller':
            w4e_factor_LC = 0.0279 # L/kWh of NG for conventional natural gas, 0.0727 for unconventional NG
            df['Total_w4e_intensity (L/kWh)'] = w4e_factor_LC
            
            chp_df = retrieve_PrimeMover_specs()
            chp_df.drop(columns=['ST1', 'ST2', 'ST3'], inplace=True)

            chp_sim_dataframes = []

            for CHP in chp_df.columns.unique():
                df = annual_df.copy()
                df['chp_id'] = CHP
                chp = chp_df[CHP]
                eta_CHP = chp['chp_EFF_LHV'] # CHP efficiency at lower heating value
                hpr_CHP = 1 / chp['phr']    # Heat to power ratio of the CHP
                
                chp_heat_demand = df['Cooling_HeatDemand_kWh']
                chp_fuel_demand = (chp_heat_demand * (1 + 1/hpr_CHP)) / eta_CHP

                df['Total_annual_water_consumption_L'] = df.MakeupWater_kg \
                    + chp_fuel_demand * w4e_factor_LC  # 1 kg = 1 L
                
                chp_sim_dataframes.append(df)
            
            df = pd.concat(chp_sim_dataframes, axis=0).reset_index(drop=True)
        
        else: 
            df = annual_df.copy()
            w4e_factor_LC = LC_water_consumption[eGRID_subregion]
            df['Total_annual_water_consumption_L'] = df.MakeupWater_kg \
                + df.Cooling_ElecDemand_kWh * w4e_factor_LC  # 1 kg = 1 L
        

        df['Total_w4e_intensity (L/kWh)'] = w4e_factor_LC

        # Total Lifecycle
        df['Total_WaterConsumption_intensity_L/kWh'] = df['Total_annual_water_consumption_L'] / \
            df['CoolingDemand_kWh']
        df['Total_WaterConsumption_intensity L_per_kWh_sqm'] = df['Total_WaterConsumption_intensity_L/kWh'] / floor_area

        w4e_df = df.copy()

    elif how == 'fuel_type':
        fuel_dataframes = []

        for fuel in LC_water_consumption.index:
            df = annual_df.copy()

            df['fuel_type'] = fuel

            w4e_factor_LC = LC_water_consumption[fuel]
            df['Total_w4e_intensity (L/kWh)'] = w4e_factor_LC

            # Total Lifecycle
            df['Total_annual_water_consumption_L'] = df.MakeupWater_kg \
                + df.Cooling_ElecDemand_kWh * w4e_factor_LC  # 1 kg = 1 L

            df['Total_WaterConsumption_intensity_L/kWh'] = df['Total_annual_water_consumption_L'] / \
                df['CoolingDemand_kWh']
            df['Total_WaterConsumption_intensity L_per_kWh_sqm'] = df['Total_WaterConsumption_intensity_L/kWh'] / floor_area

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
                 'Total_w4e_intensity (L/kWh)',
                 'CoolingDemand_intensity_kWh/sqm']].copy()

    if how == 'fuel_type':
        df = df[['city', 'building', 'CoolingDemand_kWh',
                 'floor_area_m^2', 'climate_zone', 'eGRID_subregion', 'fuel_type',
                 'Total_w4e_intensity (L/kWh)',
                 'CoolingDemand_intensity_kWh/sqm']].copy()

    df['chiller_type'] = 'AirCooledChiller'
    df['Cooling_ElecDemand_kWh'] = df['CoolingDemand_kWh'] / 3.4
    df['Cooling_HeatDemand_kWh'] = 0
    df['CoolingDemand_intensity_kWh/sqm'] = df['CoolingDemand_kWh'] / \
        df['floor_area_m^2']

    # Lifecycle Stage
    df['Total_annual_water_consumption_L'] = df['Cooling_ElecDemand_kWh'] * \
        df['Total_w4e_intensity (L/kWh)']
    df['Total_WaterConsumption_intensity_L/kWh'] = df['Total_annual_water_consumption_L'] / \
        df['CoolingDemand_kWh']
    df['Total_WaterConsumption_intensity L_per_kWh_sqm'] = df['Total_WaterConsumption_intensity_L/kWh'] / df['floor_area_m^2']

    file_path = r'model_outputs\AbsorptionChillers\water_consumption'
    file_name = F'\\LC_water_for_cooling_baseline_{how}.csv'
    df.to_csv(F'{file_path}\\{file_name}')


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
# calculate_PoG_water_consumption(chiller_type='AbsorptionChiller', how='NERC')
# calculate_PoG_water_consumption(chiller_type='WaterCooledChiller', how='NERC')
# baseline_PoG_water_consumption(how='NERC')

# print('Calculating "fuel type" water consumption')
# calculate_PoG_water_consumption(chiller_type='AbsorptionChiller', how='fuel_type')
# calculate_PoG_water_consumption(chiller_type='WaterCooledChiller', how='fuel_type')
# baseline_PoG_water_consumption(how='fuel_type')

# print('LC Impacts:')
# print('Calculating "NERC" water consumption')
# calculate_LC_water_consumption(chiller_type='AbsorptionChiller', how='NERC')
# calculate_LC_water_consumption(chiller_type='WaterCooledChiller', how='NERC')
# baseline_LC_water_consumption(how='NERC')

# print('Calculating "fuel type" water consumption')
# calculate_LC_water_consumption(chiller_type='AbsorptionChiller', how='fuel_type')
# calculate_LC_water_consumption(chiller_type='WaterCooledChiller', how='fuel_type')
# baseline_LC_water_consumption(how='fuel_type')
