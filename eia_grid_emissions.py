# Data Plotting
# My modules
from cmath import nan
import csv

from pandas.core.indexes import base
from sysClasses import *
import models
# 3rd Party Modules
import pandas as pd  # To install: pip install pandas
import numpy as np  # To install: pip install numpy
from pyarrow import feather  # Storage format

import scipy.optimize
import scipy.stats
import pathlib
from openpyxl import load_workbook
import math as m
import time
import datetime
import inspect
import os

emm_ls = ['BASN', 'CANO', 'CASO', 'FRCC', 'ISNE', 
          'MISC', 'MISE', 'MISS', 'MISW', 'NWPP',
          'NYCW', 'NYUP', 'PJMC', 'PJMD', 'PJME',
          'PJMW', 'RMRG', 'SPPC', 'SPPN', 'SPPS',
          'SRCA', 'SRCE', 'SRSE', 'SRSG', 'TRE']

def convert_MMBTU_to_kWh(MMBTU):
    return MMBTU * 10**6 / 3412.14

def convert_MMBTU_to_MWh(MMBTU):
    return MMBTU / 3.41214

def read_eia_file(emm, header=0, nrows=None, dataset='electric_power_projections', datatype='Unspecified'):
    columns = [datatype,
               '2020', '2021', '2022', '2023', '2024', '2025',	
               '2026', '2027', '2028', '2029', '2030', 
               '2031', '2032', '2033', '2034', '2035',	
               '2036', '2037', '2038', '2039', '2040',
               '2041', '2042', '2043', '2044', '2045',
               '2046', '2047', '2048', '2049', '2050',]
               #'compound_growth_percent']

    if dataset == 'electric_power_projections':
        data_path = r'data\EIA - Electric Power Projections\raw_electric_power_projections'
    elif dataset == 'renewable_power_projections':
        data_path = r'data\EIA - Electric Power Projections\raw_renewable_power_projections'
    file_name = F'\{emm}.xlsx'
    data_file = data_path + file_name

    df = pd.read_excel(io=data_file, header=header,
                       nrows=nrows,
                       usecols='B:AG')

    df.set_axis(columns, axis=1, inplace=True)

    return df


# Modify the functions below to include the renewable tables.
def get_power_projections(emm, datatype):
    # Data width [header, nrows]
    data_width = {
                  # Net Summer Generating Capacity (GW)
                  'net_capacity_GW':[16, 11],
                  'planned_additions_GW':[29, 11],
                  'unplanned_additions_GW':[41, 11],
                  'retirements_GW':[55, 11],
                  'end_use_sectors_GW':[67, 8],
                  # Electricity Sales
                  'electricity_sales_TWh':[77, 6],
                  'net_energy_for_load_TWh':[84, 8],
                  # Generation by Fuel Type TWh
                  'electric_power_generation_TWh':[95, 10],
                  'end_use_sectors_TWh':[107, 10],
                  'total_electricity_generation_TWh':[117, 1],
                  # End-Use Prices
                  'EU_price_2020_cents_per_kWh':[121, 6],
                  'EU_price_nominal_cents_per_kWh':[127, 6],
                  # Prices by Service Category
                  'SC_price_2020_cents_per_kWh':[135, 4],
                  'SC_price_nominal_cents_per_kWh':[139, 4],
                  # Fuel Consumption
                  'fuel_consumption_QuadBtu':[144, 5],
                  # Fuel Prices to Electric Power Sector
                  'fuel_price_2020_cents_per_kWh':[151, 5],
                  'fuel_price_nominal_cents_per_kWh':[156, 5],
                  # Emissions from the Electric Power Sector}
                  'emissions_from_electric_sector_Mton':[162, 5]}
    header = data_width[datatype][0]
    nrows = data_width[datatype][1]

    return read_eia_file(emm, header=header, nrows=nrows,
                         dataset='electric_power_projections', 
                         datatype=datatype)

def get_renewable_projections(emm, datatype):
    # Data width [header, nrows]
    data_width = {
                  # Net Summer Generating Capacity (GW)
                  'net_capacity_GW':[17, 10],
                  'planned_additions_GW':[29, 10],
                  # Generation by Method TWh
                  'electric_power_generation_TWh':[41, 10],
                  # Energy Consumption QuadBTU
                  'energy_consumption_QuadBtu':[53, 10],
                  }
    header = data_width[datatype][0]
    nrows = data_width[datatype][1]

    return read_eia_file(emm, header=header, nrows=nrows, 
                         datatype=datatype)


def clean_eia_data(emm, datatype, dataset='electric_power_projections'):
    raw = get_power_projections(emm, datatype)

    # TO DO
    # RECALCULATE VALUES BASED ON THE REQUIRED CONVERSIONS
    # RENAME THE FIRST COLUMN
    # TRANSPOSE THE DATA

    if dataset == 'electric_power_projections':
        rows = {# Net Summer Generating Capacity (GW)
                  'net_capacity_GW':['coal', 'oil and natural gas steam', 'combined cycle',
                      'combustion turbine/diesel', 'nuclear power', 'pumped storage',
                      'diurnal storage', 'fuel cells', 'renewable sources', 
                      'distributed generation', 'total capacity'],
                  'planned_additions_GW':['coal', 'oil and natural gas steam', 'combined cycle',
                      'combustion turbine/diesel', 'nuclear power', 'pumped storage',
                      'diurnal storage', 'fuel cells', 'renewable sources', 
                      'distributed generation', 'total planned additions'],
                  'unplanned_additions_GW':['coal', 'oil and natural gas steam', 'combined cycle',
                      'combustion turbine/diesel', 'nuclear power', 'pumped storage',
                      'diurnal storage', 'fuel cells', 'renewable sources', 
                      'distributed generation', 'total unplanned additions'],
                  'retirements_GW':['coal', 'oil and natural gas steam', 'combined cycle',
                      'combustion turbine/diesel', 'nuclear power', 'pumped storage',
                      'diurnal storage', 'fuel cells', 'renewable sources',
                      'total'],
                  'end_use_sectors_GW':['coal', 'petroleum', 'natural gas',
                                        'other gaseous fuels', 'renewable sources', 'other'],
                  # Electricity Sales
                  'electricity_sales_TWh':['residential', 'commercial/other', 'industrial',
                                        'transportation', 'total sales'],
                  'net_energy_for_load_TWh':['gross international imports', 'gross international exports', 
                                            'gross interretional electricity imports',
                                            'gross interregional electricity exports', 
                                            'purchases from combined heat and power', 
                                            'electric power sector generation for customer', 
                                            'total net energy for load'],
                  # Generation by Fuel Type TWh
                  'electric_power_generation_TWh':['coal', 'petroleum', 'natural gas', 
                        'nuclear', 'pumped storage and other', 'renewable sources', 
                        'distributed generation', 'total generation', 'sales to customers',
                        'generation for own use'],
                  'end_use_sectors_TWh':['coal', 'petroleum', 'natural gas', 
                        'other gaseous fuels', 'renewable sources', 'other',
                        'total', 'sales to the grid', 'generation for own use'],
                  'total_electricity_generation_TWh':['total_electricity_generation_TWh'],
                  # End-Use Prices
                  'EU_price_2020_cents_per_kWh':['residential', 'commercial', 'industrial',
                        'transportation', 'all sectors average'],
                  'EU_price_nominal_cents_per_kWh':['residential', 'commercial', 'industrial',
                        'transportation', 'all sectors average'],
                  # Prices by Service Category
                  'SC_price_2020_cents_per_kWh':['generation', 'transmission', 'distribution'],
                  'SC_price_nominal_cents_per_kWh':['generation', 'transmission', 'distribution'],
                  # Fuel Consumption
                  'fuel_consumption_QuadBtu':['coal', 'natural gas', 'oil', 'total'],
                  # Fuel Prices to Electric Power Sector
                  'fuel_price_2020_cents_per_kWh':['coal', 'natural gas', 'distillate fuel oil', 'residual fuel oil'],
                  'fuel_price_nominal_cents_per_kWh':['coal', 'natural gas', 'distillate fuel oil', 'residual fuel oil'],
                  # Emissions from the Electric Power Sector}
                  'emissions_from_electric_sector_Mton':['total_carbon', 'CO2', 'SO2', 'NOx', 'HG']}
    elif dataset == 'renewable_power_projections':
        rows = {# Net Summer Generating Capacity (GW)
                  'net_capacity_GW':['conventional hydroelectric power', 'geothermal',
                      'municipal waste', 'wood and other biomass', 'solar thermal',
                      'solar photovoltaic', 'wind', 'offshore wind', 'total'],
                  'planned_additions_GW':['conventional hydroelectric power', 'geothermal',
                      'municipal waste', 'wood and other biomass', 'solar thermal',
                      'solar photovoltaic', 'wind', 'offshore wind', 'total'],
                  # Electricity Sector
                  'electric_power_generation_TWh':['conventional hydroelectric power', 'geothermal',
                      'municipal waste', 'wood and other biomass', 'solar thermal',
                      'solar photovoltaic', 'wind', 'offshore wind', 'total'],
                  'energy_consumption_QuadBtu':['conventional hydroelectric power', 'geothermal',
                      'municipal waste', 'wood and other biomass', 'solar thermal',
                      'solar photovoltaic', 'wind', 'offshore wind', 'total'],
                }
    else:
        raise KeyError

    df = raw.copy()

    df[datatype] = rows[datatype]
    

    if datatype == 'emissions_from_electric_sector_Mton':
        # 1 short ton = 907.185 kg
        new_index = 'emissions_from_electric_sector_kg' 
        df.rename(columns={datatype:new_index}, inplace=True)
        df.set_index(new_index, inplace=True, drop=True)
        df = df * (907.185 * 10**6)
    else:
        df.set_index(datatype, inplace=True, drop=True)

    df = df.transpose()    

    return df


def calculate_emission_factor(emm, impact):
    generation_df = clean_eia_data(emm, 'electric_power_generation_TWh')
    emissions_df = clean_eia_data(emm, 'emissions_from_electric_sector_Mton')

    emissions_df.index = generation_df.index

    if impact == 'CO2':
        # Convert TWh to MWh - Will give us kg/MWh
        total_generation_MWh = generation_df['total generation'] * 10**6
    else:
        # Convert TWh to GWh - Will give us g / MWh
        total_generation_MWh = generation_df['total generation'] * 10**3
    emissions_kg = emissions_df[impact]


    emission_factors_df = emissions_kg / total_generation_MWh
    
    return emission_factors_df

def calculate_ISO_emission_factor(impact):
    emm_ls = ['BASN', 'CANO', 'CASO', 'FRCC', 'ISNE', 
          'MISC', 'MISE', 'MISS', 'MISW', 'NWPP',
          'NYCW', 'NYUP', 'PJMC', 'PJMD', 'PJME',
          'PJMW', 'RMRG', 'SPPC', 'SPPN', 'SPPS',
          'SRCA', 'SRCE', 'SRSE', 'SRSG', 'TRE']

    index = ['2020', '2021', '2022', '2023', '2024', '2025',	
               '2026', '2027', '2028', '2029', '2030', 
               '2031', '2032', '2033', '2034', '2035',	
               '2036', '2037', '2038', '2039', '2040',
               '2041', '2042', '2043', '2044', '2045',
               '2046', '2047', '2048', '2049', '2050',]
    
    aggregate_df = pd.DataFrame(index=index)

    for emm_region in emm_ls:
        df = calculate_emission_factor(emm_region, impact)
        aggregate_df[emm_region] = df

    return aggregate_df

def calculate_EIA_emission_factors():
    impacts = ['total_carbon', 'CO2', 'SO2', 'NOx', 'HG']
    
    eia_impacts = ['total_carbon', 'CO2', 'SO2', 'NOx', 'HG']
    calculated_impacts = ['CH4', 'CO', 'N2O', 'PM', 'VOC']

    for impact in impacts:
        if impact in eia_impacts:
            df = calculate_ISO_emission_factor(impact)

            savepath = r'data\EIA - Electric Power Projections\processed'
            savefile = F'\EIA_{impact}_emission_factors.csv'
            filename = savepath + savefile
            df.to_csv(filename)

        

    print('Completed EIA Emission Factor Calculations')    
    

def read_eia_cooling_detail():
    r'''
    Read EIA Cooling Detail Data

    https://www.eia.gov/electricity/data/water/
    '''

    data_file = r'data\Tech_specs\eia_cooling_detail_2020.csv'
    columns_to_import = ['\n \n \n \n \n \nUtility ID', 'State', 
       'Plant Code', 'Plant Name', 'Month',
       'Generator ID',
       'Generator Primary Technology',
       'Summer Capacity of Steam Turbines (MW)',
       'Gross Generation from Steam Turbines (MWh)',
       'Net Generation from Steam Turbines (MWh)',
       'Summer Capacity Associated with Single Shaft Combined Cycle Units (MW)',
       'Gross Generation Associated with Single Shaft Combined Cycle Units (MWh)',
       'Net Generation Associated with Single Shaft Combined Cycle Units (MWh)',
       'Summer Capacity Associated with Combined Cycle Gas Turbines (MW)',
       'Gross Generation Associated with Combined Cycle Gas Turbines (MWh)',
       'Net Generation Associated with Combined Cycle Gas Turbines (MWh)',
       'Water Withdrawal Volume (Million Gallons)',
       'Water Consumption Volume (Million Gallons)', 'Sector']


    df = pd.read_csv(data_file, header=2, usecols=columns_to_import)

    df.rename(columns={'\n \n \n \n \n \nUtility ID':'Utility_ID'}, inplace=True)
    # df.set_axis(columns, axis=1, inplace=True)
    columns = ['Utility_ID', 'State', 'Plant Code', 'Plant Name', 'Year', 'Month',
       'Generator ID', 'Boiler ID', 'Cooling ID',
       'Generator Primary Technology',
       'Summer Capacity of Steam Turbines (MW)',
       'Gross Generation from Steam Turbines (MWh)',
       'Net Generation from Steam Turbines (MWh)',
       'Summer Capacity Associated with Single Shaft Combined Cycle Units (MW)',
       'Gross Generation Associated with Single Shaft Combined Cycle Units (MWh)',
       'Net Generation Associated with Single Shaft Combined Cycle Units (MWh)',
       'Summer Capacity Associated with Combined Cycle Gas Turbines (MW)',
       'Gross Generation Associated with Combined Cycle Gas Turbines (MWh)',
       'Net Generation Associated with Combined Cycle Gas Turbines (MWh)',
       'Fuel Consumption from All Fuel Types (MMBTU)',
       'Fuel Consumption from Steam Turbines (MMBTU)',
       'Fuel Consumption from Single Shaft Combined Cycle Units (MMBTU)',
       'Fuel Consumption from Combined Cycle Gas Turbines (MMBTU)',
       'Coal Consumption (MMBTU)', 'Natural Gas Consumption (MMBTU)',
       'Petroleum Consumption (MMBTU)', 'Biomass Consumption (MMBTU)',
       'Other Gas Consumption (MMBTU)', 'Other Fuel Consumption (MMBTU)',
       'Water Withdrawal Volume (Million Gallons)',
       'Water Consumption Volume (Million Gallons)',
       'Water Withdrawal Intensity Rate (Gallons / MWh)',
       'Water Consumption Intensity Rate (Gallons / MWh)',
       'Water Withdrawal Rate per Fuel Consumption (Gallons / MMBTU)',
       'Water Consumption Rate per Fuel Consumption (Gallons / MMBTU)',
       'Cooling Unit Hours in Service',
       'Average Distance of Water Intake Below Water Surface (Feet)',
       '860 Cooling Type 1', '860 Cooling Type 2', '923 Cooling Type',
       'Cooling System Type', 'Water Type', 'Water Source',
       'Water Source Name', 'Water Discharge Name', 'Generator Status',
       'Generator Inservice Month', 'Generator Inservice Year',
       'Generator Retirement Month', 'Generator Retirement Year',
       'Boiler Status', 'Boiler Inservice Month', 'Boiler Inservice Year',
       'Boiler Retirement Month', 'Boiler Retirement Year', 'Cooling Status',
       'Cooling Inservice Month', 'Cooling Inservice Year', '\n', '\n.1',
       'Combined Heat and Power Generator?',
       'Generator Primary Energy Source Code', 'Generator Prime Mover Code',
       'Generator Duct Burners?', 'Sector', 'Steam Plant Type',
       'Number Operable Generators', 'Number Operable Boilers',
       'Number Operable Cooling Systems', 'Relationship Type']

    '''drop_columns = ['Year', 'Cooling ID',
       'Fuel Consumption from All Fuel Types (MMBTU)',
       'Fuel Consumption from Steam Turbines (MMBTU)',
       'Fuel Consumption from Single Shaft Combined Cycle Units (MMBTU)',
       'Fuel Consumption from Combined Cycle Gas Turbines (MMBTU)',
       'Coal Consumption (MMBTU)', 'Natural Gas Consumption (MMBTU)',
       'Petroleum Consumption (MMBTU)', 'Biomass Consumption (MMBTU)',
       'Other Gas Consumption (MMBTU)', 'Other Fuel Consumption (MMBTU)',
       'Water Withdrawal Intensity Rate (Gallons / MWh)',
       'Water Consumption Intensity Rate (Gallons / MWh)',
       'Water Withdrawal Rate per Fuel Consumption (Gallons / MMBTU)',
       'Water Consumption Rate per Fuel Consumption (Gallons / MMBTU)',
       'Cooling Unit Hours in Service',
       'Average Distance of Water Intake Below Water Surface (Feet)',
       '860 Cooling Type 1', '860 Cooling Type 2', '923 Cooling Type',
       'Cooling System Type', 'Water Type', 'Water Source',
       'Water Source Name', 'Water Discharge Name', 'Generator Status',
       'Generator Inservice Month', 'Generator Inservice Year',
       'Generator Retirement Month', 'Generator Retirement Year',
       'Boiler Status', 'Boiler Inservice Month', 'Boiler Inservice Year',
       'Boiler Retirement Month', 'Boiler Retirement Year', 'Cooling Status',
       'Cooling Inservice Month', 'Cooling Inservice Year', '\n', '\n.1',
       'Combined Heat and Power Generator?',
       'Generator Primary Energy Source Code', 'Generator Prime Mover Code',
       'Generator Duct Burners?', 'Steam Plant Type',
       'Number Operable Generators', 'Number Operable Boilers',
       'Number Operable Cooling Systems', 'Relationship Type']
    df.drop(axis=1, columns=drop_columns, inplace=True)'''

    return df


def clean_eia_cooling_detail():
    data_file = r'data\Tech_specs\eia_cooling_detail_2020_clean.csv'
    
    columns = ['Utility_ID', 'State', 'Plant Code', 'Plant Name', 'Year', 'Month',
       'Generator ID', 'Boiler ID', 'Cooling ID',
       'Generator Primary Technology',
       'Summer Capacity of Steam Turbines (MW)',
       'Gross Generation from Steam Turbines (MWh)',
       'Net Generation from Steam Turbines (MWh)',
       'Summer Capacity Associated with Single Shaft Combined Cycle Units (MW)',
       'Gross Generation Associated with Single Shaft Combined Cycle Units (MWh)',
       'Net Generation Associated with Single Shaft Combined Cycle Units (MWh)',
       'Summer Capacity Associated with Combined Cycle Gas Turbines (MW)',
       'Gross Generation Associated with Combined Cycle Gas Turbines (MWh)',
       'Net Generation Associated with Combined Cycle Gas Turbines (MWh)',
       'Fuel Consumption from All Fuel Types (MMBTU)',
       'Fuel Consumption from Steam Turbines (MMBTU)',
       'Fuel Consumption from Single Shaft Combined Cycle Units (MMBTU)',
       'Fuel Consumption from Combined Cycle Gas Turbines (MMBTU)',
       'Coal Consumption (MMBTU)', 'Natural Gas Consumption (MMBTU)',
       'Petroleum Consumption (MMBTU)', 'Biomass Consumption (MMBTU)',
       'Other Gas Consumption (MMBTU)', 'Other Fuel Consumption (MMBTU)',
       'Water Withdrawal Volume (Million Gallons)',
       'Water Consumption Volume (Million Gallons)',
       'Water Withdrawal Intensity Rate (Gallons / MWh)',
       'Water Consumption Intensity Rate (Gallons / MWh)',
       'Water Withdrawal Rate per Fuel Consumption (Gallons / MMBTU)',
       'Water Consumption Rate per Fuel Consumption (Gallons / MMBTU)',
       'Cooling Unit Hours in Service',
       'Average Distance of Water Intake Below Water Surface (Feet)',
       '860 Cooling Type 1', '860 Cooling Type 2', '923 Cooling Type',
       'Cooling System Type', 'Water Type', 'Water Source',
       'Water Source Name', 'Water Discharge Name', 'Generator Status',
       'Generator Inservice Month', 'Generator Inservice Year',
       'Generator Retirement Month', 'Generator Retirement Year',
       'Boiler Status', 'Boiler Inservice Month', 'Boiler Inservice Year',
       'Boiler Retirement Month', 'Boiler Retirement Year', 'Cooling Status',
       'Cooling Inservice Month', 'Cooling Inservice Year', '\n', '\n.1',
       'Combined Heat and Power Generator?',
       'Generator Primary Energy Source Code', 'Generator Prime Mover Code',
       'Generator Duct Burners?', 'Sector', 'Steam Plant Type',
       'Number Operable Generators', 'Number Operable Boilers',
       'Number Operable Cooling Systems', 'Relationship Type']

    df = pd.read_csv(data_file)

    drop_columns = ['Year', 'Month','Cooling ID',
        'Plant Name',
        'Boiler ID',
        'Cooling Unit Hours in Service',
        'Water Withdrawal Intensity Rate (Gallons / MWh)',
        'Water Consumption Intensity Rate (Gallons / MWh)',
        'Water Withdrawal Rate per Fuel Consumption (Gallons / MMBTU)',
        'Water Consumption Rate per Fuel Consumption (Gallons / MMBTU)',
        'Average Distance of Water Intake Below Water Surface (Feet)',
        '860 Cooling Type 1', '860 Cooling Type 2', '923 Cooling Type',
        'Cooling System Type', 'Water Type', 'Water Source',
        'Water Source Name', 'Water Discharge Name', 'Generator Status',
        'Generator Inservice Month', 'Generator Inservice Year',
        'Generator Retirement Month', 'Generator Retirement Year',
        'Boiler Status', 'Boiler Inservice Month', 'Boiler Inservice Year',
        'Boiler Retirement Month', 'Boiler Retirement Year', 'Cooling Status',
        'Cooling Inservice Month', 'Cooling Inservice Year', '\n', '\n.1',
        'Combined Heat and Power Generator?',
        'Generator Primary Energy Source Code', 'Generator Prime Mover Code',
        'Generator Duct Burners?', 'Steam Plant Type',
        'Number Operable Generators', 'Number Operable Boilers',
        'Number Operable Cooling Systems', 'Relationship Type']
    df.drop(axis=1, columns=drop_columns, inplace=True)
   
    df.rename(columns={
        'Utility ID':'utility_id',
        'State':'state',
        'Plant Code':'plant_code',
        'Generator ID':'generator_id', 
        'Generator Primary Technology':'generator_type',
        'Summer Capacity of Steam Turbines (MW)':'steam_turbine_capacity_MW',
        'Gross Generation from Steam Turbines (MWh)':'gross_gen_steam_turbine_MWh',
        'Net Generation from Steam Turbines (MWh)':'net_gen_steam_turbine_MWh',
        'Summer Capacity Associated with Single Shaft Combined Cycle Units (MW)':'sscc_capacity_MW',
        'Gross Generation Associated with Single Shaft Combined Cycle Units (MWh)':'gross_gen_sscc_MWh',
        'Net Generation Associated with Single Shaft Combined Cycle Units (MWh)':'net_gen_sscc_MWh',
        'Summer Capacity Associated with Combined Cycle Gas Turbines (MW)':'ccgt_capacity_MW',
        'Gross Generation Associated with Combined Cycle Gas Turbines (MWh)':'gross_gen_ccgt_MWh',
        'Net Generation Associated with Combined Cycle Gas Turbines (MWh)':'net_gen_ccgt_MWh',
        'Fuel Consumption from All Fuel Types (MMBTU)':'fuel_cons_total_MMBTU',
        'Fuel Consumption from Steam Turbines (MMBTU)':'fuel_cons_steam_turbines_MMBTU',
        'Fuel Consumption from Single Shaft Combined Cycle Units (MMBTU)':'fuel_cons_sscc_MMBTU',
        'Fuel Consumption from Combined Cycle Gas Turbines (MMBTU)':'fuel_cons_ccgt_MMBTU',
        'Coal Consumption (MMBTU)':'fuel_cons_coal_MMBTU', 
        'Natural Gas Consumption (MMBTU)':'fuel_cons_ng_MMBTU',
        'Petroleum Consumption (MMBTU)':'fuel_cons_petroleum_MMBTU', 
        'Biomass Consumption (MMBTU)':'fuel_cons_biogass_MMBTU',
        'Other Gas Consumption (MMBTU)':'fuel_cons_othergas_MMBTU', 
        'Other Fuel Consumption (MMBTU)':'fuel_cons_otherfuel_MMBTU',
        'Water Withdrawal Volume (Million Gallons)':'water_withdrawal_MGal',
        'Water Consumption Volume (Million Gallons)':'water_consumption_MGal'}, 
        inplace=True)
    
    df.fillna(value=0, axis=0, inplace=True)

    df = df.astype({
            'utility_id':'int64', 
            'state':'object', 
            'plant_code':'int64', 
            'generator_id':'object', 
            'generator_type':'object',
            'steam_turbine_capacity_MW':'float64', 
            'gross_gen_steam_turbine_MWh':'float64',
            'net_gen_steam_turbine_MWh':'float64', 
            'sscc_capacity_MW':'float64', 
            'gross_gen_sscc_MWh':'float64',       
            'net_gen_sscc_MWh':'float64', 
            'ccgt_capacity_MW':'float64', 
            'gross_gen_ccgt_MWh':'float64',
            'net_gen_ccgt_MWh':'float64', 
            'fuel_cons_total_MMBTU':'float64',
            'fuel_cons_steam_turbines_MMBTU':'float64', 
            'fuel_cons_sscc_MMBTU':'float64',
            'fuel_cons_ccgt_MMBTU':'float64', 
            'fuel_cons_coal_MMBTU':'float64', 
            'fuel_cons_ng_MMBTU':'float64',
            'fuel_cons_petroleum_MMBTU':'float64', 
            'fuel_cons_biogass_MMBTU':'float64',
            'fuel_cons_othergas_MMBTU':'float64', 
            'fuel_cons_otherfuel_MMBTU':'float64',
            'water_withdrawal_MGal':'float64', 
            'water_consumption_MGal':'float64', 
            'Sector':'object'})

    df['total_gen_MWh'] = df.net_gen_steam_turbine_MWh + df.net_gen_sscc_MWh + df.net_gen_ccgt_MWh

    df = df[df['total_gen_MWh'] != 0]

    # Aggregate 
    agg_df = df.groupby(['utility_id', 'plant_code', 
                         'state', 'generator_id', 'generator_type', 'Sector']).agg({'sum'})

    agg_df.columns = agg_df.columns.droplevel(1)
    return agg_df


def calc_water_for_energy():
    df = clean_eia_cooling_detail()

    # Convert water consumption to L
    df['water_consumption_ML'] = df.water_consumption_MGal * 3.78541
    
    # Calculate intensity values
    df['water_consumption_intensity_L_per_kWh'] = df.water_consumption_ML * 10**3 / df.total_gen_MWh
    df['water_consumption_rate_per_fuel_cons_L_per_kWh'] = df.water_consumption_ML * 10**6 / convert_MMBTU_to_kWh(df.fuel_cons_total_MMBTU)

    df = df[df['water_consumption_intensity_L_per_kWh'] >= 0]
    
    df.reset_index(inplace=True)
    return df

def plot_w4e():
    df = calc_water_for_energy()

    
    for generator in df.generator_type.unique():
        plt.close()
        print(generator)
        subset = df[df['generator_type'] == generator]
        sns.displot(data=subset, x='water_consumption_intensity_L_per_kWh')
        plt.show()

def boxplot_w4e():
    df = calc_water_for_energy()

    ax = sns.boxplot(y='generator_type', 
        x='water_consumption_intensity_L_per_kWh', 
        data=df, 
        linewidth=2.5,
        orient='h',
        showfliers=False)

    plt.show()

def water_for_energy_statistics():
    df = calc_water_for_energy()

    df = df[['generator_type', 'water_consumption_intensity_L_per_kWh']].copy()
    
    stats = df.groupby(['generator_type']).agg({'mean', 'std', 'count'})

    stats.to_csv(r'data\water_for_energy_eia_generators.csv')

water_for_energy_statistics()
# To Do:
# Look Up Emission Factors for each fuel type