####################################
# LIBRARIES NEEDED TO RUN THE TOOL #
####################################
from logging import error
from msilib.schema import Error
from tkinter import Label
from turtle import width
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
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns  # To install: pip install seaborn

from sysClasses import *

##########################################################################

"""
PLOTTING FUNCTIONS
==================

"""
save_path = r'model_outputs\AbsorptionChillers\Figures'


def coolingdemand_heatmap():
    '''
    This function generates a heat map of each building's cooling demand for each climate zone
    '''

    df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\AbsorptionChiller\water_for_cooling_NERC.csv')

    building_rename = {'primary_school': 'Primary School',
                       'secondary_school': 'Secondary School',
                       'hospital': 'Hospital',
                       'outpatient_healthcare': 'Outpatient Healthcare',
                       'large_hotel': 'Large Hotel',
                       'small_hotel': 'Small Hotel',
                       'warehouse': 'Warehouse',
                       'midrise_apartment': 'Midrise Apartment',
                       'large_office': 'Large Office',
                       'medium_office': 'Medium Office',
                       'small_office': 'Small Office',
                       'full_service_restaurant': 'Full Service Restaurant',
                       'quick_service_restaurant': 'Quick Serice Restaurant',
                       'stand_alone_retail': 'Stand-alone Retail',
                       'strip_mall': 'Strip Mall',
                       'supermarket': 'Supermarket'}

    df['building'] = df['building'].apply(lambda x: building_rename[x])

    custom_order = ['Primary School', 'Secondary School',
                    'Hospital', 'Outpatient Healthcare',
                    'Large Hotel', 'Small Hotel',
                    'Warehouse',
                    'Midrise Apartment',
                    'Large Office', 'Medium Office', 'Small Office',
                    'Full Service Restaurant', 'Quick Serice Restaurant',
                    'Stand-alone Retail', 'Strip Mall', 'Supermarket']

    pivot_df = df.pivot(index='building',
                        columns='climate_zone',
                        values='CoolingDemand_intensity_kWh/sqm')

    pivot_df.index = pd.CategoricalIndex(
        pivot_df.index, categories=custom_order)
    pivot_df.sort_index(level=0, inplace=True)

    # grid_kws = {'height_ratios':(0.9,0.05), 'hspace': 0.3}
    grid_kws = {'width_ratios': (0.95, 0.05), 'wspace': 0.001}
    f, (ax, cbar_ax) = plt.subplots(
        1, 2, gridspec_kw=grid_kws, figsize=(13, 10))

    ax = sns.heatmap(pivot_df,
                     vmin=0, vmax=4000,  
                     ax=ax, cbar_ax=cbar_ax,
                     cbar_kws={'orientation': 'vertical'},
                     cmap='mako',
                     square=True,
                     norm=LogNorm(),
                     )

    cbar_ax.yaxis.set_tick_params(which='both', width=1.5, labelsize=12)

    ax.set_xlabel('ASHRAE Climate Zone', fontsize=16)
    ax.set_ylabel('Building', fontsize=16)
    ax.tick_params(axis='both', width=1.5, labelsize=12)

    sns.set_context('paper')

    filename = r'CoolingDemand_HeatMap.png'
    plt.savefig(F'{save_path}\\{filename}', dpi=300)

    plt.show()


def water_consumption_intensity(how='NERC'):
    ABC_file_path = r'model_outputs\AbsorptionChillers\water_consumption\AbsorptionChiller'
    WCC_file_path = r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller'
    ACC_file_path = r'model_outputs\AbsorptionChillers\water_consumption'
    
    ABC_df = pd.read_csv(F'{ABC_file_path}\water_for_cooling_{how}.csv')
    WCC_df = pd.read_csv(F'{WCC_file_path}\water_for_cooling_{how}.csv')
    ACC_df = pd.read_csv(F'{ACC_file_path}\water_for_cooling_baseline_{how}.csv')

    dataframes = [ABC_df, WCC_df, ACC_df]
    for df in dataframes:
        try:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        except KeyError:
            pass
        df.set_index(['city', 'building'], drop=True, inplace=True)

    data = pd.concat(dataframes, axis=0)
    data.reset_index(inplace=True)
    
    if how == 'NERC':
        sns.boxplot(x=data['WaterConsumption_intensity L_per_kWh_sqm'],
                    y=data['chiller_type'],
                    hue=data['climate_zone'])

    elif how == 'fuel_type':
        sns.violinplot(x=data['WaterConsumption_intensity L_per_kWh_sqm'],
                    y=data['chiller_type'],
                    hue=data['fuel_type'])
    
    plt.show()

water_consumption_intensity(how='fuel_type')

def plot_water_cons_v_energy_dem():

    ABC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\AbsorptionChiller\water_for_cooling_NERC.csv')
    WCC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller\water_for_cooling_NERC.csv')
    ACC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\water_for_cooling_baseline_NERC.csv')

    # print(df)
    # Absorption Chiller
    sns.scatterplot(
        x=ABC_df['CoolingDemand_intensity_kWh/sqm'],
        y=ABC_df['WaterConsumption_intensity_L/kWh'])

    # Water Cooled Chiller
    sns.scatterplot(
        x=WCC_df['CoolingDemand_intensity_kWh/sqm'],
        y=WCC_df['WaterConsumption_intensity_L/kWh'])

    # Air Cooled Chiller
    sns.scatterplot(
        x=ACC_df['CoolingDemand_intensity_kWh/sqm'],
        y=ACC_df['WaterConsumption_intensity_L/kWh'])

    plt.legend(['Absorption Chiller',
                'Water Cooled Chiller',
                'Air Cooled Chiller'])
    plt.show()


def plot_percent_change():

    ABC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\AbsorptionChiller\water_for_cooling_NERC.csv')
    WCC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller\water_for_cooling_NERC.csv')
    ACC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\water_for_cooling_baseline_NERC.csv')

    # print(df)
    # Absorption Chiller
    sns.scatterplot(
        x=ABC_df['CoolingDemand_intensity_kWh/sqm'],
        y=ABC_df['WaterConsumption_intensity_L/kWh'])

    # Water Cooled Chiller
    sns.scatterplot(
        x=WCC_df['CoolingDemand_intensity_kWh/sqm'],
        y=WCC_df['WaterConsumption_intensity_L/kWh'])

    # Air Cooled Chiller
    sns.scatterplot(
        x=ACC_df['CoolingDemand_intensity_kWh/sqm'],
        y=ACC_df['WaterConsumption_intensity_L/kWh'])

    plt.legend(['Absorption Chiller',
                'Water Cooled Chiller',
                'Air Cooled Chiller'])
    plt.show()


def plot_bar_climate_zone():
    ABC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\AbsorptionChiller\water_for_cooling_NERC.csv')
    WCC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller\water_for_cooling_NERC.csv')
    ACC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\water_for_cooling_baseline_NERC.csv')

    # print(df)
    # Absorption Chiller
    sns.scatterplot(
        x=ABC_df['CoolingDemand_intensity_kWh/sqm'],
        y=ABC_df['WaterConsumption_intensity_L/kWh'])

    # Water Cooled Chiller
    sns.scatterplot(
        x=WCC_df['CoolingDemand_intensity_kWh/sqm'],
        y=WCC_df['WaterConsumption_intensity_L/kWh'])

    # Air Cooled Chiller
    sns.scatterplot(
        x=ACC_df['CoolingDemand_intensity_kWh/sqm'],
        y=ACC_df['WaterConsumption_intensity_L/kWh'])

    plt.legend(['Absorption Chiller',
                'Water Cooled Chiller',
                'Air Cooled Chiller'])
    plt.show()


def plot_distributions():
    ABC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\AbsorptionChiller\water_for_cooling_NERC.csv')
    WCC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller\water_for_cooling_NERC.csv')
    ACC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\water_for_cooling_baseline_NERC.csv')

    dataframes = [ABC_df, WCC_df, ACC_df]
    for df in dataframes:
        try:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        except KeyError:
            pass
        df.set_index(['city', 'building'], drop=True, inplace=True)

    data = pd.concat(dataframes, axis=0)
    data.reset_index(inplace=True)

    sns.boxplot(x=data['WaterConsumption_intensity L_per_kWh_sqm'],
                y=data['chiller_type'],
                hue=data['climate_zone'])

    plt.show()
    # print(df)
    # Absorption Chiller
    # sns.scatterplot(x=ABC_df['CoolingDemand_intensity_kWh/sqm'], y=ABC_df['WaterConsumption_intensity_L/kWh'])

    # # Water Cooled Chiller
    # sns.scatterplot(x=WCC_df['CoolingDemand_intensity_kWh/sqm'], y=WCC_df['WaterConsumption_intensity_L/kWh'])

    # # Air Cooled Chiller
    # sns.scatterplot(x=ACC_df['CoolingDemand_intensity_kWh/sqm'], y=ACC_df['WaterConsumption_intensity_L/kWh'])

    # plt.legend(['Absorption Chiller', 'Water Cooled Chiller', 'Air Cooled Chiller'])
    # plt.show()


def plot_percentages():
    ABC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\AbsorptionChiller\water_for_cooling_NERC.csv')
    WCC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller\water_for_cooling_NERC.csv')
    ACC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\water_for_cooling_baseline_NERC.csv')

    dataframes = [ABC_df, WCC_df, ACC_df]
    for df in dataframes:
        try:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        except KeyError:
            pass
        df.set_index(['city', 'building'], drop=True, inplace=True)

    ABC_change = (ABC_df['annual_water_consumption_L'] -
                  ACC_df['annual_water_consumption_L']) / ACC_df['annual_water_consumption_L'] * 100
    WCC_change = (WCC_df['annual_water_consumption_L'] -
                  ACC_df['annual_water_consumption_L']) / ACC_df['annual_water_consumption_L'] * 100

    sns.scatterplot(
        x=ABC_df['CoolingDemand_intensity_kWh/sqm'],
        y=ABC_change,
        hue=ABC_df['climate_zone'],
        markers=['+'])
    sns.scatterplot(
        x=ABC_df['CoolingDemand_intensity_kWh/sqm'],
        y=WCC_change,
        hue=WCC_df['climate_zone'],
        markers=['o'])

    plt.xlabel('CoolingDemand_intensity_kWh/sqm')
    plt.ylabel('Difference in Annual Water Consumption, %')

    # plt.legend(['Absorption Chiller', 'Water Cooled Chiller'])
    plt.show()
