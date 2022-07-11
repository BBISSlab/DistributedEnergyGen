####################################
# LIBRARIES NEEDED TO RUN THE TOOL #
####################################
from cProfile import label
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
from matplotlib import rcParams
import seaborn as sns
from sympy import Predicate, inverse_laplace_transform  # To install: pip install seaborn

from sysClasses import *

##########################################################################

"""
PLOTTING FUNCTIONS
==================

"""
# TODO
# 


save_path = r'model_outputs\AbsorptionChillers\Figures'


def FigS1_coolingdemand_heatmap():
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

    grid_kws = {'height_ratios':(0.03,0.95), 'hspace': 0.05}
    # grid_kws = {'width_ratios': (0.95, 0.05), 'wspace': 0.001}
    f, (cbar_ax, ax) = plt.subplots(
        2, 1, gridspec_kw=grid_kws, figsize=(13, 10))

    ax = sns.heatmap(pivot_df,
                     vmin=0, vmax=1000,  
                     ax=ax,
                     cbar_ax=cbar_ax,
                     cbar_kws= {'orientation': 'horizontal',
                               'ticks':mtick.LogLocator(),
                               'extend':'max'
                               },
                     cmap='coolwarm_r',
                     square=True,
                     norm=LogNorm(),
                     )

    cbar_ax.xaxis.set_tick_params(which='both', width=1.5, labelsize=14, 
                    bottom=False, labelbottom=False, 
                    top=True, labeltop=True)

    cbar_ax.set_title('Cooling Demand Intensity, $kWh/m^2$', fontsize=16)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=14)
    ax.set_xlabel('IECC Climate Zone', fontsize=18)

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    ax.set_ylabel('Building', fontsize=18)
    ax.tick_params(axis='both', width=1.5, labelsize=14)

    sns.set_context('paper')

    filename = r'CoolingDemand_HeatMap.png'
    plt.savefig(F'{save_path}\\{filename}', dpi=300)

    plt.show()


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

    ACC_df = pd.read_csv(F'{ACC_filepath}\{baseline_filename}', index_col=0)
    WCC_df = pd.read_csv(F'{WCC_filepath}\{chiller_filename}', index_col=0)
    ABC_df = pd.read_csv(F'{ABC_filepath}\{chiller_filename}', index_col=0)

    # Concatenate data
    df = pd.concat([ACC_df, WCC_df, ABC_df], axis=0)
    df.reset_index(inplace=True, drop=True)
    df.fillna(0, inplace=True)

    return df


def concatenate_all_data(how='NERC'):
    
    df_S2 = concatenate_dataframes(how, scope=2)
    df_S3 = concatenate_dataframes(how, scope=3)

    df_S2['scope'] = 'PoG'
    df_S3['scope'] = 'LC'

    # Rename columns
    df_S2.rename(columns={'PoG_w4e_intensity_factor_(L/kWh)':'w4e_int_factor_(L/kWhe)',	
                        'PoG_annual_water_consumption_L':'annual_water_consumption_L',		
                        'PoG_WaterConsumption_intensity_L/kWh':'WaterConsumption_int_(L/kWhr)',	
                        'PoG_WaterConsumption_intensity_L/kWh_sqm':'WaterConsumption_int_(L/kWhr_sqm)'},
                        inplace=True)

    
    df_S3.rename(columns={'Total_w4e_intensity_factor_(L/kWh)':'w4e_int_factor_(L/kWhe)',	
                        'Total_annual_water_consumption_L':'annual_water_consumption_L',		
                        'Total_WaterConsumption_intensity_L/kWh':'WaterConsumption_int_(L/kWhr)',	
                        'Total_WaterConsumption_intensity_L/kWh_sqm':'WaterConsumption_int_(L/kWhr_sqm)'},
                        inplace=True)

    
    df = pd.concat([df_S2, df_S3], axis=0).reset_index(drop=True)

    # Rename chillers


    return df


def Fig3_peak_electricity_reduction():
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates
    
    df = pd.read_csv(r'model_outputs\AbsorptionChillers\peak_electricity_reduction.csv')
    
    df['datetime'] = pd.to_datetime(df['datetime'])

    ACC_df = df[['datetime', 'Electricity_ACC_kW_m^-2']].copy()
    WCC_df = df[['datetime', 'Electricity_WCC_kW_m^-2']].copy()
    ABC_df = df[['datetime', 'Electricity_ABC_kW_m^-2']].copy()

    ACC_df.rename(columns={'Electricity_ACC_kW_m^-2':'Electricity_kW_m^-2'}, inplace=True)
    WCC_df.rename(columns={'Electricity_WCC_kW_m^-2':'Electricity_kW_m^-2'}, inplace=True)
    ABC_df.rename(columns={'Electricity_ABC_kW_m^-2':'Electricity_kW_m^-2'}, inplace=True)

    ACC_df['chiller_type'] = 'ACC'
    WCC_df['chiller_type'] = 'WCC'
    ABC_df['chiller_type'] = 'ABC'

    df = pd.concat([ACC_df, WCC_df, ABC_df], axis=0).reset_index(drop=True)

    fig, ax = plt.subplots(1,1, figsize=(8, 4))
    sns.set_context()

    colors = ['dimgrey', 'deepskyblue', 'coral']

    sns.lineplot(x=df['datetime'], y=df['Electricity_kW_m^-2'] * 1000, 
                palette=colors, hue=df['chiller_type'], style=df['chiller_type'])


    ##############
    # Formatting #
    ##############

    plt.legend(['ACC', 'WCC', 'ABC'],
                loc='upper center', ncol=3,
                bbox_to_anchor=(0.5, 1.1),
                frameon=False,
                fontsize=14)

    fig.text(0.03, 0.5, 'Peak electricity for cooling, $W_e/m^2$',
            va='center', rotation='vertical', fontsize=16)
    ax.set_ylabel('')
    ax.set_yticks(np.arange(0, 30, 10))
    ax.set_yticklabels(ax.get_yticks(), fontsize=14)
    ax.set_ylim(0, 20)
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))

    # Setting x ticks to each month
    date_form = DateFormatter('%b')
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J',
    #                     'J', 'A', 'S', 'O', 'N', 'D', 'J'])
    plt.xticks(fontsize=16)
    
    ax.set_xlabel('', fontsize=14)
    ax.set_xlim(18262, 18628)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)

    ax.tick_params('both', width=2, length=8, which='major')
    ax.tick_params('both', width=1, length=5, which='minor')

    sns.despine()

    plt.savefig(F'{save_path}\\peak_electricity_demands.png', dpi=300)
    
    plt.show()


def Fig4_water_consumption_intensity_NERC():
    df = concatenate_all_data("NERC")

    # Separate dataframes for subplots
    first_set = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                'MROE', 'MROW', 'NEWE', 'NWPP',
                'NYCW', 'NYLI', 'NYUP',] 
    second_set = ['RFCE', 'RFCM', 'RFCW', 'RMPA', 
                'SPNO', 'SPSO', 'SRMV', 'SRMW', 
                'SRSO', 'SRTV', 'SRVC']

    df_1 = df[df['eGRID_subregion'].isin(first_set)].copy()
    df_2 = df[df['eGRID_subregion'].isin(second_set)].copy()

    ordered_NERC = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                        'MROE', 'MROW', 'NEWE', 'NWPP',
                        'NYCW', 'NYLI', 'NYUP', 'RFCE',
                        'RFCM', 'RFCW', 'RMPA', 'SPNO',
                        'SPSO', 'SRMV', 'SRMW', 'SRSO',
                        'SRTV', 'SRVC']
    
    for df in [df_1, df_2]:
        df['eGRID_subregion'] = pd.Categorical(df['eGRID_subregion'], categories=ordered_NERC, ordered=True)
        df.sort_values(by='eGRID_subregion')

        df['simulation'] = df[['chiller_type', 'scope']].apply(' - '.join, axis=1)

        # Adjust consumption values        
        Y = 'WaterConsumption_int_(L/MWhr_sqm)' 

        df[Y] = df['WaterConsumption_int_(L/kWhr_sqm)'] * 1000

    X_1 = df_1['eGRID_subregion']
    X_2 = df_2['eGRID_subregion']

    x_label = 'eGRID subregion'
    x_1_labels = first_set
    x_2_labels = second_set

    # Close any previous plots
    plt.close()
    
    grid_kws = {'height_ratios':(0.1, 0.4, 0.1, 0.4)}

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,figsize=(10,4), sharex=False,
                                    gridspec_kw=grid_kws)
    
    plt.subplots_adjust(hspace=0.5)
    
    # Format fonts and style
    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    from numpy import median
    colors = ['white', 'deepskyblue', 'salmon', 'white', 'deepskyblue', 'salmon']
    
    ########################################
    # First Plot, upper bound of first set #
    ########################################

    sns.barplot(x=X_1, 
                y=Y, hue='simulation', 
                data=df_1,
                ax=ax1,
                # estimator=median,
                ci=95,
                palette=colors, 
                edgecolor='0.1',
                linewidth=1.5,
                capsize=0.05,
                )

    hatches = ['', '', '','////', '////', '////']
    
    for bars, hatch in zip(ax1.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    
    ##############
    # Formatting #
    ############## 
    # X-axis
    ax1.set_xlabel('')
    ax1.set_xticks(np.arange(0,12,1))
    ax1.set_xticklabels('')
    ax1.set_xlim(-0.5, 10.5)

    ax1.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax1.set_ylabel('')
    ax1.set_yticks(np.arange(10, 70, 20))
    ax1.set_ylim(np.min(10), np.max(50))
    ax1.set_yticklabels(['', 30, 50],fontsize=14)
    ax1.yaxis.set_minor_locator(mtick.MultipleLocator(10))

    ax1.get_legend().remove()

    ########################################
    # Second plot, first set, bottom layer #
    ########################################
    sns.barplot(x=X_1, 
                y=Y, hue='simulation', 
                data=df_1,
                ax=ax2,
                # estimator=median,
                ci=95,
                palette=colors, 
                edgecolor='0.1',
                linewidth=1.5,
                capsize=0.05,
                )

    hatches = ['', '', '','////', '////', '////']
    
    for bars, hatch in zip(ax2.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    
    ##############
    # Formatting #
    ############## 
    # X-axis
    ax2.set_xlabel('')
    ax2.set_xticks(np.arange(0,11,1))
    ax2.set_xticklabels(first_set, fontsize=14)
    ax2.set_xlim(-0.5, 10.5)

    ax2.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax2.set_ylabel('')
    ax2.set_yticks(np.arange(0, 12, 2))
    ax2.set_ylim(0, 10)
    ax2.set_yticklabels(np.arange(0, 12, 2),fontsize=14)
    ax2.yaxis.set_minor_locator(mtick.MultipleLocator(0.5))

    ax2.get_legend().remove()

    # Move bottom of ax1 to meet ax2
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    
    points1 = pos1.get_points()
    points2 = pos2.get_points()

    points1[0][1] = points2[1][1] * (1.01)
    
    pos1.set_points(points1)

    ax1.set_position(pos1)

    ###################################
    # Third Plot, second set top half #
    ###################################
    sns.barplot(x=X_2, 
                y=Y, hue='simulation', 
                data=df_2,
                ax=ax3,
                # estimator=median,
                ci=95,
                palette=colors, 
                edgecolor='0.1',
                linewidth=1.5,
                capsize=0.05,
                )
    
    for bars, hatch in zip(ax3.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    
    ##############
    # Formatting #
    ############## 
    # X-axis
    ax3.set_xticks(np.arange(10,22,1))
    ax3.set_xlim(10.5, 21.5)
    ax3.set_xlabel('')
    ax3.set_xticklabels('')
    # ax3.set_xticklabels(x_2_labels, ha='center', fontsize=14)
    ax3.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax3.set_ylabel('')
    ax3.set_yticks(np.arange(5,20,5))
    ax3.set_ylim(5, 15)
    ax3.set_yticklabels(['', 10, 15],fontsize=14)
    ax3.yaxis.set_minor_locator(mtick.MultipleLocator(1))


    ax3.get_legend().remove()
    sns.despine()

    ####################################
    # Fourth Plot, second set bot half #
    ####################################
    # ax_2 = fig.add_subplot(212)
    # plt.subplot(212)

    sns.barplot(x=X_2, 
                y=Y, hue='simulation', 
                data=df_2,
                ax=ax4,
                # estimator=median,
                ci=95,
                palette=colors, 
                edgecolor='0.1',
                linewidth=1.5,
                capsize=0.05,
                )
    
    for bars, hatch in zip(ax4.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    
    ##############
    # Formatting #
    ############## 
    # X-axis
    ax4.set_xticks(np.arange(11,22,1))
    ax4.set_xlim(10.5, 21.5)
    ax4.set_xlabel('')
    ax4.set_xticklabels(x_2_labels, ha='center', fontsize=14)
    ax4.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax4.set_ylabel('')
    ax4.set_yticks(np.arange(0,6,1))
    ax4.set_ylim(0, 5)
    ax4.set_yticklabels(np.arange(0,6,1),fontsize=14)
    ax4.yaxis.set_minor_locator(mtick.MultipleLocator(0.5))

    ax4.get_legend().remove()

    # Move bottom of ax3 to meet ax4
    pos3 = ax3.get_position()
    pos4 = ax4.get_position()
    
    points3 = pos3.get_points()
    points4 = pos4.get_points()
    
    points3[0][1] = points4[1][1] * (1.015)
    
    pos3.set_points(points3)

    ax3.set_position(pos3)

    sns.despine()

    y_label = "Water Consumption Intensity,\n           $L / (MWh_r \cdot m^2)$"
    fig.text(0.5, 0.01, x_label, ha='center', fontsize=18)
    fig.text(0.03, 0.5, y_label, va='center', rotation='vertical', fontsize=18)

    plt.savefig(F'{save_path}\water4cooling_NERC.png', dpi=300)

    plt.show()


def Fig4_alt_map():
    pass

def Fig5_water_consumption_intensity_fuel_type():
    df = concatenate_all_data("fuel_type")

    # Separate dataframes for subplots
    first_set = ['United States Overall',
                    'Ethanol', 
                    'Conventional Oil', 'Unconventional Oil', 
                    'Subbituminous Coal', 'Bituminous Coal', 'Lignite Coal',
                    'Conventional Natural Gas', 'Unconventional Natural Gas', 
                    ]

    second_set = ['Uranium', 
                'Biodiesel', 'Biogas', 'Solid Biomass and RDF', 
                'Geothermal', 'Hydropower', 
                'Solar Photovoltaic', 'Solar Thermal', 'Wind']

    df_1 = df[df['fuel_type'].isin(first_set)].copy()
    df_2 = df[df['fuel_type'].isin(second_set)].copy()

    ordered_fuel = ['United States Overall',
                    'Ethanol', 
                    'Conventional Oil', 'Unconventional Oil', 
                    'Subbituminous Coal', 'Bituminous Coal', 'Lignite Coal',
                    'Conventional Natural Gas', 'Unconventional Natural Gas', 
                    'Uranium', 
                    'Biodiesel', 'Biogas', 'Solid Biomass and RDF', 
                    'Geothermal', 'Hydropower', 'Solar Photovoltaic', 'Solar Thermal', 'Wind'
                    ]
    
    for df in [df_1, df_2]:
        df['fuel_type'] = pd.Categorical(df['fuel_type'], categories=ordered_fuel, ordered=True)
        df.sort_values(by='fuel_type')

        df['simulation'] = df[['chiller_type', 'scope']].apply(' - '.join, axis=1)

        # Adjust consumption values        
        Y = 'WaterConsumption_int_(L/MWhr_sqm)' 

        df[Y] = df['WaterConsumption_int_(L/kWhr_sqm)'] * 1000

    X_1 = df_1['fuel_type']
    X_2 = df_2['fuel_type']

    x_label = 'Fuel type'
    x_1_labels = ['US', # '\n\n' 
                    'E', #'\n\nFossil Fuels',
                    'CO', 'UO', 'SC', 'BC', 'LC', 'CNG', 'UNG']
    x_2_labels = ['U', #'\n\n'
                    'BD', #'\n\nRenewables', 
                    'BG', 'BM', 
                    'GT','H', 'SPV', 'STh', 'W'
                    ]

    # Close any previous plots
    plt.close()

    fig, ax = plt.subplots(2, 1,figsize=(10,4), sharex=False)
    plt.subplots_adjust(hspace=0.15)
    
    # Format fonts and style
    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    from numpy import median
    colors = ['white', 'deepskyblue', 'salmon', 'white', 'deepskyblue', 'salmon']
    
    y_ticks = np.arange(0, 5, 1)

    ###################################
    # First Plot, US and fossil fuels #
    ###################################
    plt.subplot(211)
    # ax_1 = fig.add_subplot(211)

    ax_1 = sns.barplot(x=X_1, 
                y=Y, hue='simulation', 
                data=df_1,
                # estimator=median,
                ci=95,
                palette=colors, 
                edgecolor='0.1',
                linewidth=1.5,
                capsize=0.05,
                )

    hatches = ['', '', '','////', '////', '////']
    
    for bars, hatch in zip(ax_1.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    
    ##############
    # Formatting #
    ############## 
    # X-axis
    ax_1.set_xlabel('')
    ax_1.set_xticks(np.arange(0,9,1))
    ax_1.set_xlim(-0.5, 8.5)
    ax_1.set_xticklabels(x_1_labels, ha='center', fontsize=14)
    ax_1.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax_1.set_ylabel('')
    ax_1.set_yticks(y_ticks)
    ax_1.set_ylim(np.min(y_ticks), np.max(y_ticks))
    ax_1.set_yticklabels(y_ticks,fontsize=14)
    ax_1.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))


    # ax_1.legend(title='Chiller - Scope',
    #           loc='upper center', 
    #           ncol=3,
    #           bbox_to_anchor=(0.5, 1.5),
    #           frameon=False)

    ax_1.get_legend().remove()

    ####################################
    # Second Plot, US and fossil fuels #
    ####################################
    # ax_2 = fig.add_subplot(212)
    plt.subplot(212)

    ax_2 = sns.barplot(x=X_2, 
                y=Y, hue='simulation', 
                data=df_2,
                # estimator=median,
                ci=95,
                palette=colors, 
                edgecolor='0.1',
                linewidth=1.5,
                capsize=0.05,
                )
    
    for bars, hatch in zip(ax_2.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    
    ##############
    # Formatting #
    ############## 
    # X-axis
    ax_2.set_xticks(np.arange(9,18,1))
    ax_2.set_xlim(8.5, 17.5)
    ax_2.set_xlabel('')
    ax_2.set_xticklabels(x_2_labels, ha='center', fontsize=14)
    ax_2.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax_2.set_ylabel('')
    ax_2.set_yticks(y_ticks)
    ax_2.set_ylim(np.min(y_ticks), np.max(y_ticks))
    ax_2.set_yticklabels(y_ticks,fontsize=14)
    ax_2.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))


    ax_2.get_legend().remove()
    sns.despine()


    y_label = "Water Consumption Intensity,\n           $L / (MWh_r \cdot m^2)$"
    fig.text(0.5, 0.01, x_label, ha='center', fontsize=18)
    fig.text(0.03, 0.5, y_label, va='center', rotation='vertical', fontsize=18)

    plt.savefig(F'{save_path}\water4cooling_fuel_type.png', dpi=300)

    plt.show()


def calculate_percent_difference(how='NERC', scope=2):
    # Read simulation files
    ACC_filepath = r'model_outputs\AbsorptionChillers\water_consumption'
    WCC_filepath = r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller'
    ABC_filepath = r'model_outputs\AbsorptionChillers\water_consumption\AbsorptionChiller'
    
    if scope == 2:
        prefix = 'PoG'
        column_prefix=prefix
    elif scope == 3:
        prefix = 'LC'
        column_prefix = 'Total'

    if how == 'NERC':
        X = 'eGRID_subregion'
        m_indeces = ['city', 'building', 'eGRID_subregion']
        abc_indeces = m_indeces
    elif how == 'fuel_type':
        X = 'fuel_type'
        m_indeces = ['city', 'building', 'eGRID_subregion', 'fuel_type']
        abc_indeces = m_indeces # .append('chp_id')

    baseline_filename = F'{prefix}_water_for_cooling_baseline_{how}.csv'
    chiller_filename = F'{prefix}_water_for_cooling_{how}.csv'

    ACC_data = pd.read_csv(F'{ACC_filepath}\{baseline_filename}', index_col=0)
    ACC_data.set_index(m_indeces, inplace=True, drop=True)

    WCC_data = pd.read_csv(F'{WCC_filepath}\{chiller_filename}', index_col=0)
    WCC_data.set_index(m_indeces, inplace=True, drop=True)

    ABC_data = pd.read_csv(F'{ABC_filepath}\{chiller_filename}', index_col=0)    
    ABC_data.set_index(abc_indeces, inplace=True, drop=True)

    # ACC_water_consumption = ACC_df[F'{column_prefix}_annual_water_consumption_L']
    # ACC_water_intensity_ref = ACC_df[F'{column_prefix}_WaterConsumption_intensity_L/kWh']
    # ACC_water_consumption_building = ACC_df[F'{column_prefix}_WaterConsumption_intensity L_per_kWh_sqm']


    comparison_metric = F'{column_prefix}_WaterConsumption_intensity_L/kWh'
    ACC_df = ACC_data[comparison_metric]
    WCC_df = WCC_data[comparison_metric]
    ABC_df = ABC_data[comparison_metric]

    df = pd.merge(ABC_df, ACC_df, 
                  left_index=True, right_index=True,
                  suffixes=('_ABC', '_ACC'))

    ACC_df = df[F'{comparison_metric}_ACC']
    ABC_df = df[F'{comparison_metric}_ABC']
    df['Percent Difference'] = (ABC_df - ACC_df) / ACC_df * 100

    df.reset_index(inplace=True)
    ordered_fuel = ['United States Overall',
                        'Ethanol', 
                        'Conventional Oil', 'Unconventional Oil', 
                        'Subbituminous Coal', 'Bituminous Coal', 'Lignite Coal',
                        'Conventional Natural Gas', 'Unconventional Natural Gas', 
                        'Uranium', 
                        'Biodiesel', 'Biogas', 'Solid Biomass and RDF', 
                        'Geothermal', 'Hydropower', 'Solar Photovoltaic', 'Solar Thermal', 'Wind'
                        ]
        
    df['fuel_type'] = pd.Categorical(df['fuel_type'], categories=ordered_fuel, ordered=True)
    df.sort_values(by='fuel_type')

    df.to_csv(r'model_outputs\AbsorptionChillers\water_consumption\test.csv')
    X = df['fuel_type']

    x_label = 'Fuel type'
    xtick_labels = ['US', # '\n\n' 
                    'E', #'\n\nFossil Fuels',
                    'CO', 'UO', 'SC', 'BC', 'LC', 'CNG', 'UNG', 
                    'U', #'\n\n'
                    'BD', #'\n\nRenewables', 
                    'BG', 'BioM', 
                    'GeoTh','Hydro', 'S.PV', 'S.Th', 'Wind'
                    ]

    # ABC_difference = (ABC_df - ACC_df) / ACC_df * 100
    # WCC_difference = (WCC_df - ACC_df) / ACC_df * 100

    # ABC_df = ABC_difference.reset_index().copy()
    # ABC_df['chiller_type'] = 'ABC'

    # WCC_df = WCC_difference.reset_index().copy()
    # WCC_df['chiller_type'] = 'WCC'

    # df = pd.concat([ABC_df, WCC_df], axis=0).reset_index(drop=True)

    # df.rename(columns={comparison_metric:'Percent Difference'}, inplace=True)

    
    plt.close()

    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)

    sns.barplot(x=X, y='Percent Difference', data=df)
    
    ax.set_xticklabels(xtick_labels)
    plt.show()
    


def water_consumption_intensity(how='NERC', scope=2):
    '''
    Inputs:
        how: "NERC" or "fuel_type"
        scope: 2 or 3
    '''
    try:
        df = concatenate_dataframes(how, scope)
    except FileNotFoundError:
        df = concatenate_dataframes('NERC', scope)

    if how == 'NERC':
        ordered_NERC = ['AZNM', 'CAMX', 'ERCT', 'FRCC', 'MROW', 
                        'NWPP', 'RFCE', 'RFCW', 'RMPA', 'SRSO']

        df['eGRID_subregion'] = pd.Categorical(df['eGRID_subregion'], categories=ordered_NERC, ordered=True)
        df.sort_values(by='eGRID_subregion')

        X = df['eGRID_subregion']

        x_label = 'eGRID subregion'
        xtick_labels = ordered_NERC

        y_ticks = np.arange(0, 9, 1)

    elif how == 'climate_zone':
        ordered_CZ = ['1A', '2A', '2B',
                    '3A', '3B', '3B-CA', '3C',
                    '4A', '4B', '4C',
                    '5A', '5B', '6A', '6B', '7']

        df['climate_zone'] = pd.Categorical(df['climate_zone'], categories=ordered_CZ, ordered=True)
        df.sort_values(by='climate_zone')

        X = df['climate_zone']

        x_label = 'Climate zone'
        xtick_labels = ordered_CZ

        y_ticks = np.arange(0, 9, 1)

    elif how == 'fuel_type':
        ordered_fuel = ['United States Overall',
                        'Ethanol', 
                        'Conventional Oil', 'Unconventional Oil', 
                        'Subbituminous Coal', 'Bituminous Coal', 'Lignite Coal',
                        'Conventional Natural Gas', 'Unconventional Natural Gas', 
                        'Uranium', 
                        'Biodiesel', 'Biogas', 'Solid Biomass and RDF', 
                        'Geothermal', 'Hydropower', 'Solar Photovoltaic', 'Solar Thermal', 'Wind'
                        ]
        
        df['fuel_type'] = pd.Categorical(df['fuel_type'], categories=ordered_fuel, ordered=True)
        df.sort_values(by='fuel_type')

        X = df['fuel_type']

        x_label = 'Fuel type'
        xtick_labels = ['US', # '\n\n' 
                        'E', #'\n\nFossil Fuels',
                        'CO', 'UO', 'SC', 'BC', 'LC', 'CNG', 'UNG', 
                        'U', #'\n\n'
                        'BD', #'\n\nRenewables', 
                        'BG', 'BioM', 
                        'GeoTh','Hydro', 'S.PV', 'S.Th', 'Wind'
                        ]

        y_ticks = np.arange(0, 5, 1)
    
    if scope == 2:
        Y = 'PoG_WaterConsumption_intensity_L_per_kWh_sqm'
        df[Y] = df[Y] * 10**3
        y_label = "PoG Water Consumption Intensity, $L / (MWh_r*m^2)$"
        
    elif scope == 3:
        Y = 'Total_WaterConsumption_intensity_L_per_kWh_sqm'
        df[Y] = df[Y] * 10**3
        y_label = "Total Water Consumption Intensity, $L / (MWh_r*m^2)$"
    

    # Close any previous plots
    plt.close()


    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)

    # Format fonts and style
    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    from numpy import median
    colors = ['white', 'skyblue', 'salmon']
    
    sns.barplot(x=X, 
                y=Y, hue='chiller_type', 
                data=df,
                # estimator=median,
                ci=95,
                palette=colors, 
                edgecolor='0.2',
                linewidth=1,
                capsize=0.2,
                )

    hatches = ['////', '', '....']
    
    for bars, hatch in zip(ax.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    
    ##############
    # Formatting #
    ############## 
    # X-axis
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_xticklabels(xtick_labels, ha='center', fontsize=14)
    ax.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_yticks(y_ticks)
    ax.set_ylim(np.min(y_ticks), np.max(y_ticks))
    ax.set_yticklabels(y_ticks,fontsize=14)
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))


    # ax.legend(title='Chiller',
    #           loc='upper center', 
    #           ncol=3,
    #           bbox_to_anchor=(0.5, 1.1),
    #           frameon=False)

    ax.get_legend().remove()

    sns.despine()

    plt.show()



def stacked_bar_evaporation_percent(how='NERC', scope=2):
    import matplotlib.patches as mpatches

    df = concatenate_dataframes(how, scope)
    df = df[(df['chiller_type'] == 'AbsorptionChiller') | (df['chiller_type'] == 'WaterCooledChiller')]
    
    if scope == 2:
        df['Total_L/MWh_sqm'] = df['PoG_WaterConsumption_intensity_L/kWh_sqm'] * 1000
    elif scope == 3:
        df['Total_L/MWh_sqm'] = df['Total_WaterConsumption_intensity_L_per_kWh_sqm'] * 1000
    df['Evaporation_L/MWh_sqm'] = df['percent_evaporation'] * df['Total_L/MWh_sqm']
    df['Power_Generation_L/MWh_sqm'] =  df['Total_L/MWh_sqm'] - df['Evaporation_L/MWh_sqm'] *1000

    df = df[['city', 'building', 'eGRID_subregion', 'chiller_type', 'climate_zone', 'Total_L/MWh_sqm', 'Evaporation_L/MWh_sqm', 'Power_Generation_L/MWh_sqm']].copy()

    ordered_NERC = ['AZNM', 'CAMX', 'ERCT', 'FRCC', 'MROW', 
                        'NWPP', 'RFCE', 'RFCW', 'RMPA', 'SRSO']

    df['eGRID_subregion'] = pd.Categorical(df['eGRID_subregion'], categories=ordered_NERC, ordered=True)
    df.sort_values(by='eGRID_subregion')

    ############
    # Plotting #
    ############
    colors = ['skyblue', 'salmon']

    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)

    # Bar Chart 1 -> Evaporation from electricity
    bar1 = sns.barplot(x='eGRID_subregion', 
                       y='Total_L/MWh_sqm', 
                       data=df, 
                       hue='chiller_type',
                       edgecolor='0.2', 
                       linewidth=1, 
                       capsize=0.2,)

    bar2 = sns.barplot(x='eGRID_subregion', 
                       y='Evaporation_L/MWh_sqm', 
                       data=df, 
                       hue='chiller_type',
                       edgecolor='0.2', 
                       ci=None)
    
    hatches = ['', '', '//', '//']

    for bars, hatch in zip(ax.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)

    ax.legend(labels=['WCC', 'ABC', 'WCC - Evaporation', 'ABC - Evaporation'],
              loc='upper center', 
              ncol=4,
              bbox_to_anchor=(0.5, 1.1),
              frameon=False)

    plt.show()
    

# stacked_bar_evaporation_percent(scope=3)

# water_consumption_intensity(how ='NERC', scope=3)

####################
# INCOMPLETE PLOTS #
####################

def Fig4_alt_map():
    '''
    This function only works locally since the shapefiles are too large for GitHub.
    Water for cooling dataframes are kept in Git, maps can be downloaded online.
    '''
    import geopandas as gp
    IECC_path = r'C:\Users\oabro\Dropbox (GaTech)\Datafiles\Climate_Zones_-_DOE_Building_America_Program'
    eGRID_path = r'C:\Users\oabro\Dropbox (GaTech)\Datafiles\egrid2020_subregions'

    plt.close()
    w4r_df = pd.read_csv(r'model_outputs\AbsorptionChillers\water_consumption\NERC_w4r_summary.csv', index_col=0)

    eGRID_subregions = gp.read_file(F'{eGRID_path}\eGRID2020_subregions.shp')
    IECC_climate_zones = gp.read_file(F'{IECC_path}\Climate_Zones_-_DOE_Building_America_Program.shp')

    eGRID_keys = [# 'AKGD', 'AKMS', 
                        'AZNM', 'CAMX', 'ERCT', 'FRCC', 
                        # 'HIMS', 'HIOA', 
                        'MROE', 'MROW', 'NEWE', 'NWPP', 
                        'NYCW', 'NYLI', 'NYUP', 'RFCE', 
                        'RFCM', 'RFCW', 'RMPA', 'SPNO',
                        'SPSO', 'SRMV', 'SRMW', 'SRSO', 'SRTV', 'SRVC', 
                        #'PRMS'
                        ]

    # Remove subregions not included in analysis
    eGRID_subregions = eGRID_subregions[eGRID_subregions['ZipSubregi'].isin(eGRID_keys)]
    eGRID_subregions.reset_index(inplace=True, drop=True)

    IECC_climate_zones = IECC_climate_zones[IECC_climate_zones['IECC_Clima'].isin(np.arange(1,8,1))]

    # IECC_climate_zones = IECC_climate_zones[IECC_climate_zones['IECC_Moist'].isin(['A', 'B', 'C'])]
    IECC_climate_zones.reset_index(inplace=True, drop=True)

    # Mercator Projection
    eGRID_subregions = eGRID_subregions.to_crs('EPSG:3395')
    IECC_climate_zones = IECC_climate_zones.to_crs('EPSG:3395')

    IECC_climate_zones['ClimateZone'] = IECC_climate_zones['IECC_Clima'].astype(str) + IECC_climate_zones['IECC_Moist']

    # Spatial join of IECC_climate_zones and the eGRID_subregions
    merged_maps = IECC_climate_zones.sjoin(eGRID_subregions, how='inner', predicate='intersects')
    merged_maps.reset_index(inplace=True, drop=True)

    # Create new ID column for both dataframes to merge
    merged_maps['eGRID-Climate'] = merged_maps['ZipSubregi'] + '-' + merged_maps['ClimateZone']
    merged_maps.rename(columns={'ZipSubregi':'eGRID_subregion',
                             'ClimateZone':'climate_zone'},
                             inplace=True)

    w4r_df['eGRID-Climate'] = w4r_df['eGRID_subregion'] + '-' + w4r_df['climate_zone']
    
    w4r_map = merged_maps.merge(w4r_df, on='eGRID_subregion') 

    # For Now Limit to LC scope
    w4r_map = w4r_map[(w4r_map['scope'] == 'LC')]

    # merged_maps[['ZipSubregi', 'ClimateZone', 'eGRID-Climate']].to_csv(r'model_outputs\AbsorptionChillers\subregion_climate.csv')
    # IECC_climate_zones.apply(lambda x: ax.annotate(s=F'{x.ClimateZone}',
    #                             xy=x.geometry.centroid.coords[0], ha='center',
    #                             fontsize=14), axis=1)

    ############
    # PLOTTING #
    ############
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 11))

    # # Plot climate zones and regions
    # merged_maps.boundary.plot(color='Black', ax=ax1)
    # IECC_climate_zones.boundary.plot(color='Grey', ax=ax2)
    # eGRID_subregions.plot(cmap='Set3', ax=ax3)

    ############
    # WCC Plot #
    ############
    WCC_map = w4r_map[w4r_map['chiller_type'] == 'WaterCooledChiller']

    WCC_map.plot(ax=ax1, column='percent_diff-mean', 
                cmap='RdYlGn_r', legend=True,
                vmin=-100, vmax=100)

    ############
    # ABC Plot #
    ############
    ABC_map = w4r_map[w4r_map['chiller_type'] == 'AbsorptionChiller']

    ABC_map.plot(ax=ax2, column='percent_diff-mean', 
                cmap='RdYlGn_r', legend=True,
                vmin=-100, vmax=100)

    # print(ABC_df)
    # WCC_df.plot(column='WC_mean')
    
    # # # Set Map Boundaries
    boundaries = eGRID_subregions.bounds

    for ax in [ax1, ax2]:
        ax.set_xlim(boundaries['minx'].min(), boundaries['maxx'].max())
        ax.set_ylim(boundaries['miny'].min(), boundaries['maxy'].max())

    plt.savefig(r'model_outputs\AbsorptionChillers\Figures\percent_diff_map.png', dpi=300)
    plt.show()

Fig4_alt_map()

def plot_water_cons_v_energy_dem():

    ABC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\AbsorptionChiller\PoG_water_for_cooling_NERC.csv')
    WCC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller\PoG_water_for_cooling_NERC.csv')
    ACC_df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\water_consumption\PoG_water_for_cooling_baseline_NERC.csv')

    # print(df)
    # Absorption Chiller
    sns.scatterplot(
        x=ABC_df['CoolingDemand_intensity_kWh/sqm'],
        y=ABC_df['PoG_WaterConsumption_intensity_L/kWh'])

    # Water Cooled Chiller
    sns.scatterplot(
        x=WCC_df['CoolingDemand_intensity_kWh/sqm'],
        y=WCC_df['PoG_WaterConsumption_intensity_L/kWh'])

    # Air Cooled Chiller
    sns.scatterplot(
        x=ACC_df['CoolingDemand_intensity_kWh/sqm'],
        y=ACC_df['PoG_WaterConsumption_intensity_L/kWh'])

    plt.legend(['Absorption Chiller',
                'Water Cooled Chiller',
                'Air Cooled Chiller'])
    plt.show()

# plot_water_cons_v_energy_dem()


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



def plot_electricity(data):
    df = pd.read_feather(data)
    df.set_index('datetime', inplace=True, drop=True)
    df.index = pd.to_datetime(df.index)

    df['ACRC_ElecDemand_kW'] = df.CoolingDemand_kW / 3.4
    # print(df.head())

    sns.lineplot(x=df.index, y=df.AbsCh_ElecDemand_kW, alpha=0.5)
    sns.lineplot(x=df.index, y=df.ACRC_ElecDemand_kW, alpha=0.5)

    plt.legend(['Absorption Chiller', 'Air Cooled Chiller'])

    plt.xlabel('Time')
    plt.ylabel('Electricity Demand, kW')
    file_path = r'model_outputs\AbsorptionChillers\Figures'
    file_name = r'hourly_E_demand.png'
    plt.savefig(F'{file_path}\{file_name}')
    plt.show()



##################################
# VOID FUNCTIONS - FOR REFERENCE #
##################################
def water_consumption_intensity_NERC_VOID():
    df = concatenate_all_data("NERC")

    # Separate dataframes for subplots
    first_set = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                'MROE', 'MROW', 'NEWE', 'NWPP',
                'NYCW', 'NYLI', 'NYUP',] 
    second_set = ['RFCE', 'RFCM', 'RFCW', 'RMPA', 
                'SPNO', 'SPSO', 'SRMV', 'SRMW', 
                'SRSO', 'SRTV', 'SRVC']

    df_1 = df[df['eGRID_subregion'].isin(first_set)].copy()
    df_2 = df[df['eGRID_subregion'].isin(second_set)].copy()

    ordered_NERC = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                        'MROE', 'MROW', 'NEWE', 'NWPP',
                        'NYCW', 'NYLI', 'NYUP', 'RFCE',
                        'RFCM', 'RFCW', 'RMPA', 'SPNO',
                        'SPSO', 'SRMV', 'SRMW', 'SRSO',
                        'SRTV', 'SRVC']
    
    for df in [df_1, df_2]:
        df['eGRID_subregion'] = pd.Categorical(df['eGRID_subregion'], categories=ordered_NERC, ordered=True)
        df.sort_values(by='eGRID_subregion')

        df['simulation'] = df[['chiller_type', 'scope']].apply(' - '.join, axis=1)

        # Adjust consumption values        
        Y = 'WaterConsumption_int_(L/MWhr_sqm)' 

        df[Y] = df['WaterConsumption_int_(L/kWhr_sqm)'] * 1000

    X_1 = df_1['eGRID_subregion']
    X_2 = df_2['eGRID_subregion']

    x_label = 'eGRID subregion'
    x_1_labels = first_set
    x_2_labels = second_set

    # Close any previous plots
    plt.close()


    fig, ax = plt.subplots(2, 1,figsize=(10,4), sharex=False)
    plt.subplots_adjust(hspace=0.15)
    
    # Format fonts and style
    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    from numpy import median
    colors = ['white', 'deepskyblue', 'salmon', 'white', 'deepskyblue', 'salmon']
    
    # y_ticks = np.arange(0, 20, 1)

    ###################################
    # First Plot, US and fossil fuels #
    ###################################
    plt.subplot(211)
    # ax_1 = fig.add_subplot(211)

    ax_1 = sns.barplot(x=X_1, 
                y=Y, hue='simulation', 
                data=df_1,
                # estimator=median,
                ci=95,
                palette=colors, 
                edgecolor='0.1',
                linewidth=1.5,
                capsize=0.05,
                )

    hatches = ['', '', '','////', '////', '////']
    
    for bars, hatch in zip(ax_1.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    
    ##############
    # Formatting #
    ############## 
    # X-axis
    ax_1.set_xlabel('')
    ax_1.set_xticks(np.arange(0,12,1))
    ax_1.set_xlim(-0.5, 10.5)

    # ax_1.set_xticklabels(x_1_labels, ha='center', fontsize=14)
    ax_1.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax_1.set_ylabel('')
    # ax_1.set_yticks(y_ticks)
    # ax_1.set_ylim(np.min(y_ticks), np.max(y_ticks))
    # ax_1.set_yticklabels(y_ticks,fontsize=14)
    ax_1.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))


    # ax_1.legend(title='Chiller - Scope',
    #           loc='upper center', 
    #           ncol=3,
    #           bbox_to_anchor=(0.5, 1.5),
    #           frameon=False)

    ax_1.get_legend().remove()

    ####################################
    # Second Plot, US and fossil fuels #
    ####################################
    # ax_2 = fig.add_subplot(212)
    plt.subplot(212)

    ax_2 = sns.barplot(x=X_2, 
                y=Y, hue='simulation', 
                data=df_2,
                # estimator=median,
                ci=95,
                palette=colors, 
                edgecolor='0.1',
                linewidth=1.5,
                capsize=0.05,
                )
    
    for bars, hatch in zip(ax_2.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    
    ##############
    # Formatting #
    ############## 
    # X-axis
    ax_2.set_xticks(np.arange(10,22,1))
    ax_2.set_xlim(10.5, 21.5)
    ax_2.set_xlabel('')
    # ax_2.set_xticklabels(x_2_labels, ha='center', fontsize=14)
    ax_2.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax_2.set_ylabel('')
    # ax_2.set_yticks(y_ticks)
    # ax_2.set_ylim(np.min(y_ticks), np.max(y_ticks))
    # ax_2.set_yticklabels(y_ticks,fontsize=14)
    ax_2.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))


    ax_2.get_legend().remove()
    sns.despine()


    y_label = "Water Consumption Intensity,\n           $L / (MWh_r \cdot m^2)$"
    fig.text(0.5, 0.01, x_label, ha='center', fontsize=18)
    fig.text(0.03, 0.5, y_label, va='center', rotation='vertical', fontsize=18)

    plt.savefig(F'{save_path}\water4cooling_NERC.png', dpi=300)

    plt.show()


def water_consumption_intensity_version2_VOID(how='NERC'):

    if how == 'NERC':
        df = concatenate_all_data(how)

        ordered_NERC = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                        'MROE', 'MROW', 'NEWE', 'NWPP',
                        'NYCW', 'NYLI', 'NYUP', 'RFCE',
                        'RFCM', 'RFCW', 'RMPA', 'SPNO',
                        'SPSO', 'SRMV', 'SRMW', 'SRSO',
                        'SRTV', 'SRVC']

        df['eGRID_subregion'] = pd.Categorical(df['eGRID_subregion'], categories=ordered_NERC, ordered=True)
        df.sort_values(by='eGRID_subregion')

        X = df['eGRID_subregion']

        x_label = 'eGRID subregion'
        xtick_labels = ordered_NERC

        y_ticks = np.arange(0, 9, 1)

    elif how == 'climate_zone':
        df = concatenate_all_data('NERC')
        ordered_CZ = ['1A', '2A', '2B',
                    '3A', '3B', '3B-CA', '3C',
                    '4A', '4B', '4C',
                    '5A', '5B', '6A', '6B', '7']

        df['climate_zone'] = pd.Categorical(df['climate_zone'], categories=ordered_CZ, ordered=True)
        df.sort_values(by='climate_zone')

        X = df['climate_zone']

        x_label = 'Climate zone'
        xtick_labels = ordered_CZ

        y_ticks = np.arange(0, 9, 1)

    elif how == 'fuel_type':
        df = concatenate_all_data(how)
        ordered_fuel = ['United States Overall',
                        'Ethanol', 
                        'Conventional Oil', 'Unconventional Oil', 
                        'Subbituminous Coal', 'Bituminous Coal', 'Lignite Coal',
                        'Conventional Natural Gas', 'Unconventional Natural Gas', 
                        'Uranium', 
                        'Biodiesel', 'Biogas', 'Solid Biomass and RDF', 
                        'Geothermal', 'Hydropower', 'Solar Photovoltaic', 'Solar Thermal', 'Wind'
                        ]
        
        df['fuel_type'] = pd.Categorical(df['fuel_type'], categories=ordered_fuel, ordered=True)
        df.sort_values(by='fuel_type')

        X = df['fuel_type']

        x_label = 'Fuel type'
        xtick_labels = ['US', # '\n\n' 
                        'E', #'\n\nFossil Fuels',
                        'CO', 'UO', 'SC', 'BC', 'LC', 'CNG', 'UNG', 
                        'U', #'\n\n'
                        'BD', #'\n\nRenewables', 
                        'BG', 'BioM', 
                        'GeoTh','Hydro', 'S.PV', 'S.Th', 'Wind'
                        ]

        y_ticks = np.arange(0, 5, 1)
    
    
    df['simulation'] = df[['chiller_type', 'scope']].apply(' - '.join, axis=1)

    Y = 'WaterConsumption_int_(L/MWhr_sqm)' 
    df[Y] = df['WaterConsumption_int_(L/kWhr_sqm)'] * 1000
    y_label = "Water Consumption Intensity, $L / (MWh_r \cdot m^2)$"

    # Close any previous plots
    plt.close()

    fig = plt.figure(figsize=(13,8))
    ax = fig.add_subplot(111)

    # Format fonts and style
    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    from numpy import median
    colors = ['white', 'deepskyblue', 'salmon', 'white', 'deepskyblue', 'salmon']
    
    sns.barplot(x=X, 
                y=Y, hue='simulation', 
                data=df,
                # estimator=median,
                ci=95,
                palette=colors, 
                edgecolor='0.1',
                linewidth=1.5,
                capsize=0.05,
                )

    hatches = ['', '', '','////', '////', '////']
    
    for bars, hatch in zip(ax.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)
    
    ##############
    # Formatting #
    ############## 
    # X-axis
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_xticklabels(xtick_labels, ha='center', fontsize=14, rotation=30)
    ax.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_yticks(y_ticks)
    ax.set_ylim(np.min(y_ticks), np.max(y_ticks))
    ax.set_yticklabels(y_ticks,fontsize=14)
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))


    ax.legend(title='Chiller - Scope',
              loc='upper center', 
              ncol=3,
              bbox_to_anchor=(0.5, 1.5),
              frameon=False)

    sns.despine()

    # plt.savefig(F'{save_path}\water4cooling_{how}.png', dpi=300)

    plt.show()
