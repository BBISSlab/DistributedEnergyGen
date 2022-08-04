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


save_path = r'model_outputs\CCHPvGrid\figures'

def get_peak_demands():
    ac_filepath = r'model_outputs\CCHPvGrid\Energy_Demands\AC'
    abc_filepath = r'model_outputs\CCHPvGrid\Energy_Demands\ABC_SS'
    
    dfs = []
    for city in city_list:        
        for building in building_type_list:
            ac_df = pd.read_feather(F'{ac_filepath}\{city}_{building}.feather')
            ac_df['chiller'] = ac_df['AC_id']
            abc_df = pd.read_feather(F'{abc_filepath}\{city}_{building}.feather')
            abc_df['chiller'] = abc_df['ABC_id']

            for df in [ac_df, abc_df]:
                agg_df = df.groupby(['City', 'Building', 'chiller']).agg({'total_electricity_demand':'max',
                                                                          'total_heat_demand':'max'})

                agg_df.rename(columns={'total_electricity_demand':'peak_electricity_demand_kW',
                                       'total_heat_demand':'peak_heat_demand_kW'}, inplace=True)
                
                agg_df.reset_index(inplace=True)
                
                dfs.append(agg_df)

    df = pd.concat(dfs, axis=0)

    df.reset_index(inplace=True, drop=True)

    return df

def pivot_peaks(df, energy='electricity', chiller='ac'):
    
    if chiller == 'ac':
        df = df[df['chiller'] == 'AC3'].copy()
    else:
        df = df[df['chiller'] == 'ABC_SS1'].copy()

    if energy == 'electricity':
        pivot_df = df.pivot(index='Building',
                            columns= 'climate_zone',
                            values = 'peak_electricity_demand_kW')
    else:
        pivot_df = df.pivot(index='Building',
                            columns= 'climate_zone',
                            values = 'peak_heat_demand_kW')
    
    return pivot_df

def peak_demand_heatmap():
    '''
    This function generates a heat map of each building's cooling demand for each climate zone
    '''
    df = get_peak_demands()

    df['climate_zone'] = df['City'].apply(lambda x: climate_zone_dictionary[x])

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

    df['Building'] = df['Building'].apply(lambda x: building_rename[x])

    custom_order = ['Primary School', 'Secondary School',
                    'Hospital', 'Outpatient Healthcare',
                    'Large Hotel', 'Small Hotel',
                    'Warehouse',
                    'Midrise Apartment',
                    'Large Office', 'Medium Office', 'Small Office',
                    'Full Service Restaurant', 'Quick Serice Restaurant',
                    'Stand-alone Retail', 'Strip Mall', 'Supermarket']

    # Electricity Pivots
    ac_E_pivot = pivot_peaks(df, 'electricity', 'ac')
    abc_E_pivot = pivot_peaks(df, 'electricity', 'abc')

    # Heat Pivots
    ac_H_pivot = pivot_peaks(df, 'heat', 'ac')
    abc_H_pivot = pivot_peaks(df, 'heat', 'abc')

    print(abc_H_pivot)

    # pivot_df = df.pivot(index='building',
    #                     columns='climate_zone',
    #                     values='CoolingDemand_intensity_kWh/sqm')

    # pivot_df.index = pd.CategoricalIndex(
    #     pivot_df.index, categories=custom_order)
    # pivot_df.sort_index(level=0, inplace=True)

    # grid_kws = {'height_ratios':(0.03,0.95), 'hspace': 0.05}
    # # grid_kws = {'width_ratios': (0.95, 0.05), 'wspace': 0.001}
    # f, (cbar_ax, ax) = plt.subplots(
    #     2, 1, gridspec_kw=grid_kws, figsize=(13, 10))

    # ax = sns.heatmap(pivot_df,
    #                  vmin=0, vmax=1000,  
    #                  ax=ax,
    #                  cbar_ax=cbar_ax,
    #                  cbar_kws= {'orientation': 'horizontal',
    #                            'ticks':mtick.LogLocator(),
    #                            'extend':'max'
    #                            },
    #                  cmap='coolwarm_r',
    #                  square=True,
    #                  norm=LogNorm(),
    #                  )

    # cbar_ax.xaxis.set_tick_params(which='both', width=1.5, labelsize=14, 
    #                 bottom=False, labelbottom=False, 
    #                 top=True, labeltop=True)

    # cbar_ax.set_title('Cooling Demand Intensity, $kWh/m^2$', fontsize=16)

    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=14)
    # ax.set_xlabel('IECC Climate Zone', fontsize=18)

    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    # ax.set_ylabel('Building', fontsize=18)
    # ax.tick_params(axis='both', width=1.5, labelsize=14)

    # sns.set_context('paper')

    # filename = r'CoolingDemand_HeatMap.png'
    # plt.savefig(F'{save_path}\\{filename}', dpi=300)

    # plt.show()

    pass

peak_demand_heatmap()