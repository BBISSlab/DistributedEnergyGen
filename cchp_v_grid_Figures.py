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

    df['floor_area_m^2'] = df['Building'].apply(lambda x: floor_area_dictionary[x]) 

    df.reset_index(inplace=True, drop=True)

    df['peak_electricity_demand_W/m^2'] = (df['peak_electricity_demand_kW'] / df['floor_area_m^2']) * 1000
    df['peak_heat_demand_W/m^2'] = (df['peak_heat_demand_kW'] / df['floor_area_m^2']) * 1000

    return df

def pivot_peaks(df, energy='electricity', chiller='ac'):
    
    if chiller == 'ac':
        df = df[df['chiller'] == 'AC3'].copy()
    else:
        df = df[df['chiller'] == 'ABC_SS1'].copy()

    if energy == 'electricity':
        pivot_df = df.pivot(index='Building',
                            columns= 'climate_zone',
                            values = 'peak_electricity_demand_W/m^2')
    else:
        pivot_df = df.pivot(index='Building',
                            columns= 'climate_zone',
                            values = 'peak_heat_demand_W/m^2')

    custom_order = ['P. School', 'S. School',
                    'Hospital', 'O. Healthcare',
                    'L. Hotel', 'S. Hotel',
                    'Warehouse',
                    'Midrise Apt',
                    'L. Office', 'M. Office', 'S. Office',
                    'F. Restaurant', 'Q. Restaurant',
                    'S. Retail', 'S. Mall', 'Supermarket']

    pivot_df.index = pd.CategoricalIndex(
        pivot_df.index, categories=custom_order)
    pivot_df.sort_index(level=0, inplace=True)

    return pivot_df

def peak_demand_heatmap():
    '''
    This function generates a heat map of each building's cooling demand for each climate zone
    '''
    plt.close()

    df = get_peak_demands()

    df['climate_zone'] = df['City'].apply(lambda x: climate_zone_dictionary[x])

    # building_rename = {'primary_school': 'Primary School',
    #                    'secondary_school': 'Secondary School',
    #                    'hospital': 'Hospital',
    #                    'outpatient_healthcare': 'Outpatient Healthcare',
    #                    'large_hotel': 'Large Hotel',
    #                    'small_hotel': 'Small Hotel',
    #                    'warehouse': 'Warehouse',
    #                    'midrise_apartment': 'Midrise Apartment',
    #                    'large_office': 'Large Office',
    #                    'medium_office': 'Medium Office',
    #                    'small_office': 'Small Office',
    #                    'full_service_restaurant': 'Full Service Restaurant',
    #                    'quick_service_restaurant': 'Quick Serice Restaurant',
    #                    'stand_alone_retail': 'Stand-alone Retail',
    #                    'strip_mall': 'Strip Mall',
    #                    'supermarket': 'Supermarket'}
    
    building_rename = {'primary_school': 'P. School',
                       'secondary_school': 'S. School',
                       'hospital': 'Hospital',
                       'outpatient_healthcare': 'O. Healthcare',
                       'large_hotel': 'L. Hotel',
                       'small_hotel': 'S. Hotel',
                       'warehouse': 'Warehouse',
                       'midrise_apartment': 'Midrise Apt',
                       'large_office': 'L. Office',
                       'medium_office': 'M. Office',
                       'small_office': 'S. Office',
                       'full_service_restaurant': 'F. Restaurant',
                       'quick_service_restaurant': 'Q. Restaurant',
                       'stand_alone_retail': 'S. Retail',
                       'strip_mall': 'S. Mall',
                       'supermarket': 'Supermarket'}


    df['Building'] = df['Building'].apply(lambda x: building_rename[x])

    # Electricity Pivots
    ac_E_pivot = pivot_peaks(df, 'electricity', 'ac')
    abc_E_pivot = pivot_peaks(df, 'electricity', 'abc')

    # Heat Pivots
    ac_H_pivot = pivot_peaks(df, 'heat', 'ac')
    abc_H_pivot = pivot_peaks(df, 'heat', 'abc')


    grid_kws = {'height_ratios':(0.5, 0.5, 0.01), 'hspace': 0.3, 'wspace':0.1}

    fig, ([ace_ax, ach_ax], [abce_ax, abch_ax], [cbar_e_ax, cbar_h_ax]) = plt.subplots(
            3, 2, gridspec_kw=grid_kws, figsize=(11, 10))

    e_vmin = 0
    e_vmax = 200
    h_vmin = 0
    h_vmax = 700

    # Electricity Heatmaps
    # AC
    sns.heatmap(ac_E_pivot, 
                vmin=e_vmin, vmax=e_vmax,
                ax=ace_ax,
                cbar_ax=cbar_e_ax,
                cbar_kws= {'orientation': 'horizontal',
                           'ticks':np.arange(0,300,100),
                        #    'label':'Electricity Demand Intensity $W_e / m^2$'
                          },
                square=True,
                cmap='YlGnBu')
    # ABC
    sns.heatmap(abc_E_pivot, 
                vmin=e_vmin, vmax=e_vmax,
                ax=abce_ax,
                cbar=False,
                square=True,
                cmap='YlGnBu')

    # Heat Heatmaps
    # AC
    sns.heatmap(ac_H_pivot, 
                vmin=h_vmin, vmax=h_vmax,
                ax=ach_ax,
                cbar_ax=cbar_h_ax,
                cbar_kws= {'orientation': 'horizontal',
                           'ticks': np.arange(0, 800, 100),
                        #    'label':'Heat Demand Intensity $W_h / m^2$'
                          },
                square=True,
                cmap='YlOrRd')
    
    # ABC
    sns.heatmap(abc_H_pivot, 
                vmin=h_vmin, vmax=h_vmax,
                ax=abch_ax,
                cbar=False,
                square=True,
                cmap='YlOrRd')

    for ax in [ace_ax, ach_ax, abce_ax, abch_ax]:
        ax.set_facecolor('black')

    for cbar_ax in [cbar_e_ax, cbar_h_ax]:
        cbar_ax.xaxis.set_tick_params(which='both', width=1.5, labelsize=14, 
                        bottom=True, labelbottom=True, 
                        top=False, labeltop=False)
        cbar_ax.xaxis.set_tick_params(which='major', length=7)
        cbar_ax.xaxis.set_tick_params(which='minor', length=4)

    cbar_e_ax.set_xticklabels(cbar_e_ax.get_xticks(), fontsize=14)
    cbar_h_ax.set_xticklabels(cbar_h_ax.get_xticks(), fontsize=14)
    
    cbar_e_ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
    cbar_h_ax.xaxis.set_minor_locator(mtick.MultipleLocator(20))

    # cbar_e_ax.label.set_size(14)
    # cbar_h_ax.label.set_size(14)

    for ax in [ace_ax, ach_ax, abce_ax, abch_ax]:
        ax.set_xlabel('')
        ax.set_ylabel('')

    for ax in [ach_ax, abch_ax]:
        ax.set_yticklabels('')
    
    for ax in [ace_ax, ach_ax]:
        ax.set_xticklabels('')

    for ax in [ace_ax, abce_ax]:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    for ax in [abce_ax, abch_ax]:
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=90)

    ace_ax.set_title('(a)', loc='left', fontsize=18)
    ach_ax.set_title('(b)', loc='left', fontsize=18)
    abce_ax.set_title('(c)', loc='left', fontsize=18)
    abch_ax.set_title('(d)', loc='left', fontsize=18)

    # ace_ax.text(0.5, 1.1, 'Electricity Demand Intensity')
    # ach_ax.text(0.5, 1.1, 'Heat Demand Intensity')

    # ace_ax.text(-0.1, 0.5, 'Air-Cooled Chiller', rotation = 90)
    # abce_ax.text(-0.1,  0.5, 'Absorption Chiller', rotation=90)
    
    sns.set_context('paper')

    filename = r'Electicity and Heat Demands.png'
    plt.savefig(F'{save_path}\\{filename}', dpi=300)

    plt.show()

peak_demand_heatmap()