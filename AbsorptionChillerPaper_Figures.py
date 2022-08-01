####################################
# LIBRARIES NEEDED TO RUN THE TOOL #
####################################
from cProfile import label
from logging import error
from msilib.schema import Error
from multiprocessing.spawn import import_main_path
from tkinter import Label
from turtle import width
import matplotlib
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
from matplotlib import gridspec, rcParams
import seaborn as sns
from sqlalchemy import column
# To install: pip install seaborn
from sympy import Predicate, inverse_laplace_transform

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

    grid_kws = {'height_ratios': (0.03, 0.95), 'hspace': 0.05}
    # grid_kws = {'width_ratios': (0.95, 0.05), 'wspace': 0.001}
    f, (cbar_ax, ax) = plt.subplots(
        2, 1, gridspec_kw=grid_kws, figsize=(13, 10))

    ax = sns.heatmap(pivot_df,
                     vmin=0, vmax=1000,
                     ax=ax,
                     cbar_ax=cbar_ax,
                     cbar_kws={'orientation': 'horizontal',
                               'ticks': mtick.LogLocator(),
                               'extend': 'max'
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

    ACC_df = pd.read_csv(F'{ACC_filepath}\\{baseline_filename}', index_col=0)
    WCC_df = pd.read_csv(F'{WCC_filepath}\\{chiller_filename}', index_col=0)
    ABC_df = pd.read_csv(F'{ABC_filepath}\\{chiller_filename}', index_col=0)

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

    df = pd.concat([df_S2, df_S3], axis=0).reset_index(drop=True)

    # Rename chillers

    return df


def Fig3_peak_electricity_reduction():
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates

    df = pd.read_csv(
        r'model_outputs\AbsorptionChillers\peak_electricity_reduction.csv')

    df['datetime'] = pd.to_datetime(df['datetime'])

    ACC_df = df[['datetime', 'Electricity_ACC_kW_m^-2']].copy()
    WCC_df = df[['datetime', 'Electricity_WCC_kW_m^-2']].copy()
    ABC_df = df[['datetime', 'Electricity_ABC_kW_m^-2']].copy()

    ACC_df.rename(
        columns={
            'Electricity_ACC_kW_m^-2': 'Electricity_kW_m^-2'},
        inplace=True)
    WCC_df.rename(
        columns={
            'Electricity_WCC_kW_m^-2': 'Electricity_kW_m^-2'},
        inplace=True)
    ABC_df.rename(
        columns={
            'Electricity_ABC_kW_m^-2': 'Electricity_kW_m^-2'},
        inplace=True)

    ACC_df['chiller_type'] = 'ACC'
    WCC_df['chiller_type'] = 'WCC'
    ABC_df['chiller_type'] = 'ABC'

    df = pd.concat([ACC_df, WCC_df, ABC_df], axis=0).reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
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
    try:
        df = pd.read_feather(
            r'model_outputs\AbsorptionChillers\water_consumption\annual_w4r_NERC.feather')
    except FileNotFoundError:
        df = concatenate_all_data("NERC")

    # Separate dataframes for subplots
    first_set = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                 'MROE', 'MROW', 'NEWE', 'NWPP',
                 'NYCW', 'NYLI', 'NYUP', ]
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
        df['eGRID_subregion'] = pd.Categorical(
            df['eGRID_subregion'], categories=ordered_NERC, ordered=True)
        df.sort_values(by='eGRID_subregion')

        df['simulation'] = df[['chiller_type', 'scope']].apply(
            ' - '.join, axis=1)

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

    grid_kws = {'height_ratios': (0.1, 0.4, 0.1, 0.4)}

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 4), sharex=False,
                                             gridspec_kw=grid_kws)

    plt.subplots_adjust(hspace=0.5)

    # Format fonts and style
    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    from numpy import median
    colors = [
        'white',
        'deepskyblue',
        'salmon',
        'white',
        'deepskyblue',
        'salmon']

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

    hatches = ['', '', '', '////', '////', '////']

    for bars, hatch in zip(ax1.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)

    ##############
    # Formatting #
    ##############
    # X-axis
    ax1.set_xlabel('')
    ax1.set_xticks(np.arange(0, 12, 1))
    ax1.set_xticklabels('')
    ax1.set_xlim(-0.5, 10.5)

    ax1.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax1.set_ylabel('')
    ax1.set_yticks(np.arange(10, 70, 20))
    ax1.set_ylim(np.min(10), np.max(50))
    ax1.set_yticklabels(['', 30, 50], fontsize=14)
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

    hatches = ['', '', '', '////', '////', '////']

    for bars, hatch in zip(ax2.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)

    ##############
    # Formatting #
    ##############
    # X-axis
    ax2.set_xlabel('')
    ax2.set_xticks(np.arange(0, 11, 1))
    ax2.set_xticklabels(first_set, fontsize=14)
    ax2.set_xlim(-0.5, 10.5)

    ax2.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax2.set_ylabel('')
    ax2.set_yticks(np.arange(0, 12, 2))
    ax2.set_ylim(0, 10)
    ax2.set_yticklabels(np.arange(0, 12, 2), fontsize=14)
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
    ax3.set_xticks(np.arange(10, 22, 1))
    ax3.set_xlim(10.5, 21.5)
    ax3.set_xlabel('')
    ax3.set_xticklabels('')

    # Y-axis
    ax3.set_ylabel('')
    ax3.set_yticks(np.arange(5, 20, 5))
    ax3.set_ylim(5, 15)
    ax3.set_yticklabels(['', 10, 15], fontsize=14)
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
    ax4.set_xticks(np.arange(11, 22, 1))
    ax4.set_xlim(10.5, 21.5)
    ax4.set_xlabel('')
    ax4.set_xticklabels(x_2_labels, ha='center', fontsize=14)

    # Y-axis
    ax4.set_ylabel('')
    ax4.set_yticks(np.arange(0, 6, 1))
    ax4.set_ylim(0, 5)
    ax4.set_yticklabels(np.arange(0, 6, 1), fontsize=14)
    ax4.yaxis.set_minor_locator(mtick.MultipleLocator(0.5))

    ax4.get_legend().remove()

    ######################
    # Special Formatting #
    ######################
    for ax in [ax1, ax3]:
        ax.spines['bottom'].set_linestyle('--')
        ax.spines['bottom'].set_linewidth(1.5)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', width=1.5)

    # Move bottom of ax3 to meet ax4
    pos3 = ax3.get_position()
    pos4 = ax4.get_position()

    points3 = pos3.get_points()
    points4 = pos4.get_points()

    points3[0][1] = points4[1][1] * (1.015)

    pos3.set_points(points3)

    ax3.set_position(pos3)

    sns.despine()

    y_label = "Water Consumption Intensity,\n           $L / (MWh_r \\cdot m^2)$"
    fig.text(0.5, 0.01, x_label, ha='center', fontsize=18)
    fig.text(0.03, 0.5, y_label, va='center', rotation='vertical', fontsize=18)

    plt.savefig(F'{save_path}\\water4cooling_NERC.png', dpi=300)

    plt.show()


def Fig5_w4e_vs_w4c(fit_intercept = True):
    """
    This function plots the linear relationship between water-for-electricity
    and water-for-cooling for all data points and all scopes.
    """
    from water_for_cooling import retrieve_water_consumption_intensities as get_w4e
    
    plt.close()
    #########################
    # Gather and clean data #
    #########################
    df = pd.read_feather(r'model_outputs\AbsorptionChillers\water_consumption\annual_w4r_NERC.feather')

    w4c_df = get_w4e("NERC")
    
    PoG_df = df[df['scope'] == 'PoG'].copy()
    LC_df = df[df['scope'] == 'LC'].copy()

    PoG_df['regional_w4e_(L/kWhe)'] = PoG_df['eGRID_subregion'].apply(lambda x: w4c_df['PoG'][x])
    LC_df['regional_w4e_(L/kWhe)'] = LC_df['eGRID_subregion'].apply(lambda x: w4c_df['total'][x])

    df = pd.concat([PoG_df, LC_df], axis=0)

    df.rename(columns={'WaterConsumption_int_(L/kWhr)':'w4c_(L/kWhc)'
                        }, inplace=True)

    #####################
    # Linear Regression #
    #####################
    regressions = w4c_regression(df)
    ############
    # PLOTTING #
    ############

    fig, ax = plt.subplots(1, 1, figsize=(11,7))

    colors = ['lightgray', 'lightskyblue', 'peachpuff']
    marker_types = {'AirCooledChiller': 'o',
                    'WaterCooledChiller': 'X',
                    'AbsorptionChiller': 'd'}
    
    scatter_points = sns.scatterplot(x = df['regional_w4e_(L/kWhe)'],
                                    y = df['w4c_(L/kWhc)'],
                                    hue = df['chiller_type'],
                                    palette=colors,
                                    style=df['chiller_type'],
                                    s=100,
                                    markers = marker_types,
                                    # facecolors=None,
                                    # edgecolors='b',
                                    alpha=0.3,
                                    ax=ax)
    
    acc_data = regressions['AirCooledChiller']
    acc_line = sns.lineplot(x=acc_data['X'],
                            y=acc_data['Y_predicted'],
                            ax=ax,
                            color='black',
                            linewidth=2.5
                            )

    wcc_data = regressions['WaterCooledChiller']
    # Will add standard error calculation method in later, for now:
    x_wcc = wcc_data['X']
    y_wcc = wcc_data['Y_predicted']
    
    # Mean
    wcc_line = sns.lineplot(x=x_wcc,
                            y=y_wcc,
                            ax=ax,
                            color='royalblue',
                            linewidth=2.5
                            )

    abc_data = regressions['AbsorptionChiller']
    # Will add standard error calculation method in later, for now:
    x_abc = abc_data['X']
    y_abc = abc_data['Y_predicted']
    
    # Mean
    abc_line = sns.lineplot(x=x_abc,
                            y=y_abc,
                            ax=ax,
                            color='orangered',
                            linewidth=2.5)

    # Line Formatting
    ax.lines[1].set_linestyle("--")
    ax.lines[2].set_linestyle(':')

    # Legend
    # Added legend before other lines to avoid issues
    # legend_title = r'$\underline{Legend}$'
    plt.legend(['$WCI_{C, ACC} = 0.302 \cdot WCI_{E}$                 $(R^2 = 1)$', 
                '$WCI_{C, WCC} = 0.206 \cdot WCI_{E} + 1.97$      $(R^2 = 0.995)$', 
                '$WCI_{C, ABC} = 0.00430 \cdot WCI_{E} + 3.68$   $(R^2 = 0.08)$', '', 
                'ACC', 'WCC', 'ABC'],
                # title = 'LEGEND',
                loc='upper center', ncol=2,
                bbox_to_anchor=(0.5, 1.05),
                frameon=False,
                facecolor='white',
                columnspacing=2,
                handletextpad=2,
                handlelength=2,
                fontsize=14)

    ###############
    # Error Bands #
    ###############
    # wcc_std = 0.015
    # lower_wcc = y_wcc - wcc_std
    # upper_wcc = y_wcc + wcc_std
    # ax.plot(x_wcc, upper_wcc, color = 'tab:royalblue', alpha=0.1)
    # ax.plot(x_wcc, lower_wcc, color = 'tab:royalblue', alpha=0.1)
    # ax.fill_between(x_wcc, lower_wcc, upper_wcc, alpha=0.2)

    # abc_std = 0.046
    # lower_abc = y_abc - abc_std
    # upper_abc = y_abc + abc_std
    # ax.plot(x_abc, upper_abc, color = 'tab:orangered', alpha=0.1)
    # ax.plot(x_abc, lower_abc, color = 'tab:orangered', alpha=0.1)
    # ax.fill_between(x_abc, lower_abc, upper_abc, alpha=0.2)

    ##############
    # Formatting #
    ##############
    sns.set_context('paper', rc={'lines.linewidth':2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    ##################################################
    # Boundaries of PoG and LC Water for Electricity #
    ##################################################

    # Draw vertical lines
    x_pog = 11.68
    x_lc = 5.6
    pog_limit = plt.vlines(x_pog, 0, 5.5, ls='-.', colors='olivedrab', linewidth=2)
    lc_limit = plt.vlines(x_lc, 0, 5.5, ls='-.', colors='firebrick', linewidth=2)

    # # Annotate vertical lines
    ax.annotate('$WCI_{PoG}^U$', xy=(x_pog, 0.1), rotation=270, 
                xytext=(2, 40), textcoords='offset points',
                fontsize=14)
    ax.annotate('$WCI_{Total}^L$', xy=(x_lc,0.1), rotation=90, 
                xytext=(-18, 5), textcoords='offset points',
                fontsize=14)

    pog_arrow = plt.arrow(x_pog, 2, -3, 0,
                        width = 0.1, 
                        # shape='left', 
                        color='olivedrab',
                        label='Upper Bound: PoG WCI')
    lc_arrow = plt.arrow(x_lc, 0.5, 3, 0,
                        width = 0.1, 
                        # shape='left', 
                        color='firebrick', 
                        label='Lower Bound: Total WCI')


    # Equilibrium Points
    # ABC - WCC
    x_eq1 = 8.48
    y_eq1 = 0.0043 * x_eq1 + 3.68

    # ABC - ACC
    x_eq2 = 12.4
    y_eq2 = 0.0043 * x_eq2 + 3.68

    # ACC - WCC
    x_eq3 = 20.5
    y_eq3 = 0.302 * x_eq3

    eq_x = [x_eq1, x_eq2, x_eq3]
    eq_y = [y_eq1, y_eq2, y_eq3]

    eq_points = plt.scatter(x = eq_x,
                            y = eq_y,
                            marker='o',
                            s=180,
                            facecolors='None',
                            linewidths=2,
                            edgecolors='r')
    
    dy = 0.4
    eq_1_annotation = plt.text(x = x_eq1,
                                 y = y_eq1 - dy,
                                 s = '$P^{eq}_1$',
                                 fontsize=14)
    eq_2_annotation = plt.text(x = x_eq2,
                                 y = y_eq2 - dy,
                                 s = '$P^{eq}_2$',
                                 fontsize=14)
    eq_3_annotation = plt.text(x = x_eq3,
                                 y = y_eq3 - dy,
                                 s = '$P^{eq}_3$',
                                 fontsize=14)                                                

    # Format x-axis 
    ax.set_xlabel('eGRID Regional water-for-electricity consumption intensity, $m^3 / MWh_e$',
                fontsize=18)
    ax.set_xlim(0,30)
    ax.set_xticks(np.arange(0,40,10))
    ax.set_xticklabels(ax.get_xticks(), fontsize=16)
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))

    # Format y-axis
    ax.set_ylim(0,10)
    ax.set_ylabel('Water-for-cooling consumption \nintensity, $m^3 / MWh_C$',
                    fontsize=18)
    ax.set_yticks(np.arange(0, 15, 5))
    ax.set_yticklabels(ax.get_yticks(), fontsize=16)
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))

    # Format both axes
    sns.despine(ax=ax)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2)
    ax.tick_params(axis='both', which='major', width = 2, length=10)
    ax.tick_params(axis='both', which='minor', width =2, length=7)

    ##############
    # INSET AXES #
    ##############
    # Inset Plot
    bounds = [0.05, 0.57, 0.25, 0.25] # Lower left coordinates, width, height
    axins = ax.inset_axes(bounds)
    
    # For now, just replot the same data, you can clean this later
    scatter_points = sns.scatterplot(x = df['regional_w4e_(L/kWhe)'],
                                    y = df['w4c_(L/kWhc)'],
                                    hue = df['chiller_type'],
                                    palette=colors,
                                    style=df['chiller_type'],
                                    s=80,
                                    markers = marker_types,
                                    # facecolors=None,
                                    # edgecolors='b',
                                    alpha=0.1,
                                    ax=axins)
    
    acc_data = regressions['AirCooledChiller']
    acc_line = sns.lineplot(x=acc_data['X'],
                            y=acc_data['Y_predicted'],
                            ax=axins,
                            color='black',
                            linewidth=2
                            )

    wcc_data = regressions['WaterCooledChiller']
    wcc_line = sns.lineplot(x=wcc_data['X'],
                            y=wcc_data['Y_predicted'],
                            ax=axins,
                            color='royalblue',
                            linewidth=2
                            )

    abc_data = regressions['AbsorptionChiller']
    abc_line = sns.lineplot(x=abc_data['X'],
                            y=abc_data['Y_predicted'],
                            ax=axins,
                            color='orangered',
                            linewidth=2)

    # Line Formatting
    axins.lines[1].set_linestyle("--")
    axins.lines[2].set_linestyle(':')

    # Format x inset axes 
    axins.set_xlabel('')
    axins.set_xlim(0,160)
    axins.set_xticks([0,160])
    axins.set_xticklabels(axins.get_xticks(), fontsize=14)
    axins.xaxis.set_minor_locator(mtick.MultipleLocator(10))

    axins.set_ylim(0,50)
    axins.set_ylabel('')
    axins.set_yticks(np.arange(0, 75, 25))
    axins.set_yticklabels(axins.get_yticks(), fontsize=14)
    axins.yaxis.set_minor_locator(mtick.MultipleLocator(5))

    for spine in ['left', 'bottom', 'top', 'right']:
        axins.spines[spine].set_linewidth(1.5)
    axins.tick_params(axis='both', which='both', width=1.5)

    axins.get_legend().remove()

    save_file= r'w4e_v_w4c.png'
    plt.savefig(F'{save_path}\{save_file}', dpi=300)
    plt.show()


def Fig6_water_consumption_intensity_fuel_type():
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
        df['fuel_type'] = pd.Categorical(
            df['fuel_type'], categories=ordered_fuel, ordered=True)
        df.sort_values(by='fuel_type')

        df['simulation'] = df[['chiller_type', 'scope']].apply(
            ' - '.join, axis=1)

        # Adjust consumption values
        Y = 'WaterConsumption_int_(L/MWhr)'

        df[Y] = df['WaterConsumption_int_(L/kWhr)'] * 1000

    X_1 = df_1['fuel_type']
    X_2 = df_2['fuel_type']

    x_label = 'Fuel type'
    x_1_labels = ['US',  # '\n\n'
                  'E',  # '\n\nFossil Fuels',
                  'CO', 'UO', 'SC', 'BC', 'LC', 'CNG', 'UNG']
    x_2_labels = ['U',  # '\n\n'
                  'BD',  # '\n\nRenewables',
                  'BG', 'BM',
                  'GT', 'H', 'SPV', 'STh', 'W'
                  ]

    # Close any previous plots
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=False)
    plt.subplots_adjust(hspace=0.15)

    # Format fonts and style
    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    from numpy import median
    colors = [
        'white',
        'deepskyblue',
        'salmon',
        'white',
        'deepskyblue',
        'salmon']

    # y_ticks = np.arange(0, 5, 1)

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

    hatches = ['', '', '', '////', '////', '////']

    for bars, hatch in zip(ax_1.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)

    ##############
    # Formatting #
    ##############
    # X-axis
    ax_1.set_xlabel('')
    ax_1.set_xticks(np.arange(0, 9, 1))
    ax_1.set_xlim(-0.5, 8.5)
    ax_1.set_xticklabels(x_1_labels, ha='center', fontsize=14)
    ax_1.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax_1.set_ylabel('')
    # ax_1.set_yticks(y_ticks)
    # ax_1.set_ylim(np.min(y_ticks), np.max(y_ticks))
    # ax_1.set_yticklabels(y_ticks, fontsize=14)
    # ax_1.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))

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
    ax_2.set_xticks(np.arange(9, 18, 1))
    ax_2.set_xlim(8.5, 17.5)
    ax_2.set_xlabel('')
    ax_2.set_xticklabels(x_2_labels, ha='center', fontsize=14)
    ax_2.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax_2.set_ylabel('')
    # ax_2.set_yticks(y_ticks)
    # ax_2.set_ylim(np.min(y_ticks), np.max(y_ticks))
    # ax_2.set_yticklabels(y_ticks, fontsize=14)
    # ax_2.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))

    ax_2.get_legend().remove()

    for ax in [ax_1, ax_2]:
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', width=1.5)

    sns.despine()

    y_label = "Water Consumption Intensity,\n           $L/MWh_r$"
    fig.text(0.5, 0.01, x_label, ha='center', fontsize=18)
    fig.text(0.03, 0.5, y_label, va='center', rotation='vertical', fontsize=18)

    plt.savefig(F'{save_path}\\water4cooling_fuel_type.png', dpi=300)

    plt.show()


# Fig6_water_consumption_intensity_fuel_type()

def Fig5_normalized_w4r_fuel_type():
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
        df['fuel_type'] = pd.Categorical(
            df['fuel_type'], categories=ordered_fuel, ordered=True)
        df.sort_values(by='fuel_type')

        df['simulation'] = df[['chiller_type', 'scope']].apply(
            ' - '.join, axis=1)

        # Adjust consumption values
        Y = 'WaterConsumption_int_(L/MWhr_sqm)'

        df[Y] = df['WaterConsumption_int_(L/kWhr_sqm)'] * 1000

    X_1 = df_1['fuel_type']
    X_2 = df_2['fuel_type']

    x_label = 'Fuel type'
    x_1_labels = ['US',  # '\n\n'
                  'E',  # '\n\nFossil Fuels',
                  'CO', 'UO', 'SC', 'BC', 'LC', 'CNG', 'UNG']
    x_2_labels = ['U',  # '\n\n'
                  'BD',  # '\n\nRenewables',
                  'BG', 'BM',
                  'GT', 'H', 'SPV', 'STh', 'W'
                  ]

    # Close any previous plots
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=False)
    plt.subplots_adjust(hspace=0.15)

    # Format fonts and style
    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    from numpy import median
    colors = [
        'white',
        'deepskyblue',
        'salmon',
        'white',
        'deepskyblue',
        'salmon']

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

    hatches = ['', '', '', '////', '////', '////']

    for bars, hatch in zip(ax_1.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)

    ##############
    # Formatting #
    ##############
    # X-axis
    ax_1.set_xlabel('')
    ax_1.set_xticks(np.arange(0, 9, 1))
    ax_1.set_xlim(-0.5, 8.5)
    ax_1.set_xticklabels(x_1_labels, ha='center', fontsize=14)
    ax_1.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax_1.set_ylabel('')
    ax_1.set_yticks(y_ticks)
    ax_1.set_ylim(np.min(y_ticks), np.max(y_ticks))
    ax_1.set_yticklabels(y_ticks, fontsize=14)
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
    ax_2.set_xticks(np.arange(9, 18, 1))
    ax_2.set_xlim(8.5, 17.5)
    ax_2.set_xlabel('')
    ax_2.set_xticklabels(x_2_labels, ha='center', fontsize=14)
    ax_2.tick_params(axis='x', which='both', length=0)

    # Y-axis
    ax_2.set_ylabel('')
    ax_2.set_yticks(y_ticks)
    ax_2.set_ylim(np.min(y_ticks), np.max(y_ticks))
    ax_2.set_yticklabels(y_ticks, fontsize=14)
    ax_2.yaxis.set_minor_locator(mtick.MultipleLocator(0.2))

    ax_2.get_legend().remove()
    sns.despine()

    y_label = "Water Consumption Intensity,\n           $L / (MWh_r \\cdot m^2)$"
    fig.text(0.5, 0.01, x_label, ha='center', fontsize=18)
    fig.text(0.03, 0.5, y_label, va='center', rotation='vertical', fontsize=18)

    plt.savefig(F'{save_path}\\water4cooling_fuel_type.png', dpi=300)

    plt.show()


def FigS4_climate_eGRID_map():
    import geopandas as gp
    import matplotlib.patches as mpatches
    map_file = r'C:\Users\oabro\Dropbox (GaTech)\Datafiles\eGRID_climate_overlay.shp'

    ###################
    # READ DATAFRAMES #
    ###################
    # Map
    grid_climate_gdf = gp.read_file(F'{map_file}')

    # Mercator Projection
    grid_climate_gdf = grid_climate_gdf.to_crs('EPSG:3395')

    # Create new ID column
    grid_climate_gdf.rename(columns={'ZipSubregi': 'eGRID_subregion',
                                     'ClimateZon': 'climate_zone'},
                            inplace=True)
    grid_climate_gdf['eGRID-Climate'] = grid_climate_gdf['eGRID_subregion'] + \
        '-' + grid_climate_gdf['climate_zone']

    cz_colors = {'1A': 'deeppink',
                 '2A': 'red',
                 '2B': 'tomato',
                 '3A': 'sienna',
                 '3B': 'sandybrown',
                 '3C': 'peachpuff',
                 '4A': 'gold',
                 '4B': 'yellow',
                 '4C': 'palegoldenrod',
                 '5A': 'lime',
                 '5B': 'palegreen',
                 '6A': 'dodgerblue',
                 '6B': 'skyblue',
                 '7': 'blueviolet'
                 }

    grid_climate_gdf['climate_zone'] = np.where(
        grid_climate_gdf['climate_zone'] == '7N/A',
        '7',
        grid_climate_gdf['climate_zone'])

    grid_climate_gdf['color'] = grid_climate_gdf['climate_zone'].apply(
        lambda x: cz_colors[x])

    # print(grid_climate_gdf.eGRID_subregion.unique())
    # print(grid_climate_gdf.climate_zone.unique())

    # hatch_style = {}

    fig, (ax1) = plt.subplots(1, figsize=(10, 5),
                              # gridspec_kw={'width_ratios':(0.9, 0.1)}
                              )

    grid_climate_gdf.plot(  # column='climate_zone',
        ax=ax1,
        categorical=True,
        color=grid_climate_gdf['color'],
        linewidth=1, edgecolor='0.2',
        legend=True)

    labels = list(cz_colors.keys())
    patches = []
    for key in labels:
        patch = mpatches.Patch(
            label=key,
            facecolor=cz_colors[key],
            edgecolor='black',
            lw=1)
        patches.append(patch)
    plt.legend(handles=patches, labels=labels,  # edgecolor='black',
               ncol=14,
               bbox_to_anchor=(1, 1.1),
               loc='best', frameon=True,
               columnspacing=1,
               handletextpad=0.5, handlelength=1,
               title='Key'
               ).get_frame().set_edgecolor('black')
    ax1.axis('off')

    save_file = 'climate_eGRID_map.png'
    plt.savefig(F'{save_path}\\{save_file}', dpi=300)

    plt.show()


def calculate_percent_difference(how='NERC', scope=2):
    # Read simulation files
    ACC_filepath = r'model_outputs\AbsorptionChillers\water_consumption'
    WCC_filepath = r'model_outputs\AbsorptionChillers\water_consumption\WaterCooledChiller'
    ABC_filepath = r'model_outputs\AbsorptionChillers\water_consumption\AbsorptionChiller'

    if scope == 2:
        prefix = 'PoG'
        column_prefix = prefix
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
        abc_indeces = m_indeces  # .append('chp_id')

    baseline_filename = F'{prefix}_water_for_cooling_baseline_{how}.csv'
    chiller_filename = F'{prefix}_water_for_cooling_{how}.csv'

    ACC_data = pd.read_csv(F'{ACC_filepath}\\{baseline_filename}', index_col=0)
    ACC_data.set_index(m_indeces, inplace=True, drop=True)

    WCC_data = pd.read_csv(F'{WCC_filepath}\\{chiller_filename}', index_col=0)
    WCC_data.set_index(m_indeces, inplace=True, drop=True)

    ABC_data = pd.read_csv(F'{ABC_filepath}\\{chiller_filename}', index_col=0)
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

    df['fuel_type'] = pd.Categorical(
        df['fuel_type'],
        categories=ordered_fuel,
        ordered=True)
    df.sort_values(by='fuel_type')

    df.to_csv(r'model_outputs\AbsorptionChillers\water_consumption\test.csv')
    X = df['fuel_type']

    x_label = 'Fuel type'
    xtick_labels = ['US',  # '\n\n'
                    'E',  # '\n\nFossil Fuels',
                    'CO', 'UO', 'SC', 'BC', 'LC', 'CNG', 'UNG',
                    'U',  # '\n\n'
                    'BD',  # '\n\nRenewables',
                    'BG', 'BioM',
                    'GeoTh', 'Hydro', 'S.PV', 'S.Th', 'Wind'
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

    fig = plt.figure(figsize=(11, 8))
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

        df['eGRID_subregion'] = pd.Categorical(
            df['eGRID_subregion'], categories=ordered_NERC, ordered=True)
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

        df['climate_zone'] = pd.Categorical(
            df['climate_zone'], categories=ordered_CZ, ordered=True)
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

        df['fuel_type'] = pd.Categorical(
            df['fuel_type'], categories=ordered_fuel, ordered=True)
        df.sort_values(by='fuel_type')

        X = df['fuel_type']

        x_label = 'Fuel type'
        xtick_labels = ['US',  # '\n\n'
                        'E',  # '\n\nFossil Fuels',
                        'CO', 'UO', 'SC', 'BC', 'LC', 'CNG', 'UNG',
                        'U',  # '\n\n'
                        'BD',  # '\n\nRenewables',
                        'BG', 'BioM',
                        'GeoTh', 'Hydro', 'S.PV', 'S.Th', 'Wind'
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

    fig = plt.figure(figsize=(11, 8))
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
    ax.set_yticklabels(y_ticks, fontsize=14)
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
    df = df[(df['chiller_type'] == 'AbsorptionChiller') |
            (df['chiller_type'] == 'WaterCooledChiller')]

    if scope == 2:
        df['Total_L/MWh_sqm'] = df['PoG_WaterConsumption_intensity_L/kWh_sqm'] * 1000
    elif scope == 3:
        df['Total_L/MWh_sqm'] = df['Total_WaterConsumption_intensity_L_per_kWh_sqm'] * 1000
    df['Evaporation_L/MWh_sqm'] = df['percent_evaporation'] * df['Total_L/MWh_sqm']
    df['Power_Generation_L/MWh_sqm'] = df['Total_L/MWh_sqm'] - \
        df['Evaporation_L/MWh_sqm'] * 1000

    df = df[['city',
             'building',
             'eGRID_subregion',
             'chiller_type',
             'climate_zone',
             'Total_L/MWh_sqm',
             'Evaporation_L/MWh_sqm',
             'Power_Generation_L/MWh_sqm']].copy()

    ordered_NERC = ['AZNM', 'CAMX', 'ERCT', 'FRCC', 'MROW',
                    'NWPP', 'RFCE', 'RFCW', 'RMPA', 'SRSO']

    df['eGRID_subregion'] = pd.Categorical(
        df['eGRID_subregion'],
        categories=ordered_NERC,
        ordered=True)
    df.sort_values(by='eGRID_subregion')

    ############
    # Plotting #
    ############
    colors = ['skyblue', 'salmon']

    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    fig = plt.figure(figsize=(11, 8))
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
#############
# Run Plots #
#############
# Fig4_water_consumption_intensity_NERC()
# Fig5_water_consumption_intensity_fuel_type()

####################
# INCOMPLETE PLOTS #
####################
def overlay_eGRID_climates():
    import geopandas as gp

    IECC_path = r'C:\Users\oabro\Dropbox (GaTech)\Datafiles\Climate_Zones_-_DOE_Building_America_Program'
    eGRID_path = r'C:\Users\oabro\Dropbox (GaTech)\Datafiles\egrid2020_subregions'

    plt.close()
    filename = r'model_outputs\AbsorptionChillers\water_consumption\NERC_w4r_summary.feather'

    ###################
    # READ DATAFRAMES #
    ###################
    # Water for refrigeration
    w4r_df = pd.read_feather(filename)

    # Maps
    eGRID_subregions = gp.read_file(F'{eGRID_path}\\eGRID2020_subregions.shp')
    IECC_climate_zones = gp.read_file(
        F'{IECC_path}\\Climate_Zones_-_DOE_Building_America_Program.shp')

    ##################################
    # Crop and Edit the Map Datasets #
    ##################################

    # Narrow the field scope of the map to the contiguous US
    eGRID_keys = [  # 'AKGD', 'AKMS',
        'AZNM', 'CAMX', 'ERCT', 'FRCC',
                        # 'HIMS', 'HIOA',
                        'MROE', 'MROW', 'NEWE', 'NWPP',
                        'NYCW', 'NYLI', 'NYUP', 'RFCE',
                        'RFCM', 'RFCW', 'RMPA', 'SPNO',
                        'SPSO', 'SRMV', 'SRMW', 'SRSO', 'SRTV', 'SRVC',
                        # 'PRMS'
    ]

    # Remove subregions not included in analysis
    eGRID_subregions = eGRID_subregions[eGRID_subregions['ZipSubregi'].isin(
        eGRID_keys)]
    eGRID_subregions.reset_index(inplace=True, drop=True)

    # Get only climate zones in the contiguous US
    IECC_climate_zones = IECC_climate_zones[IECC_climate_zones['IECC_Clima'].isin(
        np.arange(1, 8, 1))]
    IECC_climate_zones.reset_index(inplace=True, drop=True)
    IECC_climate_zones['ClimateZone'] = IECC_climate_zones['IECC_Clima'].astype(
        str) + IECC_climate_zones['IECC_Moist']

    # Mercator Projection
    eGRID_subregions = eGRID_subregions.to_crs('EPSG:3395')
    IECC_climate_zones = IECC_climate_zones.to_crs('EPSG:3395')

    overlay = eGRID_subregions.overlay(IECC_climate_zones)

    overlay.to_file(
        r'C:\Users\oabro\Dropbox (GaTech)\Datafiles\eGRID_climate_overlay.shp')


def Fig4_alt_map(scope, metric='normalized'):
    '''
    This function only works locally since the shapefiles are too large for GitHub.
    Water for cooling dataframes are kept in Git, maps can be downloaded online.
    '''
    import geopandas as gp
    from matplotlib.colors import TwoSlopeNorm
    map_file = r'C:\Users\oabro\Dropbox (GaTech)\Datafiles\eGRID_climate_overlay.shp'

    plt.close()
    filename = r'model_outputs\AbsorptionChillers\water_consumption\NERC_w4r_summary.feather'

    ###################
    # READ DATAFRAMES #
    ###################
    # Water for refrigeration
    w4r_df = pd.read_feather(filename)

    # Maps
    grid_climate_gdf = gp.read_file(F'{map_file}')

    # Mercator Projection
    grid_climate_gdf = grid_climate_gdf.to_crs('EPSG:3395')

    grid_climate_gdf.rename(columns={'ZipSubregi': 'eGRID_subregion',
                                     'ClimateZon': 'climate_zone'},
                            inplace=True)

    ##############
    # MERGE DATA #
    ##############
    # Create new ID column for both dataframes to merge
    grid_climate_gdf['climate_zone'] = np.where(
        grid_climate_gdf['climate_zone'] == '7N/A',
        '7',
        grid_climate_gdf['climate_zone'])

    grid_climate_gdf['eGRID-Climate'] = grid_climate_gdf['eGRID_subregion'] + \
        '-' + grid_climate_gdf['climate_zone']

    # Add a common column to merge data on
    w4r_df['eGRID-Climate'] = w4r_df['eGRID_subregion'] + \
        '-' + w4r_df['climate_zone']

    # Merge Water for Refrigeration data with Map
    w4r_map = grid_climate_gdf.merge(w4r_df, on=['eGRID_subregion', 'climate_zone',
                                                'eGRID-Climate'])

    # Limit data to either PoG or LC
    w4r_map = w4r_map[(w4r_map['scope'] == scope)]

    ############
    # PLOTTING #
    ############
    if metric == 'normalized':
        data = 'normalized_w4r_mean'
        vmin = 0
        if scope == 'PoG':
            vmax = 1.5
            reference_value = 2.36  # L/MWh-m^2
            cbar_ticks = np.arange(0, 2., 0.5)
            cbar_minor_ticks = 0.1
            scope_text = 'Scope 2'
        else:
            vmax = 4
            reference_value = 6.43  # L/MWh-m^2
            cbar_ticks = np.arange(0, 5, 1)
            cbar_minor_ticks = 0.2
            scope_text = 'Scope 3'
        divnorm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=1)

        cbar_label = 'Average Normalized Water-for-Cooling Consumption\n'
        cbar_label = cbar_label + \
            F'{scope_text} Reference Value: {reference_value} ' + \
            '$L/(MWh_r \\cdot m^2)$'

    elif metric == 'percent':
        data = 'percent_diff-mean'

        if scope == 'PoG':
            vmin = -50
            vmax = 300
            cbar_ticks = np.arange(-50, 350, 50)
            cbar_minor_ticks = 10
            scope_text = 'C&P'
        else:
            vmin = -100
            vmax = 100
            cbar_ticks = np.arange(-100, 150, 50)
            cbar_minor_ticks = 10
            scope_text = 'Total'
        divnorm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)

        cbar_label = F'Average Difference in {scope_text} Water-for-Cooling\n Consumption relative to Air-Cooled Chiller, %'
    sns.set_context('paper')

    grid_kws = {'height_ratios': (0.49, 0.49, 0.02), 'hspace': 0.05}
    fig, (wcc_ax, abc_ax, cbar_ax) = plt.subplots(
        3, 1, gridspec_kw=grid_kws, figsize=(8, 10))

    ############
    # WCC Plot #
    ############
    WCC_map = w4r_map[w4r_map['chiller_type'] == 'WaterCooledChiller']

    WCC_map.plot(ax=wcc_ax, column=data,
                 # cmap='RdYlGn_r',
                 cmap='Spectral_r',
                 linewidth=1.0, edgecolor='black',
                 legend=True,
                 cax=cbar_ax,
                 legend_kwds={'label': '',
                              'orientation': 'horizontal',
                              'ticks': cbar_ticks,
                              },
                 norm=divnorm,
                 vmin=vmin, vmax=vmax
                 )

    ############
    # ABC Plot #
    ############
    ABC_map = w4r_map[w4r_map['chiller_type'] == 'AbsorptionChiller']

    ABC_map.plot(ax=abc_ax, column=data,
                 # cmap='RdYlGn_r',
                 cmap='Spectral_r',
                 legend=False,
                 linewidth=1.0, edgecolor='black',
                 norm=divnorm,
                 vmin=vmin, vmax=vmax
                 )

    # # # Set Map Boundaries
    # boundaries = eGRID_subregions.bounds

    for ax in [wcc_ax, abc_ax]:
        # ax.set_xlim(boundaries['minx'].min(), boundaries['maxx'].max())
        # ax.set_ylim(boundaries['miny'].min(), boundaries['maxy'].max())
        ax.spines['left'].set_linewidth(0)
        ax.spines['bottom'].set_linewidth(0)
        # ax.set_axis_off()
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax.tick_params(width=0)
    sns.despine()

    if scope == 'PoG':
        wcc_label = '(a)'
        abc_label = '(c)'
    else:
        wcc_label = '(b)'
        abc_label = '(d)'

    wcc_ax.set_title(
        F'{wcc_label} Water-cooled Chiller - {scope_text}',
        fontsize=14,
        loc='left')
    abc_ax.set_title(
        F'{abc_label} Absorption Chiller - {scope_text}',
        fontsize=14,
        loc='left')
    #############
    # COLOR BAR #
    #############
    # cbar_ax.set_ticks(cbar_ticks)
    # cbar_ax.set_xticklabels(cbar_ticks)
    cbar_ax.xaxis.set_minor_locator(mtick.MultipleLocator(cbar_minor_ticks)
                                    )
    cbar_ax.xaxis.set_tick_params(which='both', width=1.5, labelsize=14,
                                  bottom=False, labelbottom=False,
                                  top=True, labeltop=True)

    cbar_ax.set_xlabel(cbar_label, fontsize=14)
    # cbar_ax.set_title(cbar_label,# rotation=270,
    #                    fontsize=14,
    #                 #    labelpad=35
    #                    )
    cbar_ax.xaxis.set_ticks_position('top')

    # plt.tight_layout()

    save_file = F'{scope}_{metric}_map.png'
    plt.savefig(F'{save_path}\\{save_file}', dpi=300)
    plt.show()


def Fig4_grid_map(scope):
    '''
    This function only works locally since the shapefiles are too large for GitHub.
    Water for cooling dataframes are kept in Git, maps can be downloaded online.
    '''
    import geopandas as gp
    from matplotlib.colors import TwoSlopeNorm
    map_file = r'C:\Users\oabro\Dropbox (GaTech)\Datafiles\eGRID_climate_overlay.shp'

    plt.close()
    filename = r'model_outputs\AbsorptionChillers\water_consumption\NERC_w4r_summary.feather'

    ###################
    # READ DATAFRAMES #
    ###################
    # Water for refrigeration
    w4r_df = pd.read_feather(filename)

    # Maps
    grid_climate_gdf = gp.read_file(F'{map_file}')

    # Mercator Projection
    grid_climate_gdf = grid_climate_gdf.to_crs('EPSG:3395')

    grid_climate_gdf.rename(columns={'ZipSubregi': 'eGRID_subregion',
                                     'ClimateZon': 'climate_zone'},
                            inplace=True)

    ##############
    # MERGE DATA #
    ##############
    # Create new ID column for both dataframes to merge
    grid_climate_gdf['climate_zone'] = np.where(
        grid_climate_gdf['climate_zone'] == '7N/A',
        '7',
        grid_climate_gdf['climate_zone'])

    grid_climate_gdf['eGRID-Climate'] = grid_climate_gdf['eGRID_subregion'] + \
        '-' + grid_climate_gdf['climate_zone']

    # Add a common column to merge data on
    w4r_df['eGRID-Climate'] = w4r_df['eGRID_subregion'] + \
        '-' + w4r_df['climate_zone']

    # Merge Water for Refrigeration data with Map
    w4r_map = grid_climate_gdf.merge(w4r_df, on='eGRID_subregion')

    # Limit data to either PoG or LC
    w4r_map = w4r_map[(w4r_map['scope'] == scope)]

    # Labels
    # IECC_climate_zones.apply(lambda x: ax.annotate(s=F'{x.ClimateZone}',
    #                             xy=x.geometry.centroid.coords[0], ha='center',
    #                             fontsize=14), axis=1)

    ############
    # PLOTTING #
    ############
    # Plot Conditions
    norm_data = 'normalized_w4r_mean'
    p_data = 'percent_diff-mean'

    n_vmin = 0

    n_vcenter = 1
    p_vcenter = 0
    if scope == 'PoG':
        n_vmax = 1.5
        reference_value = 2.36  # L/MWh-m^2
        n_cbar_ticks = np.arange(0, 2., 0.5)
        n_cbar_minor_ticks = 0.1

        p_vmin = -50
        p_vmax = 300
        p_cbar_ticks = np.arange(-50, 350, 50)
        p_cbar_minor_ticks = 10

        scope_text = 'Point-of-Generation'
    else:
        n_vmax = 4
        reference_value = 6.43  # L/MWh-m^2
        n_cbar_ticks = np.arange(0, 5, 1)
        n_cbar_minor_ticks = 0.2

        p_vmin = -100
        p_vmax = 100
        p_cbar_ticks = np.arange(-100, 150, 50)
        p_cbar_minor_ticks = 10

        scope_text = 'Lifecycle'

    # Color gradient format
    n_divnorm = TwoSlopeNorm(vmin=n_vmin, vmax=n_vmax, vcenter=n_vcenter)
    p_divnorm = TwoSlopeNorm(vmin=p_vmin, vmax=p_vmax, vcenter=p_vcenter)

    n_cbar_label = 'Average Normalized Water for Refrigeration Consumption\n'
    n_cbar_label = n_cbar_label + \
        F'{scope_text} Reference Value: {reference_value} ' + \
        '$L/(MWh_r \\cdot m^2)$'

    p_cbar_label = F'Average {scope_text} Difference in Water for Refrigeration\n Consumption relative to Air-Cooled Chiller, %'

    ############
    # PLOTTING #
    ############

    sns.set_context('paper')

    grid_kws = {'height_ratios': (0.49, 0.49, 0.02), 'hspace': 0.05}
    fig, ((pax_wcc, nax_wcc), (pax_abc, nax_abc), (pax_cbar, nax_cbar)) = plt.subplots(3, 2,
                                                                                       gridspec_kw=grid_kws,
                                                                                       figsize=(11, 7))

    ############
    # WCC Plots #
    ############
    WCC_map = w4r_map[w4r_map['chiller_type'] == 'WaterCooledChiller']

    # Percent
    WCC_map.plot(ax=pax_wcc, column=p_data,
                 # cmap='RdYlGn_r',
                 cmap='Spectral_r',
                 linewidth=1.0, edgecolor='black',
                 legend=True,
                 cax=pax_cbar,
                 legend_kwds={'label': '',
                              'orientation': 'horizontal',
                              'ticks': p_cbar_ticks,
                              },
                 norm=p_divnorm,
                 vmin=p_vmin, vmax=p_vmax
                 )

    # Normalized
    WCC_map.plot(ax=nax_wcc, column=norm_data,
                 # cmap='RdYlGn_r',
                 cmap='Spectral_r',
                 linewidth=1.0, edgecolor='black',
                 legend=True,
                 cax=nax_cbar,
                 legend_kwds={'label': '',
                              'orientation': 'horizontal',
                              'ticks': n_cbar_ticks,
                              },
                 norm=n_divnorm,
                 vmin=n_vmin, vmax=n_vmax
                 )

    #############
    # ABC Plots #
    #############
    ABC_map = w4r_map[w4r_map['chiller_type'] == 'AbsorptionChiller']

    # Percent
    ABC_map.plot(ax=pax_abc, column=p_data,
                 # cmap='RdYlGn_r',
                 cmap='Spectral_r',
                 legend=False,
                 linewidth=1.0, edgecolor='black',
                 norm=p_divnorm,
                 vmin=p_vmin, vmax=p_vmax
                 )

    # Normalized
    ABC_map.plot(ax=nax_abc, column=norm_data,
                 # cmap='RdYlGn_r',
                 cmap='Spectral_r',
                 legend=False,
                 linewidth=1.0, edgecolor='black',
                 norm=n_divnorm,
                 vmin=n_vmin, vmax=n_vmax
                 )

    for ax in [pax_wcc, pax_abc, nax_wcc, nax_abc]:
        # ax.set_xlim(boundaries['minx'].min(), boundaries['maxx'].max())
        # ax.set_ylim(boundaries['miny'].min(), boundaries['maxy'].max())
        ax.spines['left'].set_linewidth(0)
        ax.spines['bottom'].set_linewidth(0)
        # ax.set_axis_off()
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax.tick_params(width=0)
    sns.despine()

    pax_wcc.set_title('a) Water-cooled Chiller', fontsize=14, loc='left')
    nax_wcc.set_title('b) Water-cooled Chiller', fontsize=14, loc='left')
    pax_abc.set_title('c) Absorption Chiller', fontsize=14, loc='left')
    nax_abc.set_title('d) Absorption Chiller', fontsize=14, loc='left')

    #############
    # COLOR BAR #
    #############
    pax_cbar.xaxis.set_minor_locator(mtick.MultipleLocator(p_cbar_minor_ticks))
    nax_cbar.xaxis.set_minor_locator(mtick.MultipleLocator(n_cbar_minor_ticks))

    pax_cbar.set_xlabel(p_cbar_label, fontsize=14)
    nax_cbar.set_xlabel(n_cbar_label, fontsize=14)

    for ax in [pax_cbar, nax_cbar]:
        ax.xaxis.set_tick_params(which='both', width=1.5, labelsize=14,
                                 bottom=False, labelbottom=False,
                                 top=True, labeltop=True)
        ax.xaxis.set_ticks_position('top')

    save_file = F'{scope}_grid_map.png'
    plt.savefig(F'{save_path}\\{save_file}', dpi=300)
    plt.show()


# Plot maps
# Fig4_alt_map(scope='PoG', metric='normalized')
# Fig4_alt_map(scope='LC', metric='normalized')
Fig4_alt_map(scope='PoG', metric='percent')
Fig4_alt_map(scope='LC', metric='percent')


def absolute_values_map(scope, metric='L/MWh_sqm'):
    import geopandas as gp
    from matplotlib.colors import TwoSlopeNorm
    map_file = r'C:\Users\oabro\Dropbox (GaTech)\Datafiles\eGRID_climate_overlay.shp'

    plt.close()
    if metric == 'L/MWh_sqm':    
        data_col = 'WC_mean'
        unit_label = '$m^3 / (kWh_C \\cdot m^2)$'
        save_label = 'energy_and_area'
    elif metric == 'L/kWh':
        data_col = 'W4C_mean'
        unit_label = '$m^3/MWh_C $'
        save_label = 'energy'
    elif metric == 'L/sqm':
        data_col = 'WC_2_mean'
        unit_label = '$m^3$ of water / $m^2$ of floor area'
        save_label = 'area'
    
    filename = r'model_outputs\AbsorptionChillers\water_consumption\total_w4r_NERC.feather'

    ###################
    # READ DATAFRAMES #
    ###################
    # Water for refrigeration
    w4r_df = pd.read_feather(filename)

    # Maps
    grid_climate_gdf = gp.read_file(F'{map_file}')

    # Mercator Projection
    grid_climate_gdf = grid_climate_gdf.to_crs('EPSG:3395')

    grid_climate_gdf.rename(columns={'ZipSubregi': 'eGRID_subregion',
                                     'ClimateZon': 'climate_zone'},
                            inplace=True)

    ##############
    # MERGE DATA #
    ##############
    # Create new ID column for both dataframes to merge
    grid_climate_gdf['climate_zone'] = np.where(
        grid_climate_gdf['climate_zone'] == '7N/A',
        '7',
        grid_climate_gdf['climate_zone'])

    grid_climate_gdf['eGRID-Climate'] = grid_climate_gdf['eGRID_subregion'] + \
        '-' + grid_climate_gdf['climate_zone']

    # Add a common column to merge data on
    w4r_df['eGRID-Climate'] = w4r_df['eGRID_subregion'] + \
        '-' + w4r_df['climate_zone']

    w4r_df['WC_2_mean'] = w4r_df['WC_2_mean'] / 1000

    w4r_df = w4r_df[['eGRID_subregion', 'climate_zone', 'chiller_type',
                     'scope', 'eGRID-Climate', data_col]].copy()
    
    w4r_df.reset_index(inplace=True)

    # Merge Water for Refrigeration data with Map
    w4r_map = grid_climate_gdf.merge(w4r_df, on=['eGRID_subregion', 'climate_zone', 'eGRID-Climate'])

    # Limit data to either PoG or LC
    w4r_map = w4r_map[(w4r_map['scope'] == scope)]

    ############
    # PLOTTING #
    ############

    if scope == 'PoG':
        if metric == 'L/MWh_sqm':
            vmin = 0
            vmax = 2.5
            vcenter = 1.25
            cbar_ticks = np.arange(0, 3, 0.5)
            cbar_minor_ticks = 0.1
        elif metric == 'L/kWh':
            vmin = 0
            vmax = 6
            vcenter = 3
            cbar_ticks = np.arange(0, 7, 1)
            cbar_minor_ticks = 0.2
        elif metric == 'L/sqm':
            vmin = 0
            vmax = 1.5
            vcenter = 0.75
            cbar_ticks = np.arange(0, 2, 0.500)
            cbar_minor_ticks = 0.1
        
        scope_text = 'C & P'
    
    else:
        if metric == 'L/MWh_sqm':
            vmin = 0
            vmax = 35
            vcenter = 35/2
            cbar_ticks = np.arange(0, 40, 5)
            cbar_minor_ticks = 1
        elif metric == 'L/kWh':
            vmin = 0
            vmax = 50
            vcenter = 25
            cbar_ticks = np.arange(0, 60, 10)
            cbar_minor_ticks = 2
        elif metric == 'L/sqm':
            vmin = 0
            vmax = 14
            vcenter = 7
            cbar_ticks = np.arange(0, 16, 2)
            cbar_minor_ticks = 0.5
        
        scope_text = 'Total'
    divnorm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)

    cbar_label = F'Average {scope_text} Water-for-Cooling\nConsumption {unit_label}'

    sns.set_context('paper')

    grid_kws = {'height_ratios': (0.33, 0.33, 0.33, 0.01)}
    fig, (acc_ax, wcc_ax, abc_ax, cbar_ax) = plt.subplots(
        4, 1, gridspec_kw=grid_kws, figsize=(8, 11))

    ACC_map = w4r_map[w4r_map['chiller_type'] == 'AirCooledChiller'].copy()
    WCC_map = w4r_map[w4r_map['chiller_type'] == 'WaterCooledChiller'].copy()
    ABC_map = w4r_map[w4r_map['chiller_type'] == 'AbsorptionChiller'].copy()

    ############
    # ACC Plot #
    ############
    ACC_map.plot(ax=acc_ax, 
                column=data_col,
                 cmap='Spectral_r',
                 linewidth=1.0, edgecolor='black',
                 legend=True,
                 cax=cbar_ax,
                 legend_kwds={'label': '',
                              'orientation': 'horizontal',
                              'ticks': cbar_ticks,
                              },
                 norm=divnorm,
                 vmin=vmin, vmax=vmax
                 )

    ############
    # WCC Plot #
    ############

    WCC_map.plot(ax=wcc_ax, 
                column=data_col,
                 cmap='Spectral_r',
                 linewidth=1.0, edgecolor='black',
                # legend=True,
                 legend=False,
                 norm=divnorm,
                 vmin=vmin, vmax=vmax
                 )

    ############
    # ABC Plot #
    ############

    ABC_map.plot(ax=abc_ax, 
                column=data_col,
                 cmap='Spectral_r',                
                 linewidth=1.0, edgecolor='black',
                # legend=True,
                 legend=False,
                 norm=divnorm,
                 vmin=vmin, vmax=vmax
                 )

    for ax in [acc_ax, wcc_ax, abc_ax]:
        ax.spines['left'].set_linewidth(0)
        ax.spines['bottom'].set_linewidth(0)
        # ax.set_axis_off()
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax.tick_params(width=0)
        ax.set_xlim(-1.4 * 10**7, -0.72 * 10**7)
        
    sns.despine()

    # Chart Titles
    acc_ax.set_title('a) Air-cooled Chiller', fontsize=14, loc='left')
    wcc_ax.set_title('b) Water-cooled Chiller', fontsize=14, loc='left')
    abc_ax.set_title('c) Absorption Chiller', fontsize=14, loc='left')

    #############
    # COLOR BAR #
    #############
    #  
    cbar_ax.xaxis.set_minor_locator(mtick.MultipleLocator(cbar_minor_ticks))
    cbar_ax.xaxis.set_tick_params(which='both', width=1.5, labelsize=14,
                                  bottom=False, labelbottom=False,
                                  top=True, labeltop=True)

    cbar_ax.set_xlabel(cbar_label, fontsize=14)
    cbar_ax.xaxis.set_ticks_position('top')

    # plt.tight_layout()

    save_file = F'{scope}_totals_map_{save_label}.png'
    plt.savefig(F'{save_path}\\{save_file}', dpi=300)
    plt.show()

# Plotting the totals
# metrics = [
#             # 'L/MWh_sqm', 
#             'L/kWh', 
#            'L/sqm'
#             ]
# for metric in metrics:
#     absolute_values_map('PoG', metric)
#     absolute_values_map('LC', metric)

def w4c_regression(df, fit_intercept=True):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    import statsmodels.api as sm
    

    regressions ={}
    for chiller in df.chiller_type.unique():
        data = df[df['chiller_type'] == chiller].copy()
        data.reset_index(inplace=True, drop=True)
        
        model = LinearRegression(fit_intercept=True)

        x = data['regional_w4e_(L/kWhe)']
        X = x.values.reshape(len(x.index), 1)

        y = data['w4c_(L/kWhc)']
        Y = y.values.reshape(len(y.index), 1)

        model.fit(X,Y) 

        X_vals = np.arange(0, 160, 1)
        if fit_intercept is True:
            m = model.coef_[0][0]
            b = model.intercept_[0]
            r_squared = model.score(X, Y)

            Y_predicted = m * X_vals + b
            regression_dict = {'coef': m,
                            'intercept': b,
                            'score': r_squared,
                            'X':X_vals,
                            'Y_predicted':Y_predicted}
        else:
            m = model.coef_[0][0]
            b = 0
            r_squared = model.score(X, Y)

            Y_predicted = m * X_vals + b

            regression_dict = {'coef': model.coef_[0][0],
                            'intercept': 0,
                            'score': r_squared,
                            'X':X_vals,
                            'Y_predicted':Y_predicted}

        # ols = sm.OLS(X, Y)
        # ols_result = ols.fit()
        # print(ols_result.summary())
        regressions[chiller] = regression_dict

        print(F'{chiller}: y = {m}x + {b}, $R^2 = {r_squared}$')   

    return regressions     


def water_consumption_heatmap(scope):
    plt.close()
    filename = r'model_outputs\AbsorptionChillers\water_consumption\total_w4r_NERC.feather'

    ###################
    # READ DATAFRAMES #
    ###################
    # Water for refrigeration
    data = pd.read_feather(filename)

    subset = data[(data['scope'] == scope) & (
        data['climate_zone'] != '3B-CA')].copy()
    ACC_df = subset[subset['chiller_type'] == 'AirCooledChiller'].copy()
    WCC_df = subset[subset['chiller_type'] == 'WaterCooledChiller'].copy()
    ABC_df = subset[subset['chiller_type'] == 'AbsorptionChiller'].copy()

    # Organize the dataframes
    subregion_names = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                       'MROE', 'MROW', 'NEWE', 'NWPP',
                       'NYCW', 'NYLI', 'NYUP', 'RFCE',
                       'RFCM', 'RFCW', 'RMPA',
                       'SPNO', 'SPSO', 'SRMV', 'SRMW',
                       'SRSO', 'SRTV', 'SRVC']

    climate_zones = ['1A',
                     '2A', '2B',
                     '3A', '3B', '3C',
                     '4A', '4B', '4C',
                     '5A', '5B',
                     '6A', '6B',
                     '7']

    if scope == 'PoG':
        vmin = 0
        vmax = 6
        cbar_ticks = np.arange(0, 7, 1)
        cbar_mticks = 0.2
    else:
        vmin = 0
        vmax = 50
        cbar_ticks = np.arange(0, 60, 10)
        cbar_mticks = 2

    for df in [ACC_df, WCC_df, ABC_df]:
        df['eGRID_subregion'] = pd.Categorical(
            df['eGRID_subregion'], categories=subregion_names)
        df['climate_zone'] = pd.Categorical(
            df['climate_zone'], categories=climate_zones)

    grid_kws = {'width_ratios': (0.33, 0.33, 0.33, 0.01), 'wspace': 0.02}
    fig, (acc_ax, wcc_ax, abc_ax, cbar_ax) = plt.subplots(1, 4,
                                                          gridspec_kw=grid_kws,
                                                          # sharey=True,
                                                          figsize=(11, 5.1))

    ACC_df = ACC_df.pivot(index='eGRID_subregion',
                          columns='climate_zone',
                          values='W4C_mean')

    WCC_df = WCC_df.pivot(index='eGRID_subregion',
                          columns='climate_zone',
                          values='W4C_mean')

    ABC_df = ABC_df.pivot(index='eGRID_subregion',
                          columns='climate_zone',
                          values='W4C_mean')

    # pivot_df.fillna(0, inplace=True)

    ACC_df.sort_index(level=0, inplace=True)
    WCC_df.sort_index(level=0, inplace=True)
    ABC_df.sort_index(level=0, inplace=True)

    sns.heatmap(ACC_df,
                vmin=vmin,
                vmax=vmax,
                ax=acc_ax,
                cbar_ax=cbar_ax,
                cbar_kws={'orientation': 'vertical',
                          'ticks': cbar_ticks,
                          },
                cmap='Spectral_r',
                annot=True,
                annot_kws={'fontsize': 8,
                           'rotation': 45},
                fmt='.2g',
                # square=True
                )

    sns.heatmap(WCC_df,
                vmin=vmin,
                vmax=vmax,
                ax=wcc_ax,
                cbar=False,
                cmap='Spectral_r',
                annot=True,
                annot_kws={'fontsize': 8,
                           'rotation': 45},
                fmt='.2g',
                # square=True
                )

    sns.heatmap(ABC_df,
                vmin=vmin,
                vmax=vmax,
                ax=abc_ax,
                cbar=False,
                cmap='Spectral_r',
                annot=True,
                annot_kws={'fontsize': 8,
                           'rotation': 45},
                fmt='.2g',
                # square=True
                )

    for ax in [acc_ax, wcc_ax, abc_ax]:
        ax.set_facecolor('black')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=14)

    for ax in [wcc_ax, abc_ax]:
        ax.set_yticklabels('')
        ax.set_ylabel('')

    acc_ax.set_ylabel('eGRID Subregion', fontsize=16)
    acc_ax.set_yticklabels(acc_ax.get_yticklabels(), fontsize=14)

    acc_ax.set_title('(a) Air-cooled Chiller', fontsize=16, loc='left')
    wcc_ax.set_title('(b) Water-cooled Chiller', fontsize=16, loc='left')
    abc_ax.set_title('(c) Absorption Chiller', fontsize=16, loc='left')

    fig.text(0.5, 0.0, 'Climate Zone', ha='center', fontsize=15)
    cbar_ax.yaxis.set_minor_locator(mtick.MultipleLocator(cbar_mticks))
    cbar_ax.yaxis.set_tick_params(which='both', width=1.5, labelsize=14)
    cbar_ax.set_ylabel(
        'Average Water-for-Cooling Consumption, \n$10^{-3}  m^3/(kWh_c)$',
        fontsize=14)

    save_file = F'WC heatmap {scope}'
    plt.savefig(F'{save_path}\\{save_file}', dpi=300)
    plt.show()


# water_consumption_heatmap('PoG')
# water_consumption_heatmap('LC')

def calc_map_areas():
    import geopandas as gp
    # Read Mapfile
    map_file = r'C:\Users\oabro\Dropbox (GaTech)\Datafiles\eGRID_climate_overlay.shp'
    gdf = gp.read_file(map_file)

    # Mercator Projection
    gdf = gdf.to_crs('EPSG:6933')

    # Clean Map
    gdf.rename(columns={'ZipSubregi': 'eGRID_subregion',
                                     'ClimateZon': 'climate_zone'},
                            inplace=True)

    gdf['climate_zone'] = np.where(
        gdf['climate_zone'] == '7N/A',
        '7',
        gdf['climate_zone'])

    gdf.set_index(['climate_zone', 'eGRID_subregion'], inplace=True, drop=True)
    gdf['area'] = gdf.area
    gdf.reset_index(inplace=True)

    area_df = gdf[['climate_zone', 'eGRID_subregion', 'area']].copy()

    return area_df

def calc_map_stats(how='total'):
    from sigfig import round

    #################################
    # ANNUAL WATER CONSUMPTION DATA #
    #################################
    # Read Datafile
    datafile = r'model_outputs\AbsorptionChillers\water_consumption\total_w4r_NERC.feather'
    w4r_df = pd.read_feather(datafile)

    w4r_df['WC_var'] = w4r_df['WC_std'] ** 2
    w4r_df['%_evap_var'] = w4r_df['%_evap_std'] ** 2

    area_df = calc_map_areas()


    ##############
    # MERGE DATA #
    ##############
    # Merge Water for Refrigeration data with Map
    w4r_df = w4r_df.merge(area_df, on=['eGRID_subregion', 'climate_zone'])

    w4r_df['WC_mean_area_product'] = w4r_df['WC_mean'] * w4r_df['area']
    w4r_df['WC_var_area_product'] = w4r_df['WC_var'] * w4r_df['area']
    w4r_df['%_evap_mean_area_product'] = w4r_df['%_evap_mean'] * w4r_df['area']
    w4r_df['%_evap_var_area_product'] = w4r_df['%_evap_var'] * w4r_df['area']

    chillers = ['AirCooledChiller', 'WaterCooledChiller', 'AbsorptionChiller']

    subregion_names = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                       'MROE', 'MROW', 'NEWE', 'NWPP',
                       'NYCW', 'NYLI', 'NYUP', 'RFCE',
                       'RFCM', 'RFCW', 'RMPA',
                       'SPNO', 'SPSO', 'SRMV', 'SRMW',
                       'SRSO', 'SRTV', 'SRVC']

    climate_zones = ['1A',
                     '2A', '2B',
                     '3A', '3B', '3C',
                     '4A', '4B', '4C',
                     '5A', '5B',
                     '6A', '6B',
                     '7']
    
    scopes = ['PoG', 'LC']

    w4r_df['chiller_type'] = pd.Categorical(w4r_df['chiller_type'], categories=chillers)
    w4r_df['scope'] = pd.Categorical(w4r_df['scope'], categories=scopes)
    w4r_df['eGRID_subregion'] = pd.Categorical(w4r_df['eGRID_subregion'], categories=subregion_names)
    w4r_df['climate_zone'] = pd.Categorical(w4r_df['climate_zone'], categories=climate_zones)

    if how == 'total':
    
        df = w4r_df.groupby(['chiller_type', 'scope']).agg({'WC_mean_area_product':['sum'],
                                                            'WC_var_area_product':['sum'],
                                                            '%_evap_mean_area_product':['sum'],
                                                            '%_evap_var_area_product':['sum'],
                                                            'area':['sum']})
    elif how == 'eGRID':
        df = w4r_df.groupby(['eGRID_subregion', 'chiller_type', 'scope']).agg({'WC_mean_area_product':['sum'],
                                                            'WC_var_area_product':['sum'],
                                                            '%_evap_mean_area_product':['sum'],
                                                            '%_evap_var_area_product':['sum'],
                                                            'area':['sum']})
    elif how == 'climate_zone':
        df = w4r_df.groupby(['climate_zone', 'chiller_type', 'scope']).agg({'WC_mean_area_product':['sum'],
                                                            'WC_var_area_product':['sum'],
                                                            '%_evap_mean_area_product':['sum'],
                                                            '%_evap_var_area_product':['sum'],
                                                            'area':['sum']})
    elif how == 'eGRID-climate':
        df = w4r_df.groupby(['eGRID_subregion','climate_zone', 'chiller_type', 'scope']).agg({'WC_mean_area_product':['sum'],
                                                            'WC_var_area_product':['sum'],
                                                            '%_evap_mean_area_product':['sum'],
                                                            '%_evap_var_area_product':['sum'],
                                                            'area':['sum']})
    
    df.columns = df.columns.map('-'.join)
    df.rename(columns={'WC_mean_area_product-sum':'WC_mean_area_product',
                        'WC_var_area_product-sum':'WC_var_area_product',
                        '%_evap_mean_area_product-sum':'%_evap_mean_area_product',
                        '%_evap_var_area_product-sum':'%_evap_var_area_product',
                        'area-sum':'area'}, inplace=True)
    df['WC_mean'] = df['WC_mean_area_product'] / df['area']
    df['WC_var'] = df['WC_var_area_product'] / df['area']
    df['WC_std'] = df['WC_var'] ** (1/2)
    df['%_evap_mean'] = (df['%_evap_mean_area_product'] / df['area']) * 100
    df['%_evap_var'] = (df['%_evap_var_area_product'] / df['area']) * 100
    df['%_evap_std'] = (df['%_evap_var'] ** (1/2)) * 100

    keep_cols = ['WC_mean', 'WC_std', '%_evap_mean', '%_evap_std']
    df = df[keep_cols].copy()

    try:
        df = df.applymap(lambda x: round(x, sigfigs = 3))
    except ValueError:
        df['WC_mean'].apply(lambda x: round(x, sigfigs = 3))
        df['WC_std'].apply(lambda x: round(x, sigfigs = 3))
        df['%_evap_mean'].apply(lambda x: round(x, sigfigs = 3))
        df['%_evap_std'].apply(lambda x: round(x, sigfigs = 3))

    df[keep_cols] = df[keep_cols].astype(str)

    df['Water Consumption, $L/MWh_C m^2$'] = df['WC_mean'] + " +/- " +df['WC_std'] 
    df['Percent Evaporation'] = df['%_evap_mean'] + " +/- " +df['%_evap_std']

    save_path = r'model_outputs\AbsorptionChillers\water_consumption'
    save_file = F'water_for_cooling_stats_{how}.csv'
    df.to_csv(F'{save_path}\{save_file}')

def calculate_hypothetical_wcc(how='total'):
    from sigfig import round
    
    '''
    This function calculates the hypothetical water consumption of a water-cooled chiller
    with a COP of 7.8 (which is shown by DOE values)
    '''
    data = pd.read_feather(r'model_outputs\AbsorptionChillers\water_consumption\annual_w4r_NERC.feather')
    data = data[data['chiller_type'] == 'WaterCooledChiller'].copy()

    copy_cols = ['eGRID_subregion', 'climate_zone', 'city', 'building',
                'CoolingDemand_kWh', 'floor_area_m^2', 'grid_loss_factor',
                'w4e_int_factor_(L/kWhe)', 'CoolingDemand_intensity_kWh/sqm',
                'chiller_type', 'Cooling_ElecDemand_kWh', 'Cooling_HeatDemand_kWh',
                'annual_water_consumption_L', 'WaterConsumption_int_(L/kWhr)',
                'WaterConsumption_int_(L/kWhr_sqm)', 'MakeupWater_kg',
                'percent_evaporation', 'scope']

    df = data[copy_cols].copy()

    df.reset_index(inplace=True, drop=True)

    # 1 ton_r = 3.5 kW_r
    df['ton-h_r'] = df['CoolingDemand_kWh'] / 3.5 
    # Industry standard is 3 gallons per minute of cooling water per ton of refrigeration
    df['cooling_water_gpm'] = df['ton-h_r'] * 3 
    # Industry standard is about 2% of the cooling water is lost to evaporation
    df['gallons_evaporated'] = df['cooling_water_gpm'] * 0.02 * 60 # 60 min / h

    # Update values
    df['MakeupWater_kg'] =  df['gallons_evaporated'] * 3.78541 # assume 1 kg = 1 L
    df['Cooling_ElecDemand_kWh'] = df['CoolingDemand_kWh'] / (7.8 * (1 - df['grid_loss_factor']))
    df['CoolingDemand_intensity_kWh/sqm'] = df['CoolingDemand_kWh'] / \
        df['floor_area_m^2']

    # Update consumption values
    df['annual_water_consumption_L'] = (df['Cooling_ElecDemand_kWh'] * \
        df['w4e_int_factor_(L/kWhe)']) + df['MakeupWater_kg']
    df['WaterConsumption_int_(L/kWhr)'] = df['annual_water_consumption_L'] / \
        df['CoolingDemand_kWh']
    df['WaterConsumption_int_(L/kWhr_sqm)'] = df['WaterConsumption_int_(L/kWhr)'] / df['floor_area_m^2']

    df['percent_evaporation'] = df['MakeupWater_kg'] / df['annual_water_consumption_L']

    df = df[['climate_zone', 'eGRID_subregion', 'scope', 'building', 'WaterConsumption_int_(L/kWhr_sqm)', 'percent_evaporation']].copy()

    df.rename(columns={'WaterConsumption_int_(L/kWhr_sqm)':'WC',
                        'percent_evaporation':'%_evap'}, inplace=True)
    
    df['WC'] = df['WC'] * 1000 # Convert from L/kWh to L/MWh

    subregion_names = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                       'MROE', 'MROW', 'NEWE', 'NWPP',
                       'NYCW', 'NYLI', 'NYUP', 'RFCE',
                       'RFCM', 'RFCW', 'RMPA',
                       'SPNO', 'SPSO', 'SRMV', 'SRMW',
                       'SRSO', 'SRTV', 'SRVC']

    climate_zones = ['1A',
                     '2A', '2B',
                     '3A', '3B', '3C',
                     '4A', '4B', '4C',
                     '5A', '5B',
                     '6A', '6B',
                     '7']
    
    scopes = ['PoG', 'LC']

    df = df.groupby(['climate_zone', 'eGRID_subregion', 'scope']).agg({'WC':['mean', 'var'],
                                                                        '%_evap':['mean','var']})


    # df['scope'] = pd.Categorical(df['scope'], categories=scopes)
    # df['eGRID_subregion'] = pd.Categorical(df['eGRID_subregion'], categories=subregion_names)
    # df['climate_zone'] = pd.Categorical(df['climate_zone'], categories=climate_zones)

    df.columns = df.columns.map('_'.join)
    ################
    # Do Map Stats #
    ################
    area_df = calc_map_areas()

    df.reset_index(inplace=True)
    df = df.merge(area_df, on=['climate_zone', 'eGRID_subregion'])

    df['WC_mean_area_product'] = df['WC_mean'] * df['area']
    df['WC_var_area_product'] = df['WC_var'] * df['area']
    df['%_evap_mean_area_product'] = df['%_evap_mean'] * df['area']
    df['%_evap_var_area_product'] = df['%_evap_var'] * df['area']

    if how == 'total':
    
        df = df.groupby(['scope']).agg({'WC_mean_area_product':['sum'],
                                                            'WC_var_area_product':['sum'],
                                                            '%_evap_mean_area_product':['sum'],
                                                            '%_evap_var_area_product':['sum'],
                                                            'area':['sum']})
    elif how == 'eGRID':
        df = df.groupby(['eGRID_subregion','scope']).agg({'WC_mean_area_product':['sum'],
                                                            'WC_var_area_product':['sum'],
                                                            '%_evap_mean_area_product':['sum'],
                                                            '%_evap_var_area_product':['sum'],
                                                            'area':['sum']})
    elif how == 'climate_zone':
        df = df.groupby(['climate_zone', 'scope']).agg({'WC_mean_area_product':['sum'],
                                                            'WC_var_area_product':['sum'],
                                                            '%_evap_mean_area_product':['sum'],
                                                            '%_evap_var_area_product':['sum'],
                                                            'area':['sum']})
    elif how == 'eGRID-climate':
        df = df.groupby(['eGRID_subregion','climate_zone', 'scope']).agg({'WC_mean_area_product':['sum'],
                                                            'WC_var_area_product':['sum'],
                                                            '%_evap_mean_area_product':['sum'],
                                                            '%_evap_var_area_product':['sum'],
                                                            'area':['sum']})
    
    df.columns = df.columns.map('-'.join)
    df.rename(columns={'WC_mean_area_product-sum':'WC_mean_area_product',
                        'WC_var_area_product-sum':'WC_var_area_product',
                        '%_evap_mean_area_product-sum':'%_evap_mean_area_product',
                        '%_evap_var_area_product-sum':'%_evap_var_area_product',
                        'area-sum':'area'}, inplace=True)

    df['WC_mean'] = df['WC_mean_area_product'] / df['area']
    df['WC_var'] = df['WC_var_area_product'] / df['area']
    df['WC_std'] = df['WC_var'] ** (1/2)
    df['%_evap_mean'] = (df['%_evap_mean_area_product'] / df['area']) * 100
    df['%_evap_var'] = (df['%_evap_var_area_product'] / df['area']) * 100
    df['%_evap_std'] = (df['%_evap_var'] ** (1/2)) * 100

    keep_cols = ['WC_mean', 'WC_std', '%_evap_mean', '%_evap_std']
    df = df[keep_cols].copy()

    try:
        df = df.applymap(lambda x: round(x, sigfigs = 3))
    except ValueError:
        df['WC_mean'].apply(lambda x: round(x, sigfigs = 3))
        df['WC_std'].apply(lambda x: round(x, sigfigs = 3))
        df['%_evap_mean'].apply(lambda x: round(x, sigfigs = 3))
        df['%_evap_std'].apply(lambda x: round(x, sigfigs = 3))

    df[keep_cols] = df[keep_cols].astype(str)


    print(df)


def calc_diff_metric(how='total'):
    from sigfig import round
    
    '''
    
    '''
    data = pd.read_feather(r'model_outputs\AbsorptionChillers\water_consumption\annual_w4r_NERC.feather')

    copy_cols = ['eGRID_subregion', 'climate_zone', 'city', 'building',
                'CoolingDemand_kWh', 'floor_area_m^2', 'grid_loss_factor',
                'w4e_int_factor_(L/kWhe)', 'CoolingDemand_intensity_kWh/sqm',
                'chiller_type', 'Cooling_ElecDemand_kWh', 'Cooling_HeatDemand_kWh',
                'WaterConsumption_int_(L/kWhr)',
                'percent_evaporation', 'scope', 'percent_diff']

    df = data[copy_cols].copy()

    df.reset_index(inplace=True, drop=True)

    df['WaterConsumption_int_(L/m^2)'] = df['WaterConsumption_int_(L/kWhr)'] * df['CoolingDemand_intensity_kWh/sqm']


    df = df[['climate_zone', 'eGRID_subregion', 'scope', 'building', 'chiller_type', 'WaterConsumption_int_(L/m^2)', 'WaterConsumption_int_(L/kWhr)', 'percent_evaporation', 'percent_diff']].copy()

    df['MakeupWater_int_(L/kWhr)'] = df['WaterConsumption_int_(L/kWhr)'] * df['percent_evaporation']

    df.rename(columns={'WaterConsumption_int_(L/m^2)':'WC_int',
                        'WaterConsumption_int_(L/kWhr)':'W4C',
                        'MakeupWater_int_(L/kWhr)':'MuW',
                        'percent_evaporation':'%_evap',
                        'percent_diff':'%_diff'}, inplace=True)

    subregion_names = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                       'MROE', 'MROW', 'NEWE', 'NWPP',
                       'NYCW', 'NYLI', 'NYUP', 'RFCE',
                       'RFCM', 'RFCW', 'RMPA',
                       'SPNO', 'SPSO', 'SRMV', 'SRMW',
                       'SRSO', 'SRTV', 'SRVC']

    climate_zones = ['1A',
                     '2A', '2B',
                     '3A', '3B', '3C',
                     '4A', '4B', '4C',
                     '5A', '5B',
                     '6A', '6B',
                     '7']
    
    scopes = ['PoG', 'LC']

    chillers = ['AirCooledChiller', 'WaterCooledChiller', 'AbsorptionChiller']

    df = df.groupby(['climate_zone', 'eGRID_subregion', 'chiller_type', 'scope']).agg({'WC_int':['mean', 'var'],
                                                                        'W4C':['mean', 'var'],
                                                                        'MuW':['mean', 'var'],
                                                                        '%_evap':['mean','var'],
                                                                        '%_diff':['mean','var']})

    df.columns = df.columns.map('_'.join)

    df.reset_index(inplace=True)

    df['chiller_type'] = pd.Categorical(df['chiller_type'], categories=chillers)
    df['scope'] = pd.Categorical(df['scope'], categories=scopes)
    df['eGRID_subregion'] = pd.Categorical(df['eGRID_subregion'], categories=subregion_names)
    df['climate_zone'] = pd.Categorical(df['climate_zone'], categories=climate_zones)

    
    ################
    # Do Map Stats #
    ################
    area_df = calc_map_areas()

    df.reset_index(inplace=True)
    df = df.merge(area_df, on=['climate_zone', 'eGRID_subregion'])

    df['WC_mean_area_product'] = df['WC_int_mean'] * df['area']
    df['WC_var_area_product'] = df['WC_int_var'] * df['area']
    df['W4C_mean_area_product'] = df['W4C_mean'] * df['area']
    df['W4C_var_area_product'] = df['W4C_var'] * df['area']
    df['MuW_mean_area_product'] = df['MuW_mean'] * df['area']
    df['MuW_var_area_product'] = df['MuW_var'] * df['area']
    df['%_evap_mean_area_product'] = df['%_evap_mean'] * df['area']
    df['%_evap_var_area_product'] = df['%_evap_var'] * df['area']
    df['%_diff_mean_area_product'] = df['%_diff_mean'] * df['area']
    df['%_diff_var_area_product'] = df['%_diff_var'] * df['area']

    if how == 'total':
        group = ['chiller_type', 'scope']
    elif how == 'eGRID':
        group = ['eGRID_subregion', 'chiller_type','scope']
    elif how == 'climate_zone':
        group = ['climate_zone', 'chiller_type', 'scope']
    elif how == 'eGRID-climate':
        group = ['eGRID_subregion','climate_zone', 'chiller_type', 'scope']
        
    df = df.groupby(group).agg({'WC_mean_area_product':['sum'],
                                'WC_var_area_product':['sum'],
                                'W4C_mean_area_product':['sum'],
                                'W4C_var_area_product':['sum'],
                                'MuW_mean_area_product':['sum'],
                                'MuW_var_area_product':['sum'],
                                '%_evap_mean_area_product':['sum'],
                                '%_evap_var_area_product':['sum'],
                                '%_diff_mean_area_product':['sum'],
                                '%_diff_var_area_product':['sum'],
                                'area':['sum']})
    df.columns = df.columns.map('-'.join)
    df.rename(columns={'WC_mean_area_product-sum':'WC_mean_area_product',
                        'WC_var_area_product-sum':'WC_var_area_product',
                        'W4C_mean_area_product-sum':'W4C_mean_area_product',
                        'W4C_var_area_product-sum':'W4C_var_area_product',
                        'MuW_mean_area_product-sum':'MuW_mean_area_product',
                        'MuW_var_area_product-sum':'MuW_var_area_product',
                        '%_evap_mean_area_product-sum':'%_evap_mean_area_product',
                        '%_evap_var_area_product-sum':'%_evap_var_area_product',
                        '%_diff_mean_area_product-sum':'%_diff_mean_area_product',
                        '%_diff_var_area_product-sum':'%_diff_var_area_product',
                        'area-sum':'area'}, inplace=True)

    # Water Consumption Intensity (L/m^2)
    df['WC_mean'] = df['WC_mean_area_product'] / df['area']
    df['WC_var'] = df['WC_var_area_product'] / df['area']
    df['WC_std'] = df['WC_var'] ** (1/2)
    # Water for Cooling (L/kWh_r)
    df['W4C_mean'] = df['W4C_mean_area_product'] / df['area']
    df['W4C_var'] = df['W4C_var_area_product'] / df['area']
    df['W4C_std'] = df['W4C_var'] ** (1/2)
    # Makeup Water (L/kWh_r)
    df['MuW_mean'] = df['MuW_mean_area_product'] / df['area']
    df['MuW_var'] = df['MuW_var_area_product'] / df['area']
    df['MuW_std'] = df['MuW_var'] ** (1/2)
    # Percentage of water consumed by the cooling tower
    df['%_evap_mean'] = (df['%_evap_mean_area_product'] / df['area']) * 100
    df['%_evap_var'] = (df['%_evap_var_area_product'] / df['area']) * 100
    df['%_evap_std'] = (df['%_evap_var'] ** (1/2)) * 100
    # Percentage of water consumed by the cooling tower
    df['%_diff_mean'] = (df['%_diff_mean_area_product'] / df['area'])
    df['%_diff_var'] = (df['%_diff_var_area_product'] / df['area'])
    df['%_diff_std'] = (df['%_diff_var'] ** (1/2))


    keep_cols = ['WC_mean', 'WC_std', 'W4C_mean', 'W4C_std', 'MuW_mean', 'MuW_std', '%_evap_mean', '%_evap_std', '%_diff_mean', '%_diff_std']
    df = df[keep_cols].copy()

    if how == 'eGRID-climate':
        save_path = r'model_outputs\AbsorptionChillers\water_consumption'
        save_file = F'NEW_water_for_cooling_stats_{how}.csv'
        df.to_csv(F'{save_path}\{save_file}')
        
    try:
        df = df.applymap(lambda x: round(x, sigfigs = 3))
    except ValueError:
        df['WC_mean'].apply(lambda x: round(x, sigfigs = 3))
        df['WC_std'].apply(lambda x: round(x, sigfigs = 3))
        df['W4C_mean'].apply(lambda x: round(x, sigfigs = 3))
        df['W4C_std'].apply(lambda x: round(x, sigfigs = 3))
        df['MuW_mean'].apply(lambda x: round(x, sigfigs = 3))
        df['MuW_std'].apply(lambda x: round(x, sigfigs = 3))
        df['%_evap_mean'].apply(lambda x: round(x, sigfigs = 3))
        df['%_evap_std'].apply(lambda x: round(x, sigfigs = 3))
        df['%_diff_mean'].apply(lambda x: round(x, sigfigs = 3))
        df['%_diff_std'].apply(lambda x: round(x, sigfigs = 3))

    df[keep_cols] = df[keep_cols].astype(str)


    df['Water Consumption Intensity, L/m^2'] = df['WC_mean'] + " +/- " +df['WC_std'] 
    df['Water for Cooling, L/kWh_C'] = df['W4C_mean'] + " +/- " +df['W4C_std'] 
    df['MakeupWater, L/kWh_C'] = df['MuW_mean'] + " +/- " + df['MuW_std']
    df['Percent Evaporation'] = df['%_evap_mean'] + " +/- " +df['%_evap_std']
    df['Percent Difference'] = df['%_diff_mean'] + " +/- " +df['%_diff_std']

    save_path = r'model_outputs\AbsorptionChillers\water_consumption'
    save_file = F'NEW_water_for_cooling_stats_{how}.csv'
    df.to_csv(F'{save_path}\{save_file}')

    print(df)

# Calculate the L/kWhr and L/m^2 of water consumption for each system
# calc_diff_metric('total')
# calc_diff_metric('eGRID')
# calc_diff_metric('climate_zone')
# calc_diff_metric('eGRID-climate')

# calculate_hypothetical_wcc()

# calc_map_stats('total')
# calc_map_stats('eGRID')
# calc_map_stats('climate_zone')
# calc_map_stats('eGRID-climate')

def plot_building_heatmaps(chiller, scope):
    #WHY WONT IT WORK!!!!!!!!!!
    data = pd.read_feather(r'model_outputs\AbsorptionChillers\water_consumption\annual_w4r_NERC.feather')

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

    data['building'] = data['building'].apply(lambda x: building_rename[x])

    custom_order = ['Primary School', 'Secondary School',
                    'Hospital', 'Outpatient Healthcare',
                    'Large Hotel', 'Small Hotel',
                    'Warehouse',
                    'Midrise Apartment',
                    'Large Office', 'Medium Office', 'Small Office',
                    'Full Service Restaurant', 'Quick Serice Restaurant',
                    'Stand-alone Retail', 'Strip Mall', 'Supermarket']

    subregion_names = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                       'MROE', 'MROW', 'NEWE', 'NWPP',
                       'NYCW', 'NYLI', 'NYUP', 'RFCE',
                       'RFCM', 'RFCW', 'RMPA',
                       'SPNO', 'SPSO', 'SRMV', 'SRMW',
                       'SRSO', 'SRTV', 'SRVC']

    climate_zones = ['1A',
                     '2A', '2B',
                     '3A', '3B', '3C',
                     '4A', '4B', '4C',
                     '5A', '5B',
                     '6A', '6B',
                     '7']

    # data['eGRID_subregion'] = pd.Categorical(
    #         data['eGRID_subregion'], categories=subregion_names, ordered=True)
    # data['climate_zone'] = pd.Categorical(
    #         data['climate_zone'], categories=climate_zones, ordered=True)

    data['WaterConsumption_int_(L/MWhr_sqm)'] = data['WaterConsumption_int_(L/kWhr_sqm)'] * 1000

    print(data['WaterConsumption_int_(L/kWhr)'])
    
    df = data[data['scope'] == 'LC']
    df['x_vals'] = df['climate_zone'] + '-' + df['eGRID_subregion']
    sns.lineplot(x = df.x_vals,
                y = df['WaterConsumption_int_(L/kWhr)'],# * df['CoolingDemand_intensity_kWh/sqm'],
                hue = df['chiller_type'])
    plt.show()
    exit()
    '''
    CONTINUE
    '''
    df = data.groupby(['climate_zone', 'eGRID_subregion', 'building', 'chiller_type', 'scope'])
    df.reset_index(inplace=True)
    print(df)
    exit()

    df = df[df['climate_zone'] != '3B-CA'].copy()
    df = df[(df['chiller_type'] == chiller) & (df['scope'] == scope)].copy()
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    fig = plt.figure(figsize=(11, 8))

    i = 1
    vmax = 200
    for building in custom_order:
        ax = fig.add_subplot(4, 4, i)
        subset = df[df['building'] == building].copy()
        subset = subset[['climate_zone', 'eGRID_subregion', 'WaterConsumption_int_(L/MWhr_sqm)']].copy()

        subset.reset_index(inplace=True, drop=True)

        print(subset.eGRID_subregion.unique())
        print(subset.climate_zone.unique())


        pivot_df = df.pivot(index='eGRID_subregion',
                            columns = 'climate_zone', 
                            values = 'WaterConsumption_int_(L/MWhr_sqm)')

        pivot_df.sort_index(level=0, inplace=True)

        sns.heatmap(pivot_df,
                    vmin=0,
                    vmax=vmax,
                    ax=ax,
                    cmap='Spectral_r',
                    cbar=False)
        
        i+=1

    plt.show()



    # for chiller in data['chiller_type'].unique():
    #     for scope in data['scope'].unique():
    #         df = data[(data['chiller_type'] == chiller) & (data['scope'] == scope)]
    #         minimum_w4r = df['WaterConsumption_int_(L/MWhr_sqm)'].min()
    #         print('\n\n' + chiller.upper())
            
    #         print(df[df['WaterConsumption_int_(L/MWhr_sqm)'] == minimum_w4r])

    #         maximum_w4r = df['WaterConsumption_int_(L/MWhr_sqm)'].max()
    #         print(df[df['WaterConsumption_int_(L/MWhr_sqm)'] == maximum_w4r])
    
    #         print(F'{chiller}, {scope}, min = {minimum_w4r}; max = {maximum_w4r}')
            
            

# plot_building_heatmaps(chiller='AirCooledChiller', scope='LC')

def plot_egrid_climate_stats(): 
    data = pd.read_feather(r'model_outputs\AbsorptionChillers\water_consumption\annual_w4r_NERC.feather')
    data['Water Consumption L / MWhr_sqm'] = data['WaterConsumption_int_(L/kWhr_sqm)'] * 1000

    df = data[['climate_zone', 'eGRID_subregion', 'building', 'chiller_type', 'scope', 'Water Consumption L / MWhr_sqm', 'percent_evaporation']].copy()
    
    df = df[df['climate_zone'] != '3B-CA'].copy()

    subregion_names = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                       'MROE', 'MROW', 'NEWE', 'NWPP',
                       'NYCW', 'NYLI', 'NYUP', 'RFCE',
                       'RFCM', 'RFCW', 'RMPA',
                       'SPNO', 'SPSO', 'SRMV', 'SRMW',
                       'SRSO', 'SRTV', 'SRVC']

    climate_zones = ['1A',
                     '2A', '2B',
                     '3A', '3B', '3C',
                     '4A', '4B', '4C',
                     '5A', '5B',
                     '6A', '6B',
                     '7']

    df['eGRID_subregion'] = pd.Categorical(df['eGRID_subregion'], categories=subregion_names, ordered=True)
    df['climate_zone'] = pd.Categorical(df['climate_zone'], categories=climate_zones, ordered=True)

    df.sort_values(by=['climate_zone', 'eGRID_subregion'], inplace=True)

    df['climate-grid'] = df[['climate_zone', 'eGRID_subregion']].apply('-'.join, axis=1)

    df.sort_values(by='Water Consumption L / MWhr_sqm', inplace=True)

    df = df[df['scope'] == 'PoG']

    for chiller in ['AirCooledChiller', 'WaterCooledChiller', 'AbsorptionChiller']:
        fig, (ax1) = plt.subplots(1, figsize=(8, 11))
        df2 = df[df['chiller_type'] == chiller].copy()
        df2.sort_values(by='Water Consumption L / MWhr_sqm', inplace=True)    
        sns.lineplot(x=df2['climate-grid'],
                    y=df2['Water Consumption L / MWhr_sqm'],
                    hue=df2['building'],
                    ax=ax1)

        ax1.set_xticklabels(df2['climate-grid'], rotation=90, fontsize=14)


        plt.show()

    # data['climate-eGRID']
    # print(data.isna().any())

# plot_egrid_climate_stats()



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
    plt.close()
    filename = r'model_outputs\AbsorptionChillers\water_consumption\total_w4r_NERC.feather'

    ###################
    # READ DATAFRAMES #
    ###################
    # Water for refrigeration
    w4r_df = pd.read_feather(filename)
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
    plt.savefig(F'{file_path}\\{file_name}')
    plt.show()


##################################
# VOID FUNCTIONS - FOR REFERENCE #
##################################
def water_consumption_intensity_NERC_VOID():
    df = concatenate_all_data("NERC")

    # Separate dataframes for subplots
    first_set = ['AZNM', 'CAMX', 'ERCT', 'FRCC',
                 'MROE', 'MROW', 'NEWE', 'NWPP',
                 'NYCW', 'NYLI', 'NYUP', ]
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
        df['eGRID_subregion'] = pd.Categorical(
            df['eGRID_subregion'], categories=ordered_NERC, ordered=True)
        df.sort_values(by='eGRID_subregion')

        df['simulation'] = df[['chiller_type', 'scope']].apply(
            ' - '.join, axis=1)

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

    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=False)
    plt.subplots_adjust(hspace=0.15)

    # Format fonts and style
    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    from numpy import median
    colors = [
        'white',
        'deepskyblue',
        'salmon',
        'white',
        'deepskyblue',
        'salmon']

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

    hatches = ['', '', '', '////', '////', '////']

    for bars, hatch in zip(ax_1.containers, hatches):
        for bar in bars:
            bar.set_hatch(hatch)

    ##############
    # Formatting #
    ##############
    # X-axis
    ax_1.set_xlabel('')
    ax_1.set_xticks(np.arange(0, 12, 1))
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
    ax_2.set_xticks(np.arange(10, 22, 1))
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

    y_label = "Water Consumption Intensity,\n           $L / (MWh_r \\cdot m^2)$"
    fig.text(0.5, 0.01, x_label, ha='center', fontsize=18)
    fig.text(0.03, 0.5, y_label, va='center', rotation='vertical', fontsize=18)

    plt.savefig(F'{save_path}\\water4cooling_NERC.png', dpi=300)

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

        df['eGRID_subregion'] = pd.Categorical(
            df['eGRID_subregion'], categories=ordered_NERC, ordered=True)
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

        df['climate_zone'] = pd.Categorical(
            df['climate_zone'], categories=ordered_CZ, ordered=True)
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

        df['fuel_type'] = pd.Categorical(
            df['fuel_type'], categories=ordered_fuel, ordered=True)
        df.sort_values(by='fuel_type')

        X = df['fuel_type']

        x_label = 'Fuel type'
        xtick_labels = ['US',  # '\n\n'
                        'E',  # '\n\nFossil Fuels',
                        'CO', 'UO', 'SC', 'BC', 'LC', 'CNG', 'UNG',
                        'U',  # '\n\n'
                        'BD',  # '\n\nRenewables',
                        'BG', 'BioM',
                        'GeoTh', 'Hydro', 'S.PV', 'S.Th', 'Wind'
                        ]

        y_ticks = np.arange(0, 5, 1)

    df['simulation'] = df[['chiller_type', 'scope']].apply(' - '.join, axis=1)

    Y = 'WaterConsumption_int_(L/MWhr_sqm)'
    df[Y] = df['WaterConsumption_int_(L/kWhr_sqm)'] * 1000
    y_label = "Water Consumption Intensity, $L / (MWh_r \\cdot m^2)$"

    # Close any previous plots
    plt.close()

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111)

    # Format fonts and style
    sns.set_style('ticks', {'axes.facecolor': '1'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    from numpy import median
    colors = [
        'white',
        'deepskyblue',
        'salmon',
        'white',
        'deepskyblue',
        'salmon']

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

    hatches = ['', '', '', '////', '////', '////']

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
    ax.set_yticklabels(y_ticks, fontsize=14)
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))

    ax.legend(title='Chiller - Scope',
              loc='upper center',
              ncol=3,
              bbox_to_anchor=(0.5, 1.5),
              frameon=False)

    sns.despine()

    # plt.savefig(F'{save_path}\water4cooling_{how}.png', dpi=300)

    plt.show()
