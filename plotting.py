# Data Plotting
# My modules
import csv
from sysClasses import *
# 3rd Party Modules
import pandas as pd  # To install: pip install pandas
import numpy as np  # To install: pip install numpy
from pyarrow import feather  # Storage format
import seaborn as sns  # To install: pip install seaborn

#
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter

import scipy.optimize
import scipy.stats
import pathlib
from openpyxl import load_workbook
import math as m
import time
import datetime
import inspect
import os


def plot_all_impacts(data, impact,
                     save=False, show=False,
                     building=''):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.ticker import (
        MultipleLocator, FormatStrFormatter, AutoMinorLocator)
    from data_analysis import lin_reg, clean_impact_data

    ################################
    # DATA CLEANING & ORGANIZATION #
    ################################
    data = clean_impact_data(data)

    # The Hybrid Energy System (hes) is anywhere alpha_CHP == 1 and/or
    # beta_ABC = 1
    hes_df = data[(data.alpha_CHP == 1) | (
        data.beta_ABC == 1)].copy()
    # Merge hes_df with the CHP efficiency
    pm_df = pd.read_csv(r'data\Tech_specs\PrimeMover_specs.csv', header=2)
    pm_df['CHP_efficiency'] = pm_df[['chp_EFF_LHV', 'chp_EFF_HHV']].max(axis=1)

    hes_df = pd.merge(
        hes_df, pm_df[['PM_id', 'technology']], on='PM_id', how='left').fillna('None')
    hes_df = pd.merge(
        hes_df, pm_df[['PM_id', 'CHP_efficiency']], on='PM_id', how='left').fillna(1)

    # The Conventional Energy System (ces) is anywhere alpha_CHP = 0 and
    # beta_ABC = 0
    ces_df = data[(data.alpha_CHP == 0) & (data.beta_ABC == 0)].copy()
    # Linear regression to plot the baseline (CES)
    X = np.arange(1, 2600, 10)

    if impact in ['TFCE']:
        log_reg_dict = {'full_service_restaurant': [0.2301, -0.9362],
                        'hospital': [0.2204, -0.7394],
                        'hotel': [0.1399, -0.1498],
                        'office': [0.2299, -0.5822],
                        'midrise_apartment': [0.2279, -0.4587],
                        'outpatient_healthcare': [0.1847, -0.523],
                        'school': [0.221, -0.5393],
                        'quick_service_restaurant': [0.2358, -1.0697],
                        'retail': [0.2232, -0.5758],
                        'supermarket': [0.2801, -1.1821],
                        'warehouse': [0.1931, -0.2435]}

        tfce_params = log_reg_dict[building]
        trendline = (tfce_params[0] * np.log(X) + tfce_params[1]) * 100
    else:
        regression_dict = lin_reg(ces_df, impact)
        trendline = regression_dict['coef'] * X + regression_dict['intercept']

    ############
    # Plotting #
    ############
    # Close any previous plots
    plt.close()

    # Format fonts and style
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')
    sns.set_style('ticks', {'axes.facecolor': '0.8'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)

    # Tick Marks
    tick_dict = {
        # Emissions
        'co2_int': np.arange(0, 1800, 200),
        'ch4_int': np.arange(0, 30, 5),
        'n2o_int': np.arange(0, 11, 1),
        'co_int': np.arange(-20, 320, 20),
        'nox_int': np.arange(-10, 140, 10),
        'pm_int': np.arange(0, 32, 2),
        'so2_int': np.arange(0, 6.5, 0.5),
        'voc_int': np.arange(0, 110, 10),
        'GHG_int_100': np.arange(0, 2000, 200),
        'GHG_int_20': np.arange(0, 2000, 200),
        'NG_int': np.arange(0, 4500, 500),
        # Fuel and TFCE
        'TFCE': np.arange(50, 105, 5),
        'trigen_efficiency': np.arange(50, 105, 5),
        # Relative Change
        'percent_change_co2_int': np.arange(-200, 1200, 200),
        'percent_change_ch4_int': np.arange(-100, 10, 10),
        'percent_change_n2o_int': np.arange(-100, 10, 10),
        'percent_change_co_int': np.arange(-100, 500, 50),
        'percent_change_nox_int': np.arange(-100, 600, 100),
        'percent_change_pm_int': np.arange(-100, 10, 10),
        'percent_change_so2_int': np.arange(-100, 10, 10),
        'percent_change_voc_int': np.arange(-100, 1600, 200),
        'percent_change_GHG_int_100': np.arange(-50, 700, 50),
        'percent_change_GHG_int_20': np.arange(-50, 450, 50),
        'percent_change_NG_int': np.arange(-20, 140, 20)}

    minorticks_dict = {
        #################
        # Absolute values
        #################
        # With CHP Credit
        'co2_int': 50.,
        'ch4_int': 1,
        'n2o_int': 0.5,
        'co_int': 10.,
        'nox_int': 5.,
        'pm_int': 1,
        'so2_int': 0.1,
        'voc_int': 5,
        'GHG_int_100': 100,
        'GHG_int_20': 100,
        'NG_int': 250,
        'TFCE': 5,
        # Relative Change
        'percent_change_co2_int': 50,
        'percent_change_ch4_int': 5,
        'percent_change_n2o_int': 5,
        'percent_change_co_int': 10,
        'percent_change_nox_int': 50,
        'percent_change_pm_int': 5,
        'percent_change_so2_int': 5,
        'percent_change_voc_int': 50,
        'percent_change_GHG_int_100': 10,
        'percent_change_GHG_int_20': 10,
        'percent_change_NG_int': 5}

    # Making the FacetGrid
    abc_df = hes_df[hes_df.alpha_CHP == 0].copy()
    chp_df = hes_df[hes_df.alpha_CHP > 0].copy()

    # Make subplots
    fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 3))

    ################
    # LEFT SUBPLOT #
    ################
    # No ABC
    ax = plt.subplot(1, 2, 1)
    subset = hes_df[hes_df.beta_ABC == 0]

    # Plot HES Scatter
    # CHP provides energy
    sns.scatterplot(x=subset['energy_demand_int'],
                    y=subset[impact],
                    style=subset['technology'],
                    markers={'Fuel Cell': 'P', 'Reciprocating Engine': 's',
                             'Gas Turbine': 'X', 'Microturbine': '.'},
                    hue=subset["CHP_efficiency"],
                    palette='YlOrRd',
                    alpha=0.4,
                    s=80
                    )

    # Plot CES Baseline
    sns.lineplot(x=X, y=trendline, color='black')
    if impact == 'GHG_int_20':
        regression_dict = lin_reg(ces_df, 'GHG_int_100')
        trendline2 = regression_dict['coef'] * X + regression_dict['intercept']
        ax3 = sns.lineplot(x=X, y=trendline2, color='black')
        ax3.lines[0].set_linestyle("--")

    # Formatting
    if impact in ['TFCE']:
        x_ticks = {'full_service_restaurant': np.arange(600, 1900, 100),
                   'hospital': np.arange(300, 600, 50),
                   'hotel': np.arange(100, 700, 50),
                   'midrise_apartment': np.arange(0, 300, 50),
                   'office': np.arange(100, 260, 10),
                   'outpatient_healthcare': np.arange(400, 620, 20),
                   'school': np.arange(100, 420, 20),
                   'quick_service_restaurant': np.arange(700, 2400, 100),
                   'retail': np.arange(100, 520, 20),
                   'supermarket': np.arange(400, 860, 20),
                   'warehouse': np.arange(40, 260, 20)}
        xticklabels = x_ticks[building] / 100
        ax.set_xlim(np.min(x_ticks[building]), np.max(x_ticks[building]))
    else:
        xticklabels = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
        ax.set_xticks(np.arange(0, 3000, 500))
        ax.set_xlim(0, 2500)
        ax.xaxis.set_minor_locator(MultipleLocator(100))

    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('')

    ax.set_yticks(tick_dict[impact])
    if impact in ['co2_int', 'GHG_int_100', 'GHG_int_20', 'NG_int']:
        ax.set_yticklabels(tick_dict[impact] / 1000)
    else:
        ax.set_yticklabels(tick_dict[impact])
    ax.set_ylim(np.min(tick_dict[impact]), np.max(tick_dict[impact]))
    ax.yaxis.set_minor_locator(MultipleLocator(minorticks_dict[impact]))
    ax.set_ylabel('')

    ax.legend([], frameon=False)

    #################
    # RIGHT SUBPLOT #
    #################
    # Represents when ABC is used
    ax2 = plt.subplot(1, 2, 2)
    subset = hes_df[(hes_df.beta_ABC == 1) & (hes_df.alpha_CHP > 0)]
    abc_subset = hes_df[(hes_df.beta_ABC == 1) & (hes_df.alpha_CHP == 0)]

    sns.scatterplot(x=subset['energy_demand_int'],
                    y=subset[impact],
                    style=subset['technology'],
                    markers={'Fuel Cell': 'P', 'Reciprocating Engine': 's',
                             'Gas Turbine': 'X', 'Microturbine': '.'},
                    hue=subset["CHP_efficiency"],
                    palette='YlOrRd',
                    alpha=0.4,
                    s=80
                    )

    # No CHP only ABC
    sns.scatterplot(x=abc_df['energy_demand_int'],
                    y=abc_df[impact],
                    marker='.',
                    s=50,
                    color='royalblue',
                    alpha=0.05)

    # Plot CES Baseline
    sns.lineplot(x=X, y=trendline, color='black')
    if impact == 'GHG_int_20':
        regression_dict = lin_reg(ces_df, 'GHG_int_100')
        trendline2 = regression_dict['coef'] * X + regression_dict['intercept']
        ax3 = sns.lineplot(x=X, y=trendline2, color='black')
        ax3.lines[0].set_linestyle("--")

    # Formatting
    if impact in ['TFCE']:
        x_ticks = {'full_service_restaurant': np.arange(600, 1900, 100),
                   'hospital': np.arange(300, 600, 50),
                   'hotel': np.arange(100, 700, 50),
                   'midrise_apartment': np.arange(0, 300, 50),
                   'office': np.arange(100, 260, 10),
                   'outpatient_healthcare': np.arange(400, 620, 20),
                   'school': np.arange(100, 420, 20),
                   'quick_service_restaurant': np.arange(700, 2400, 100),
                   'retail': np.arange(100, 520, 20),
                   'supermarket': np.arange(400, 860, 20),
                   'warehouse': np.arange(40, 260, 20)}
        xticklabels = x_ticks[building] / 100
        ax2.set_xlim(np.min(x_ticks[building]), np.max(x_ticks[building]))
    else:
        xticklabels = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
        ax2.set_xticks(np.arange(0, 3000, 500))
        ax2.set_xlim(0, 2500)
        ax2.xaxis.set_minor_locator(MultipleLocator(100))

    ax2.set_xticklabels(xticklabels)
    ax2.set_xlabel('')

    ax2.set_yticks(tick_dict[impact])
    ax2.set_yticklabels([])
    ax2.set_ylim(np.min(tick_dict[impact]), np.max(tick_dict[impact]))
    ax2.yaxis.set_minor_locator(MultipleLocator(minorticks_dict[impact]))

    ax2.set_ylabel('')

    ax2.legend([], frameon=False)

    plt.subplots_adjust(wspace=0.1)

    sns.despine(fig)

    if save is True:
        save_path = r'model_outputs\plots'
        if building is not None:
            save_name = F'{impact}_{building}'
        else:
            save_name = F'{impact}'
        save_file = F'{save_path}\\{save_name}.png'
        plt.savefig(save_file, dpi=300)

    if show is True:
        plt.show()

    return


def energy_demand_plots():
    """
    This function plots the distribution of energy demands for all buildings
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    plt.close()

    data = pd.read_feather(
        'PhD_Code\\Outputs\\Feather\\Data_normalized_impacts.feather')

    climate_zone_dictionary = {'City': ['albuqu erque', 'atlanta', 'baltimore', 'chicago',
                                        'denver', 'duluth', 'fairbanks', 'helena',
                                        'houston', 'las_vegas', 'los_angeles', 'miami',
                                        'minneapolis', 'phoenix', 'san_francisco', 'seattle'],
                               'Climate Zone': ['4B', '3A', '4A', '5A',
                                                '5B', '7', '8', '6B',
                                                '2A', '3B', '3B-CA', '1A',
                                                '6A', '2B', '3C', '4C']}
    climates = pd.DataFrame.from_dict(climate_zone_dictionary)
    df = data[['City', 'HDD', 'CDD',
               'Building', 'floor_area',
               'beta_ABC', 'electricity_demand_int', 'heat_demand_int', 'cooling_demand_int',
               'CHP_heat_surplus', 'alpha_CHP', 'GHG_int']]

    df['energy_demand_int'] = df.electricity_demand_int + df.heat_demand_int
    df['heat_to_power_ratio'] = df.heat_demand_int / df.electricity_demand_int
    df = pd.merge(df, climates)
    # df.drop_duplicates(inplace=True)

    rcParams['font.family'] = 'Times New Roman'
    plt.rc('font', family='serif')

    sns.set_context('paper', rc={"lines.linewidth": 1.2})

    fig, ax = plt.subplots(4, 4, sharey=True, figsize=(20, 10))
    x_axis_plots = [13, 14, 15, 16]
    y_axis_plots = [1, 5, 9, 13]

    i = 1
    building_order = ['primary_school', 'secondary_school',
                      'hospital', 'outpatient_healthcare',
                      'large_hotel', 'small_hotel',
                      'warehouse',
                      'midrise_apartment',
                      'large_office', 'medium_office', 'small_office',
                      'full_service_restaurant', 'quick_service_restaurant',
                      'stand_alone_retail', 'strip_mall', 'supermarket']
    for building in building_order:
        ax = plt.subplot(4, 4, i)

        subset = df[df.Building == building]

        ###############
        # Scatterplot #
        ###############
        scatter = False
        if scatter is True:
            sns.scatterplot(x='electricity_demand_int', y='heat_demand_int',
                            hue='Climate Zone', style='beta_ABC',
                            data=subset,
                            palette='RdBu_r',
                            s=100)
            i += 1

            ######################
            # Subplot Formatting #
            ######################

            # Text
            ax.set_title(plot_building_dict[building])
            ax.set_xlabel('')
            ax.set_ylabel('')
            # ax.legend([], frameon=False)

            # Ticks #
            #########
            tick_dict = {'primary_school': (np.arange(0, 220, 20), np.arange(0, 600, 100)),
                         'secondary_school': (np.arange(0, 350, 50), np.arange(0, 1600, 200)),
                         'hospital': (np.arange(0, 550, 50), np.arange(0, 2250, 250)),
                         'outpatient_healthcare': (np.arange(0, 500, 50), np.arange(0, 2000, 250)),
                         'large_hotel': (np.arange(0, 300, 50), np.arange(0, 1100, 100)),
                         'small_hotel': (np.arange(0, 220, 20), np.arange(0, 700, 100)),
                         'warehouse': (np.arange(0, 110, 10), np.arange(0, 350, 50)),
                         'midrise_apartment': (np.arange(0, 150, 25), np.arange(0, 400, 50)),
                         'large_office': (np.arange(0, 220, 20), np.arange(0, 700, 100)),
                         'medium_office': (np.arange(0, 220, 20), np.arange(0, 550, 50)),
                         'small_office': (np.arange(0, 220, 20), np.arange(0, 550, 50)),
                         'full_service_restaurant': (np.arange(0, 800, 100), np.arange(0, 2500, 250)),
                         'quick_service_restaurant': (np.arange(0, 1100, 100), np.arange(0, 3000, 250)),
                         'stand_alone_retail': (np.arange(0, 220, 20), np.arange(0, 600, 50)),
                         'strip_mall': (np.arange(0, 220, 20), np.arange(0, 800, 100)),
                         'supermarket': (np.arange(0, 550, 50), np.arange(0, 1100, 100))
                         }
            ax.set_xticklabels(tick_dict[building][0])
            ax.set_xticks(tick_dict[building][0])
            ax.set_yticks(tick_dict[building][1])
            ax.set_yticklabels(tick_dict[building][1] / 1000)

        ############
        # Bar Plot #
        ############
        bar = False
        if bar is True:

            subset.sort_values(by='Climate Zone', inplace=True)
            sns.bar(x='Climate Zone',
                    y='energy_demand_int',
                    hue='beta_ABC',
                    data=subset,
                    palette='muted')

            ######################
            # Subplot Formatting #
            ######################

            # Text
            ax.set_title(plot_building_dict[building])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.legend([], frameon=False)

            # Ticks #
            #########
            tick_dict = {'primary_school': np.arange(0, 600, 100),
                         'secondary_school': np.arange(0, 1400, 200),
                         'hospital': np.arange(0, 2250, 250),
                         'outpatient_healthcare': np.arange(0, 1400, 100),
                         'large_hotel': np.arange(0, 1000, 100),
                         'small_hotel': np.arange(0, 550, 50),
                         'warehouse': np.arange(0, 275, 25),
                         'midrise_apartment': np.arange(0, 350, 50),
                         'large_office': np.arange(0, 550, 50),
                         'medium_office': np.arange(0, 550, 50),
                         'small_office': np.arange(0, 550, 50),
                         'full_service_restaurant': np.arange(0, 2250, 250),
                         'quick_service_restaurant': np.arange(0, 2750, 250),
                         'stand_alone_retail': np.arange(0, 550, 50),
                         'strip_mall': np.arange(0, 550, 50),
                         'supermarket': np.arange(0, 1000, 100)
                         }

            # Y Ticks for absolute values #
            ax.set_yticks(tick_dict[building])
            ax.set_yticklabels(tick_dict[building] / 1000)

            # Y Ticks for Ratio Values #
            # plt.yscale('log')
            # ax.set_ylim([0.01, 10])

            # Axes labels
            if i in x_axis_plots:
                # ax.set_xticklabels(fontsize=10)
                plt.xticks(rotation='vertical')
            else:
                ax.set_xticklabels('')
            if i in y_axis_plots:
                pass
            else:
                # ax.set_yticklabels('')
                pass

            i += 1

        ############
        # Box Plot #
        ############
        box = True
        if box is True:
            subset = subset[subset.alpha_CHP == 1]
            subset.sort_values(by='Climate Zone', inplace=True)
            sns.boxplot(x='Climate Zone',
                        # y='CHP_heat_surplus',
                        y='GHG_int',
                        hue='beta_ABC',
                        data=subset,
                        palette='muted')

            ######################
            # Subplot Formatting #
            ######################

            # Text
            ax.set_title(plot_building_dict[building])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.legend([], frameon=False)

            # Ticks #
            #########
            # For Heat Surplus
            surplus = False
            if surplus is True:
                tick_dict = {'primary_school': np.arange(0, 450, 50),
                             'secondary_school': np.arange(0, 800, 100),
                             'hospital': np.arange(0, 1100, 100),
                             'outpatient_healthcare': np.arange(0, 900, 100),
                             'large_hotel': np.arange(0, 550, 50),
                             'small_hotel': np.arange(0, 450, 50),
                             'warehouse': np.arange(0, 240, 20),
                             'midrise_apartment': np.arange(0, 300, 50),
                             'large_office': np.arange(0, 500, 50),
                             'medium_office': np.arange(0, 450, 50),
                             'small_office': np.arange(0, 450, 50),
                             'full_service_restaurant': np.arange(0, 1750, 250),
                             'quick_service_restaurant': np.arange(0, 2500, 250),
                             'stand_alone_retail': np.arange(0, 500, 50),
                             'strip_mall': np.arange(0, 500, 50),
                             'supermarket': np.arange(0, 1400, 200)
                             }
            GHG = True
            if GHG is True:
                tick_dict = {'primary_school': np.arange(0, 350, 50),
                             'secondary_school': np.arange(0, 500, 50),
                             'hospital': np.arange(0, 900, 100),
                             'outpatient_healthcare': np.arange(0, 800, 100),
                             'large_hotel': np.arange(0, 550, 50),
                             'small_hotel': np.arange(0, 350, 50),
                             'warehouse': np.arange(0, 180, 20),
                             'midrise_apartment': np.arange(0, 220, 20),
                             'large_office': np.arange(0, 350, 50),
                             'medium_office': np.arange(0, 350, 50),
                             'small_office': np.arange(0, 350, 50),
                             'full_service_restaurant': np.arange(0, 1300, 100),
                             'quick_service_restaurant': np.arange(0, 2000, 200),
                             'stand_alone_retail': np.arange(0, 400, 50),
                             'strip_mall': np.arange(0, 400, 50),
                             'supermarket': np.arange(0, 900, 100)
                             }

            # Y Ticks for absolute values #
            ax.set_yticks(tick_dict[building])
            ax.set_yticklabels(tick_dict[building] / 1000)

            # Y Ticks for Ratio Values #
            # plt.yscale('log')
            # ax.set_ylim([0, 1000])

            # Axes labels
            if i in x_axis_plots:
                # ax.set_xticklabels(fontsize=10)
                plt.xticks(rotation='vertical')
            else:
                ax.set_xticklabels('')
            if i in y_axis_plots:
                pass
            else:
                # ax.set_yticklabels('')
                pass

            i += 1

    fig.text(0.05, 0.5, r'Annual GHG Emissions Intensity, $tCO_2eq-m^2$',
                        ha='center', va='center', rotation='vertical', fontsize=14)
    plt.subplots_adjust(wspace=0.15, hspace=0.2)

    # Save Figure
    save_path = 'PhD_Code\\Outputs\\Figures'
    save_file = F'{save_path}\\Heat_Surplus_Bar.png'
    # plt.savefig(save_file, dpi=300)
    print(F'Saved {save_file}')

    plt.show()


def energy_demand_violin_plots():
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    data = pd.read_feather(
        'PhD_Code\\Outputs\\Feather\\Data_normalized_impacts.feather')
    df = data[['City', 'HDD', 'CDD',
               'Building', 'floor_area',
               'beta_ABC', 'electricity_demand_int', 'heat_demand_int', 'cooling_demand_int']]

    df['energy_demand_int'] = df.electricity_demand_int + df.heat_demand_int
    df.drop_duplicates(inplace=True)

    rcParams['font.family'] = 'Times New Roman'
    plt.rc('font', family='serif')

    sns.set_context('paper', rc={"lines.linewidth": 1.2})

    fig, ax = plt.subplots(4, 4, sharey=True, figsize=(10, 15))

    i = 1
    building_order = ['primary_school', 'secondary_school',
                      'hospital', 'outpatient_healthcare',
                      'large_hotel', 'small_hotel',
                      'warehouse',
                      'midrise_apartment',
                      'large_office', 'medium_office', 'small_office',
                      'full_service_restaurant', 'quick_service_restaurant',
                      'stand_alone_retail', 'strip_mall', 'supermarket']
    for building in building_order:
        ax = plt.subplot(4, 4, i)

        subset = df[df.Building == building]

        sns.violinplot(x='Building', y='electricity_demand_int',
                       hue='beta_ABC', data=subset,
                       palette='RdBu_r',
                       despine=True, split=True, inner='quartile')

        sns.violinplot(x='Building', y='heat_demand_int',
                       hue='beta_ABC', data=subset,
                       palette='RdBu_r',
                       despine=True, split=True, inner='quartile')
        i += 1

        ######################
        # Subplot Formatting #
        ######################

        # Text
        ax.set_title(plot_building_dict[building])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend([], frameon=False)

        # Ticks #
        #########
        ax.set_xticklabels('')
        ax.set_xticks([])

        y_tick_dict = {'primary_school': np.arange(0, 600, 100),
                       'secondary_school': np.arange(0, 1600, 200),
                       'hospital': np.arange(0, 2250, 250),
                       'outpatient_healthcare': np.arange(0, 2000, 250),
                       'large_hotel': np.arange(0, 1100, 100),
                       'small_hotel': np.arange(0, 700, 100),
                       'warehouse': np.arange(0, 350, 50),
                       'midrise_apartment': np.arange(0, 400, 50),
                       'large_office': np.arange(0, 700, 100),
                       'medium_office': np.arange(0, 550, 50),
                       'small_office': np.arange(0, 550, 50),
                       'full_service_restaurant': np.arange(0, 2500, 250),
                       'quick_service_restaurant': np.arange(0, 3000, 250),
                       'stand_alone_retail': np.arange(0, 600, 50),
                       'strip_mall': np.arange(0, 800, 100),
                       'supermarket': np.arange(0, 1100, 100)
                       }

        ax.set_yticks(y_tick_dict[building])
        ax.set_yticklabels(y_tick_dict[building] / 1000)

    fig.text(0.05, 0.5, r'Annual Energy Demand Intensity, $MWh-m^2$',
             ha='center', va='center', rotation='vertical', fontsize=14)

    plt.subplots_adjust(wspace=0.3)

    # Save Figure
    save_path = 'PhD_Code\\Outputs\\Figures'
    save_file = F'{save_path}\\Energy_Demand_Violins.png'
    # plt.savefig(save_file, dpi=300)
    print(F'Saved {save_file}')

    plt.show()


def TOC_art(violin=False, box=True, bar=False):
    """
    This function plots the art for the TOC. Guidelines are below:
    Text should usually be limited to the labeling of compounds, reaction arrows, and diagrams. Long
    phrases or sentences should be avoided.
    • Submit the graphic at the actual size to be used for the TOC so that it will fit in an area no larger than
      3.25 inches by 1.75 inches (approx. 8.25 cm by 4.45 cm).
    • Use a sans serif font type such as Helvetica, preferably at 8 pt. but no smaller than 6 pt. Do not make
      the reader strain to read it. The type should be of high quality in order to reproduce well.
    • The graphic file should be saved as either:
      TIFF at 300 dpi for color and at 1200 dpi for black and white.
      EPS in RGB document color mode with all fonts converted to outlines or embedded in the file.
    • Label the graphic “For Table of Contents Only” and provide it on the last page of the submitted
      manuscript.
    """

    import matplotlib.pyplot as plt
    from matplotlib.ticker import (
        MultipleLocator, FormatStrFormatter, AutoMinorLocator)
    import pandas as pd
    import seaborn as sns

    plt.close()

    data = pd.read_feather(
        r'model_outputs\impacts\All_impacts.feather')

    climate_zone_dictionary = {'City': ['albuquerque', 'atlanta', 'baltimore', 'chicago',
                                        'denver', 'duluth', 'fairbanks', 'helena',
                                        'houston', 'las_vegas', 'los_angeles', 'miami',
                                        'minneapolis', 'phoenix', 'san_francisco', 'seattle'],
                               'Climate Zone': ['4B', '3A', '4A', '5A',
                                                '5B', '7', '8', '6B',
                                                '2A', '3B', '3B-CA', '1A',
                                                '6A', '2B', '3C', '4C']}
    climates = pd.DataFrame.from_dict(climate_zone_dictionary)
    df = data[['City', 'HDD', 'CDD',
               'Building', 'floor_area',
               'alpha_CHP', 'beta_ABC', 'technology',
               'electricity_demand_int', 'heat_demand_int', 'cooling_demand_int',
               'GHG_int', 'perc_change_GHG_w_credit']]

    df['energy_demand_int'] = df.electricity_demand_int + df.heat_demand_int
    df['heat_to_power_ratio'] = df.heat_demand_int / df.electricity_demand_int
    df = pd.merge(df, climates)
    # df.drop_duplicates(inplace=True)

    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.5)

    fig, ax = plt.subplots(1, 1, figsize=(6.5 / 2, 3.5))

    ###############
    # Violin Plot #
    ###############
    if violin is True:
        subset = df[df.alpha_CHP == 1]
        subset.sort_values(by='technology', inplace=True)
        sns.violinplot(x='technology',
                       y='perc_change_GHG_w_credit',
                       hue='beta_ABC',
                       data=subset,
                       palette='muted',
                       split=True)

        ##############
        # Formatting #
        ##############
        # Text
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend([], frameon=False)

        # Ticks
        ax.set_yticks(np.arange(-150, 600, 50))

        # Save Figure
        save_path = 'PhD_Code\\Outputs\\Figures'
        save_file = F'{save_path}\\TOC_violin.png'
        plt.savefig(save_file, dpi=300)
        print(F'Saved {save_file}')

        plt.show()

    ############
    # Box Plot #
    ############

    if box is True:
        subset = df[df.alpha_CHP == 1]
        subset.sort_values(by='technology', inplace=True)
        sns.boxplot(x='technology',
                    y='perc_change_GHG_w_credit',
                    hue='beta_ABC',
                    data=subset,
                    palette='muted',
                    showfliers=False)

        ##############
        # Formatting #
        ##############
        # Text
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend([], frameon=False)

        # Ticks
        ax.set_yticks(np.arange(-200, 450, 100))
        ax.set_yticklabels(['' for i in np.arange(-200, 400, 100)])
        ax.set_xticklabels(['', '', '', ''])
        ax.yaxis.set_minor_locator(MultipleLocator(50))
        # ax.set_xticklabels(['Fuel\nCells', 'Gas\nTurbine',
        # 'Micro-\n-turbine', 'Reciproc. \nEngine'])

        # Save Figure
        save_path = 'PhD_Code\\Outputs\\Figures'
        save_file = F'{save_path}\\TOC_box.png'
        plt.savefig(save_file, dpi=300)
        print(F'Saved {save_file}')

        plt.show()

    ############
    # Bar Plot #
    ############

    if bar is True:
        subset = df[(df.alpha_CHP == 0) & (df.alpha_CHP == 1)]
        subset.sort_values(by='technology', inplace=True)
        sns.barplot(x='technology',
                    y='GHG_int',
                    # hue='beta_ABC',
                    data=subset[subset.alpha_CHP == 0])
        sns.barplot(x='technology',
                    y='GHG_int',
                    # hue='beta_ABC',
                    data=subset[subset.alpha_CHP == 1])

        ##############
        # Formatting #
        ##############
        # Text
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend([], frameon=False)

        # Ticks
        ax.set_yticks(np.arange(-150, 550, 50))

        # Save Figure
        save_path = 'model_outputs\\plots'
        save_file = F'{save_path}\\TOC_box.png'
        plt.savefig(save_file, dpi=300)
        print(F'Saved {save_file}')

        plt.show()


def execute_impact_plot(type='impact'):
    if type == 'impact':
        data = pd.read_feather(r'model_outputs\impacts\All_impacts.feather')

        impacts = ['co2_int', 'n2o_int', 'ch4_int',
                'co_int', 'nox_int', 'pm_int', 'so2_int', 'voc_int',
                'GHG_int_100', 'GHG_int_20', 'NG_int']

        for impact in impacts:
            plot_all_impacts(data=data, impact=impact,
                        save=True, show=True, building=None)
    if type == 'percent':
        data = pd.read_feather(r'model_outputs\impacts\percent_change.feather')
        rel_impacts = ['percent_change_co2_int',
                'percent_change_ch4_int',
                'percent_change_n2o_int',
                'percent_change_co_int',
                'percent_change_nox_int',
                'percent_change_pm_int',
                'percent_change_so2_int',
                'percent_change_voc_int',
                'percent_change_GHG_int_100',
                'percent_change_GHG_int_20',
                'percent_change_NG_int'] 

        for impact in rel_impacts:
            plot_all_impacts(data=data, impact=impact,
                    save=True, show=False, building=None)
    if type == 'TFCE':
        building_cat = {  # 'full_service_restaurant': ['full_service_restaurant'],
            # 'hospital': ['hospital'],
            # 'hotel': ['large_hotel', 'small_hotel'],
            # 'midrise_apartment': ['midrise_apartment'],
            # 'office': ['large_office', 'medium_office', 'small_office'],
            # 'outpatient_healthcare': ['outpatient_healthcare'],
            # 'school': ['primary_school', 'secondary_school'],
            # 'quick_service_restaurant': ['quick_service_restaurant'],
            'retail': ['stand_alone_retail', 'strip_mall'],
            # 'supermarket': ['supermarket'],
            # 'warehouse': ['warehouse']
                }

        # for category in building_cat:
        #print(F'{category}: {building_cat[category]}')
        '''subset = data[(data.Building == 'stand_alone_retail')
                        | (data.Building =='strip_mall')]'''
                        # | (data.Building == 'small_office')]
    

#########################
# Running Plot Programs #
#########################
# data = pd.read_feather(r'model_outputs\impacts\percent_change.feather')

execute_impact_plot(type='percent')


# TOC_art(violin=False, box=True, bar=False)
# energy_demand_violin_plots()
# energy_demand_plots()
