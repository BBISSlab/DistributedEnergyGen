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

def plot_all_impacts(impact, negatives=False, save_name=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.ticker import (
        MultipleLocator, FormatStrFormatter, AutoMinorLocator)

    # Get Data
    data = pd.read_feather(
        'model_outputs\impacts\Impacts.feather')
    df['energy_demand_int'] = df.electricity_demand_int + df.heat_demand_int

    df.drop(df[(df.Building == 'outpatient_healthcare') &
               (df.City == 'duluth')].index, inplace=True)

    # For Now, Only keep data where alpha_CHP == 1 and beta_ABC = 1
    df.drop(df[(df.alpha_CHP == 0)].index, inplace=True)

    baseline_df = pd.read_feather(
        'PhD_Code\Outputs\Feather\Baseline_data.feather')
    baseline_df['energy_demand_int'] = baseline_df.electricity_demand_int + \
        baseline_df.heat_demand_int
    baseline_df.drop(baseline_df[(baseline_df.Building == 'outpatient_healthcare') &
                                 (baseline_df.City == 'duluth')].index, inplace=True)

    ############
    # Plotting #
    ############
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')
    sns.set_style('ticks', {'axes.facecolor': '0.8'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.3)
    for i in [1]:  # building in df.Building.unique():
        plt.close()

        if negatives is True:
            df = df[df[impact] <= 0]
            tick_dict = {
                ################
                # Percent Change
                ################
                'perc_change_co2': np.arange(-100, 10, 10),
                'perc_change_co2_w_credit': np.arange(-200, 500, 50),
                'perc_leak_ch4': np.arange(-100, 10, 10),
                'perc_leak_ch4_w_credit': np.arange(0, 5.5, 0.5),
                'perc_change_ch4': np.arange(-100, 10, 10),
                'perc_change_ch4_w_credit': np.arange(-100, 175, 25),
                'perc_change_n2o': np.arange(-100, 10, 10),
                'perc_change_n2o_w_credit': np.arange(-100, 500, 50),
                'perc_change_co': np.arange(-100, 10, 10),
                'perc_change_co_w_credit': np.arange(-400, 550, 50),
                'perc_change_nox': np.arange(-250, 1750, 250),
                'perc_change_nox_w_credit': np.arange(-1000, 1750, 250),
                'perc_change_pm': np.arange(0, 100, 10),
                'perc_change_pm_w_credit': np.arange(0, 100, 10),
                'perc_change_so2': np.arange(0, 100, 10),
                'perc_change_so2_w_credit': np.arange(0, 100, 10),
                'perc_change_voc': np.arange(-250, 1750, 250),
                'perc_change_voc_w_credit': np.arange(-250, 1500, 250),
                'perc_change_w4e': np.arange(0, 100, 10),
                'perc_change_w4e_w_credit': np.arange(0, 100, 10),
                'perc_leak_GHG': np.arange(0, 100, 10),
                'perc_leak_GHG_w_credit': np.arange(0, 100, 10),
                'perc_change_GHG': np.arange(-100, 900, 100),
                'perc_change_GHG_w_credit': np.arange(-200, 700, 100),
                'perc_change_Fuel_int': np.arange(-40, 160, 20),
                'perc_change_Fuel_int_w_credit': np.arange(-40, 150, 10)
            }

        # Making the FacetGrid
        g = sns.FacetGrid(df,
                          col="beta_ABC",  # row="beta_ABC", margin_titles=True,
                          hue="chp_EFF",
                          palette='YlOrRd',
                          despine=True)
        g.map(sns.scatterplot,
              'energy_demand_int',
              impact,
              style=df.technology,
              markers={'Fuel Cell': 'P', 'Reciprocating Engine': 's',
                       'Gas Turbine': 'X', 'Microturbine': '.'},
              alpha=0.4,
              s=80
              )

        X = np.arange(0, 2600, 100)
        trendlines = {'GHG_int_kg_m^-2_w_credit': 0.2675 * X,
                      'nox_int_g_m^-2_w_credit': 0.0788 * X,
                      'voc_int_g_m^-2_w_credit': 0.0078 * X,
                      'Fuel_int_total_w_credit': 1.6166 * X}
        g.map(sns.lineplot,
              x=X,
              y=trendlines[impact],
              color='black'
              # markers={'D'},
              # alpha=0.4,
              # s=40
              )

        g.set_axis_labels('',  # X-axis
                          ''  # y-axis
                          )

        # g.add_legend()

        # Formatting FacetGrid
        if negatives is False:
            tick_dict = {
                ################
                # Percent Change
                ################
                'perc_change_co2': np.arange(-200, 700, 100),
                'perc_change_co2_w_credit': np.arange(-150, 800, 50),
                'perc_leak_ch4': np.arange(0, 100, 10),
                'perc_leak_ch4_w_credit': np.arange(0, 100, 10),
                'perc_change_ch4': np.arange(0, 100, 10),
                'perc_change_ch4_w_credit': np.arange(-50, 175, 25),
                'perc_change_n2o': np.arange(0, 100, 10),
                'perc_change_n2o_w_credit': np.arange(-130, 20, 10),
                'perc_change_co': np.arange(0, 100, 10),
                'perc_change_co_w_credit': np.arange(-350, 500, 50),
                'perc_change_nox': np.arange(-250, 1750, 250),
                'perc_change_nox_w_credit': np.arange(-1000, 1750, 250),
                'perc_change_pm': np.arange(0, 100, 10),
                'perc_change_pm_w_credit': np.arange(-140, 20, 20),
                'perc_change_so2': np.arange(0, 100, 10),
                'perc_change_so2_w_credit': np.arange(-140, 20, 20),
                'perc_change_voc': np.arange(-250, 1750, 250),
                'perc_change_voc_w_credit': np.arange(-300, 1400, 100),
                'w4e_int_g_m^-2': np.arange(0, 100, 10),
                'w4e_int_g_m^-2_w_credit': np.arange(0, 100, 10),
                'perc_change_w4e': np.arange(0, 100, 10),
                'perc_change_w4e_w_credit': np.arange(0, 100, 10),
                'GHG_leak_total': np.arange(0, 100, 10),
                'perc_leak_GHG': np.arange(0, 100, 10),
                'perc_leak_GHG_w_credit': np.arange(0, 100, 10),
                'perc_change_GHG': np.arange(-100, 900, 100),
                'perc_change_GHG_w_credit': np.arange(-150, 550, 50),
                ############
                # Normalized
                ############
                #  GHGs
                'co2 (kg kWh^-1 m^-2)': np.arange(0, 10**4, 100),
                'co2_w_credit (kg kWh^-1 m^-2)': np.arange(0, 10**4, 100),
                'ch4 (g kWh^-1 m^-2)':  np.arange(0, 10**4, 100),
                'ch4_w_credit (g kWh^-1 m^-2)':  np.arange(0, 10**4, 100),
                'n2o (g kWh^-1 m^-2)': np.arange(0, 10, 1),
                'n2o_w_credit (g kWh^-1 m^-2)': np.arange(0, 10, 1),
                # CAPs
                'co (g kWh^-1 m^-2)': np.arange(0, 10**3, 100),
                'co_w_credit (g kWh^-1 m^-2)': np.arange(0, 10**3, 100),
                'nox (g kWh^-1 m^-2)': np.arange(0, 10**3, 10),
                'nox_w_credit (g kWh^-1 m^-2)': np.arange(0, 10**3, 10),
                'pm (g kWh^-1 m^-2)': np.arange(0, 10, 1),
                'pm_w_credit (g kWh^-1 m^-2)': np.arange(0, 10, 1),
                'so2 (g kWh^-1 m^-2)': np.arange(0, 1, 0.1),
                'so2_w_credit (g kWh^-1 m^-2)': np.arange(0, 1, 0.1),
                'voc (g kWh^-1 m^-2)': np.arange(0, 100, 10),
                'voc_w_credit (g kWh^-1 m^-2)': np.arange(0, 100, 10),
                'GHG (kg kWh^-1 m^-2)': np.arange(0, 10**4, 100),
                'GHG_w_credit (kg kWh^-1 m^-2)': np.arange(0, 100, 10),
                'Fuel (kWh kWh^-1 m^-2)': np.arange(0, 100, 10),
                #################
                # Absolute values
                #################
                # Without Credit
                'co2_int_kg_m^-2': np.arange(0, 275, 25),
                'ch4_int_g_m^-2': np.arange(0, 850, 50),
                'n2o_int_g_m^-2': np.arange(-0.25, 1.75, 0.25),
                'co_int_g_m^-2': np.arange(0, 45, 5),
                'nox_int_g_m^-2': np.arange(0, 35, 5),
                'pm_int_g_m^-2': np.arange(-1, 3.5, 0.5),
                'so2_int_g_m^-2': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'voc_int_g_m^-2': np.arange(0, 16, 1),
                'GHG_int_kg_m^-2': np.arange(0, 325, 25),
                'Fuel_int_total': np.arange(0, 650, 50),
                # With CHP Credit
                'CHP_heat_surplus': np.arange(0, 275, 25),
                'co2_int_kg_m^-2_w_credit': np.arange(-100, 1700, 100),
                'ch4_int_g_m^-2_w_credit': np.arange(0, 8500, 500),
                'n2o_int_g_m^-2_w_credit': np.arange(-2, 11, 1),
                'co_int_g_m^-2_w_credit': np.arange(-200, 350, 50),
                'nox_int_g_m^-2_w_credit': np.arange(-200, 300, 50),
                'pm_int_g_m^-2_w_credit': np.arange(-6, 24, 2),
                'so2_int_g_m^-2_w_credit': np.arange(-2, 6.5, 0.5),
                'voc_int_g_m^-2_w_credit': np.arange(-10, 110, 10),
                'GHG_int_kg_m^-2_w_credit': np.arange(-200, 2000, 200),
                'Fuel_int_total_w_credit': np.arange(0, 4500, 500),
                # Fuel and TFCE
                'Fuel_Grid_int': np.arange(0, 100, 10),
                'Fuel_CHP_int': np.arange(0, 100, 10),
                'Fuel_Furnace_int': np.arange(0, 100, 10),
                'perc_change_Fuel_int': np.arange(-40, 160, 20),
                'perc_change_Fuel_int_w_credit': np.arange(-40, 150, 10),
                'TFCE': np.arange(30, 110, 10),
                'TFCE_w_credit': np.arange(50, 105, 5)}

            minorticks_dict = {
                #################
                # Absolute values
                #################
                # With CHP Credit
                'CHP_heat_surplus': np.arange(0, 275, 25),
                'co2_int_kg_m^-2_w_credit': np.arange(-100, 1700, 100),
                'ch4_int_g_m^-2_w_credit': np.arange(0, 8500, 500),
                'n2o_int_g_m^-2_w_credit': np.arange(-2, 11, 1),
                'co_int_g_m^-2_w_credit': np.arange(-200, 350, 50),
                'nox_int_g_m^-2_w_credit': 25,
                'pm_int_g_m^-2_w_credit': np.arange(-6, 24, 2),
                'so2_int_g_m^-2_w_credit': np.arange(-2, 6.5, 0.5),
                'voc_int_g_m^-2_w_credit': 5,
                'GHG_int_kg_m^-2_w_credit': 100,
                'Fuel_int_total_w_credit': 250,
                # Fuel and TFCE
                'Fuel_Grid_int': np.arange(0, 100, 10),
                'Fuel_CHP_int': np.arange(0, 100, 10),
                'Fuel_Furnace_int': np.arange(0, 100, 10),
                'perc_change_Fuel_int': np.arange(-40, 160, 20),
                'perc_change_Fuel_int_w_credit': np.arange(-40, 150, 10),
                'TFCE': np.arange(30, 110, 10),
                'TFCE_w_credit': np.arange(50, 105, 5)}
        g.fig.subplots_adjust(wspace=0.1, hspace=0.05)
        g.set_titles(
            col_template='',  # 'alpha = {col_name}',
            row_template='')  # 'beta = {row_name}')
        try:
            y_min = np.min(tick_dict[impact])
            y_max = np.max(tick_dict[impact])

            if impact in ['co2_int_kg_m^-2_w_credit', 'GHG_int_kg_m^-2_w_credit',
                          'co2_int_kg_m^-2', 'GHG_int_kg_m^-2',
                          'ch4_int_g_m^-2_w_credit', 'c44_int_g_m^-2',
                          'Fuel_int_total_w_credit']:
                yticklabels = tick_dict[impact] / 1000
            else:
                yticklabels = tick_dict[impact]
            g.set(xlim=(0, 2500),
                  ylim=(y_min, y_max),
                  yticks=tick_dict[impact],
                  yticklabels=yticklabels,
                  xticks=np.arange(0, 2600, 100),
                  xticklabels=[0, '', '', '', '', 0.5, '', '', '', '', 1.0,
                               '', '', '', '', 1.5, '', '', '', '', 2.0,
                               '', '', '', '', 2.5])

        except KeyError:
            g.set(xlim=(0, 2500),
                  xticks=np.arange(0, 2600, 100),
                  xticklabels=[0, '', '', '', '', 0.5, '', '', '', '', 1.0,
                               '', '', '', '', 1.5, '', '', '', '', 2.0,
                               '', '', '', '', 2.5])

        g.set(xlim=(0, 2500),
              xticks=np.arange(0, 3000, 500),
              xticklabels=[0, 0.5, 1.0, 1.5, 2.0, 2.5])

        g.axes[0, 0].yaxis.set_minor_locator(
            MultipleLocator(minorticks_dict[impact]))
        g.axes[0, 0].xaxis.set_minor_locator(MultipleLocator(100))

        if save_name is not None:
            save_path = 'PhD_Code\Outputs\Figures'
            save_file = F'{save_path}\Edited_{save_name}.png'
            plt.savefig(save_file, dpi=300)

        plt.show()


impacts = ['GHG_int_kg_m^-2_w_credit', 'nox_int_g_m^-2_w_credit', 'voc_int_g_m^-2_w_credit',
           'Fuel_int_total_w_credit']
name = {'GHG_int_kg_m^-2_w_credit': 'GHG_int_w_credit', 'nox_int_g_m^-2_w_credit': 'nox_int_w_credit', 'voc_int_g_m^-2_w_credit': 'voc_int_w_credit',
        'Fuel_int_total_w_credit': 'Fuel_int_w_credit'}
# for i in impacts:
i = impacts[2]
plot_all_impacts(i,
                 negatives=False,
                 save_name=name[i])


def energy_demand_plots():
    """
    This function plots the distribution of energy demands for all buildings
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    plt.close()

    data = pd.read_feather(
        'PhD_Code\Outputs\Feather\Data_normalized_impacts.feather')

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
               'CHP_heat_surplus', 'alpha_CHP', 'GHG_int_kg_m^-2_w_credit']]

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
            ax.set_yticklabels(tick_dict[building][1]/1000)

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
            ax.set_yticklabels(tick_dict[building]/1000)

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
                        y='GHG_int_kg_m^-2_w_credit',
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
            ax.set_yticklabels(tick_dict[building]/1000)

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
    save_path = 'PhD_Code\Outputs\Figures'
    save_file = F'{save_path}\Heat_Surplus_Bar.png'
    # plt.savefig(save_file, dpi=300)
    print(F'Saved {save_file}')

    plt.show()


# energy_demand_plots()


def energy_demand_violin_plots():
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    data = pd.read_feather(
        'PhD_Code\Outputs\Feather\Data_normalized_impacts.feather')
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
        ax.set_yticklabels(y_tick_dict[building]/1000)

    fig.text(0.05, 0.5, r'Annual Energy Demand Intensity, $MWh-m^2$',
             ha='center', va='center', rotation='vertical', fontsize=14)

    plt.subplots_adjust(wspace=0.3)

    # Save Figure
    save_path = 'PhD_Code\Outputs\Figures'
    save_file = F'{save_path}\Energy_Demand_Violins.png'
    # plt.savefig(save_file, dpi=300)
    print(F'Saved {save_file}')

    plt.show()


# energy_demand_violin_plots()

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
        'PhD_Code\Outputs\Feather\Compiled_data_sample.feather')

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
               'GHG_int_kg_m^-2_w_credit', 'perc_change_GHG_w_credit']]

    df['energy_demand_int'] = df.electricity_demand_int + df.heat_demand_int
    df['heat_to_power_ratio'] = df.heat_demand_int / df.electricity_demand_int
    df = pd.merge(df, climates)
    # df.drop_duplicates(inplace=True)

    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')

    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=1.5)

    fig, ax = plt.subplots(1, 1, figsize=(6.5/2, 3.5))

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
        save_path = 'PhD_Code\Outputs\Figures'
        save_file = F'{save_path}\TOC_violin.png'
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
        save_path = 'PhD_Code\Outputs\Figures'
        save_file = F'{save_path}\TOC_box.png'
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
                    y='GHG_int_kg_m^-2_w_credit',
                    # hue='beta_ABC',
                    data=subset[subset.alpha_CHP == 0])
        sns.barplot(x='technology',
                    y='GHG_int_kg_m^-2_w_credit',
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
        save_path = 'PhD_Code\Outputs\Figures'
        save_file = F'{save_path}\TOC_box.png'
        plt.savefig(save_file, dpi=300)
        print(F'Saved {save_file}')

        plt.show()


# TOC_art(violin=False, box=True, bar=False)
