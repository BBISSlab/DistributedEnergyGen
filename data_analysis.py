# Data Plotting
# My modules
import csv
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


def clean_impact_data(data):
    # Duluth Outpatient Heathcare Appears to have an error in calculation
    # Drop from the data
    data.drop(data[(data.Building == 'outpatient_healthcare') &
                   (data.City == 'duluth')].index, inplace=True)

    if 'percent_change_co2_int' in data.columns:
        pass
    else:
        # Convert CO2 and GHGs from g into kg
        for impact in ['co2_int', 'GHG_int_100', 'GHG_int_20']:
            data[impact] = data[impact] / 1000

        for impact in ['TFCE', 'trigen_efficiency']:
            data[impact] = data[impact] * 100
    return data


def lin_reg(data, impact):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge

    # Save data as a dictionary
    if 'percent_change_co2_int' in data.columns:
        regression_dict = {'coef': 0,
                           'intercept': 0,
                           'score': 1}
    else:
        # Linear Regression
        model = LinearRegression(fit_intercept=True)
        x = data.energy_demand_int
        X = x.values.reshape(len(x.index), 1)

        # Select the items you want to get data for
        impacts = ['co2_int', 'n2o_int', 'ch4_int',
                   'co_int', 'nox_int', 'pm_int', 'so2_int', 'voc_int',
                   'GHG_int_100', 'GHG_int_20', 'NG_int']

        # Perform regression for each impact
        y = data[impact]
        Y = y.values.reshape(len(y.index), 1)

        model.fit(X, Y)
        Y_predicted = model.predict(X)

        regression_dict = {'coef': model.coef_[0][0],
                           'intercept': model.intercept_[0],
                           'score': model.score(X, Y)}

    return regression_dict


def get_regression_results(data):
    data = clean_impact_data(data)
    ces_df = data[(data.alpha_CHP == 0) & (data.beta_ABC == 0)].copy()

    impacts = ['co2_int', 'n2o_int', 'ch4_int',
               'co_int', 'nox_int', 'pm_int', 'so2_int', 'voc_int',
               'GHG_int_100', 'GHG_int_20', 'NG_int']

    impact_names = []
    slopes = []
    intercepts = []
    scores = []

    for impact in impacts:
        reg_dict = lin_reg(ces_df, impact)
        impact_names.append(impact)
        slopes.append(reg_dict['coef'])
        intercepts.append(reg_dict['intercept'])
        scores.append(reg_dict['score'])

    dictionary = {
        'impact': impact_names,
        'slope': slopes,
        'intercept': intercepts,
        'score': scores}
    df = pd.DataFrame.from_dict(dictionary)
    df.to_csv(r'model_outputs\testing\Regression_results_w_intercept.csv')
    print(df)


def calculate_building_relative_change(data, city, building, impact):
    # Loop through each city and building
    subset = data[(data.City == city) & (data.Building == building)]
    subset = subset[['City', 'Building', 'PM_id', 'alpha_CHP',
                     'beta_ABC', 'energy_demand_int', impact]].copy()

    baseline_df = subset[(subset.alpha_CHP == 0) & (
        subset.beta_ABC == 0)]
    baseline = baseline_df[impact].mean()
    subset[F'baseline_{impact}'] = baseline
    subset[F'absolute_change_{impact}'] = subset[impact] - \
        subset[F'baseline_{impact}']
    subset[F'percent_change_{impact}'] = (
        subset[F'absolute_change_{impact}'] / subset[F'baseline_{impact}']) * 100

    # subset.drop([impact, 'baseline', 'absolute_change'], axis=1, inplace=True)
    subset.drop(baseline_df.index, inplace=True, axis=0)

    return subset[[F'percent_change_{impact}']].copy()


def calc_city_relative_change(data, city):

    subset = data[data.City == city]

    impacts = ['co2_int', 'n2o_int', 'ch4_int',
               'co_int', 'nox_int', 'pm_int', 'so2_int', 'voc_int',
               'GHG_int_100', 'GHG_int_20', 'NG_int']

    super_x = []
    for building in building_type_list:
        building_df = subset[['City', 'Building', 'PM_id',
                              'alpha_CHP', 'beta_ABC', 'energy_demand_int']].copy()
        for impact in impacts:
            df = calculate_building_relative_change(
                subset, city, building, impact)
            building_df = building_df.merge(df, left_index=True,
                                            how='left', right_index=True)
        super_x.append(building_df)
    city_df = pd.concat(super_x, axis=0)
    city_df.dropna(axis=0, inplace=True)
    return city_df


def calculate_relative_change(data):
    super_y = []
    for city in city_list:
        city_df = calc_city_relative_change(data, city)
        super_y.append(city_df)
    all_percents_df = pd.concat(super_y, axis=0)

    return all_percents_df


def calculate_percent_change(new_value, baseline_value):
    absolute_change = new_value - baseline_value
    percent_change = (absolute_change / baseline_value) * 100

    return percent_change


def leakage_sensitivity():
    import models

    # Load Energy Supply and Baseline Impacts
    supply_data = pd.read_feather(
        r'model_outputs\energy_supply\All_supply_data.feather')
    try:
        baseline_df = pd.read_feather(
            r'model_outputs\impacts\All_impacts.feather')
    except FileNotFoundError:
        baseline_df = models.impacts_sim(supply_data)

    columns_to_copy = ['City', 'Building', 'PM_id',
                       'alpha_CHP', 'beta_ABC',
                       'GHG_int_100',
                       'ch4_leak_int', 'energy_demand_int']
    baseline_df = baseline_df[columns_to_copy].copy()

    #################################
    # ALTERNATE LEAKAGE SIMULATIONS #
    #################################
    # No Leakage
    no_leakage_df = models.impacts_sim(supply_data, leakage_factor=0)
    no_leakage_df = no_leakage_df[columns_to_copy].copy()

    # 10% increased leakage
    higher_leakage_df = models.impacts_sim(supply_data, leakage_factor=1.1)
    higher_leakage_df = higher_leakage_df[columns_to_copy].copy()

    sensitivity_df = merge_leakage_dataframes(
        baseline_df, no_leakage_df, higher_leakage_df)

    # Merge with prime mover dataframe to get technology type
    pm_df = pd.read_csv(r'data\\Tech_specs\\PrimeMover_specs.csv', header=2)
    sensitivity_df = pd.merge(sensitivity_df, pm_df[['PM_id', 'technology']],
                              on='PM_id', how='left').fillna('None')

    ####################
    # CALCULATE % LEAK #
    ####################
    sensitivity_df['percent_leak_base_case'] = sensitivity_df.GHG_leak_base_case / \
        sensitivity_df.GHG_base_case * 100
    sensitivity_df['percent_leak_sensitivity'] = sensitivity_df.GHG_leak_sensitivity / \
        sensitivity_df.GHG_int_sensitivity * 100

    #########################
    # CALCULATE DIFFERENCES #
    #########################

    # Calculate change from baseline emissions
    super_x = []
    for city in baseline_df.City.unique():
        for building in baseline_df.Building.unique():
            ces_df = baseline_df[(baseline_df.City == city)
                                 & (baseline_df.Building == building)
                                 & (baseline_df.alpha_CHP == 0)
                                 & (baseline_df.beta_ABC == 0)]
            subset = sensitivity_df[(higher_leakage_df.City == city)
                                       & (higher_leakage_df.Building == building)]

            subset['GHG_leak_ces'] = models.calculate_GHG(ch4=ces_df['ch4_leak_int']).mean()

            super_x.append(subset)

    sensitivity_df = pd.concat(super_x, axis=0)

    # Calculate % Change from ces
    sensitivity_df['percent_change_leak_base_case_to_ces'] = calculate_percent_change(sensitivity_df.GHG_leak_base_case, sensitivity_df.GHG_leak_ces)
    sensitivity_df['percent_change_leak_to_ces'] = calculate_percent_change(sensitivity_df.GHG_leak_sensitivity, sensitivity_df.GHG_leak_ces)
    sensitivity_df['percent_change_leak_to_base_case'] = calculate_percent_change(sensitivity_df.GHG_leak_sensitivity, sensitivity_df.GHG_leak_base_case)
    stats_df = sensitivity_df.groupby(['technology', 'alpha_CHP', 'beta_ABC'
        ]).agg({'GHG_leak_ces':['mean', 'std'],
            'GHG_base_case':['mean', 'std'],
                'GHG_leak_base_case':['mean', 'std'],
                'percent_leak_base_case':['mean', 'std'],
                'GHG_no_leak':['mean', 'std'],
                'GHG_int_sensitivity':['mean', 'std'], 
                'GHG_leak_sensitivity':['mean', 'std'],
                'percent_leak_sensitivity':['mean', 'std'],
                'percent_change_leak_base_case_to_ces':['mean','std'],
                'percent_change_leak_to_ces':['mean', 'std'],
                'percent_change_leak_to_base_case':['mean', 'std']})

    print(stats_df)
    return sensitivity_df, stats_df


def merge_leakage_dataframes(
        baseline_data, no_leakage_data, increased_leakage_data):
    baseline_data['GHG_int_leakage'] = models.calculate_GHG(
        ch4=baseline_data['ch4_leak_int'])
    baseline_data.rename(columns={'GHG_int_100': 'GHG_base_case',
                                  'GHG_int_leakage': 'GHG_leak_base_case'}, inplace=True)

    no_leakage_data.rename(
        columns={
            'GHG_int_100': 'GHG_no_leak'},
        inplace=True)

    increased_leakage_data['GHG_int_leakage'] = models.calculate_GHG(
        ch4=increased_leakage_data['ch4_leak_int'])
    increased_leakage_data.rename(columns={'GHG_int_100': 'GHG_int_sensitivity',
                                           'GHG_int_leakage': 'GHG_leak_sensitivity'}, inplace=True)

    index_columns = ['City', 'Building', 'PM_id', 'alpha_CHP', 'beta_ABC']
    merged_data = pd.merge(
        baseline_data,
        no_leakage_data,
        left_on=index_columns,
        right_on=index_columns)

    merged_data = pd.merge(
        merged_data,
        increased_leakage_data,
        left_on=index_columns,
        right_on=index_columns)

    merged_data = merged_data[['City', 'Building', 'PM_id',
                               'alpha_CHP', 'beta_ABC',
                               'GHG_base_case', 'GHG_leak_base_case', 'GHG_no_leak',
                               'GHG_int_sensitivity', 'GHG_leak_sensitivity']].copy()

    merged_data.drop_duplicates(inplace=True)
    merged_data.reset_index(inplace=True, drop=True)

    return merged_data


def run_perc_change():
    data = pd.read_feather(r'model_outputs\impacts\All_impacts.feather')
    data = clean_impact_data(data)

    percent_change = calculate_relative_change(data)
    percent_change.reset_index(inplace=True, drop=True)

    pm_df = pd.read_csv(r'data\Tech_specs\PrimeMover_specs.csv', header=2)
    pm_df['CHP_efficiency'] = pm_df[['chp_EFF_LHV', 'chp_EFF_HHV']].max(axis=1)

    percent_change = pd.merge(
        percent_change, pm_df[['PM_id', 'technology']], on='PM_id', how='left').fillna('None')
    percent_change = pd.merge(
        percent_change, pm_df[['PM_id', 'CHP_efficiency']], on='PM_id', how='left').fillna(1)

    percent_change.to_feather(r'model_outputs\impacts\percent_change.feather')
    percent_change.to_csv(r'model_outputs\impacts\percent_change.csv')


def calculate_percent_contribution(numerator, denominator):
    return numerator / denominator * 100


def calculate_ces_reference(data):
    ces_df = data[(data.alpha_CHP == 0) & (data.beta_ABC == 0)].copy()

    super_x = []
    for city in ces_df.City.unique():
        for building in ces_df.Building.unique():
            subset = ces_df[(ces_df.City == city) & (
                data.Building == building)].copy()

            grouped_df = subset.groupby(['City', 'Building',
                                         # These are not needed, but keeping for
                                         # merging purposes
                                         'PM_id', 'alpha_CHP', 'beta_ABC'])

            mean_df = grouped_df.mean()

            super_x.append(mean_df)

    baseline_data = pd.concat(super_x, axis=0)

    return baseline_data.reset_index()


def sim_statistics(raw_data):
    '''
    Need:
    Number of cases in which tech is shown
    Average results and standard dev
    How many reduce the impact relative to the baseline
    '''
    raw_data = clean_impact_data(raw_data)

    pm_df = pd.read_csv(r'data\Tech_specs\PrimeMover_specs.csv', header=2)
    pm_df['CHP_efficiency'] = pm_df[['chp_EFF_LHV', 'chp_EFF_HHV']].max(axis=1)

    raw_data = pd.merge(
        raw_data, pm_df[['PM_id', 'technology']], on='PM_id', how='left').fillna('None')
    raw_data = pd.merge(
        raw_data, pm_df[['PM_id', 'CHP_efficiency']], on='PM_id', how='left').fillna(1)
    raw_data['technology'] = np.where((raw_data.alpha_CHP == 0)
                                      & (raw_data.beta_ABC == 1), 'ABC Only', raw_data.technology)

    means_df = raw_data.groupby(['technology',
                                 # These are not needed, but keeping for
                                 # merging purposes
                                 'alpha_CHP', 'beta_ABC'])
    means_df = means_df.mean()

    std_df = raw_data.groupby(['technology',
                               # These are not needed, but keeping for
                               # merging purposes
                               'alpha_CHP', 'beta_ABC'])
    std_df = std_df.std()

    statistics_df = pd.merge(means_df, std_df,
                             on=['technology', 'alpha_CHP', 'beta_ABC'],
                             how='left',
                             suffixes=['_mean', '_std'],
                             sort=True)

    return statistics_df


def percentage_stats(method='technology', reductions=False):
    percent_data = pd.read_feather(r'model_outputs\impacts\percent_change.feather')
    data = clean_impact_data(percent_data)

    data.drop(['CHP_efficiency', 'energy_demand_int'], axis=1, inplace=True)
    data.drop_duplicates(inplace=True)

    if method == 'technology':
        group = ['alpha_CHP', 'beta_ABC', 'technology']
    if method == 'PM_id':
        group = ['alpha_CHP', 'beta_ABC', 'PM_id']
    
    percents_df = data.groupby(group)
    mean = percents_df.mean()
    std = percents_df.std()

    stats_df = pd.merge(mean, std,
                               on=group,
                               how='left',
                               suffixes=['_mean', '_std'],
                               sort=True)

    # Saving file
    savepath = r'model_outputs\impacts'
    savefile = F'avg_percents_{method}.csv'
    stats_df.to_csv(F'{savepath}\{savefile}')

    if reductions is True:
        impacts = ['percent_change_co2_int',
                'percent_change_n2o_int', 'percent_change_ch4_int',
                'percent_change_co_int', 'percent_change_nox_int', 'percent_change_pm_int',
                'percent_change_so2_int', 'percent_change_voc_int',
                'percent_change_GHG_int_100', 'percent_change_GHG_int_20', 'percent_change_NG_int']

        # totals = data.groupby(['alpha_CHP', 'beta_ABC', 'technology'])
        # print(totals.count())
        super_y = []
        for impact in impacts:
            reductions = data[['alpha_CHP', 'beta_ABC', 'technology', impact]].copy()
            reductions = reductions[reductions[impact] < -0.005]
            grouped_df = reductions.groupby(['alpha_CHP', 'beta_ABC', 'technology'])
            super_y.append(grouped_df.count())
            # print(grouped_df.count())

        reductions_df = pd.concat(super_y, axis=1)
        reductions_df.fillna(value=0, inplace=True)
        # print(reductions_df)
        return reductions_df


def GWP_sensitivity(data):
    # Load CHP Dataframe
    pm_df = pd.read_csv('data\\Tech_specs\\PrimeMover_specs.csv', header=2)

    # Copy only relevant data
    GHG_df = data[['PM_id', 'alpha_CHP', 'beta_ABC',
                   'co2_int', 'ch4_int', 'n2o_int']].copy()

    GHG_df = pd.merge(
        GHG_df, pm_df[['PM_id', 'technology']], on='PM_id', how='left').fillna('None')

    GHG_df['GHG_int_100'] = models.calculate_GHG(
        co2=GHG_df.co2_int,
        ch4=GHG_df.ch4_int,
        n2o=GHG_df.n2o_int,
        GWP_factor=1)
    GHG_df['GHG_int_10%'] = models.calculate_GHG(
        co2=GHG_df.co2_int,
        ch4=GHG_df.ch4_int,
        n2o=GHG_df.n2o_int,
        GWP_factor=1.1)

    GHG_df['delta_GHG'] = GHG_df['GHG_int_10%'] - GHG_df.GHG_int_100
    GHG_df['percent_change'] = GHG_df.delta_GHG / GHG_df.GHG_int_100 * 100

    GHG_df.drop(['co2_int', 'ch4_int', 'n2o_int'], inplace=True, axis=1)

    GHG_df = GHG_df.groupby(['technology', 'alpha_CHP', 'beta_ABC']).agg({
        'GHG_int_100': ['mean', 'std'],
        'GHG_int_10%': ['mean', 'std'],
        'delta_GHG': ['mean', 'std'],
        'percent_change': ['mean', 'std']})

    return GHG_df

##########################
# Generate CES Reference #
##########################


##############
# Statistics #
##############
# Average and Standard Deviation
# raw_data = pd.read_feather(r'model_outputs\impacts\All_impacts.feather')

# stats = sim_statistics(raw_data)
# stats.to_csv(r'model_outputs\impacts\statistics.csv')

# Percent stats
# percentage_stats(method='PM_id')

# run_perc_change()

#######################
# Leakage Sensitivity #
#######################
# sensitivity, sensitivity_stats= leakage_sensitivity()
# sensitivity.to_csv(r'model_outputs\testing\leakage_sensitivity.csv')
# sensitivity_stats.to_csv(r'model_outputs\testing\leakage_stats.csv')

###################
# GWP Sensitivity #
###################
def run_gwp_sensitivity():
    data = pd.read_feather(r'model_outputs\impacts\All_impacts.feather')
    GWP_df = GWP_sensitivity(data)
    GWP_df.to_csv(r'model_outputs\impacts\GWP_sensitivity.csv')

run_gwp_sensitivity()