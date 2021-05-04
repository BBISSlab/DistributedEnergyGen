# Data Plotting
# My modules
import csv
from sysClasses import *
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


def leakage_sensitivity(data):
    pass


def run_perc_change():
    data = pd.read_feather(r'model_outputs\impacts\All_impacts.feather')

    percent_change = calculate_relative_change(data)
    percent_change.reset_index(inplace=True, drop=True)

    percent_change.to_feather(r'model_outputs\impacts\percent_change.feather')

run_perc_change()