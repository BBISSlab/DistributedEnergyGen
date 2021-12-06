# Data Plotting
import seaborn as sns  # To install: pip install seaborn

#
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MultipleLocator
from pandas.core.indexes import base
import pandas as pd  # To install: pip install pandas
import numpy as np  # To install: pip install numpy
from pyarrow import feather  # Storage format

import scipy.stats
from openpyxl import load_workbook
import math as m


def lin_reg(data, fit_intercept=True):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge

    model = LinearRegression(fit_intercept=fit_intercept)

    # Linear Regression
    model = LinearRegression(fit_intercept=fit_intercept)
    x = data.waterflow_kg_per_hour
    X = x.values.reshape(len(x.index), 1)

    # Perform regression for each impact
    y = data['airflow_kg_per_hour']
    Y = y.values.reshape(len(y.index), 1)

    model.fit(X, Y)
    Y_predicted = model.predict(X)
    if fit_intercept is True:
        regression_dict = {'coef': model.coef_[0][0],
                        'intercept': model.intercept_[0],
                        'score': model.score(X, Y)}
    else:
        regression_dict = {'coef': model.coef_[0][0],
                        'intercept': 0,
                        'score': model.score(X, Y)}

    return regression_dict


def get_regression_results(fit_intercept=True):

    data = pd.read_csv(r'data\Tech_specs\CoolingTower_specs.csv')

    dependent_var = []
    slopes = []
    intercepts = []
    scores = []

    reg_dict = lin_reg(data, fit_intercept)
    dependent_var.append(variable)
    slopes.append(reg_dict['coef'])
    intercepts.append(reg_dict['intercept'])
    scores.append(reg_dict['score'])

    dictionary = {
        'dependent_var': variables,
        'slope': slopes,
        'intercept': intercepts,
        'score': scores}
    df = pd.DataFrame.from_dict(dictionary)

    savepath = r'model_outputs\CCHPvGrid'
    savefile = F'Regression_results_interc_{type}.csv'

    df.to_csv(F'{savepath}\{savefile}')
    print(type)
    print(df)

# get_regression_results(fit_intercept=False, type='MT')


def plot_regression(fit_intercept=False):
    
    data = pd.read_csv(r'data\Tech_specs\CoolingTower_specs.csv')

    ############
    # Plotting #
    ############
    # Close any previous plots
    plt.close()

    # Format fonts and style
    rcParams['font.family'] = 'Helvetica'
    plt.rc('font', family='sans-serif')
    sns.set_style('ticks')#, {'axes.facecolor': '0.99'})
    sns.set_context('paper', rc={"lines.linewidth": 1.2}, font_scale=2.0)


    fig, axn = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9, 7))
    y_ticks = np.arange(0, 2200000, 200000)
    x_ticks = np.arange(0, 70000, 10000)
    ##########################
    # TOP SUBPLOT - PLR Fuel #
    ##########################
    ax = plt.subplot(1, 1, 1)

    sns.scatterplot(x=data['waterflow_kg_per_hour'],
                    y=data['airflow_kg_per_hour'],
                    alpha=0.4,
                    s=80,
                    color='cornflowerblue')

    # Plot Trendline
    regression_dict = lin_reg(data, fit_intercept)
    
    X = np.arange(0, 60100, 100)
    slope = regression_dict['coef']
    intercept = regression_dict['intercept']
    score = regression_dict['score']
    trendline = slope * X + intercept

    sns.lineplot(x=X, y=trendline, color='black', label='y = {:.2f}x + {:.2f}, $R^2$ = {:.3f}'.format(slope, intercept, score))

    # Formatting
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks / 100000)    
    ax.set_ylim(np.min(y_ticks), np.max(y_ticks))
    ax.yaxis.set_minor_locator(MultipleLocator(50000))
    ax.set_ylabel('Air Mass Flowrate $10^5 kg / hr$')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks / 1000)
    ax.set_xlim(np.min(x_ticks), np.max(x_ticks))
    ax.xaxis.set_minor_locator(MultipleLocator(2000))
    ax.set_xlabel('Water Mass Flowrate $10^3 kg / hr$')

    # ax.get_legend().remove()
    sns.despine()
    
    savepath = r'model_outputs\AbsorptionChillers\Figures'
    savefile = F'{savepath}\Cooling_Tower_Reg.png'
    plt.savefig(savefile, dpi=300)
    
    plt.show()

plot_regression(fit_intercept=True)

    