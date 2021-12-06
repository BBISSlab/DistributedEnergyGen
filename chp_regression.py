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


def lin_reg(data, variable, fit_intercept=True):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge

    model = LinearRegression(fit_intercept=fit_intercept)

    # Linear Regression
    model = LinearRegression(fit_intercept=fit_intercept)
    x = data.PLR_E
    X = x.values.reshape(len(x.index), 1)

    # Select the items you want to get data for
    variables = ['PLR_H', 'PLR_F']

    # Perform regression for each impact
    y = data[variable]
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


def get_regression_results(fit_intercept=True, type='MT'):
    variables = ['PLR_H', 'PLR_F']

    if type == 'MT':
        data = pd.read_csv(r'data\Tech_specs\Microturbine_PLR_data.csv')
    elif type == 'FC':
        data = pd.read_csv(r'data\Tech_specs\PAFC_PLR_data.csv')

    dependent_var = []
    slopes = []
    intercepts = []
    scores = []

    for variable in variables:
        reg_dict = lin_reg(data, variable, fit_intercept)
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


def plot_regression(type='MT', fit_intercept=False):
    if type == 'MT':
        data = pd.read_csv(r'data\Tech_specs\Microturbine_PLR_data.csv')
    elif type == 'FC':
        data = pd.read_csv(r'data\Tech_specs\PAFC_PLR_data.csv')

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


    fig, axn = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(7, 8))
    y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_ticks = [0, 0.2,0.4, 0.6, 0.8,  1.0]
    ##########################
    # TOP SUBPLOT - PLR Fuel #
    ##########################
    ax = plt.subplot(2, 1, 1)

    sns.scatterplot(x=data['PLR_E'],
                    y=data['PLR_F'],
                    alpha=0.4,
                    s=80,
                    color='indigo')

    # Plot Trendline
    regression_dict = lin_reg(data, 'PLR_F', False)
    
    X = np.arange(0, 1, 0.02)
    slope = regression_dict['coef']
    intercept = regression_dict['intercept']
    score = regression_dict['score']
    trendline = slope * X + intercept

    sns.lineplot(x=X, y=trendline, color='black', label='y = {:.3f}x, $R^2$ = {:.3f}'.format(slope, score))

    # Formatting
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)    
    ax.set_ylim(np.min(y_ticks), np.max(y_ticks))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_ylabel('$F$ / $F_{nominal}$')

    ax.set_xticklabels('')
    ax.set_xlabel('')

    ax.legend(frameon=False, loc=2)

    ##########################
    # BOT SUBPLOT - PLR Heat #
    ##########################
    ax2 = plt.subplot(2, 1, 2)

    sns.scatterplot(x=data['PLR_E'],
                    y=data['PLR_H'],
                    alpha=0.4,
                    s=80,
                    color = 'orangered')

    # Plot Trendline
    regression_dict = lin_reg(data, 'PLR_H', False)
    
    X = np.arange(0, 1, 0.02)
    slope = regression_dict['coef']
    intercept = regression_dict['intercept']
    score = regression_dict['score']
    trendline = slope * X + intercept

    sns.lineplot(x=X, y=trendline, color='black', label='y = {:.3f}x, $R^2$ = {:.3f}'.format(slope, score))

    # Formatting
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_ticks)
    
    ax2.set_ylim(np.min(y_ticks), np.max(y_ticks))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax2.set_ylabel('$Q$ / $Q_{nominal}$')

    ax2.legend(frameon=False, loc=2)

    sns.despine(fig)

    for axis in [ax, ax2]:
        axis.set_xlim(0, 1)
        axis.set_xticks(x_ticks)
        axis.xaxis.set_minor_locator(MultipleLocator(0.05))
        axis.set_xticklabels('')
        axis.set_xlabel('')
    
    ax2.set_xticklabels(x_ticks)
    ax2.set_xlabel('$E$ / $E_{nominal}$')

    plt.subplots_adjust(hspace = 0.1)
    
    savepath = r'model_outputs\CCHPvGrid\figures'
    savefile = F'{savepath}\Regression_{type}.png'
    plt.savefig(savefile, dpi=300)
    
    plt.show()

plot_regression(type='FC')

    