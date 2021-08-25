from pvlib.inverter import sandia
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.spa import solar_position

# from openpyxl import load_workbook
# import pathlib
from IPython.display import display

import pandas as pd
import math as m
import numpy as np

import pvlib

# Writing format
import pyarrow

# Financing
def monthly_payment(principal, interest_rate, time_of_loan):
    # interest rate in % / yr
    numerator = principal * (interest_rate / 12) * ((1 + interest_rate / 12) ** (12 * time_of_loan))
    denominator = ((1 + interest_rate / 12) ** (12 * time_of_loan)) - 1
    return numerator / denominator

def annual_depreciation(capital_cost, salvage_value, lifetime):
    return (capital_cost - salvage_value) / lifetime

def life_cycle_cost(annual_costs, discount_rate, total_savings):
    
    pass

def discounted_value(value, year, discount_rate):
    # Calculate the discount factor
    discount_factor = (1 + discount_rate) ** year
    # multiply the cost at the time * discount factor
    discounted_value = value / discount_factor
    return discounted_value

def net_present_value(discounted_values):
    if isinstance(discounted_values, pd.Series):
        return discounted_values.sum()
    if isinstance(discounted_values, list):
        return sum(discounted_values)

def levelized_cost_of_energy(net_present_value, net_present_energy):
    return net_present_value / net_present_energy