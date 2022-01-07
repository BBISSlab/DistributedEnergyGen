####################################
# LIBRARIES NEEDED TO RUN THE TOOL #
####################################
from logging import error
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
import seaborn as sns  # To install: pip install seaborn

# PV library and classes for Object-oriented
import pvlib  # [1] To install: pip install pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem

# Thermofluids modules
from iapws import iapws97
from iapws import humidAir
##########################################################################

# Properties of Humid Air - might make into a class later

def convert_mbar_to_kPa(pressure):
    # 10 millibar = 1 kPa
    return pressure / 10

def convert_C_to_K(temperature):
    return temperature + 273.15

def calculate_abs_humidity(RH=0, P_atm=101.325, temperature=0):
    # RH should be a fraction
    # Pressure values must be in kPa
    # Temperature input in Celsius
    if (RH > 1) or (RH < 0):
        print('Enter value between 0 and 1')
        exit()
    temp_K = convert_C_to_K(temperature)
    P_sat = calculate_P_sat(temp_K)
    abs_humidity = (0.622 * RH * P_sat) / (P_atm - P_sat)
    return abs_humidity

def calculate_P_sat(temperature):
    # temperature must be in Kelvin
    P_sat_MPa = iapws97._PSat_T(temperature)
    P_sat_kPa = P_sat_MPa * 1000
    return P_sat_kPa

def calculate_enthalpy():
    # Create Humid Air Class from iawps
    # 
    pass

'''
Humid air class with complete functionality

Parameters:	
T (float) – Temperature, [K]
P (float) – Pressure, [MPa]
rho (float) – Density, [kg/m³]
v (float) – Specific volume, [m³/kg]
A (float) – Mass fraction of dry air in humid air, [kg/kg]
xa (float) – Mole fraction of dry air in humid air, [-]
W (float) – Mass fraction of water in humid air, [kg/kg]
xw (float) – Mole fraction of water in humid air, [-]
HR (float) – Humidity ratio, Mass fraction of water in dry air, [kg/kg]
Notes

It needs two incoming properties of T, P, rho.
v as a alternate input parameter to rho
For composition need one of A, xa, W, xw.
'''

w = calculate_abs_humidity(RH=0.4, temperature=25)

Humid_Air = humidAir.HumidAir(T = 298.15, P = 0.101325, W=w)
RH = Humid_Air.RH

print(RH)

