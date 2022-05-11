####################################
# LIBRARIES NEEDED TO RUN THE TOOL #
####################################
from logging import error
from msilib.schema import Error
from tkinter import Label
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
from sympy import QQ_gmpy

from sysClasses import *

##########################################################################

# TO DO
#   Function to create databases with Q_e, relative humidity, drybulb temperature, and pressure
def generate_EES_inputs():
    save_path = r'model_outputs\AbsorptionChillers\cooling_demand'
    for city in city_list:
        for building in building_type_list:
            # building_datapath = city_building_dictionary[city][building]
            # climate_datapath = processed_tmy3_dictionary[city]
            City_ = City(city)
            City_._infer_tmy_data()

            Building_ = Building(name=building, building_type=building, City_=City_)

            cooling_demand = Building_.cooling_demand

            ees_df = organize_EES_inputs(cooling_demand, City_.tmy_data)

            cols = ['hour', 'CoolingDemand_kW', 'DryBulb_C', 'Pressure_kPa', 'RHum']
            ees_df = ees_df[cols]

            save_file = F'{save_path}\{city}_{building}_CoolDem.csv'
            ees_df.to_csv(save_file)
            print(F'Saved {city} {building}')

def organize_EES_inputs(building_cooling_demand, climate_data):
    ees_df = climate_data[['DryBulb','RHum','Pressure']].copy()
    ees_df['Pressure_kPa'] = ees_df['Pressure'] / 10
    ees_df['RHum'] = ees_df['RHum'] / 100
    ees_df['CoolingDemand_kW'] = building_cooling_demand
    ees_df.rename(columns={'DryBulb':'DryBulb_C'}, inplace=True)
    ees_df.drop(columns=['Pressure'], inplace=True)

    ees_df['hour'] = np.arange(1, 8761, 1)

    return ees_df

generate_EES_inputs()

