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


def organize_EES_outputs(city_name):
    filepath = r'model_outputs\AbsorptionChillers\cooling_supply'
    filename = F'{filepath}\{city_name}.csv'

    building_path = r'model_outputs\AbsorptionChillers\cooling_demand'
    building_file = F'{building_path}\{city_name}_hospital_CoolDem.csv'

    building_df = pd.read_csv(building_file, index_col='datetime')

    cols = ['Qe', 'T_db', 'Patm', 'RH', 'Qfrac', 'Qheat_kW', 'Welec_kW', 'makeup_water_kg_per_s']
    df = pd.read_csv(filename, names=cols)

    try:
        df['datetime'] = building_df.index
        df.set_index('datetime', inplace=True, drop=True)
    except ValueError:
        print(F'{city_name} has mismatched indices')
    return df

def clean_EES_outputs():
    cities = city_list
    cities.remove('fairbanks')
    
    filepath = r'model_outputs\AbsorptionChillers\cooling_supply'
    
    for city in cities:        
        df = organize_EES_outputs(city)
        df.to_csv(F'{filepath}\{city}_supply.csv')


def calculate_EES_building_output(city_df, building_df):
    city_df['datetime'] = building_df['datetime']
    city_df.set_index('datetime', inplace=True, drop=True)

    supply_df = building_df.copy()
    supply_df.set_index('datetime', inplace=True, drop=True)

    supply_df['Qfrac'] = supply_df.CoolingDemand_kW / city_df.Qe
    supply_df['AbsCh_HeatDemand_kW'] = city_df.Qheat_kW * supply_df.Qfrac
    supply_df['AbsCh_ElecDemand_kW'] = city_df.Welec_kW * supply_df.Qfrac
    supply_df['MakeupWater_kph'] = city_df.makeup_water_kg_per_s * supply_df.Qfrac * 3600
    supply_df.drop(columns=['hour'], inplace=True)

    return supply_df

def annual_building_sim():
    demand_path = r'model_outputs\AbsorptionChillers\cooling_demand'
    supply_path = r'model_outputs\AbsorptionChillers\cooling_supply'

    cities = city_list
    cities.remove('fairbanks')
        
    for city in cities:
        for building in building_type_list:
            city_df = pd.read_csv(F'{supply_path}\{city}_supply.csv')
            demand_df = pd.read_csv(F'{demand_path}\{city}_{building}_CoolDem.csv')
            supply_df = calculate_EES_building_output(city_df, demand_df)

            #supply_df.to_feather()
