# My modules
from sysClasses import *
from models import *
# 3rd Party Modules
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
import pvlib  # [1]
import pandas as pd  # To install: pip install pandas
import numpy as np  # To install: pip install numpy
from pyarrow import feather  # Storage format
#
import scipy.optimize
import scipy.stats
import pathlib
from openpyxl import load_workbook
import math as m
import time
import datetime
import inspect
import os


def execute_energy_demand_sim():
    all_cities = int(input('All cities?:\n 1) True\n 2) False\n'))
    '''
    OBJECT GENERATOR
    ----------------
    '''
    if all_cities == 1:
        system_dict = generate_objects(all_cities=True)
    else:
        cities_to_simulate = []
        n = int(input('Enter number of cities: '))
        for i in range(0, n):
            city_input = str(input(
                'Type in the name of the {}th city: '.format(i+1)))
            cities_to_simulate.append(city_input)
        print('Simulating the following list: {}'.format(cities_to_simulate))
        system_dict = generate_objects(
            all_cities=False, selected_cities=cities_to_simulate)

    City_dict = system_dict['City_dict']
    Grid_dict = system_dict['Grid_dict']
    PrimeMover_dict = system_dict['PrimeMover_dict']
    BES_dict = system_dict['BatteryStorage_dict']
    Furnace_dict = system_dict['Furnace_dict']
    AC_dict = system_dict['AC_dict']
    ABC_dict = system_dict['ABC_dict']

    # Drop residential AC units and exhaust-fired ABCs
    AC_drop = ['AC1', 'AC2', 'CH1', 'CH2', 'CH3', 'CH4']
    ABC_drop = ['ABC_SS1', 'ABC_SS3', 'ABC_TS1', 'ABC_TS2', 'ABC_TS3', 'ABC_TS4']
    for key in AC_drop:
        AC_dict.pop(key)
    for key in ABC_drop:
        ABC_dict.pop(key)

    # Code Starts
    ts = time.gmtime()
    print('Start Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", ts)))

    # beta_ABC_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    beta_ABC_range = [0.0, 1.0]

    city_number = 1
    for city in City_dict:

        City_ = City_dict[city]

        building_number = 1
        for building in building_type_list:

            dataframes_ls = []

            Building_ = Building(
                name=building, building_type=building, City_=City_dict[city])
            for beta in beta_ABC_range:
                for ac in AC_dict:
                    AC_ = AC_dict[ac]
                    ac_number = 1
                    for abc in ABC_dict:
                        abc_number = 1
                        ABC_ = ABC_dict[abc]

                        print('City: {}, {} of 16 | Building: {}, {} of 16 | Beta_ABC = {} | AC {}: {} of 3 | ABC {}: {} of 4 | Time: {}'.format(
                            city, city_number, building, building_number, beta, ac, ac_number, abc, abc_number, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())), end='\r')

                        df = energy_demand_sim(Building_=Building_,
                                               City_=City_,
                                               AC_=AC_,
                                               beta_ABC=beta,
                                               ABC_=ABC_)

                        df.reset_index(inplace=True)

                        dataframes_ls.append(df)

                        # ABC Loop
                        abc_number += 1
                    # AC Loop
                    ac_number += 1
                # Beta loop
            # Building Loop
            building_agg = pd.concat(dataframes_ls, axis=0).reset_index()
            building_agg.to_feather(
                r'model_outputs\energy_demands\Hourly_'+city+'_'+building+'_energy_dem.feather')
            building_number += 1
        # City Loop
        city_number += 1
    print('\nCompleted Simulation')


def execute_energy_supply_sim():
    all_cities = int(input('All cities?:\n 1) True\n 2) False\n'))
    '''
    OBJECT GENERATOR
    ----------------
    '''
    if all_cities == 1:
        system_dict = generate_objects(all_cities=True)
    else:
        cities_to_simulate = []
        n = int(input('Enter number of cities: '))
        for i in range(0, n):
            city_input = str(input(
                'Type in the name of the {}th city: '.format(i+1)))
            cities_to_simulate.append(city_input)
        print('Simulating the following list: {}'.format(cities_to_simulate))
        system_dict = generate_objects(
            all_cities=False, selected_cities=cities_to_simulate)

    City_dict = system_dict['City_dict']
    Grid_dict = system_dict['Grid_dict']
    PrimeMover_dict = system_dict['PrimeMover_dict']
    BES_dict = system_dict['BatteryStorage_dict']
    Furnace_dict = system_dict['Furnace_dict']
    AC_dict = system_dict['AC_dict']
    ABC_dict = system_dict['ABC_dict']

    # Just look at two furnaces, one electric and one gas
    Furnace_drop = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'B1']
    for key in Furnace_drop:
        Furnace_dict.pop(key)

    # Code Starts
    ts = time.gmtime()
    print('Start Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", ts)))

    alpha_CHP_range = [0.0, #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                        1.0]

    city_number = 1
    for city in City_dict:

        City_ = City_dict[city]

        building_number = 1
        for building in building_type_list:
            dataframes_ls = []

            Building_ = Building(
                name=building, building_type=building, City_=City_dict[city])

            pm_number = 1
            for pm in PrimeMover_dict:
                PrimeMover_ = PrimeMover_dict[pm]
                Furnace_number = 1

                for furnace in Furnace_dict:
                    Furnace_ = Furnace_dict[furnace]

                    for alpha in alpha_CHP_range:

                        print('City: {}, {} of 16 | Building: {}, {} of 16 | Alpha_CHP = {} | Furnace {}: {} of 5 | CHP {}: {} of 22 | Time: {}'.format(
                            city, city_number, building, building_number, alpha, furnace, Furnace_number, pm, pm_number, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())), end='\r')

                        df = energy_supply_sim(Building_=Building_,
                                           City_=City_,
                                           Furnace_=Furnace_,
                                           PrimeMover_=PrimeMover_,
                                           alpha_CHP=alpha)

                        df.reset_index(inplace=True)

                        dataframes_ls.append(df)

                        # alpha Loop

                    # Furnace Loop
                    Furnace_number += 1
                # PrimeMover loop
                pm_number += 1
            # Building Loop
            building_number += 1
            building_agg = pd.concat(dataframes_ls, axis=0).reset_index(drop=True)
            building_agg.to_feather(
                r'model_outputs\energy_supply' + F'\Annual_{city}_{building}_energy_sup.feather')
        # City Loop
        city_number += 1
    print('\nCompleted Simulation')


def execute_impacts_sim(data, leakage_factor=1):
    impacts = impacts_sim(data, leakage_factor)
    impacts.to_feather(r'model_outputs\impacts\All_impacts.feather')

    impacts.to_csv(r'model_outputs\testing\All_impacts.csv')
    print("Impacts Sim Complete")
    return impacts

def compile_data(all_cities=True, file_type='supply'):
    if all_cities is True:
        system_dict = generate_objects(all_cities=True)
    else:
        cities_to_simulate = []
        n = int(input('Enter number of cities: '))
        for i in range(0, n):
            city_input = str(input(
                'Type in the name of the {}th city: '.format(i+1)))
            cities_to_simulate.append(city_input)
        print('Simulating the following list: {}'.format(cities_to_simulate))
        system_dict = generate_objects(
            all_cities=False, selected_cities=cities_to_simulate)
    
    City_dict = system_dict['City_dict']

    if file_type == 'supply':
        filepath = 'model_outputs\energy_supply'

    dataframes = []
    for city in City_dict:
        for building in building_type_list:
            filename = F'Annual_{city}_{building}_energy_sup.feather'
            df = pd.read_feather(F'{filepath}\{filename}')
            dataframes.append(df)

    compiled_df = pd.concat(dataframes, axis=0).reset_index(drop=True)

    # Some cleaning that should be incorporated into models.py
    compiled_df.drop(['index','level_8'], axis=1, inplace=True)

    return compiled_df




# execute_energy_demand_sim()
# execute_energy_supply_sim()
# df = compile_data()
# df.to_feather(r'model_outputs\energy_supply\All_supply_data.feather')
# df.to_csv(r'model_outputs\testing\All_supply_data.csv')

data = pd.read_feather(r'model_outputs\energy_supply\All_supply_data.feather')
# print(data.head())
execute_impacts_sim(data=data)

def test_supply():

    cities_to_simulate = ['albuquerque']
    system_dict = generate_objects(
            all_cities=False, selected_cities=cities_to_simulate)

    City_dict = system_dict['City_dict']
    Grid_dict = system_dict['Grid_dict']
    PrimeMover_dict = system_dict['PrimeMover_dict']
    BES_dict = system_dict['BatteryStorage_dict']
    Furnace_dict = system_dict['Furnace_dict']
    AC_dict = system_dict['AC_dict']
    ABC_dict = system_dict['ABC_dict']

    Building_ = Building(name='Test', building_type='full_service_restaurant', City_=City_dict['albuquerque'])
    City_ = City_dict['albuquerque']
    Furnace_ = Furnace_dict['B2']
    PrimeMover_ = PrimeMover_dict['RE1']
    alpha = 0
    df = energy_supply_sim(Building_=Building_,
                                           City_=City_,
                                           Furnace_=Furnace_,
                                           PrimeMover_=PrimeMover_,
                                           alpha_CHP=alpha)

    df.to_csv(r'model_outputs\testing\Energy_supply.csv')

    print('DONE')

# test_supply()