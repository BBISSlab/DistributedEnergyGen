# My modules
from energy_storage import design_BES
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


def execute_energy_demand_sim(thermal_distribution_loss_factor=1.0):
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
                'Type in the name of the {}th city: '.format(i + 1)))
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
    ABC_drop = [
        'ABC_SS1',
        'ABC_SS3',
        'ABC_TS1',
        'ABC_TS2',
        'ABC_TS3',
        'ABC_TS4']
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
                                               ABC_=ABC_,
                                               thermal_distribution_loss_factor=thermal_distribution_loss_factor)

                        df.reset_index(inplace=True)

                        dataframes_ls.append(df)

                        # ABC Loop
                        abc_number += 1
                    # AC Loop
                    ac_number += 1
                # Beta loop
            # Building Loop
            building_agg = pd.concat(dataframes_ls, axis=0).reset_index()

            if thermal_distribution_loss_factor == 1.0:
                building_agg.to_feather(
                    r'model_outputs\energy_demands\Hourly_' + city + '_' + building + '_energy_dem.feather')
            else:
                building_agg.to_feather(
                    r'model_outputs\distribution_sensitivity\Hourly_' + city + '_' + building + '_energy_dem_dist_sens.feather')
            building_number += 1
        # City Loop
        city_number += 1
    print('\nCompleted Simulation')


def execute_energy_supply_sim(thermal_distribution_loss_factor=1.0):
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
                'Type in the name of the {}th city: '.format(i + 1)))
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

    alpha_CHP_range = [0.0,  # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
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
                                               alpha_CHP=alpha,
                                               thermal_distribution_loss_factor=thermal_distribution_loss_factor)

                        df.reset_index(inplace=True)

                        dataframes_ls.append(df)

                        # alpha Loop

                    # Furnace Loop
                    Furnace_number += 1
                # PrimeMover loop
                pm_number += 1
            # Building Loop
            building_number += 1
            building_agg = pd.concat(
                dataframes_ls,
                axis=0).reset_index(
                drop=True)
            if thermal_distribution_loss_factor == 1:
                building_agg.to_feather(
                    r'model_outputs\energy_supply' + F'\\Annual_{city}_{building}_energy_sup.feather')
            else:
                building_agg.to_feather(
                    r'model_outputs\distribution_sensitivity' + F'\\Annual_{city}_{building}_energy_sup_dist_sens.feather')
        # City Loop
        city_number += 1
    print('\nCompleted Simulation')


def execute_impacts_sim(data, leakage_factor=1,
                        sensitivity=None):
    impacts = impacts_sim(data, leakage_factor)

    if sensitivity is None:
        impacts.to_feather(r'model_outputs\impacts\All_impacts.feather')

        impacts.to_csv(r'model_outputs\testing\All_impacts.csv')

    if sensitivity == 'DS':
        impacts.to_feather(
            r'model_outputs\distribution_sensitivity\All_impacts_DS.feather')

        impacts.to_csv(r'model_outputs\testing\All_impacts_DS.csv')
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
                'Type in the name of the {}th city: '.format(i + 1)))
            cities_to_simulate.append(city_input)
        print('Simulating the following list: {}'.format(cities_to_simulate))
        system_dict = generate_objects(
            all_cities=False, selected_cities=cities_to_simulate)

    City_dict = system_dict['City_dict']

    if file_type == 'supply':
        filepath = r'model_outputs\energy_supply'

        dataframes = []
        for city in City_dict:
            for building in building_type_list:
                filename = F'Annual_{city}_{building}_energy_sup.feather'
                df = pd.read_feather(F'{filepath}\\{filename}')
                dataframes.append(df)

    if file_type == 'distribution_sensitivity':
        filepath = r'model_outputs\distribution_sensitivity'

        dataframes = []
        for city in City_dict:
            for building in building_type_list:
                filename = F'Annual_{city}_{building}_energy_sup_dist_sens.feather'
                df = pd.read_feather(F'{filepath}\\{filename}')
                dataframes.append(df)

    compiled_df = pd.concat(dataframes, axis=0).reset_index(drop=True)

    # Some cleaning that should be incorporated into models.py
    compiled_df.drop(['index', 'level_8'], axis=1, inplace=True)

    return compiled_df


# execute_energy_demand_sim(thermal_distribution_loss_factor=1.1)
# execute_energy_supply_sim(thermal_distribution_loss_factor=1.1)
# df = compile_data(file_type='distribution_sensitivity')
# df.to_feather(r'model_outputs\distribution_sensitivity\All_supply_data_DS.feather')
# df.to_csv(r'model_outputs\testing\All_supply_data_DS.csv')

# data = pd.read_feather(r'model_outputs\distribution_sensitivity\All_supply_data_DS.feather')
# print(data.head())
# execute_impacts_sim(data=data, sensitivity=None)

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

    Building_ = Building(
        name='Test',
        building_type='full_service_restaurant',
        City_=City_dict['albuquerque'])
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


def test_pv():
    import energy_storage

    City_ = City(name='atlanta',
                 nerc_region=nerc_region_dictionary['atlanta'],
                 tmy3_file=tmy3_city_dictionary['atlanta'])
    City_._get_data(City_.tmy3_file)

    Building_ = Building(
        name='medium_office',
        building_type='medium_office',
        City_=City_)

    airconditioners = retrieve_AirConditioners()
    furnaces = retrieve_Furnaces()

    # Temporarily, make the ac and the furnace a dictionary object, rather
    # than an object.
    Furnace_ = furnaces['B1']
    AC_ = airconditioners['CH2']

    PVSystem_ = design_building_PV(Building_, Furnace_, AC_)

    building_pv_sim = building_pv_energy_sim(Building_=Building_,
                                             City_=City_,
                                             PVSystem_=PVSystem_,
                                             Furnace_=Furnace_,
                                             AC_=AC_,
                                             oversize_factor=1.3)

    electricity_load = building_pv_sim.electricity_surplus
    V_pv = nominal_voltage(PVSystem_)

    BES = design_BES('Li7', electricity_load, V_pv, 12)

    energy_input_output = building_pv_sim.electricity_surplus + building_pv_sim.electricity_deficit
    
    BES_power = BES.BES_storage_simulation(energy_input_output)
    # Separate function for energy supplied by the BES
    
    sns.lineplot(x=BES_power.index, y=BES_power.BES_energy_io)
    plt.show()
    print(BES_power)
 
    # building_pv_sim.to_csv(r'model_outputs\testing\building_pv.csv')

    return # building_pv_sim


def building_pv_sim(building_type, city_name,
                    AC_model, Furnace_model,
                    PVSystem_=None, oversize_factor=1,
                    ):
    # ToDo:
    # Generate city and building objects
    City_ = City(name=city_name,
                 nerc_region=nerc_region_dictionary[city_name],
                 tmy3_file=tmy3_city_dictionary[city_name])
    City_._get_data(City_.tmy3_file)

    Building_ = Building(
        name=building_type,
        building_type=building_type,
        City_=City_)

    # ToDo: Convert AC and Furnace into objects rather than dictionaries
    airconditioners = retrieve_AirConditioners()
    furnaces = retrieve_Furnaces()
    Furnace_ = furnaces[Furnace_model]
    AC_ = airconditioners[AC_model]

    PVSystem_ = design_building_PV(Building_, Furnace_, AC_, oversize_factor)

    building_pv_sim = building_pv_energy_sim(Building_=Building_,
                                             City_=City_,
                                             PVSystem_=PVSystem_,
                                             Furnace_=Furnace_,
                                             AC_=AC_,
                                             oversize_factor=oversize_factor)


    pass


###############
# CCHP v Grid #
###############
# create your scenarios here
Furnace_ = Furnace(Furnace_id='F4')
AC_ = AirConditioner(AC_id='AC3')
ABC_ = AbsorptionChiller(ABC_id='ABC_SS1')
# Current default PV module
pv_module = 'Silevo_Triex_U300_Black__2014_'

reference_case = {'AC':AC_, 'Furnace':Furnace_, 
                  'CHP':None, 'ABC':None,  
                  'pv_module':None}
cchp_case = {'AC':None, 'Furnace':None, 
             'CHP':None, 'ABC':ABC_,  
             'pv_module':None}
pv_case = {'AC':None, 'Furnace':None, 
             'CHP':None, 'ABC':ABC_,  
             'pv_module':None}
pv_cchp_case = {'AC':None, 'Furnace':None, 
             'CHP':None, 'ABC':ABC_,  
             'pv_module':None}


supply_scenarios = {}

def simulate_energy_demands(thermal_distribution_loss_rate=0.1,
                            thermal_distribution_loss_factor=1.0):
    all_cities = int(input('All cities?:\n 1) True\n 2) False\n'))
    
    from sysClasses import _generate_Cities
    '''
    OBJECT GENERATOR
    ----------------
    '''
    if all_cities == 1:
        City_dict = _generate_Cities(all_cities=True)
    else:
        cities_to_simulate = []
        n = int(input('Enter number of cities: '))
        for i in range(0, n):
            city_input = str(input(
                'Type in the name of the {}th city: '.format(i + 1)))
            cities_to_simulate.append(city_input)
        print('Simulating the following list: {}'.format(cities_to_simulate))
        City_dict = _generate_Cities(
            all_cities=False, selected_cities=cities_to_simulate)

    # Demand Scenarios
    AC_ = AirConditioner(AC_id='AC3')
    ABC_ = AbsorptionChiller(ABC_id='ABC_SS1')
    electric_cooling = {'AC':AC_, 'ABC':None}
    absorption_cooling = {'AC':None, 'ABC': ABC_}
    demand_scenarios = {'electric_cooling':electric_cooling,
                        'absorption_cooling':absorption_cooling}

    # Code Starts
    ts = time.gmtime()
    print('Start Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", ts)))

    city_number = 1
    for city in City_dict:

        City_ = City_dict[city]

        building_number = 1
        for building in building_type_list:

            dataframes_ls = []

            Building_ = Building(
                name=building, building_type=building, City_=City_dict[city])
            for scenario in demand_scenarios:
                ac = demand_scenarios[scenario]['AC']
                abc = demand_scenarios[scenario]['ABC']
                
                print('City: {}, {} of 16 | Building: {}, {} of 16 | Demand Scenario: {} | Time: {}'.format(
                            city, city_number, building, building_number, scenario, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())), end='\r')                

                df = calculate_energy_demands(Building_=Building_,
                                               City_=City_,
                                               AC_=ac,
                                               ABC_=abc,
                                               thermal_distribution_loss_rate=thermal_distribution_loss_rate,
                                               thermal_distribution_loss_factor=thermal_distribution_loss_factor)

                df.reset_index(inplace=True)

                dataframes_ls.append(df)

            # Building Loop
            building_agg = pd.concat(dataframes_ls, axis=0).reset_index(drop=True)

            if thermal_distribution_loss_factor == 1.0:
                building_agg.to_feather(
                    r'model_outputs\CCHPvGrid\energy_demands\Demand' + city + '_' + building + '.feather')
                # building_agg.to_csv(r'model_outputs\CCHPvGrid\energy_demands\Demand' + city + '_' + building + '.csv')
            else:
                building_agg.to_feather(
                    r'model_outputs\CCHPvGrid\energy_demands\Demand_' + city + '_' + building + '_energy_dem_dist_sens.feather')
            building_number += 1
        # City Loop
        city_number += 1
    print('\nCompleted Simulation')


def simulate_energy_supply(thermal_distribution_loss_rate=0.1,
                            thermal_distribution_loss_factor=1.0):
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
                'Type in the name of the {}th city: '.format(i + 1)))
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

    alpha_CHP_range = [0.0,  # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
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
                                               alpha_CHP=alpha,
                                               thermal_distribution_loss_factor=thermal_distribution_loss_factor)

                        df.reset_index(inplace=True)

                        dataframes_ls.append(df)

                        # alpha Loop

                    # Furnace Loop
                    Furnace_number += 1
                # PrimeMover loop
                pm_number += 1
            # Building Loop
            building_number += 1
            building_agg = pd.concat(
                dataframes_ls,
                axis=0).reset_index(
                drop=True)
            if thermal_distribution_loss_factor == 1:
                building_agg.to_feather(
                    r'model_outputs\energy_supply' + F'\\Annual_{city}_{building}_energy_sup.feather')
            else:
                building_agg.to_feather(
                    r'model_outputs\distribution_sensitivity' + F'\\Annual_{city}_{building}_energy_sup_dist_sens.feather')
        # City Loop
        city_number += 1
    print('\nCompleted Simulation')



simulate_energy_demands()