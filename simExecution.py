# My modules
from abc import ABC
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

    energy_input_output = building_pv_sim.electricity_surplus + \
        building_pv_sim.electricity_deficit

    BES_power = BES.BES_storage_simulation(energy_input_output)
    # Separate function for energy supplied by the BES

    sns.lineplot(x=BES_power.index, y=BES_power.BES_energy_io)
    plt.show()
    print(BES_power)

    # building_pv_sim.to_csv(r'model_outputs\testing\building_pv.csv')

    return  # building_pv_sim


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
def simulate_energy_flows(thermal_distribution_loss_rate=0.1,
                          thermal_distribution_loss_factor=1.0,
                          scenario='Reference',
                          years=1):
    print(scenario)
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

    # Standard cooling and heating systems
    AC_ = AirConditioner(AC_id='AC3')
    # ABC_ = AbsorptionChiller(ABC_id='ABC_SS1')
    Furnace_ = Furnace(Furnace_id='F4')
    CHP_list = retrieve_PrimeMover_specs()
    # Drop steam turbines
    CHP_list.drop(columns=['ST1', 'ST2', 'ST3'], inplace=True)

    # Scenarios
    reference_case = {'AC': AC_, 'Furnace': Furnace_,
                      'ABC': None, 'CHP': None,
                      'PV': None, 'BES': None}
    cchp_case = {'AC': None, 'Furnace': None,
                 'ABC': None, 'CHP': None,
                 'PV': None, 'BES': None}
    pv_case = {'AC': AC_, 'Furnace': Furnace_,
               'ABC': None, 'CHP': None,
               'PV': None, 'BES': None}
    cchp_pv_case = {'AC': None, 'Furnace': None,
                    'ABC': None, 'CHP': None,
                    'PV': None, 'BES': None}

    scenarios = {'Reference': reference_case,
                 'CCHP': cchp_case,
                 'PV': pv_case,
                 'CCHP_PV': cchp_pv_case}

    # Code Starts
    ts = time.gmtime()
    print('Start Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", ts)))

    design_ls = []

    city_number = 1
    for city in City_dict:

        City_ = City_dict[city]

        building_number = 1
        for building in building_type_list:

            dataframes_ls = []

            Building_ = Building(
                name=building, building_type=building, City_=City_dict[city])

            ac = scenarios[scenario]['AC']
            furnace = scenarios[scenario]['Furnace']

            EnergySystem_ = None

            copy_year = True

            if scenario == 'Reference':
                # Rewrite this so that it essentially repeats.
                if copy_year is True:
                    df, EnergySystem_ = calculate_energy_flows(Building_=Building_,
                                                                City_=City_,
                                                                EnergySystem_=EnergySystem_,
                                                                AC_=ac, Furnace_=furnace)

                    annual_df = aggregate_energy_flows(df, aggregate='A')
                    for year in range(years + 1):
                        print('Time: {} | City: {}, {} of 16 | Building: {}, {} of 16 | Scenario: {} | Year: {} of {}|||||||||||||'.format(
                            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), city, city_number, building, building_number, scenario, year, years), end='\r')
                        annual_df['year'] = 2020 + year

                        dataframes_ls.append(annual_df)

                        EnergySystem_ = EnergySystem_.increase_age()

                else:
                    for year in range(years + 1):
                        print('Time: {} | City: {}, {} of 16 | Building: {}, {} of 16 | Scenario: {} | Year: {} of {}|||||||||||||'.format(
                            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), city, city_number, building, building_number, scenario, year, years), end='\r')

                        df, EnergySystem_ = calculate_energy_flows(Building_=Building_,
                                                                City_=City_,
                                                                EnergySystem_=EnergySystem_,
                                                                AC_=ac, Furnace_=furnace)

                        annual_df = aggregate_energy_flows(df, aggregate='A')
                        annual_df['year'] = 2020 + year

                        dataframes_ls.append(annual_df)

                        EnergySystem_ = EnergySystem_.increase_age()

                design_ls.append(EnergySystem_.design_dataframe())

            elif scenario == 'CCHP':
                for chp in CHP_list.columns.unique():
                    CHP_ = PrimeMover(PM_id=chp)
                    chp_type = CHP_.module_parameters['technology']

                    if chp_type == 'Reciprocating Engine' or chp_type == 'Gas Turbine':
                        ABC_ = AbsorptionChiller(ABC_id='ABC_TS1')
                    else:
                        ABC_ = AbsorptionChiller(ABC_id='ABC_SS1')

                    EnergySystem_ = None
                    
                    if copy_year is True:

                        df, EnergySystem_ = calculate_energy_flows(Building_=Building_,
                                                                City_=City_,
                                                                EnergySystem_=EnergySystem_,
                                                                PrimeMover_=CHP_,
                                                                ABC_=ABC_)

                        annual_df = aggregate_energy_flows(df, aggregate='A')
                        annual_df['PM_id'] = chp

                        for year in range(years + 1):
                            print('Time: {} | City: {}, {} of 16 | Building: {}, {} of 16 | Scenario: {} | CHP: {} | Year: {} of {}|||||||||||||'.format(
                                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), city, city_number, building, building_number, scenario, chp, year, years), end='\r')
                            
                            copied_df = annual_df.copy()
                            copied_df['year'] = 2020 + year

                            dataframes_ls.append(copied_df)

                        EnergySystem_ = EnergySystem_.increase_age()

                    else:

                        for year in range(years + 1):
                            print('Time: {} | City: {}, {} of 16 | Building: {}, {} of 16 | Scenario: {} | CHP: {} | Year: {} of {}|||||||||||||'.format(
                                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), city, city_number, building, building_number, scenario, chp, year, years), end='\r')
                            df, EnergySystem_ = calculate_energy_flows(Building_=Building_,
                                                                    City_=City_,
                                                                    EnergySystem_=EnergySystem_,
                                                                    PrimeMover_=CHP_,
                                                                    ABC_=ABC_)

                            annual_df = aggregate_energy_flows(df, aggregate='A')
                            annual_df['year'] = 2020 + year

                            dataframes_ls.append(annual_df)

                            EnergySystem_ = EnergySystem_.increase_age()

                    design_ls.append(EnergySystem_.design_dataframe())

            elif scenario == 'PV':
                for year in range(years + 1):
                    print('Time: {} | City: {}, {} of 16 | Building: {}, {} of 16 | Scenario: {} | Year: {} of {}|||||||||||||'.format(
                        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), city, city_number, building, building_number, scenario, year, years), end='\r')

                    df, EnergySystem_ = calculate_energy_flows(Building_=Building_,
                                                               City_=City_,
                                                               EnergySystem_=EnergySystem_,
                                                               AC_=ac, Furnace_=furnace,
                                                               pv_module='Silevo_Triex_U300_Black__2014_',
                                                               battery='Li3')
                    annual_df = aggregate_energy_flows(df, aggregate='A')
                    annual_df['year'] = 2020 + year

                    dataframes_ls.append(annual_df)

                    EnergySystem_ = EnergySystem_.increase_age()

                design_ls.append(EnergySystem_.design_dataframe())

            elif scenario == 'CCHP_PV':
                for chp in CHP_list.columns.unique():
                    CHP_ = PrimeMover(PM_id=chp)
                    chp_type = CHP_.module_parameters['technology']

                    if chp_type == 'Reciprocating Engine' or chp_type == 'Gas Turbine':
                        ABC_ = AbsorptionChiller(ABC_id='ABC_TS1')
                    else:
                        ABC_ = AbsorptionChiller(ABC_id='ABC_SS1')

                    EnergySystem_ = None

                    for year in range(years + 1):
                        print('Time: {} | City: {}, {} of 16 | Building: {}, {} of 16 | Scenario: {} | CHP: {} | Year: {} of {}|||||||||||||'.format(
                            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), city, city_number, building, building_number, scenario, chp, year, years), end='\r')

                        df, EnergySystem_ = calculate_energy_flows(Building_=Building_,
                                                                   City_=City_,
                                                                   EnergySystem_=EnergySystem_,
                                                                   PrimeMover_=CHP_,
                                                                   ABC_=ABC_,
                                                                   pv_module='Trina_TSM_240PA05__2013_',
                                                                   battery='Li3')
                        annual_df = aggregate_energy_flows(df, aggregate='A')
                        annual_df['year'] = 2020 + year
                        annual_df['PM_id'] = chp

                        dataframes_ls.append(annual_df)

                        EnergySystem_ = EnergySystem_.increase_age()

                    design_ls.append(EnergySystem_.design_dataframe())

            # Building Loop
            building_agg = pd.concat(
                dataframes_ls,
                axis=0).reset_index(
                drop=True)
            building_agg.to_feather(
                F'model_outputs\\CCHPvGrid\\{scenario}\\{city}_{building}.feather')

            building_number += 1
        # City Loop
        city_number += 1

    design_agg = pd.concat(design_ls, axis=0).reset_index(drop=True)
    design_agg.to_feather(
        F'model_outputs\\CCHPvGrid\\{scenario}\\{scenario}_design.feather')

    print(building_agg.head())
    print(design_agg.head())
    print('\nCompleted Simulation')


def simulate_energy_demands(thermal_distribution_loss_rate=0.1,
                            thermal_distribution_loss_factor=1.0,
                            scenario='AC'):
    print(scenario)
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

    # Standard cooling and heating systems
    AC_ = AirConditioner(AC_id='AC3')
    ABC_SS = AbsorptionChiller(ABC_id='ABC_SS1')
    ABC_TS = AbsorptionChiller(ABC_id='ABC_TS1')
    Furnace_ = Furnace(Furnace_id='F4')

    # Scenarios
    ac_case = {'AC': AC_,
               'ABC': None}
    abc_ss_case = {'AC': None,
                   'ABC': ABC_SS}
    abc_ts_case = {'AC': None,
                   'ABC': ABC_TS}

    scenarios = {'AC': ac_case,
                 'ABC_SS': abc_ss_case,
                 'ABC_TS': abc_ts_case}
    cooling_system = {'AC': 'AC',
                      'ABC_SS': 'ABC_SS',
                      'ABC_TS': 'ABC_TS'}

    # Code Starts
    ts = time.gmtime()
    print('Start Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", ts)))

    design_ls = []

    city_number = 1
    for city in City_dict:

        City_ = City_dict[city]

        building_number = 1
        for building in building_type_list:

            dataframes_ls = []

            Building_ = Building(
                name=building, building_type=building, City_=City_dict[city])

            AC_ = scenarios[scenario]['AC']
            ABC_ = scenarios[scenario]['ABC']

            print('Time: {} | City: {}, {} of 16 | Building: {}, {} of 16 | Scenario: {}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), city, city_number, building, building_number, scenario), end='\r')

            system_design = {
                'City': [
                    City_.name], 'Building': [
                    Building_.building_type]}
            # Size and Design Cooling System
            if AC_ is None:
                ABC_ = ABC_.size_system(Building_.cooling_demand.max())

                system_design['AC_id'] = ['None']
                system_design['num_AC_modules'] = [0]
                system_design['ABC_id'] = [ABC_.ABC_id]
                system_design['num_ABC_modules'] = [ABC_.number_of_modules]

            else:
                AC_ = AC_.size_system(Building_.cooling_demand.max())

                system_design['AC_id'] = [AC_.AC_id]
                system_design['num_AC_modules'] = [AC_.number_of_modules]
                system_design['ABC_id'] = ['None']
                system_design['num_ABC_modules'] = [0]

            df = calculate_energy_demands(Building_=Building_, City_=City_,
                                          AC_=AC_, ABC_=ABC_,
                                          thermal_distribution_loss_rate=thermal_distribution_loss_rate,
                                          thermal_distribution_loss_factor=thermal_distribution_loss_factor)

            df.reset_index(inplace=True)
            dataframes_ls.append(df)

            system_design_df = pd.DataFrame.from_dict(system_design)
            design_ls.append(system_design_df)

            # Building Loop
            building_agg = pd.concat(
                dataframes_ls,
                axis=0).reset_index(
                drop=True)
            building_agg.to_feather(
                F'model_outputs\\CCHPvGrid\\Energy_Demands\\{cooling_system[scenario]}\\{city}_{building}.feather')

            building_number += 1
        # City Loop
        city_number += 1

    design_agg = pd.concat(design_ls, axis=0).reset_index(drop=True)
    design_agg.to_feather(
        F'model_outputs\\CCHPvGrid\\Energy_Demands\\{cooling_system[scenario]}\\design.feather')

    print(building_agg.head())
    print(design_agg.head())
    print('\nCompleted Simulation')


def simulate_Furnace_supply():
    pass


def simulate_PV_supply(pv_module):
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

    # Code Starts
    ts = time.gmtime()
    print('Standalone PV Sim')
    print('Start Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", ts)))

    design_ls = []

    city_number = 1
    for city in City_dict:

        City_ = City_dict[city]

        building_number = 1
        for building in building_type_list:

            dataframes_ls = []

            Building_ = Building(
                name=building, building_type=building, City_=City_dict[city])

            print('Time: {} | City: {}, {} of 16 | Building: {}, {} of 16'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), city, city_number, building, building_number), end='\r')

            system_design = {
                'City': [
                    City_.name], 'Building': [
                    Building_.building_type]}
            # Size and Design Cooling System
            PVSystem_ = design_PVSystem(module=pv_module,
                                        method='design_area',
                                        design_area=Building_.roof_area * 0.8,
                                        surface_tilt=City_.latitude)

            system_design['num_PV_modules'] = [total_pv_modules(PVSystem_)]

            pv_df = pv_simulation(PVSystem_, City_)

            # Power ouputs from PV system are given in W
            pv_df['electricity_PV'] = pv_df['p_ac'] / 1000

            pv_df.reset_index(inplace=True)
            dataframes_ls.append(pv_df)

            system_design_df = pd.DataFrame.from_dict(system_design)
            design_ls.append(system_design_df)

            # Building Loop
            building_agg = pd.concat(
                dataframes_ls,
                axis=0).reset_index(
                drop=True)
            building_agg.to_feather(
                F'model_outputs\\CCHPvGrid\\Energy_Supply\\PV\\{city}_{building}.feather')

            building_number += 1
        # City Loop
        city_number += 1

    design_agg = pd.concat(design_ls, axis=0).reset_index(drop=True)
    design_agg.to_feather(
        F'model_outputs\\CCHPvGrid\\Energy_Supply\\PV\\design.feather')

    print(building_agg.head())
    print(design_agg.head())
    print('\nCompleted Simulation')


def simulate_CHP_supply(operation_mode='FTL',
                        thermal_distribution_loss_rate=0.1,
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

    # Code Starts
    ts = time.gmtime()
    print('Standalone PV Sim')
    print('Start Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", ts)))

    CHP_list = retrieve_PrimeMover_specs()
    # Drop steam turbines
    CHP_list.drop(columns=['ST1', 'ST2', 'ST3'], inplace=True)

    design_ls = []

    city_number = 1
    for city in City_dict:

        City_ = City_dict[city]

        building_number = 1
        for building in building_type_list:

            dataframes_ls = []

            Building_ = Building(
                name=building, building_type=building, City_=City_dict[city])

            system_design = {
                'City': [
                    City_.name], 'Building': [
                    Building_.building_type]}

            # Read Dataframes
            for chp in CHP_list.columns.unique():
                print('Time: {} | City: {}, {} of 16 | Building: {}, {} of 16 | CHP: {} ||||||||||||||'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), city, city_number, building, building_number, chp), end='\r')

                CHP_ = PrimeMover(PM_id=chp)
                chp_type = CHP_.module_parameters['technology']

                if chp_type == 'Reciprocating Engine' or chp_type == 'Gas Turbine':
                    filepath = r'model_outputs\CCHPvGrid\Energy_Demands\ABC_TS'

                else:
                    filepath = r'model_outputs\CCHPvGrid\Energy_Demands\ABC_SS'

                filename = F'\\{city}_{building}.feather'
                datafile = filepath + filename
                energy_demand_df = pd.read_feather(datafile)
                energy_demand_df.set_index('datetime', inplace=True, drop=True)

                if operation_mode == 'FTL':
                    FTL = True
                else:
                    FTL = False

                # Size CHP
                CHP_ = size_chp(CHP_, energy_demand_df, City_,
                                operation_mode,
                                thermal_distribution_loss_rate,
                                thermal_distribution_loss_factor)

                chp_df = CHP_energy(electricity_demand=energy_demand_df.total_electricity_demand,
                                    heat_demand=energy_demand_df.total_heat_demand,
                                    PrimeMover_=CHP_, FTL=FTL,
                                    thermal_distribution_loss_rate=thermal_distribution_loss_rate,
                                    thermal_distribution_loss_factor=thermal_distribution_loss_factor)

                chp_df['PM_id'] = CHP_.PM_id
                chp_df.reset_index(inplace=True)
                dataframes_ls.append(chp_df)

                # Update the system Design
                system_design = {
                    'City': [
                        City_.name], 'Building': [
                        Building_.building_type]}
                system_design['Furnace_id'] = ['None']
                system_design['num_Furnace_modules'] = [0]
                system_design['PM_id'] = [CHP_.PM_id]
                system_design['num_CHP_modules'] = [CHP_.number_of_modules]
                system_design['oversize_ratio_electricity'] = CHP_.nominal_capacity(
                ) / energy_demand_df.total_electricity_demand.max()
                system_design['oversize_ratio_heat'] = CHP_.nominal_heat_capacity(
                ) / energy_demand_df.total_heat_demand.max()

                system_design_df = pd.DataFrame.from_dict(system_design)
                design_ls.append(system_design_df)

            # Building Loop
            building_agg = pd.concat(
                dataframes_ls,
                axis=0).reset_index(
                drop=True)
            building_agg.to_feather(
                F'model_outputs\\CCHPvGrid\\Energy_Supply\\CHP\\{city}_{building}.feather')

            building_number += 1
        # City Loop
        city_number += 1

    design_agg = pd.concat(design_ls, axis=0).reset_index(drop=True)
    design_agg.to_feather(
        F'model_outputs\\CCHPvGrid\\Energy_Supply\\CHP\\design.feather')

    print(building_agg.head())
    print(design_agg.head())
    print('\nCompleted Simulation')


def balance_energy_supply_and_demand(scenario='Reference'):
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

    # Code Starts
    ts = time.gmtime()
    print('Start Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", ts)))

    CHP_list = retrieve_PrimeMover_specs()
    # Drop steam turbines
    CHP_list.drop(columns=['ST1', 'ST2', 'ST3'], inplace=True)

    design_ls = []

    city_number = 1
    for city in City_dict:

        City_ = City_dict[city]

        building_number = 1
        for building in building_type_list:

            dataframes_ls = []

            Building_ = Building(
                name=building, building_type=building, City_=City_dict[city])

            system_design = {
                'City': [
                    City_.name], 'Building': [
                    Building_.building_type]}

            if scenario == 'Reference':
                demand_filepath = r'model_outputs\CCHPvGrid\Energy_Demands\AC'
                demand_filename = F'\\{city}_{building}.feather'
                demand_file = demand_filepath + demand_filename
                energy_demands_df = pd.read_feather(demand_file)

            if scenario == 'PV':
                demand_filepath = F''
                demand_filename = F''
                supply_filename = F''
                supply_filepath = F''
            if scenario == 'CCHP':
                demand_filepath = F''
                demand_filename = F''
                supply_filename = F''
                supply_filepath = F''
            if scenario == 'CCHP_PV':
                demand_filepath = F''
                demand_filename = F''
                supply_filename = F''
                supply_filepath = F''


#######
# Run #
#######

'''
Energy Demands:
Scenarios: AC, ABC_SS, and ABC_TS
'''
# simulate_energy_demands(scenario='ABC_TS')

'''
Energy Supply:
'''
# simulate_PV_supply(pv_module='Silevo_Triex_U300_Black__2014_')
# simulate_CHP_supply()

'''
The Whole Thing
'''
simulate_energy_flows(scenario='CCHP', years=30)
