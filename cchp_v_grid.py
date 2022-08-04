from math import sqrt
import os
import inspect
import datetime
import time
from openpyxl import load_workbook
import pathlib
from pyarrow import feather

import numpy as np
import pandas as pd

from sysClasses import *
from pv_system import *
from energy_storage import *



def calculate_energy_deficit(energy_demand, energy_supply, ):
    '''
    This function calculates the energy deficit, which is a difference
    between the energy demand and supply.

    If there is enough supply to meet the demand (i.e., supply - demand >= 0),
    then the function returns 0. 

    NOTICE THAT IT RETURNS A NEGATIVE NUMBER
    '''
    net_energy = energy_supply - energy_demand
    deficit_energy = np.where(net_energy <= 0,
                              net_energy,
                              0)
    return deficit_energy


def calculate_energy_surplus(energy_demand, energy_supply):
    '''
    This function calculates the energy surplus, which is a difference
    between the energy demand and supply.

    If there is not enough supply to meet the demand (i.e., supply - demand < 0),
    then the function returns 0. 
    '''
    net_energy = energy_supply - energy_demand
    surplus_energy = np.where(net_energy > 0,
                              net_energy, 0)
    return surplus_energy


#####################################################
# Functions and Models for PV-Energy Storage Models #
#####################################################


def building_pv_energy_sim(Building_,
                           City_,
                           Furnace_=None,
                           AC_=None,
                           PVSystem_=None,
                           oversize_factor=1):

    df = electrify_building_demands(Building_, Furnace_, AC_)

    if PVSystem_ is None:
        PVSystem_ = design_building_PV(Building_, Furnace_, AC_, df,
                                       oversize_factor=oversize_factor)

    # Run PV supply
    pv_energy_output = pv_simulation(PVSystem_=PVSystem_, City_=City_)
    df.index = pv_energy_output.index

    pv_energy_output.to_csv(r'model_outputs\testing\pv_output.csv')
    # Convert PV outputs into kW
    df['pv_dc'] = pv_energy_output['p_dc'] / 1000
    df['pv_ac'] = pv_energy_output['p_ac'] / 1000
    df['clipped_p_dc'] = pv_energy_output['clipped_p_dc'] / 1000
    df['clipped_p_ac'] = pv_energy_output['clipped_p_ac'] / 1000
    df['inverter_efficiency'] = pv_energy_output['inverter_efficiency']

    df['electricity_surplus'] = calculate_energy_surplus(
        df['net_electricity_demand'], df['pv_ac'])
    df['electricity_deficit'] = calculate_energy_deficit(
        df['net_electricity_demand'], df['pv_ac'])

    return df


def electrify_building_demands(Building_, Furnace_=None, AC_=None):
    """
    This function converts building loads to electricity if an electric
    furnace or chiller is used.
    """
    
    df = pd.DataFrame()

    # Read building demands. All demands are in kWh
    df['electricity_demand'] = Building_.electricity_demand
    df['heat_demand'] = Building_.heat_demand
    df['cooling_demand'] = Building_.cooling_demand

    # Electrify building loads
    if Furnace_ is not None:
        df['heat_electricity'] = Building_.thermal_to_electricity(
            df.heat_demand, efficiency=Furnace_['efficiency'])
    else:
        df['heat_electricity'] = 0

    if AC_ is not None:
        df['cooling_electricity'] = Building_.thermal_to_electricity(
            df.cooling_demand, efficiency=AC_['COP_full_load'])
    else:
        df['cooling_electricity'] = 0

    # Calculate Net electricity demand
    df['net_electricity_demand'] = df.electricity_demand + \
        df.heat_electricity + df.cooling_electricity

    return df


def design_building_PV(Building_, Furnace_=None, AC_=None,
                       energy_demands_df=None,
                       oversize_factor=1):
    r'''
    Design Steps
    ------------
    1) Determine peak energy demand
    2) Design PV Array:
        2.1 Calculate peak rating of PV module
        2.2 Calculate total number of modules needed
    3) Size the inverter
        3.1 For grid tied or grid connected systems, the input rating
        of the inverter should be the same as the PV array rating to
        allow for safe & efficient operation
    4) Battery Sizing
        4.1 Calculate total energy consumption per day (Wh)
        4.2 Divide 4.1 by round-trip efficiency
        4.3 Divide 4.2 by depth of discharge
        4.4 Divide 4.3 by nominal battery voltage
        4.5 Multiply 4.4 by the days of autonomy to get the capacity
    '''

    City_ = Building_.City_
    City_._get_data(City_.tmy3_file)

    if energy_demands_df is None:
        energy_demands_df = electrify_building_demands(
            Building_, Furnace_, AC_)

    peak_electricity_demand = energy_demands_df.net_electricity_demand.max() * \
        1000  # in W

    # Current default PV module
    module = 'Silevo_Triex_U300_Black__2014_'

    PVSystem_ = design_PVSystem(design_load=peak_electricity_demand,
                                module=module,
                                surface_azimuth=180,
                                surface_tilt=City_.latitude,
                                oversize_factor=oversize_factor,
                                name=F'{City_.name}_{Building_.name}')

    return PVSystem_


def pv_energy_simulation():
    pass


###########################
# CCHP vs U.S. Grid Paper #
###########################

class EnergySystem:

    def __init__(self,
                 system_id,
                 Building_=None,
                 City_=None,
                 AC_=None,
                 ABC_=None,
                 Furnace_=None,
                 PrimeMover_=None,
                 PVSystem_=None,
                 PVsys_age=0,
                 BES_=None,
                 age=0):

        self.system_id = system_id
        self.City = City_
        self.Building = Building_
        self.AC = AC_
        self.ABC = ABC_
        self.Furnace = Furnace_
        self.PrimeMover = PrimeMover_
        self.PVSystem = PVSystem_
        self.PVsys_age = PVsys_age
        self.BES = BES_
        self.age = age

    '''def __repr__(self):
        print('EnergySystem class: \n')
        return self.print_system_design()'''

    def increase_age(self, years=1):
        if self.AC is None:
            pass
        else:
            self.AC = self.AC.increase_system_age(years)

        if self.ABC is None:
            pass
        else:
            self.ABC = self.ABC.increase_system_age(years)

        if self.Furnace is None:
            pass
        else:
            self.Furnace = self.Furnace.increase_system_age(years)

        if self.PrimeMover is None:
            pass
        else:
            self.PrimeMover = self.PrimeMover.increase_system_age(years)

        if self.PVSystem is None:
            pass
        else:
            self.PVsys_age += years  # Check if the PV system has a built-in age

        if self.BES is None:
            pass
        else:
            self.BES = self.BES.increase_system_age(years)

        return self

    def system_design(self):
        design_dictionary = {'City': [self.City.name],
                             'Building': [self.Building.building_type]}

        if self.AC is None:
            design_dictionary['AC'] = [None]
            design_dictionary['num_AC_modules'] = [0]
        else:
            design_dictionary['AC'] = [self.AC.AC_id]
            design_dictionary['num_AC_modules'] = [self.AC.number_of_modules]

        if self.ABC is None:
            design_dictionary['ABC'] = [None]
            design_dictionary['num_ABC_modules'] = [0]
        else:
            design_dictionary['ABC'] = [self.ABC.ABC_id]
            design_dictionary['num_ABC_modules'] = [self.ABC.number_of_modules]

        if self.Furnace is None:
            design_dictionary['Furnace'] = [None]
            design_dictionary['num_Furnace_modules'] = [0]
        else:
            design_dictionary['Furnace'] = [self.Furnace.Furnace_id]
            design_dictionary['num_Furnace_modules'] = [
                self.Furnace.number_of_modules]

        if self.PrimeMover is None:
            design_dictionary['PrimeMover'] = [None]
            design_dictionary['num_PM_modules'] = [0]
        else:
            design_dictionary['PrimeMover'] = [self.PrimeMover.PM_id]
            design_dictionary['num_PM_modules'] = [
                self.PrimeMover.number_of_modules]

        if self.PVSystem is None:
            design_dictionary['PV Module'] = [None]
            design_dictionary['num_PV_modules'] = [0]
        else:
            design_dictionary['PV Module'] = [self.PVSystem.module]
            design_dictionary['num_PV_modules'] = [
                total_pv_modules(self.PVSystem)]

        if self.BES is None:
            design_dictionary['BES'] = [None]
            design_dictionary['num_BES_modules'] = [0]
        else:
            design_dictionary['BES'] = [self.BES.BES_id]
            design_dictionary['num_BES_modules'] = [
                self.BES.number_of_batteries()]

        return design_dictionary

    def print_system_design(self):
        system_design_dict = self.system_design()
        for key in system_design_dict.keys():
            print(F'{key}: {system_design_dict[key]}')

    def design_dataframe(self):
        return pd.DataFrame.from_dict(self.system_design())

    def get_demand_file(self):
        if self.AC is None:
            if self.ABC.technology == 'single_stage':
                filepath = r'model_outputs\CCHPvGrid\Energy_Demands\ABC_SS'
            else:
                filepath = r'model_outputs\CCHPvGrid\Energy_Demands\ABC_TS'
        else:
            filepath = r'model_outputs\CCHPvGrid\Energy_Demands\AC'

        filename = F'\\{self.City.name}_{self.Building.building_type}.feather'
        datafile = filepath + filename

        df = pd.read_feather(datafile)
        df.set_index('datetime', inplace=True, drop=True)

        return df

    def from_dictionary(self):
        pass


def size_energy_system(system_ID,
                       Building_, City_,
                       AC_=None,
                       Furnace_=None,
                       pv_module=None,
                       ABC_=None,
                       PrimeMover_=None, CHP_mode='FTL',
                       thermal_distribution_loss_rate=0.1,
                       thermal_distribution_loss_factor=1):

    df = pd.DataFrame()

    # Initializing Demands. All demands are in kWh
    df['electricity_demand'] = Building_.electricity_demand
    df['heat_demand'] = Building_.heat_demand
    df['cooling_demand'] = Building_.cooling_demand

    # Metadata
    df['City'] = City_.name
    df['Building'] = Building_.building_type

    # df['DryBulb_C'] = City_.tmy_data['DryBulb']
    # df['DewPoint_C'] = City_.tmy_data['DewPoint']
    # df['RHum'] = City_.tmy_data['RHum']
    # df['Pressure_mbar'] = City_.tmy_data['Pressure']

    # Design Cooling System
    if AC_ is None:
        df['AC_id'] = 'None'
    else:
        df['AC_id'] = AC_.AC_id
        AC_ = AC_.size_system(df.cooling_demand.max())
        df['heat_cooling'] = 0
        df['electricity_cooling'] = df.cooling_demand / \
            AC_.module_parameters['COP']

    if ABC_ is None:
        df['ABC_id'] = 'None'
    else:
        df['ABC_id'] = ABC_.ABC_id
        ABC_ = ABC_.size_system(df.cooling_demand.max())
        df['heat_cooling'] = df.cooling_demand / (ABC_.module_parameters['COP'](
            1 - (thermal_distribution_loss_rate * thermal_distribution_loss_factor)))
        df['electricity_cooling'] = 0

    df['total_electricity_demand'] = df['electricity_demand'] + \
        df['electricity_cooling']
    df['total_heat_demand'] = df['heat_demand'] + df['heat_cooling']

    # Design Furnace
    if Furnace_ is None:
        pass
    else:
        Furnace_ = Furnace_.size_system(peak_load=df.total_heat_demand.max())
    # Design CHP
    if PrimeMover_ is None:
        pass
    else:
        altitude = City_.metadata['altitude']
        dry_bulb_temp = City_.tmy_data['DryBulb']
        derated_power_capacity = PrimeMover_.derate(altitude, dry_bulb_temp)

        # Determine peak load and derated capacity of system
        if CHP_mode == 'FTL':
            peak_load = df.total_heat_demand.max()
            peak_load = peak_load * \
                (1 + thermal_distribution_loss_rate *
                 thermal_distribution_loss_factor)
            design_capacity = derated_power_capacity * \
                PrimeMover_.module_parameters['hpr']
        else:
            peak_load = df.total_electricity_demand.max()
            design_capacity = derated_power_capacity

        PrimeMover_ = PrimeMover_.size_chp(peak_load, design_capacity)

    # Design PV System
    if pv_module is None:
        PVSystem_ = None
    else:
        PVSystem_ = design_PVSystem(module=pv_module,
                                    method='design_area',
                                    design_area=Building_.roof_area)

    EnergySystem_ = EnergySystem(
        system_ID,
        Building_,
        AC_,
        Furnace_,
        ABC_,
        PrimeMover_,
        PVSystem_)
    return EnergySystem_


# Recoded energy demand and supply functions
def calculate_energy_demands(Building_,
                             City_,
                             AC_=None,
                             ABC_=None,
                             thermal_distribution_loss_rate=0.1,
                             thermal_distribution_loss_factor=1.0,
                             memory={}):
    df = pd.DataFrame()

    # Initializing Demands. All demands are in kWh
    df['electricity_demand'] = Building_.electricity_demand
    df['heat_demand'] = Building_.heat_demand
    df['cooling_demand'] = Building_.cooling_demand

    # Metadata
    df['City'] = City_.name
    df['Building'] = Building_.building_type

    # df['DryBulb_C'] = City_.tmy_data['DryBulb']
    # df['DewPoint_C'] = City_.tmy_data['DewPoint']
    # df['RHum'] = City_.tmy_data['RHum']
    # df['Pressure_mbar'] = City_.tmy_data['Pressure']

    if AC_ is None:
        df['AC_id'] = 'None'
    else:
        df['AC_id'] = AC_.AC_id
        df['heat_cooling'] = 0
        df['electricity_cooling'] = df.cooling_demand / \
            AC_.module_parameters['COP']

    if ABC_ is None:
        df['ABC_id'] = 'None'
    else:
        df['ABC_id'] = ABC_.ABC_id
        df['heat_cooling'] = df.cooling_demand / (ABC_.module_parameters['COP'] * (
            1 - (thermal_distribution_loss_rate * thermal_distribution_loss_factor)))
        df['electricity_cooling'] = 0

    # Size AC or ABC
    df['total_electricity_demand'] = df.electricity_demand + df.electricity_cooling
    df['total_heat_demand'] = df.heat_demand + df.heat_cooling

    return df


def calculate_energy_flows(Building_, City_,
                           EnergySystem_=None,
                           AC_=None, Furnace_=None,
                           PrimeMover_=None, ABC_=None,
                           operation_mode='FTL',
                           pv_module=None, battery=None,
                           pv_degradation_rate=0.005, year=0,
                           thermal_distribution_loss_rate=0.1,
                           thermal_distribution_loss_factor=1.0):

    if EnergySystem_ is None:
        ###########################
        # Calculate Energy Demand #
        ###########################
        system_id = ''
        EnergySystem_ = EnergySystem(system_id, Building_, City_)

        # Size and Design Cooling System
        if AC_ is None:
            ABC_ = ABC_.size_system(Building_.cooling_demand.max())

            EnergySystem_.ABC = ABC_

        else:
            AC_ = AC_.size_system(Building_.cooling_demand.max())

            EnergySystem_.AC = AC_

        # Determine energy demands
        try:
            energy_df = EnergySystem_.get_demand_file()
        except FileNotFoundError:
            energy_df = calculate_energy_demands(Building_=Building_, City_=City_,
                                                 AC_=AC_, ABC_=ABC_,
                                                 thermal_distribution_loss_rate=thermal_distribution_loss_rate,
                                                 thermal_distribution_loss_factor=thermal_distribution_loss_factor)

        ###########################
        # Calculate Energy Supply #
        ###########################
        energy_balance_df = pd.DataFrame(index=energy_df.index)
        energy_balance_df['surplus_electricity'] = 0
        energy_balance_df['deficit_electricity'] = energy_df['total_electricity_demand']
        energy_balance_df['surplus_heat'] = 0
        energy_balance_df['deficit_heat'] = energy_df['total_heat_demand']

        # Primary Heating System
        if PrimeMover_ is None:
            Furnace_ = Furnace_.size_system(energy_df.total_heat_demand.max())

            EnergySystem_.Furnace = Furnace_

            furnace_df = Furnace_energy(
                energy_df.total_heat_demand, Furnace_=Furnace_)
            energy_df['heat_Furnace'] = furnace_df
            energy_df['PM_id'] = 'None'
            energy_df['heat_CHP'] = 0
            energy_df['electricity_CHP'] = 0
            energy_df['fuel_CHP'] = 0

            # Add to total supply
            energy_balance_df['surplus_heat'] = calculate_energy_surplus(
                energy_balance_df['deficit_heat'], energy_df['heat_Furnace'])
            energy_balance_df['deficit_heat'] = calculate_energy_deficit(
                energy_balance_df['deficit_heat'], energy_df['heat_Furnace'])
        else:
            energy_df['heat_Furnace'] = 0
            energy_df['PM_id'] = PrimeMover_.PM_id

            PrimeMover_ = size_chp(PrimeMover_,
                                   energy_df,
                                   City_,
                                   operation_mode,
                                   thermal_distribution_loss_rate,
                                   thermal_distribution_loss_factor)

            EnergySystem_.PrimeMover = PrimeMover_

            if operation_mode == "FTL":
                FTL = True
            else:
                FTL = False

            try:
                filepath = r'model_outputs\CCHPvGrid\Energy_Supply\CHP'
                filename = F'\\{City_.name}_{Building_.building_type}.feather'
                datafile = filepath + filename
                chp_df = pd.read_feather(datafile)
                # Filter out for only the CHP
                chp_df = chp_df[(chp_df['PM_id'] == PrimeMover_.PM_id)]
            except FileNotFoundError:
                chp_df = CHP_energy(electricity_demand=energy_df.total_electricity_demand,
                                    heat_demand=energy_df.total_heat_demand,
                                    PrimeMover_=PrimeMover_, FTL=FTL,
                                    thermal_distribution_loss_rate=thermal_distribution_loss_rate,
                                    thermal_distribution_loss_factor=thermal_distribution_loss_factor)
                chp_df = chp_df[(chp_df['PM_id'] == PrimeMover_.PM_id)]

            chp_df.set_index('datetime', inplace=True)
            energy_df['heat_CHP'] = chp_df['heat_CHP']
            energy_df['electricity_CHP'] = chp_df['electricity_CHP']
            energy_df['fuel_CHP'] = chp_df['fuel_CHP']

            energy_balance_df['surplus_heat'] = calculate_energy_surplus(
                energy_balance_df['deficit_heat'], energy_df['heat_CHP'])
            energy_balance_df['deficit_heat'] = calculate_energy_deficit(
                energy_balance_df['deficit_heat'], energy_df['heat_CHP'])
            energy_balance_df['surplus_electricity'] = calculate_energy_surplus(
                energy_balance_df['deficit_electricity'], energy_df['electricity_CHP'])
            energy_balance_df['deficit_electricity'] = calculate_energy_deficit(
                energy_balance_df['deficit_electricity'], energy_df['electricity_CHP'])

        # Renewables
        if pv_module is None:
            energy_df['electricity_PV'] = 0
        else:
            PVSystem_ = design_PVSystem(module=pv_module,
                                        method='design_area',
                                        design_area=Building_.roof_area,
                                        surface_tilt=City_.latitude)

            EnergySystem_.PVSystem = PVSystem_
            pv_age = EnergySystem_.PVsys_age

            pv_degradation_factor = (1 - (pv_age * pv_degradation_rate))

            try:
                filepath = r'model_outputs\CCHPvGrid\Energy_Supply\PV'
                filename = F'\\{City_.name}_{Building_.building_type}.feather'
                datafile = filepath + filename
                pv_df = pd.read_feather(datafile)
            except FileNotFoundError:
                pv_df = pv_simulation(PVSystem_, City_)

            # Need to adjust index from PV simulation, or else you will get
            # NaNs
            pv_df.index = energy_df.index

            # Power ouputs from PV system are given in W
            energy_df['electricity_PV'] = (
                pv_df['p_ac'] / 1000) * pv_degradation_factor

            energy_balance_df['surplus_electricity'] = calculate_energy_surplus(
                energy_balance_df['deficit_electricity'], energy_df['electricity_PV'])
            energy_balance_df['deficit_electricity'] = calculate_energy_deficit(
                energy_balance_df['deficit_electricity'], energy_df['electricity_PV'])

        # Battery Energy System
        if (battery is None) or (
                energy_balance_df.surplus_electricity.mean() == 0):
            energy_df['electricity_BES_in'] = 0
            energy_df['electricity_BES_out'] = 0
        else:
            BES_ = design_BES(battery=battery,
                              method='surplus',
                              electricity_demand=energy_balance_df['deficit_electricity'],
                              electricity_input=energy_balance_df['surplus_electricity'],
                              hours=24,
                              design_voltage=nominal_voltage(PVSystem_))

            EnergySystem_.BES = BES_

            bes_df = BES_storage_simulation(BES_,
                                            energy_balance_df['surplus_electricity'],
                                            energy_balance_df['deficit_electricity'])

            energy_df['electricity_BES_in'] = bes_df['electricity_BES_in']
            energy_df['electricity_BES_out'] = bes_df['electricity_BES_out']

            energy_balance_df['surplus_electricity'] = calculate_energy_surplus(
                energy_balance_df['deficit_electricity'], energy_df['electricity_BES_out'])
            energy_balance_df['deficit_electricity'] = calculate_energy_deficit(
                energy_balance_df['deficit_electricity'], energy_df['electricity_BES_out'])

        energy_df['electricity_Grid'] = energy_balance_df['deficit_electricity']

    else:
        energy_df = EnergySystem_.get_demand_file()

        # Energy Balance Dataframe
        energy_balance_df = pd.DataFrame(index=energy_df.index)
        energy_balance_df['surplus_electricity'] = 0
        energy_balance_df['deficit_electricity'] = energy_df['total_electricity_demand']
        energy_balance_df['surplus_heat'] = 0
        energy_balance_df['deficit_heat'] = energy_df['total_heat_demand']

        if EnergySystem_.PrimeMover is None:
            furnace_df = Furnace_energy(
                energy_df.total_heat_demand,
                Furnace_=EnergySystem_.Furnace)
            energy_df['heat_Furnace'] = furnace_df
            energy_df['PM_id'] = 'None'
            energy_df['heat_CHP'] = 0
            energy_df['electricity_CHP'] = 0
            energy_df['fuel_CHP'] = 0

            # Add to total supply
            energy_balance_df['surplus_heat'] = calculate_energy_surplus(
                energy_balance_df['deficit_heat'], energy_df['heat_Furnace'])
            energy_balance_df['deficit_heat'] = calculate_energy_deficit(
                energy_balance_df['deficit_heat'], energy_df['heat_Furnace'])
        else:
            energy_df['heat_Furnace'] = 0
            energy_df['PM_id'] = EnergySystem_.PrimeMover.PM_id

            if operation_mode == "FTL":
                FTL = True
            else:
                FTL = False

            try:
                filepath = r'model_outputs\CCHPvGrid\Energy_Supply\CHP'
                filename = F'{City_.name}_{Building_.building_type}.feather'
                datafile = filepath + filename
                chp_df = pd.read_feather(datafile)
                chp_df = chp_df[chp_df['PM_id'] == EnergySystem_.PrimeMover.PM_id]

            except FileNotFoundError:
                chp_df = CHP_energy(electricity_demand=energy_df.total_electricity_demand,
                                    heat_demand=energy_df.total_heat_demand,
                                    PrimeMover_=EnergySystem_.PrimeMover, FTL=FTL,
                                    thermal_distribution_loss_rate=thermal_distribution_loss_rate,
                                    thermal_distribution_loss_factor=thermal_distribution_loss_factor)
                chp_df = chp_df[chp_df['PM_id'] == EnergySystem_.PrimeMover.PM_id]

            chp_df.set_index('datetime', inplace=True)
            energy_df['heat_CHP'] = chp_df['heat_CHP']
            energy_df['electricity_CHP'] = chp_df['electricity_CHP']
            energy_df['fuel_CHP'] = chp_df['fuel_CHP']

            energy_balance_df['surplus_heat'] = calculate_energy_surplus(
                energy_balance_df['deficit_heat'], energy_df['heat_CHP'])
            energy_balance_df['deficit_heat'] = calculate_energy_deficit(
                energy_balance_df['deficit_heat'], energy_df['heat_CHP'])
            energy_balance_df['surplus_electricity'] = calculate_energy_surplus(
                energy_balance_df['deficit_electricity'], energy_df['electricity_CHP'])
            energy_balance_df['deficit_electricity'] = calculate_energy_deficit(
                energy_balance_df['deficit_electricity'], energy_df['electricity_CHP'])

        # Renewables
        if EnergySystem_.PVSystem is None:
            energy_df['electricity_PV'] = 0
        else:
            pv_age = EnergySystem_.PVsys_age

            pv_degradation_factor = (1 - (pv_age * pv_degradation_rate))

            try:
                filepath = r'model_outputs\CCHPvGrid\Energy_Supply\PV'
                filename = F'\\{City_.name}_{Building_.building_type}.feather'
                datafile = filepath + filename
                pv_df = pd.read_feather(datafile)
            except FileNotFoundError:
                pv_df = pv_simulation(EnergySystem_.PVSystem, City_)

            # Need to adjust index from PV simulation, or else you will get
            # NaNs
            pv_df.index = energy_df.index

            # Power ouputs from PV system are given in W
            energy_df['electricity_PV'] = (
                pv_df['p_ac'] / 1000) * pv_degradation_factor

            energy_balance_df['surplus_electricity'] = calculate_energy_surplus(
                energy_balance_df['deficit_electricity'], energy_df['electricity_PV'])
            energy_balance_df['deficit_electricity'] = calculate_energy_deficit(
                energy_balance_df['deficit_electricity'], energy_df['electricity_PV'])

        # Battery Energy System
        if (EnergySystem_.BES is None) or (
                energy_balance_df.surplus_electricity.mean() == 0):
            energy_df['electricity_BES_in'] = 0
            energy_df['electricity_BES_out'] = 0
        else:
            bes_df = BES_storage_simulation(EnergySystem_.BES,
                                            energy_balance_df['surplus_electricity'],
                                            energy_balance_df['deficit_electricity'])

            energy_df['electricity_BES_in'] = bes_df['electricity_BES_in']
            energy_df['electricity_BES_out'] = bes_df['electricity_BES_out']

            energy_balance_df['surplus_electricity'] = calculate_energy_surplus(
                energy_balance_df['deficit_electricity'], energy_df['electricity_BES_out'])
            energy_balance_df['deficit_electricity'] = calculate_energy_deficit(
                energy_balance_df['deficit_electricity'], energy_df['electricity_BES_out'])

        energy_df['electricity_Grid'] = energy_balance_df['deficit_electricity']

    return energy_df, EnergySystem_


def energy_flows(EnergySystem_=None):
    if EnergySystem_ is None:
        pass
    else:
        calculate_energy_flows()

    return EnergySystem_


def balance_energy_flows(scenario,
                         energy_demand_df,
                         demand_design_df,
                         supply_design_df,
                         Building_, City_,
                         AC_=None, Furnace_=None,
                         PrimeMover_=None, ABC_=None,
                         year=0,
                         pv_module=None, battery=None,
                         thermal_distribution_loss_rate=0.1,
                         thermal_distribution_loss_factor=1.0,
                         aggregate='A'):

    sys_design_ls = []
    system_design = pd.merge(
        demand_design_df,
        supply_design_df,
        left_on=[
            'Building',
            'City'])

    energy_demand_df.set_index('datetime', inplace=True, drop=True)
    # Cooling System
    ac_id = demand_design_df['AC_id'].item()
    if ac_id == 'None':
        abc_id = demand_design_df['ABC_id'].item()
        abc_modules = demand_design_df['num_ABC_modules'].item()
    else:
        ac_id = demand_design_df['AC_id'].item()
        ac_modules = demand_design_df['num_AC_modules'].item()

    if scenario == 'Reference':
        Furnace_ = Furnace_.size_system(
            energy_demand_df.total_heat_demand.max())
        system_design['Furnace_id'] = [Furnace_.Furnace_id]
        system_design['num_Furnace_modules'] = [Furnace_.number_of_modules]
        system_design['CHP_id'] = ['None']
        system_design['num_CHP_modules'] = [0]

        # Add furnace heat
        furnace_df = Furnace_energy(energy_demand_df.total_heat_demand,
                                    Furnace_)

        energy_flows_df = pd.merge(
            energy_demand_df,
            furnace_df,
            left_index=True,
            right_index=True)

        # Set CHP values to 0
        energy_flows_df['heat_CHP'] = 0
        energy_flows_df['electricity_CHP'] = 0

        # set PV values to 0
        energy_flows_df['electricity_PV'] = 0

        # set BES values to 0
        energy_flows_df['electricity_BES_in'] = 0
        energy_flows_df['electricity_BES_out'] = 0

        #

    if scenario == 'CCHP':
        # Merge the Dataframes
        # get energy demands
        pass
    if scenario == 'PV':
        # get energy demands
        # simulate BES
        pass
    if scenario == 'CCHP_PV':
        # get energy demands for cchp and pv
        # simulate BES

        pass

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

    # Determine energy demands
    energy_df = calculate_energy_demands(Building_=Building_, City_=City_,
                                         AC_=AC_, ABC_=ABC_,
                                         thermal_distribution_loss_rate=thermal_distribution_loss_rate,
                                         thermal_distribution_loss_factor=thermal_distribution_loss_factor)

    ###########################
    # Calculate Energy Supply #
    ###########################
    energy_balance_df = pd.DataFrame(index=energy_df.index)
    energy_balance_df['surplus_electricity'] = 0
    energy_balance_df['deficit_electricity'] = energy_df['total_electricity_demand']
    energy_balance_df['surplus_heat'] = 0
    energy_balance_df['deficit_heat'] = energy_df['total_heat_demand']

    # Primary Heating System
    if PrimeMover_ is None:
        Furnace_ = Furnace_.size_system(energy_df.total_heat_demand.max())

        system_design['Furnace_id'] = [Furnace_.Furnace_id]
        system_design['num_Furnace_modules'] = [Furnace_.number_of_modules]
        system_design['CHP_id'] = ['None']
        system_design['num_CHP_modules'] = [0]

        furnace_df = Furnace_energy(
            energy_df.total_heat_demand,
            Furnace_=Furnace_)
        energy_df['heat_Furnace'] = furnace_df
        energy_df['PM_id'] = 'None'
        energy_df['heat_CHP'] = 0
        energy_df['electricity_CHP'] = 0

        # Add to total supply
        energy_balance_df['surplus_heat'] = calculate_energy_surplus(
            energy_balance_df['deficit_heat'], energy_df['heat_Furnace'])
        energy_balance_df['deficit_heat'] = calculate_energy_deficit(
            energy_balance_df['deficit_heat'], energy_df['heat_Furnace'])
    else:
        energy_df['heat_Furnace'] = 0
        energy_df['PM_id'] = PrimeMover_.PM_id

        PrimeMover_ = size_chp(PrimeMover_,
                               energy_df,
                               City_,
                               operation_mode,
                               thermal_distribution_loss_rate,
                               thermal_distribution_loss_factor)

        system_design['Furnace_id'] = ['None']
        system_design['num_Furnace_modules'] = [0]
        system_design['CHP_id'] = [PrimeMover_.PM_id]
        system_design['num_CHP_modules'] = [PrimeMover_.number_of_modules]

        if operation_mode == "FTL":
            FTL = True
        else:
            FTL = False

        chp_df = CHP_energy(electricity_demand=energy_df.total_electricity_demand,
                            heat_demand=energy_df.total_heat_demand,
                            PrimeMover_=PrimeMover_, FTL=FTL,
                            thermal_distribution_loss_rate=thermal_distribution_loss_rate,
                            thermal_distribution_loss_factor=thermal_distribution_loss_factor)

        energy_df['heat_CHP'] = chp_df['heat_CHP']
        energy_df['electricity_CHP'] = chp_df['electricity_CHP']

        energy_balance_df['surplus_heat'] = calculate_energy_surplus(
            energy_balance_df['deficit_heat'], energy_df['heat_CHP'])
        energy_balance_df['deficit_heat'] = calculate_energy_deficit(
            energy_balance_df['deficit_heat'], energy_df['heat_CHP'])
        energy_balance_df['surplus_electricity'] = calculate_energy_surplus(
            energy_balance_df['deficit_electricity'], energy_df['electricity_CHP'])
        energy_balance_df['deficit_electricity'] = calculate_energy_deficit(
            energy_balance_df['deficit_electricity'], energy_df['electricity_CHP'])

    # Renewables
    if pv_module is None:
        energy_df['electricity_PV'] = 0

        system_design['num_PV_modules'] = [0]
    else:
        PVSystem_ = design_PVSystem(module=pv_module,
                                    method='design_area',
                                    design_area=Building_.roof_area,
                                    surface_tilt=City_.latitude)

        system_design['num_PV_modules'] = [total_pv_modules(PVSystem_)]

        pv_df = pv_simulation(PVSystem_, City_)

        # Need to adjust index from PV simulation, or else you will get NaNs
        pv_df.index = energy_df.index

        # Power ouputs from PV system are given in W
        energy_df['electricity_PV'] = pv_df['p_ac'] / 1000

        energy_balance_df['surplus_electricity'] = calculate_energy_surplus(
            energy_balance_df['deficit_electricity'], energy_df['electricity_PV'])
        energy_balance_df['deficit_electricity'] = calculate_energy_deficit(
            energy_balance_df['deficit_electricity'], energy_df['electricity_PV'])

    # Battery Energy System
    if (battery is None) or (energy_balance_df.surplus_electricity.mean() == 0):
        energy_df['electricity_BES_in'] = 0
        energy_df['electricity_BES_out'] = 0

        system_design['BES_id'] = ['None']
        system_design['num_BES_modules'] = [0]
    else:
        BES_ = design_BES(battery=battery,
                          method='surplus',
                          electricity_demand=energy_balance_df['deficit_electricity'],
                          electricity_input=energy_balance_df['surplus_electricity'],
                          hours=24,
                          design_voltage=nominal_voltage(PVSystem_))

        system_design['BES_id'] = [BES_.BES_id]
        system_design['num_BES_modules'] = [BES_.total_BES_modules()]

        bes_df = BES_storage_simulation(BES_,
                                        energy_balance_df['surplus_electricity'],
                                        energy_balance_df['deficit_electricity'])

        energy_df['electricity_BES_in'] = bes_df['electricity_BES_in']
        energy_df['electricity_BES_out'] = bes_df['electricity_BES_out']

        energy_balance_df['surplus_electricity'] = calculate_energy_surplus(
            energy_balance_df['deficit_electricity'], energy_df['electricity_BES_out'])
        energy_balance_df['deficit_electricity'] = calculate_energy_deficit(
            energy_balance_df['deficit_electricity'], energy_df['electricity_BES_out'])

    energy_df['electricity_Grid'] = energy_balance_df['deficit_electricity']

    system_design_df = pd.DataFrame.from_dict(system_design)

    return energy_df, system_design_df


def Furnace_energy(heat_demand,
                   Furnace_=None):
    furnace_df = pd.DataFrame(index=heat_demand.index)

    if Furnace_ is None:
        furnace_df['heat_furnace'] = 0
    else:
        # Size Furnace
        furnace_df['heat_furnace'] = heat_demand

    return furnace_df


def CHP_energy(electricity_demand, heat_demand,
               PrimeMover_=None,
               FTL=True,
               thermal_distribution_loss_rate=0.1,
               thermal_distribution_loss_factor=1.0):

    loss_factor = thermal_distribution_loss_factor * thermal_distribution_loss_rate

    chp_df = pd.DataFrame(index=electricity_demand.index)

    # heat_to_power_ratio = PrimeMover_.module_parameters['hpr']

    """
    ENERGY SUPPLY SIMULATION
    """
    if FTL is True:
        heat_chp = heat_demand
        
        # Heat is lost in distribution system
        heat_gen = heat_chp / (1 - loss_factor)
        electricity_chp = PrimeMover_.heat_to_electricity(heat_gen)

    else:
        electricity_chp = electricity_demand
        heat_gen = PrimeMover_.electricity_to_heat(electricity_chp)
        heat_chp = heat_gen * (1 - loss_factor)

    fuel_chp = PrimeMover_.electricity_to_fuel(electricity_chp)
    
    chp_df['electricity_CHP'] = electricity_chp
    chp_df['heat_CHP'] = heat_chp
    chp_df['fuel_CHP'] = fuel_chp
    chp_df['PM_id'] = PrimeMover_.PM_id

    chp_df.index = electricity_demand.index

    return chp_df


def Furnace_sim():
    Furnace_ = Furnace_.size_system(energy_df.total_heat_demand.max())

    system_design['Furnace_id'] = [Furnace_.Furnace_id]
    system_design['num_Furnace_modules'] = [Furnace_.number_of_modules]
    system_design['CHP_id'] = ['None']
    system_design['num_CHP_modules'] = [0]

    furnace_df = Furnace_energy(energy_df.total_heat_demand, Furnace_=Furnace_)
    energy_df['heat_Furnace'] = furnace_df
    energy_df['PM_id'] = 'None'
    energy_df['heat_CHP'] = 0
    energy_df['electricity_CHP'] = 0

    # Add to total supply
    energy_balance_df['surplus_heat'] = calculate_energy_surplus(
        energy_balance_df['deficit_heat'], energy_df['heat_Furnace'])
    energy_balance_df['deficit_heat'] = calculate_energy_deficit(
        energy_balance_df['deficit_heat'], energy_df['heat_Furnace'])


def CHP_sim(Building_, City_, PrimeMover_):
    # Primary Heating System
    if PrimeMover_ is None:
        Furnace_ = Furnace_.size_system(energy_df.total_heat_demand.max())

        system_design['Furnace_id'] = [Furnace_.Furnace_id]
        system_design['num_Furnace_modules'] = [Furnace_.number_of_modules]
        system_design['CHP_id'] = ['None']
        system_design['num_CHP_modules'] = [0]

        furnace_df = Furnace_energy(
            energy_df.total_heat_demand,
            Furnace_=Furnace_)
        energy_df['heat_Furnace'] = furnace_df
        energy_df['PM_id'] = 'None'
        energy_df['heat_CHP'] = 0
        energy_df['electricity_CHP'] = 0

        # Add to total supply
        energy_balance_df['surplus_heat'] = calculate_energy_surplus(
            energy_balance_df['deficit_heat'], energy_df['heat_Furnace'])
        energy_balance_df['deficit_heat'] = calculate_energy_deficit(
            energy_balance_df['deficit_heat'], energy_df['heat_Furnace'])
    else:
        energy_df['heat_Furnace'] = 0
        energy_df['PM_id'] = PrimeMover_.PM_id

        PrimeMover_ = size_chp(PrimeMover_,
                               energy_df,
                               City_,
                               operation_mode,
                               thermal_distribution_loss_rate,
                               thermal_distribution_loss_factor)

        system_design['Furnace_id'] = ['None']
        system_design['num_Furnace_modules'] = [0]
        system_design['CHP_id'] = [PrimeMover_.PM_id]
        system_design['num_CHP_modules'] = [PrimeMover_.number_of_modules]

        if operation_mode == "FTL":
            FTL = True
        else:
            FTL = False

        chp_df = CHP_energy(electricity_demand=energy_df.total_electricity_demand,
                            heat_demand=energy_df.total_heat_demand,
                            PrimeMover_=PrimeMover_, FTL=FTL,
                            thermal_distribution_loss_rate=thermal_distribution_loss_rate,
                            thermal_distribution_loss_factor=thermal_distribution_loss_factor)

        energy_df['heat_CHP'] = chp_df['heat_CHP']
        energy_df['electricity_CHP'] = chp_df['electricity_CHP']

        energy_balance_df['surplus_heat'] = calculate_energy_surplus(
            energy_balance_df['deficit_heat'], energy_df['heat_CHP'])
        energy_balance_df['deficit_heat'] = calculate_energy_deficit(
            energy_balance_df['deficit_heat'], energy_df['heat_CHP'])
        energy_balance_df['surplus_electricity'] = calculate_energy_surplus(
            energy_balance_df['deficit_electricity'], energy_df['electricity_CHP'])
        energy_balance_df['deficit_electricity'] = calculate_energy_deficit(
            energy_balance_df['deficit_electricity'], energy_df['electricity_CHP'])


def PV_sim():
    pass


def BES_sim():
    pass


def aggregate_energy_flows(df, aggregate):
    df.index = pd.to_datetime(df.index)
    agg_df = df.groupby(['City', 'Building',
                         'PM_id', 'AC_id', 'ABC_id'
                         ]).resample(F'{aggregate}').agg({
                             # DEMANDS
                             'electricity_demand': ['sum'],
                             'cooling_demand': ['sum'],
                             'heat_demand': ['sum'],
                             # Adjusted total electricity and heat
                             'total_electricity_demand': ['sum'],
                             'total_heat_demand': ['sum'],
                             'electricity_cooling': ['sum'],
                             'heat_cooling': ['sum'],
                             # SUPPLY
                             # Electricity
                             'electricity_CHP': ['sum'],
                             'electricity_PV': ['sum'],
                             'electricity_BES_in': ['sum'],
                             'electricity_BES_out': ['sum'],
                             'electricity_Grid': ['sum'],
                             # Heat
                             'heat_Furnace': ['sum'],
                             'heat_CHP': ['sum'],
                             # Fuel
                             'fuel_CHP': ['sum']
                             })

    agg_df.columns = agg_df.columns.map('_'.join)
    # agg_df.columns = agg_df.columns.droplevel(1)
    agg_df.rename(columns={'electricity_demand_sum': 'electricity_demand',
                           'cooling_demand_sum': 'cooling_demand',
                           'heat_demand_sum': 'heat_demand',
                           'total_electricity_demand_sum': 'total_electricity_demand',
                           'total_heat_demand_sum': 'total_heat_demand',
                           'electricity_cooling_sum': 'electricity_cooling',
                           'heat_cooling_sum': 'heat_cooling',
                           'electricity_CHP_sum': 'electricity_CHP',
                           'electricity_PV_sum': 'electricity_PV',
                           'electricity_BES_in_sum': 'electricity_BES_in',
                           'electricity_BES_out_sum': 'electricity_BES_out',
                           'electricity_Grid_sum': 'electricity_Grid',
                           'heat_Furnace_sum': 'heat_Furnace',
                           'heat_CHP_sum': 'heat_CHP',
                           'fuel_CHP_sum': 'fuel_CHP'}, inplace=True)

    agg_df.reset_index(inplace=True, drop=True)

    return agg_df


def emissions(energy_demand_df, year):
    pass


def inspect_files():

    save_path = r'model_outputs\CCHPvGrid'

    files_to_inspect = {'ac_demands': r'model_outputs\CCHPvGrid\Energy_Demands\AC\albuquerque_full_service_restaurant.feather',
                        'abc_demands':r'model_outputs\CCHPvGrid\Energy_Demands\ABC_SS\albuquerque_full_service_restaurant.feather',
                        'chp_supply':r'model_outputs\CCHPvGrid\Energy_Supply\CHP\albuquerque_full_service_restaurant.feather',
                        'pv_supply':r'model_outputs\CCHPvGrid\Energy_Supply\PV\albuquerque_full_service_restaurant.feather',
                        'cchp':r'model_outputs\CCHPvGrid\CCHP\albuquerque_full_service_restaurant.feather',
                        'pv':r'model_outputs\CCHPvGrid\PV\albuquerque_full_service_restaurant.feather',
                        'reference':r'model_outputs\CCHPvGrid\Reference\albuquerque_full_service_restaurant.feather',
                        'cchp_design':r'model_outputs\CCHPvGrid\CCHP\CCHP_design.feather',
                        'chp_design':r'model_outputs\CCHPvGrid\Energy_Supply\CHP\design.feather',
                        'abc_ss_design':r'model_outputs\CCHPvGrid\Energy_Demands\ABC_SS\ABC_SS_design.feather',
                        'ac_design':r'model_outputs\CCHPvGrid\Energy_Demands\AC\AC_design.feather',
                        'pv_design':r'model_outputs\CCHPvGrid\PV\PV_design.feather',
                        'pv_design_2':r'model_outputs\CCHPvGrid\Energy_Supply\PV\design.feather'}

    for i in files_to_inspect:
        df = pd.read_feather(files_to_inspect[i])
        df.to_csv(F'{save_path}\inspect_{i}.csv')

inspect_files()



########################
# Simulation Functions #
########################

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

