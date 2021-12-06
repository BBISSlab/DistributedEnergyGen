####################################
# LIBRARIES NEEDED TO RUN THE TOOL #
####################################
import iapws.iapws97 as thermo
import pathlib
from openpyxl import load_workbook
import math as m
import time
import datetime
import inspect
import os

# Scientific python add-ons
import pandas as pd
from pvlib.pvsystem import PVSystem     # To install: pip install pandas
import scipy.optimize
import scipy.stats
import numpy as np

from pv_system import design_PVSystem
from sysClasses import Building      # To install: pip install numpy


########################
# BatteryStorage Class #
########################

class BatteryStorage:
    """
    Because we are typically given the roundtrip efficiencies, this method assumes that we can store all the electricity
    input. The loss of energy is assumed to occur during discharge. This assumption holds if the maximum charging and
    discharging are equal.
    """

    def __init__(self, BES_id, name=None, battery_type=None,
                 module_parameters=None, cost_parameters=None,
                 inverter='', inverter_parameters=None,
                 batteries_per_string=1, parallel_battery_strings=1,
                 interface='dc', age=0):
        self.BES_id = BES_id
        self.name = name
        self.battery_type = battery_type
        self.module_parameters = module_parameters
        self.cost_parameters = cost_parameters
        self.inverter = inverter
        self.inverter_parameters = inverter_parameters
        self.batteries_per_string = batteries_per_string
        self.parallel_battery_strings = parallel_battery_strings
        self.interface = interface
        self.age = age

        # Module parameters: operational parameters
        if module_parameters is None:
            try:
                self.module_parameters = self._infer_battery_module_parameters()
            except KeyError:
                self.module_parameters = {}

        # Cost parameters:
        if cost_parameters is None:
            try:
                self.cost_parameters = self._infer_battery_cost_parameters()
            except KeyError:
                self.cost_parameters = {}

    def __repr__(self):
        attrs = ['BES_id', 'name', 'battery_type',
                 'module_parameters', 'cost_parameters',
                 'batteries_per_string', 'parallel_battery_strings',
                 'interface', 'age']
        return ('BES: \n ' + ' \n '.join('{}: {}'.format(attr,
                getattr(self, attr)) for attr in attrs))

    def _get_data(self):
        """
        This method extracts data from a dataframe or an csv file to populate the attributes of each BES.
        """
        batteries = retrieve_battery_specs()
        battery = batteries[self.BES_id]
        # GENERAL INFO
        self.name = f"{battery['manufacturer']}_{battery['model']}"
        self.battery_type = battery['chemistry']
        self.module_parameters = self._infer_battery_module_parameters()
        self.cost_parameters = self._infer_battery_cost_parameters()

        return self

    def _infer_battery_module_parameters(self, source='doe'):
        battery = retrieve_battery_specs()[self.BES_id]
        doe_battery = retrieve_doe_op_specs()[battery['technology']]

        module_parameters = {'technology': battery['technology'],
                             'chemistry': battery['chemistry'],
                             'capacity_kWh': battery['capacity_kWh'],
                             'capacity_Ah': battery['capacity_Ah'],
                             'depth_of_discharge_perc': doe_battery['depth_of_discharge_percent'],
                             'depth_of_discharge_kWh': battery['capacity_kWh'] * doe_battery['depth_of_discharge_percent'] / 100,
                             'peak_power_kW': battery['peak_power_kW'],
                             'power_cont_kW': battery['power_continuous_kW'],
                             # Convert from percent to fraction
                             'degradation_rate': doe_battery['degradation_rate_percent'] / 100,
                             'roundtrip_efficiency': doe_battery['roundtrip_efficiency'] / 100,
                             'voltage_nom': battery['volt_nominal_V'],
                             'endoflife_capacity_kWh': doe_battery['end_of_life_capacity_percent'] * battery['capacity_kWh'] / 100,
                             'lifetime_yr': doe_battery['lifetime_yr'],
                             'cycling_times': battery['cycling_times'],
                             }
        return module_parameters

    def _infer_battery_cost_parameters(self):
        battery = retrieve_battery_specs()[self.BES_id]
        doe_battery = retrieve_doe_op_specs()[battery['technology']]

        cost_parameters = {'A0': doe_battery['A0'],
                           'A1': doe_battery['A1'],
                           'om_cost_fraction': doe_battery['om_percent'] / 100
                           }
        return cost_parameters

    def increase_system_age(self, years=1):
        self.age += years
        if self.age >= self.module_parameters['lifetime_yr']:
            self.age = 0
            return self
        else:
            return self
    ##############################
    # BES Operational Parameters #
    ##############################

    def nominal_capacity(self):
        battery_capacity = self.module_parameters['capacity_kWh']
        return battery_capacity * self.number_of_batteries()

    def voltage(self):
        battery_voltage = self.module_parameters['voltage_nom']
        return battery_voltage * self.batteries_per_string

    def number_of_batteries(self):
        return self.parallel_battery_strings * self.batteries_per_string

    def power(self):
        unit_nominal_power = self.module_parameters['power_cont_kW']
        return unit_nominal_power * self.number_of_batteries()

    def depth_of_discharge(self, age=0):
        r'''
        Returns the depth of discharge of the battery corrected by the state
        of health.

        Parameters
        ----------

        '''
        unit_depth_of_discharge = self.module_parameters['depth_of_discharge_kWh']
        nominal_depth_of_discharge = self.number_of_batteries() * unit_depth_of_discharge
        return nominal_depth_of_discharge * self.state_of_health(age)

    def state_of_charge(self, BES_current_capacity, age=0):
        r'''
        Calculate the state of charge, relative to system capacity.

        Parameters
        ----------

        '''
        # print(F'BES_cap: {BES_current_capacity}')
        SoC = BES_current_capacity / self.state_of_health_capacity(age)
        if SoC is None:
            print('SoC is None\n')
            print(F'BES_current Cap: {BES_current_capacity}')
            print(F'SoC cap: {self.state_of_health_capacity(age)}')
            exit()
        if SoC > 1:
            SoC = 1
            return SoC
        else:
            return SoC

    def state_of_health(self, age=0.):
        r'''
        Calculate the state of health of the BES, relative to the initial
        system capacity.

        Parameters
        ----------

        '''
        degradation_factor = age * self.module_parameters['degradation_rate']
        return (1 - degradation_factor)

    def state_of_health_capacity(self, age=0):
        r'''
        Calculate the SoH capacity, considering the maximum capacity with aging.

        Parameters
        ----------

        '''
        return self.nominal_capacity() * self.state_of_health(age)

    def min_capacity(self, age=0):
        return (self.state_of_health_capacity(
            age) - self.depth_of_discharge(age)) / self.state_of_health_capacity(age)

    def min_state_of_charge(self, age=0):
        minimum_capacity = self.min_capacity(age)
        '''print('Min Capacity parameters')
        print(F"SoH Cap: {self.state_of_health_capacity(age)}")
        print(F"DoD: {self.depth_of_discharge(age)}")'''
        return self.state_of_charge(minimum_capacity, age)

    #######################
    # BES Cost Parameters #
    #######################
    def specific_cost(self):
        a0 = self.cost_parameters['A0']
        a1 = self.cost_parameters['A1']
        capacity = self.nominal_capacity()
        return a0 + a1 * np.log(capacity)  # $ / kWh

    def om_cost(self):
        specific_cost = self.specific_cost()
        return self.cost_parameters['om_cost_fraction'] * specific_cost

    def calculate_installation_cost(self):
        return self.specific_cost() * self.nominal_capacity()

    ######################
    # BES Sizing Methods #
    ######################

    def backup_BES_capacity(self, electricity_demand,
                            hours_of_autonomy=72, how='mean'):
        """
        This method determines the BES required to supply energy for a consecutive number of
        storage hours. This is determined by looking at the minimum, maximum, and mean sum of electricity
        used within the consecutive hours for the time specified.
        """
        # To Do:
        # how: peak shaving and other methods
        # The storage list contains all of the required storage values for a
        # consecutive number of storage hours.
        storage = []

        """
        The for-loop below and accompanying if-statement calculate the total electricity demand for all consecutive hours
        within the specified timeframe. For example, for 24 hours of storage, it will calculate the total energy storage for
        any consecutive 24 hour period.
        """
        for i in range(0, len(electricity_demand), hours_of_autonomy):
            if (i + hours_of_autonomy) < len(electricity_demand):
                storage.append(
                    sum(electricity_demand[i:i + hours_of_autonomy]))

        storage = np.array(storage)

        if how == 'mean':
            return storage.mean()
        if how == 'max':
            return storage.max()
        if how == 'min':
            return storage.min()

    def surplus_BES_capacity(self, electricity_supply,
                             hours_of_storage=72, how='mean'):
        """
        This method determines the BES required to supply energy for a consecutive number of
        storage hours. This is determined by looking at the minimum, maximum, and mean sum of electricity
        used within the consecutive hours for the time specified.
        """
        # To Do:
        # how: peak shaving and other methods
        # The storage list contains all of the required storage values for a
        # consecutive number of storage hours.
        storage = []

        """
        The for-loop below and accompanying if-statement calculate the total electricity demand for all consecutive hours
        within the specified timeframe. For example, for 24 hours of storage, it will calculate the total energy storage for
        any consecutive 24 hour period.
        """
        electricity_input_output = electricity_supply - electricity_demand

        for i in range(0, len(electricity_demand), hours_of_autonomy):
            if (i + hours_of_autonomy) < len(electricity_demand):
                storage.append(
                    sum(electricity_demand[i:i + hours_of_autonomy]))

        storage = np.array(storage)

        if how == 'mean':
            return storage.mean()
        if how == 'max':
            return storage.max()
        if how == 'min':
            return storage.min()

    def calculate_BES_voltage(self, PVSystem_, interface='dc'):
        from pv_system import nominal_voltage
        if interface == 'dc':
            return nominal_voltage(PVSystem_)
        if interface == 'ac':
            # System has built in inverter.
            return

    def calculate_batteries_per_string(self, BES_voltage=48):
        unit_battery_voltage = self.module_parameters['voltage_nom']
        return m.floor(BES_voltage / unit_battery_voltage)

    def calculate_total_parallel_strings(
            self, backup_BES_capacity, how='high'):
        unit_battery_capacity = self.module_parameters['capacity_kWh']
        min_number_of_batteries = backup_BES_capacity / unit_battery_capacity
        # Manufacturers typically limit maximum number of parallel strings to 3
        if how == 'high':
            return m.ceil(min_number_of_batteries / self.batteries_per_string)
        elif how == 'low':
            return m.floor(min_number_of_batteries / self.batteries_per_string)

    def calculate_min_bank_size(self, available_energy):
        # Available energy in kWh
        roundtrip_efficiency = self.module_parameters['roundtrip_efficiency']

        DoD_percent = self.module_parameters['depth_of_discharge_kWh'] / \
            self.module_parameters['capacity_kWh']

        # Divide the bank size by the RT efficiency and depth of discharge to obtain
        # the bank size for the available capacity.

        # ISSUE: available_energy a Nonetype

        bank_size = available_energy / \
            (roundtrip_efficiency * DoD_percent)  # kWh

        return bank_size

    def size_backup_BES_array(self, electricity_load,
                              hours, method='mean',
                              design_voltage=48):
        """
        This method determines the number of storage units required to supply energy for a consecutive number of
        'storage_hours' to the Building. This is determined by looking at the minimum, maximum, and mean sum of electricity
        used within the consecutive hours for the time specified.

        NEXT UPDATE MUST CHECK FOR THE NUMBER OF UNITS IN PARALLEL AND IN SERIES. CONNECTING BATTERIES IN PARALLEL DOUBLES
        THE POWER OUTPUT BUT MAINTAINS THE SAME VOLTAGE. CONNECTING THEM IN SERIES DOUBLES THE VOLTAGE, BUT MAINTAINS THE
        SAME POWER OUTPUT. We can call batteries in series a train.
        """
        # Calculate required size
        required_available_energy = calculate_BES_capacity(
            electricity_load, hours, method)

        bank_size = self.calculate_min_bank_size(required_available_energy)

        self.batteries_per_string = self.calculate_batteries_per_string(
            design_voltage)
        self.parallel_battery_strings = self.calculate_total_parallel_strings(
            bank_size)

        return self

    def size_surplus_BES_array(self, electricity_surplus,
                               hours, method='mean',
                               design_voltage=48):
        r"""
        REWORD
        This method determines the number of storage units required to store surplus energy for a consecutive number of
        'storage_hours' to the Building. This is determined by looking at the minimum, maximum, and mean sum of electricity
        used within the consecutive hours for the time specified.

        NEXT UPDATE MUST CHECK FOR THE NUMBER OF UNITS IN PARALLEL AND IN SERIES. CONNECTING BATTERIES IN PARALLEL DOUBLES
        THE POWER OUTPUT BUT MAINTAINS THE SAME VOLTAGE. CONNECTING THEM IN SERIES DOUBLES THE VOLTAGE, BUT MAINTAINS THE
        SAME POWER OUTPUT. We can call batteries in series a train.
        """
        # Calculate required size
        required_available_energy = calculate_BES_capacity(
            electricity_surplus, hours, method)
        bank_size = self.calculate_min_bank_size(required_available_energy)
        self.batteries_per_string = self.calculate_batteries_per_string(
            design_voltage)
        self.parallel_battery_strings = self.calculate_total_parallel_strings(
            bank_size)

        return self

    def size_BES_inverter(self, dc_power=0, ac_voltage=240):
        from pvlib import pvsystem
        sapm_inverters = pvsystem.retrieve_sam('cecinverter')

        df = sapm_inverters.transpose().reset_index().rename(
            columns={'index': 'inverter'})

        df = df[df['Vac']] == ac_voltage

        pass
        # print(df)

    def total_BES_modules(self):
        return self.batteries_per_string * self.parallel_battery_strings
    #################
    # BES Operation #
    #################
    '''
    To Do:
    - BES algorithm
    '''

    def charge(self, SoC, electricity_input=0, age=0, time_step=1):
        """
        Three cases exist for charging your battery:
        1) The battery is at capacity and cannot hold charge;
        2) The battery can hold all of the energy input;
        3) The battery can hold a portion of the energy input, but not all.
        """

        # Null Case
        if electricity_input == 0:
            return 0

        # Case 1
        elif (SoC) == 1:
            return 0

        # Cases 2 and 3. We assume the battery is not at capacity
        else:
            """
            The charging of the battery depends on its power-rating. We cannot force more electricity into the battery
            within a specified timeframe that exceeds the power rating. Accordingly, if we cannot store all of the energy
            within the timeframe, we may still have excess energy. The power-in is the average energy input over the
            specified timeframe.
            """
            electricity_stored = self.taper_power(electricity_input, time_step)

            electricity_stored = self.storage_limitation(
                SoC, electricity_stored, age)

            # Roundtrip efficiency considered here instead of at discharge
            return electricity_stored * \
                self.module_parameters['roundtrip_efficiency']

    def discharge(self, SoC, electricity_output=0, age=0, time_step=1):
        r"""
        Three cases exist for discharging your battery:
        1) The battery is depleted;
        2) The battery can meet all of the energy demand for a time period;
        3) The battery can meet a portion of the energy demand, but not all.

        Assumptions:
        The round-trip efficiency includes the DC/AC conversion losses of the inverter
        """

        # Null Case
        if electricity_output == 0:
            return 0

        # Case 1
        # BES is at or below minimum SoC
        if SoC <= self.min_state_of_charge(age):
            return 0

        # Case 2: The BES has enough energy to meet the demand for time t
        # Case 3: The BES can satisfy some of the demand, but not all, for time
        # t
        else:
            """
            The discharging of the battery depends on its power-rating. We cannot force provide more electricity from the
            battery within a specified timeframe that exceeds the power rating. Therefore, even if we have enough energy to
            meet the demand, we may still have a deficit IF we cannot meet it fast enough. Here, we assume that the
            power_output is the average energy demand over the specified timeframe
            """
            electricity_discharged = self.taper_power(
                electricity_output, time_step)

            electricity_discharged = electricity_discharged * -1.
            electricity_discharged = self.storage_limitation(
                SoC, electricity_discharged, age)

            # Roundtrip efficiency losses considered at the charging point
            return electricity_discharged

    # When discharging the BES, the power must also be converted from DC to AC.
    def BES_dc_to_ac(self, p_dc, inverter_parameters):
        from pvlib import inverter
        # Will be completed later
        # return inverter.sandia(self.voltage, p_dc, inverter_parameters)

    def update_BES_energy(self, initial_SoC, energy_input_output, age=0):
        initial_energy = initial_SoC * self.state_of_health_capacity(age)
        updated_energy = initial_energy + energy_input_output
        if self.state_of_charge(updated_energy, age) is None:
            print('SoC is None')
            print(F'initial SoC: {initial_SoC}')
            print(F'SoH Cap: {self.state_of_health_capacity(age)}')
            print(F'initial energy: {initial_energy}')
            print(F'E io: {energy_input_output}')
            print(F'updated energy: {updated_energy}')
            exit()
        return self.state_of_charge(updated_energy, age)

    def storage_limitation(self, SoC_initial, energy_change, age=0):
        r'''
        Checks that the energy change to the BES stays within the bounds of the
        maximum capacity and depth of discharge, with respect to the system's age

        Parameters
        ----------

        '''
        # Check that the SOC is less than or equal to 1
        # Check that the Energy is above the DoD
        maximum_capacity = self.state_of_health_capacity(age)
        minimum_capacity = self.min_capacity(age)

        energy_level_initial = SoC_initial * maximum_capacity
        energy_level_new = energy_level_initial + energy_change

        # Check lower bound
        if energy_level_new < minimum_capacity:
            allowable_energy_change = energy_level_initial - minimum_capacity
        # Check upper bound
        elif energy_level_new > maximum_capacity:
            allowable_energy_change = maximum_capacity - energy_level_initial
        else:
            allowable_energy_change = energy_change

        return allowable_energy_change

    def taper_power(self, electricity_input_output=0, time_step=1):
        # Ensure that the power input/output is positive to avoid errors
        power_input_output = np.abs(electricity_input_output / time_step)

        if power_input_output < self.power():
            return power_input_output * time_step
        elif power_input_output > self.power():
            # Any power greater than the power capacity of the BES is curtailed
            return self.power() * time_step
        else:
            return 0

    def self_discharge(self):
        # Values are currently included in the roundtrip efficiency
        pass

    def BES_storage_simulation(self, energy_io,
                               initial_state_of_charge=1):
        # Make sure that input_output is a list
        if isinstance(energy_io, pd.Series) or isinstance(
                energy_io, pd.DataFrame):
            energy_input_output = energy_io.to_list()
        else:
            energy_input_output = energy_io

        SoC = initial_state_of_charge
        # Initialize battery capacity
        BES_SoC = []  # [initial_state_of_charge]
        BES_io = []  # [0]

        # you will go by index
        for i in energy_input_output:
            # print(F'input SOC, BES_io, age: {SoC}, {BES_io}, {age}')
            if i > 0:
                # Charge
                energy_charged = self.charge(SoC, i, self.age)
                BES_io.append(energy_charged)
                SoC = self.update_BES_energy(SoC, energy_charged, self.age)
                if SoC is None:
                    print('Charging')
                    print(F'SoC: {SoC}')
                    print(F'E io: {i}')
                BES_SoC.append(SoC)
            elif i < 0:
                # Discharge
                energy_discharged = self.discharge(
                    SoC, i, self.age)  # discharge returns negative value
                BES_io.append(energy_discharged)
                SoC = self.update_BES_energy(SoC, energy_discharged, self.age)
                if SoC is None:
                    print('Discharging')
                    print(F'SoC: {SoC}')
                    print(F'E io: {i}')
                BES_SoC.append(SoC)
            else:
                BES_io.append(0)
                # Add a function for self discharge here
                SoC - self.update_BES_energy(SoC, i, self.age)
                if SoC is None:
                    print('Neither')
                    print(F'SoC: {SoC}')
                    print(F'E io: {i}')
                BES_SoC.append(SoC)
            # Add a case for self discharge
            # Maybe some self discharge
        # print(F'BES_io: {len(BES_io)}')
        # print(F'BES_SoC: {len(BES_SoC)}')
        # print(F'{energy_io.index}')
        # Need to restructure the data
        BES_power = pd.DataFrame(list(zip(BES_io, BES_SoC)), index=energy_io.index, columns=[
                                 'BES_energy_io', 'BES_SoC'])
        return BES_power


def retrieve_battery_specs():
    csvdata = 'data\\Tech_specs\\Battery_specs_2.0.csv'
    return _parse_raw_BES_df(csvdata)


def retrieve_doe_op_specs():
    csvdata = 'data\\Tech_specs\\doe_energy_storage_operational_params.csv'
    return _parse_raw_BES_df(csvdata)


def _parse_raw_BES_df(csvdata):
    df = pd.read_csv(csvdata, index_col=0, skiprows=0)
    df.columns = df.columns.str.replace(' ', '_')
    df = df.transpose()
    return df


def calculate_BES_capacity(electricity_load,
                           hours=72,
                           method='mean'):
    r'''
    This function calculates the total amount of energy required to satisfy
    X-hours of autonomous operation.

    '''

    bank_size = []

    for i in range(0, len(electricity_load), hours):
        if (i + hours) < len(electricity_load):
            bank_size.append(sum(electricity_load[i: i + hours]))

    bank_size = np.array(bank_size)

    if method == 'max':
        return bank_size.max()
    if method == 'min':
        return bank_size.min()
    if method == 'mean':
        return bank_size.mean()


def battery_cost_regression(csvdata, technology):
    from sklearn.linear_model import LinearRegression

    df = pd.read_csv(csvdata)
    X = df['technology', 'system_size_MW', 'storage_duration_hr']
    Y = df['installed_sys_cost_USD_per_kWh']
    model = LinearRegression()

    pass
########
# TEST #
########


def design_BES(battery,
               method='backup',
               electricity_demand=None,
               electricity_input=None,
               design_voltage=48,
               hours=72,
               interface='dc'):
    # All PV Energy
    # Backup days
    # Voltage Requirements (V)
    # Power Requirements
    # Capacity Requirements (Ah)
    # Method: Determines the design load
    # - Daily Consumption
    # - Peak Load
    # - Days of Autonomy
    # - Sun-hours
    # Method for full-system backup, partial-system backup & to maximize home usage
    # Design for AC and DC interfaces

    # Create BES
    BES = BatteryStorage(BES_id=battery)

    # Size the BES
    if method == 'backup':
        BES = BES.size_backup_BES_array(electricity_load=electricity_demand,
                                        hours=hours,
                                        design_voltage=design_voltage)
    if method == 'surplus':
        BES = BES.size_surplus_BES_array(electricity_surplus=electricity_input,
                                         hours=hours,
                                         design_voltage=design_voltage)

    return BES


def BES_storage_simulation(BES_,
                           energy_input, energy_output,
                           initial_state_of_charge=1):

    # Energy input values will be positive if flowing into the battery
    # and negative if flowing out.
    # Accordingly, energy_output should be negative values.
    energy_input_output = energy_input + energy_output

    BES_df = BES_.BES_storage_simulation(energy_input_output,
                                         initial_state_of_charge)

    electricity_BES_in = np.where(BES_df.BES_energy_io > 0,
                                  BES_df.BES_energy_io, 0)
    electricity_BES_out = np.where(BES_df.BES_energy_io < 0,
                                   BES_df.BES_energy_io, 0)
    electricity_BES_out = electricity_BES_out * -1

    df = pd.DataFrame(index=BES_df.index)
    df['electricity_BES_out'] = electricity_BES_out
    df['electricity_BES_in'] = electricity_BES_in

    return df


def calculate_peak_shaving():
    pass
# Sample PV output


def run_test():
    # Load PV output
    try:
        df = pd.read_csv(
            r'model_outputs\testing\building_pv.csv',
            index_col='datetime')
    except KeyError:
        df = test_pv()

    df['net_electricity'] = df.electricity_surplus + df.electricity_deficit
    print(df.net_electricity)

    battery = BatteryStorage(BES_id='Li4')
    print(battery)

    # print(df)


def test_BES_sim():
    '''
    Temporary function to make sure that the BES storage simulation is working properly.
    '''
    # Design PV System
    df = pd.read_csv(
        r'model_outputs\testing\building_pv.csv',
        index_col='datetime')

    '''from sysClasses import _generate_Cities
    City_dict = _generate_Cities(all_cities=False, selected_cities=['atlanta'])
    city = City_dict['atlanta']
    building = Building(
                    name='test', building_type='medium_office', City_=city)
    PVSystem_ = design_PVSystem(module ='Silevo_Triex_U300_Black__2014_',
                                method='design_area',
                                design_area=building.roof_area,
                                surface_tilt=city.latitude)
    '''

    BES_ = design_BES('Li3', method='surplus',
                      electricity_input=df.electricity_surplus,
                      hours=24,
                      design_voltage=400)

    bes_df = BES_storage_simulation(
        BES_, df.electricity_surplus, df.electricity_deficit)
    bes_df.to_csv(r'model_outputs\testing\BES_test.csv')


def _generate_BatteryStorage(csv_file, sheet_name=None, header=1):
    """
    Similar to the PrimeMover class, this function generaes a series of Battery Storage classes from a CSV file.
    The CSV file is read and  processed by the _generate_BES_dataframe function.
    """
    dataframe = _generate_BES_dataframe(csv_file=csv_file,
                                        sheet_name=sheet_name,
                                        header=header)
    i = 0
    BatteryStorage_dictionary = {}
    while i < dataframe.BES_id.count():

        # Create new Battery object
        current_BatteryStorage = BatteryStorage(dataframe.BES_id[i])
        current_BatteryStorage._get_data(dataframe=dataframe, index=i)
        BatteryStorage_dictionary[current_BatteryStorage.BES_id] = current_BatteryStorage
        i += 1

    return BatteryStorage_dictionary

##########################################
# End BatteryStorage Class and Functions #
##########################################

################################
# Thermal Energy Storage Class #
################################


class WaterTank:
    # To Do:
    # - Lower Bound 40 C
    # - Upper Bound 90 C
    # - Charging
    # - Discharging
    # - Energy
    # - kWh to Btu converter
    #

    def __init__(self, TES_id, name=None,
                 module_parameters=None, cost_parameters=None,
                 number_of_tanks=0, age=0,
                 heat_pump=None):
        # BASIC INFORMATION
        self.TES_id = TES_id
        self.name = name

        # PERFORMANCE METRICS
        # Capacity, DoD in kWh
        self.module_parameters = module_parameters
        self.cost_parameters = cost_parameters
        self.number_of_tanks = number_of_tanks

        if module_parameters is None:
            try:
                self.module_parameters = self._infer_water_tank_parameters()
            except KeyError:
                self.module_parameters = {}
        if cost_parameters is None:
            try:
                self.cost_parameters = self._infer_water_tank_cost_parameters()
            except KeyError:
                self.cost_parameters = {}

    def __repr__(self):
        attrs = ['TES_id', 'name',
                 'module_parameters', 'cost_parameters',
                 'number_of_tanks', 'age']
        return ('WaterTank: \n ' + ' \n '.join('{}: {}'.format(attr,
                getattr(self, attr)) for attr in attrs))

    def _get_data(self):
        pass

    def _infer_water_tank_parameters(self):
        pass

    def _infer_water_tank_cost_parameters(self):
        pass

    #####################################
    # Water Tank Operational Parameters #
    #####################################

    def storage_medium_mass(self, density=1000  # kg/L
                            ):
        volume = self.module_parameters['tank_volume']  # Tank volume in L
        return volume * density

    def nominal_capacity(self):
        r'''
        The difference in energy between the upper boundary and the lower boundary
        '''
        maximum_temp = self.module_parameters['max_temperature_C']
        minimum_temp = self.module_parameters['min_temperature_C']
        mass = self.storage_medium_mass()
        c_p = 4.184  # kJ / (kg * K)

        return mass * c_p * (maximum_temp - minimum_temp)

    def power(self):
        r'''
        This may depend on the coupled system
        '''
        pass

    def depth_of_discharge(self):
        pass

    def state_of_charge(self):
        r'''The temperature of the system, and the amound of energy that can be extracted.
        The SOC is 0 when the system is at the lower boundary
        '''
        pass

    def size_WaterTank(self):
        # Determine the required capacity
        # Make sure that
        pass

    def charge(self, temperature_initial, energy_in):
        pass

    def discharge(self, temperature_initial, energy_out):
        pass

    def change_in_heat(self, temperature_initial, temperature_final):
        r'''
        Calculate the change in temperature in the water tank
        as a result of heat input or output.

        delta_Q = m * c_p * delta_T

        Parameters
        ----------
        delta_Q : change in heat

        Output
        ------
        delta_T : change in temperature, in K or C

        '''

    def change_in_temperature(self, delta_Q, units='kWh'):
        r'''
        Calculate the change in temperature in the water tank
        as a result of heat input or output.

        delta_Q = m * c_p * delta_T

        Parameters
        ----------
        delta_Q : change in heat

        Output
        ------
        delta_T : change in temperature, in K or C

        '''

        import iapws.iapws97
        # Q = m * c_p * delta_T

        if units == 'kWh':
            delta_Q = 3600 * delta_Q

        c_p = 4.184  # kJ / (kg * K)
        mass = self.storage_medium_mass()

        delta_T = delta_Q / (mass * c_p)

        return delta_T


class ThermalBattery:
    pass


class ChemicalStorage:
    pass


# Testing
# water_heater = WaterTank(TES_id='TEST')

# print(thermo())
