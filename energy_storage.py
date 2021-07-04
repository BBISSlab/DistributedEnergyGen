####################################
# LIBRARIES NEEDED TO RUN THE TOOL #
####################################
import pathlib
from openpyxl import load_workbook
import math as m
import time
import datetime
import inspect
import os

# Scientific python add-ons
import pandas as pd     # To install: pip install pandas
import scipy.optimize
import scipy.stats
import numpy as np      # To install: pip install numpy


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
                batteries_per_string=1, parallel_battery_strings=1,
                interface='dc', age=0):
        # BASIC INFORMATION
        self.BES_id = BES_id
        self.name = name
        self.battery_type = battery_type
        self.module_parameters = module_parameters
        self.cost_parameters = cost_parameters
        self.batteries_per_string = batteries_per_string
        self.parallel_battery_strings = parallel_battery_strings
        self.interface = interface
        self.age = age

        if module_parameters is None:
            try:
                self.module_parameters = self._infer_battery_module_params()
            except KeyError:
                self.module_parameters = {}

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
        self.module_parameters = self._infer_battery_module_params()
        self.cost_parameters = self._infer_battery_cost_parameters()

        return self

    def _infer_battery_module_params(self):
        battery = retrieve_battery_specs()[self.BES_id]
        module_params = {'capacity_kWh': battery['capacity_kWh'],
                              'depth_of_discharge_kWh': battery['depth_of_discharge_kWh'],
                              'peak_power_kW': battery['peak_power_kW'],
                              'power_cont_kW': battery['power_continuous_kW'],
                              'degradation_rate': battery['degradation_rate'],
                              'roundtrip_efficiency': battery['roundtrip_efficiency'],
                              'voltage_nom': battery['volt_nominal_V'],
                              'endoflife_capacity_kWh': battery['end_of_life_capacity'],
                              'lifetime_yr': battery['lifetime_yr'],
                              'warranty': battery['warranty'],
                              'cycling_times': battery['cycling_times']
                              }
        return module_params

    def _infer_battery_cost_parameters(self):
        battery = retrieve_battery_specs()[self.BES_id]
        cost_params = {'capital_cost': battery['battery_cost'],
                       'install_cost': battery['install_cost'],
                       'total_cost': battery['total_cost'],
                       'specific_cost_$_per_kWh': battery['specific_cost']
                       }
        return cost_params


    ##########################
    # BES Operational Params #
    ##########################

    def BES_capacity(self):
        battery_capacity = self.module_parameters['capacity_kWh']
        return battery_capacity * self.number_of_batteries()
    
    def BES_voltage(self):
        battery_voltage = self.module_parameters['voltage_nom']
        return battery_voltage * self.batteries_per_string

    def number_of_batteries(self):
        return self.parallel_battery_strings * self.batteries_per_string

    def BES_power(self):
        unit_nominal_power = self.module_parameters['power_cont_kW']
        return unit_nominal_power * self.number_of_batteries()

    ######################
    # BES Sizing Methods #
    ######################

    def required_BES_capacity(self, electricity_demand, 
                            hours_of_autonomy=72, how = 'mean'):
        """
        This method determines the BES required to supply energy for a consecutive number of
        storage hours. This is determined by looking at the minimum, maximum, and mean sum of electricity
        used within the consecutive hours for the time specified.
        """

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
                storage.append(sum(electricity_demand[i:i + hours_of_autonomy]))

        storage = np.array(storage)
        
        if how == 'mean':
            return storage.mean()
        if how == 'max':
            return storage.max()
        if how == 'min':
            return storage.min()

    def calculate_BES_voltage(self):
        pass

    def calculate_batteries_per_string(self, BES_voltage):
        unit_battery_voltage = self.module_parameters['voltage_nom']
        # voltage must be smaller than the input voltage
        return m.floor(BES_voltage / unit_battery_voltage)

    def calculate_total_parallel_strings(self, required_BES_capacity):
        unit_battery_voltage = self.module_parameters['voltage_nom']
        # Manufacturers typically limit maximum number of parallel strings to 3
        return required_BES_capacity / unit_battery_voltage

    #################
    # BES Operation #
    #################

    def charge(self, surplus_electricity=0, time=1):
        """
        Three cases exist for charging your battery:
        1) The battery is at capacity and cannot hold charge;
        2) The battery can hold all of the energy input;
        3) The battery can hold a portion of the energy input, but not all.

        """
        # Null Case
        if surplus_electricity == 0:
            electricity_stored = 0

        # Case 1
        elif (self.sysSoC) == (self.sysCapacity):
            rejectedEnergyIn = surplus_electricity * self.num_units
            electricity_stored = 0

        # Cases 2 and 3. We assume the battery is not at capacity
        else:
            """
            The charging of the battery depends on its power-rating. We cannot force more electricity into the battery
            within a specified timeframe that exceeds the power rating. Accordingly, if we cannot store all of the energy
            within the timeframe, we may still have excess energy. The power-in is the average energy input over the
            specified timeframe.
            """
            power_in = surplus_electricity / time
            if power_in < self.sysPower:
                delta_electricity_in = power_in * time
            if power_in > self.sysPower:
                delta_electricity_in = self.sysPower * time
                rejectedElectricityIn = (power_in - self.sysPower) * time

            # Corrected for rountrip efficiency HERE, rather than in discharge
            delta_electricity_in = delta_electricity_in * self.RTE
            potential_sysSOC = self.sysSoC + delta_electricity_in

            # Case 2
            if (potential_sysSOC) < self.sysCapacity:
                self.sysSoC = potential_sysSOC
                electricity_stored = delta_electricity_in

            # Case 3
            else:
                self.sysSoC = self.sysCapacity
                rejectedElectricityIn = potential_sysSOC - self.sysCapacity
                electricity_stored = surplus_electricity - rejectedElectricityIn

        self.SoC = self.sysSoC / self.num_units
        self.sysDoD = self.DoD * \
            (self.sysSoC / self.sysCapacity) * self.num_units

        return electricity_stored

    def discharge(self, demanded_electricity=0, time=1):
        # NEED TO INCLUDE RT EFF
        """
        Three cases exist for discharging your battery:
        1) The battery is depleted;
        2) The battery can meet all of the energy demand for a time period;
        3) The battery can meet a portion of the energy demand, but not all.

        """
        power_out = demanded_electricity / time

        # Null Case
        if demanded_electricity == 0:
            electricity_supplied = 0
        # Case 1
        if self.sysSoC == 0:
            electricity_supplied = 0

        # Cases 2 and 3. We assume that the SOC > 0.
        else:
            """
            The discharging of the battery depends on its power-rating. We cannot force provide more electricity from the
            battery within a specified timeframe that exceeds the power rating. Therefore, even if we have enough energy to
            meet the demand, we may still have a deficit IF we cannot meet it fast enough. Here, we assume that the
            power_output is the average energy demand over the specified timeframe
            """
            if power_out < self.sysPower:
                delta_electricity_out = (power_out * time)
            else:
                # This assumes power_out > maximum discharge power
                delta_electricity_out = (self.sysPower * time)

            potential_sysSOC = self.sysSoC - delta_electricity_out

            # Case 2
            if potential_sysSOC >= self.sysDoD:  # If our battery has enough charge. DoD should be updated when charging the battery
                self.sysSoC = potential_sysSOC
                electricity_supplied = delta_electricity_out

                self.sysDoD -= electricity_supplied
            # Case 3
            else:  # We don't have enough (potential_SOC is negative)
                electricity_supplied = self.sysDoD  # Fully discharge
                self.sysSoC = 0
                self.sysDoD = 0

            electricity_deficit = demanded_electricity - electricity_supplied
            self.SoC = self.sysSoC / self.num_units
            electricity_supplied = electricity_supplied

        return electricity_supplied

def retrieve_battery_specs():
    csvdata = 'data\\Tech_specs\\Battery_specs.csv'
    return _parse_raw_BES_df(csvdata)


def _parse_raw_BES_df(csvdata):
    df = pd.read_csv(csvdata, index_col=0, skiprows=1)
    df.columns = df.columns.str.replace(' ', '_')
    df = df.transpose()
    return df


########
# TEST #
########

battery = BatteryStorage(BES_id='Li5')
print(battery)


def size_BatteryStorage(battery, design_voltage, method,
                        days_of_autonomy, interface='dc'):
    # All PV Energy
    # Backup days
    # Voltage Requirements (V)
    # Power Requirements
    # Capacity Requirements (Ah)
    # Method: Determines the design load
    # - Daily Consumption
    # - Peak Load
    # - Days of Autonomy
    if method == 'days_of_autonomy':
        required_BES_capacity = size_by_autonomous_days()
    # - Sun-hours
    # Method for full-system backup, partial-system backup & to maximize home usage
    # Design for AC and DC interfaces
    required_BES_capacity = (design_load) / \
        (depth_of_discharge) * temperature_modifier
    parallel_battery_strings = required_BES_capacity / unit_battery_capacity
    batteries_per_string = BES_voltage / unit_battery_voltage

    pass


def size_by_autonomous_days(electricity_demand, battery, storage_hours=72):
    """
    This method determines the number of storage units required to supply energy for a consecutive number of
    'storage_hours' to the Building. This is determined by looking at the minimum, maximum, and mean sum of electricity
    used within the consecutive hours for the time specified.

    NEXT UPDATE MUST CHECK FOR THE NUMBER OF UNITS IN PARALLEL AND IN SERIES. CONNECTING BATTERIES IN PARALLEL DOUBLES
    THE POWER OUTPUT BUT MAINTAINS THE SAME VOLTAGE. CONNECTING THEM IN SERIES DOUBLES THE VOLTAGE, BUT MAINTAINS THE
    SAME POWER OUTPUT. We can call batteries in series a train.
    """

    # The storage list contains all of the required storage values for a
    # consecutive number of storage hours.
    storage = []

    """
        The for-loop below and accompanying if-statement calculate the total electricity demand for all consecutive hours
        within the specified timeframe. For example, for 24 hours of storage, it will calculate the total energy storage for
        any consecutive 24 hour period.
        """
    for i in range(0, len(electricity_demand), storage_hours):
        if (i + storage_hours) < len(electricity_demand):
            storage.append(sum(electricity_demand[i:i + storage_hours]))

    storage = np.array(storage)

    if method == 'max':
        num_units = storage.max() // self.nom_capacity
    if method == 'min':
        num_units = storage.min() // self.nom_capacity
    if method == 'mean':
        num_units = storage.mean() // self.nom_capacity

    storage_capacity = num_units * self.nom_capacity

    self.num_units = num_units
    self.sysCapacity = self.nom_capacity * self.num_units
    self.sysPower = self.power_cont * self.num_units
    self.sysPower = self.power_cont * self.num_units

    capital_cost_BES = self.sysCapacity * self.specific_cost

    return storage_capacity, num_units, capital_cost_BES

# End BatteryStorage Methods #
##############################


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

    def __init__(self, TES_id, model='', manufacturer='',
                 nom_capacity=0, power=0,
                 degradation_rate=0, roundtripEfficiency=1,
                 lifetime=10, age=0, cycling_times=0, end_of_life_capacity=1, warranty=0,
                 volt_nom=0, battery_cost=0, install_cost=0, total_cost=0, specific_cost=0,
                 stateOfCharge=0, num_units=1):
        # BASIC INFORMATION
        self.BES_id = BES_id
        self.model = model
        self.manufacturer = manufacturer
        # PERFORMANCE METRICS
        # Capacity, DoD in kWh
        self.nom_capacity = nom_capacity
        # Power in kW
        self.power = power
        # Degradation rate in fraction of capacity loss
        self.degradation_rate = degradation_rate
        # RTE is a % but set here as a fraction
        self.RTE = roundtripEfficiency
        # Voltage
        self.volt_nom = volt_nom
        # End of Life is a fraction of total capacity
        self.EoL_cap = end_of_life_capacity
        # LIFECYCLE METRICS
        # Lifetime, warranty, and age in years, cycling times in cycles
        self.lifetime = lifetime
        self.warranty = warranty
        self.cycling_times = cycling_times
        self.age = age
        # COSTS
        # Battery, install, and capital costs in $. Specific cost in $/kWh
        self.battery_cost = battery_cost
        self.install_cost = install_cost
        self.total_cost = total_cost
        # OPERATIONAL PARAMETERS
        # SoC and all other parameters in kWh; number of units is integer
        self.temperature = temperature
        self.num_units = num_units
        self.sysCapacity = self.nom_capacity * self.num_units
        self.sysDoD = self.DoD * self.num_units
        self.sysSoC = self.SoC * self.num_units
        self.sysPower = self.power_cont * self.num_units
        # rejectedElectricityIn
        # self.rejectedElectricityIn = 0

    def __repr__(self):
        attrs = ['BES_id', ' model', 'manufacturer', 'chemistry', 'application',
                 'nom_capacity', 'DoD', 'peak_power', 'power_cont',
                 'degradation_rate', 'RTE', 'volt_nom', 'EoL_cap',
                 'lifetime', 'warranty', 'cycling_times', 'age',
                 'battery_cost', 'install_cost', 'total_cost', 'specific_cost',
                 'SoC', 'num_units', 'sysCapacity', 'sysDoD', 'sysSoC', 'sysPower']
        return ('Battery Storage: \n ' + ' \n '.join('{}: {}'.format(attr,
                getattr(self, attr)) for attr in attrs))

    def size_WaterTank():
        pass


class ThermalBattery:
    pass


class ChemicalStorage:
    pass
