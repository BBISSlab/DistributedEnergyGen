####################################
# LIBRARIES NEEDED TO RUN THE TOOL #
####################################
from logging import error
from msilib.schema import Error
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

from ypstruct import structure

##########################################################################

####################
# GLOBAL VARIABLES #
####################

specific_heat_water = 4.186  # kJ / (kg K)
specific_heat_water_vapor = 1.86  # kJ / (kg K)

# Unit Conversions


def convert_mbar_to_kPa(pressure):
    r'''
    Convert pressure value from mbar to kPa

    Parameters
    ----------
    pressure: air pressure (mbar)

    Output
    ------
    P_kPa: air pressure in kPa

    Conversion
    ----------
    10 millibar = 1 kPa
    '''
    return pressure / 10


def convert_C_to_K(temperature):
    return temperature + 273.15


def convert_K_to_C(temperature):
    return temperature - 273.15

###########################
# PROPERTIES OF HUMID AIR #
###########################

# Relative humidity calculations


def humidity_ratio(
        temperature=0, P_atm=101.325, relative_humidity=0):
    r'''
    Calculate the humidity ratio (or specific humidity) of air (kg water / kg dry air)

    Parameters
    ----------
    temperature: temperature of the air (C)
    P_atm: atmospheric pressure of air (kPa)
    relative_humidity: relative_humidity of the air, as a fraction [0 - 1]

    Output
    ------
    humidity_ratio: the mass fraction of water in humid air
    '''

    if (relative_humidity > 1):
        raise ValueError('Relative Humidity > 1')
    elif (relative_humidity < 0):
        raise ValueError('Relative Humidity < 0')
    else:
        P_sat = calculate_P_sat(temperature, 'C')
        humidity_ratio = (0.622 * relative_humidity * P_sat) / (P_atm - P_sat)
        return humidity_ratio


def mass_fraction_of_water_in_humid_air(humidity_ratio):
    return (1 / (1 + (1 / humidity_ratio)))

# Thermodynamic properties of humid air


def humid_air_enthalpy(temperature, pressure, relative_humidity,
                       method='iawps'):
    r'''
    Calculate the enthalpy (kJ/kg) of the humid air.

    Parameters
    ----------
    temperature: temperature of the air (C)
    pressure: pressure of the air (kPa)
    relative_humidity: relative_humidity of the air, as a fraction [0 - 1]

    Output
    ------
    enthalpy: enthalpy (kJ/kg) of humid air

    '''
    # Calculate the  humidity ratio
    HR = humidity_ratio(temperature=temperature,
                        P_atm=pressure,
                        relative_humidity=relative_humidity)

    if method == 'iawps':
        # Calcualte the mass fraction of water in humid air
        W = mass_fraction_of_water_in_humid_air(HR)

        # Convert pressure from kPa to MPA
        P_MPa = pressure / 1000
        # Convert temperature from C to K
        temp_K = convert_C_to_K(temperature)

        # Create Humid Air Class from iawps
        Humid_Air = humidAir.HumidAir(T=temp_K, P=P_MPa, W=W)

        return Humid_Air.h
    elif method == 'cengel':
        enthalpy_dry_air = 1.005 * temperature
        enthalpy_water_vapor = saturated_vapor_enthalpy(temperature)
        return enthalpy_dry_air + HR * enthalpy_water_vapor

    else:
        print('Choose iawps or cengel')

#####################################
# THERMODYNAMIC PROPERTIES OF WATER #
#####################################
# Generic thermodynamic property functions


def calculate_P_sat(temperature, temp_units="C"):
    r'''
    Calculate the saturation pressure of water.

    Parameters
    ----------
    temperature: temperature of the air, in C or K

    Output
    ------
    P_sat_kPa: saturation pressure of water (kPa)
    '''

    if temp_units == "K":
        P_sat_MPa = iapws97._PSat_T(temperature)
        P_sat_kPa = P_sat_MPa * 1000
        return P_sat_kPa
    elif temp_units == "C":
        temp_K = convert_C_to_K(temperature)
        return calculate_P_sat(temp_K, "K")
    else:
        print("Unit Error - enter temperature in C or K")
        exit()


def saturated_liquid_enthalpy(temperature):
    r'''
        Calculates the enthalpy of liquid water

        Parameters
        ----------
        temperature: liquid water temperature (C)

        Output
        ------
        enthalpy: enthalpy of saturated liquid water (kJ / kg)
    '''
    temp_K = convert_C_to_K(temperature)
    P_MPa = calculate_P_sat(temperature) / 1000
    saturated_liquid = iapws97._Region1(temp_K, P_MPa)

    enthalpy = saturated_liquid['h']

    return enthalpy


def saturated_vapor_enthalpy(temperature):
    r'''
        Calculates the enthalpy of saturated water vapor

        Parameters
        ----------
        temperature: saturated water vapor temperature (C)

        Output
        ------
        enthalpy: enthalpy of saturated water vapor (kJ / kg)
    '''
    temp_K = convert_C_to_K(temperature)
    P_MPa = calculate_P_sat(temperature) / 1000
    saturated_vapor = iapws97._Region2(temp_K, P_MPa)

    enthalpy = saturated_vapor['h']

    return enthalpy


def superheated_steam_enthalpy(temperature, pressure):
    T = convert_C_to_K(temperature)
    P = pressure / 1000
    superheated_steam = iapws97._Region2(T, P)

    return superheated_steam['h']


def latent_heat(temperature):
    h_f = saturated_liquid_enthalpy(temperature)
    h_g = saturated_vapor_enthalpy(temperature)
    return h_g - h_f


def _sat_liquid_obj_fcn(temperature, enthalpy):
    return saturated_liquid_enthalpy(temperature) - enthalpy


def saturated_liquid_temperature(enthalpy,
                                 percent_error_allowed=0.01):
    # Initialize Values using upper and lower bounds for
    # the current system
    t_U = convert_C_to_K(97)
    t_L = convert_C_to_K(4)

    f_U = _sat_liquid_obj_fcn(t_U, enthalpy)
    f_L = _sat_liquid_obj_fcn(t_L, enthalpy)

    # First Guess
    t_r = t_U - ((f_U * (t_L - t_U)) / (f_L - f_U))

    # Absolute Percent Relative Error = APRE
    apre = 100

    # Iteration counters
    i_U = 0
    i_L = 0

    iteration = 0
    while apre > percent_error_allowed:

        t_r_old = t_r
        f_r = _sat_liquid_obj_fcn(t_r, enthalpy)

        test = f_L * f_r
        if test < 0:
            t_U = t_r
            f_U = _sat_liquid_obj_fcn(t_U, enthalpy)
            i_U = 0
            i_L += 1
            if i_L >= 2:
                f_L = f_L / 2
        elif test > 0:
            t_L = t_r
            f_L = _sat_liquid_obj_fcn(t_L, enthalpy)
            i_L = 0
            i_U += 1
            if i_U >= 2:
                f_U = f_U / 2
        else:
            apre = 0

        t_r = t_U - ((f_U * (t_L - t_U)) / (f_L - f_U))

        apre = absolute_percent_relative_error(t_r_old, t_r)

        iteration += 1

    return round(t_r, 2)


def sat_liquid_temp_from_pressure(pressure):
    P = pressure / 1000
    T = iapws97._TSat_P(P)
    return convert_K_to_C(T)


def absolute_total_error(old_value, new_value):
    difference = old_value - new_value
    return abs(difference)


def absolute_percent_relative_error(old_value, new_value):
    absolute_relative_error = abs(
        absolute_total_error(
            old_value,
            new_value) / new_value)
    return absolute_relative_error * 100

####################
# Statepoint Class #
####################


class StatePoint:

    def __init__(self, position, temperature, temperature_min,
                 temperature_max, pressure, enthalpy, fluid_type, mass_flowrate):
        self.position = position
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.pressure = pressure
        self.enthalpy = enthalpy
        self.fluid_type = fluid_type
        self.mass_flowrate = mass_flowrate

    def __repr__(self):
        attrs = ['position', 'fluid_type', 'mass_flowrate'
                 'temperature', 'temperature_min', 'temperature_max',
                 'pressure', 'enthalpy']
        return ('Statpoint: \n ' + ' \n '.join('{}: {}'.format(attr, getattr(self, attr))
                                               for attr in attrs))

    def check_min_temp(self):
        pass

    def check_max_temp(self):
        pass


class Water:

    def __init__(self, temperature, 
                pressure, state,
                temperature_min=3, temperature_max=120, 
                enthalpy=None):
        r'''
            t:  solution temperature, deg C
            t': refrigerant temperature, deg C
            T': refrigerant temperature, K
            P:  saturation pressure, kPa
        '''
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.pressure = pressure
        self.state = state
        self.enthalpy = enthalpy

    def Psat_T(self, temp_units="C"):
        r'''
        Calculate the saturation pressure of water.

        Parameters
        ----------
        temperature: temperature of the air, in C or K

        Output
        ------
        P_sat_kPa: saturation pressure of water (kPa)
        '''

        if temp_units == "K":
            P_sat_MPa = iapws97._PSat_T(self.temperature)
            P_sat_kPa = P_sat_MPa * 1000
            return P_sat_kPa
        elif temp_units == "C":
            temp_K = convert_C_to_K(self.temperature)
            return calculate_P_sat(temp_K, "K")
        else:
            print("Unit Error - enter temperature in C or K")
            exit()

    def saturated_liquid_enthalpy(self):
        r'''
            Calculates the enthalpy of liquid water

            Parameters
            ----------
            temperature: liquid water temperature (C)

            Output
            ------
            enthalpy: enthalpy of saturated liquid water (kJ / kg)
        '''
        temp_K = convert_C_to_K(self.temperature)
        P_MPa = calculate_P_sat(self.temperature) / 1000
        saturated_liquid = iapws97._Region1(temp_K, P_MPa)

        enthalpy = saturated_liquid['h']

        return enthalpy

    def saturated_vapor_enthalpy(self):
        r'''
            Calculates the enthalpy of saturated water vapor

            Parameters
            ----------
            temperature: saturated water vapor temperature (C)

            Output
            ------
            enthalpy: enthalpy of saturated water vapor (kJ / kg)
        '''
        temp_K = convert_C_to_K(self.temperature)
        P_MPa = calculate_P_sat(self.temperature) / 1000
        saturated_vapor = iapws97._Region2(temp_K, P_MPa)

        enthalpy = saturated_vapor['h']

        return enthalpy

    def superheated_steam_enthalpy(self):
        T = convert_C_to_K(self.temperature)
        P = self.pressure / 1000
        superheated_steam = iapws97._Region2(T, P)

        return superheated_steam['h']

    def _sat_liquid_obj_fcn(self, enthalpy):
        return saturated_liquid_enthalpy(self.temperature) - enthalpy

    def saturated_liquid_temperature(enthalpy,
                                     percent_error_allowed=0.01):
        # Initialize Values using upper and lower bounds for
        # the current system
        t_U = convert_C_to_K(97)
        t_L = convert_C_to_K(4)

        f_U = _sat_liquid_obj_fcn(t_U, enthalpy)
        f_L = _sat_liquid_obj_fcn(t_L, enthalpy)

        # First Guess
        t_r = t_U - ((f_U * (t_L - t_U)) / (f_L - f_U))

        # Absolute Percent Relative Error = APRE
        apre = 100

        # Iteration counters
        i_U = 0
        i_L = 0

        iteration = 0
        while apre > percent_error_allowed:

            t_r_old = t_r
            f_r = _sat_liquid_obj_fcn(t_r, enthalpy)

            test = f_L * f_r
            if test < 0:
                t_U = t_r
                f_U = _sat_liquid_obj_fcn(t_U, enthalpy)
                i_U = 0
                i_L += 1
                if i_L >= 2:
                    f_L = f_L / 2
            elif test > 0:
                t_L = t_r
                f_L = _sat_liquid_obj_fcn(t_L, enthalpy)
                i_L = 0
                i_U += 1
                if i_U >= 2:
                    f_U = f_U / 2
            else:
                apre = 0

            t_r = t_U - ((f_U * (t_L - t_U)) / (f_L - f_U))

            apre = absolute_percent_relative_error(t_r_old, t_r)

            iteration += 1

        return round(t_r, 2)

    def sat_liquid_temp_from_pressure(pressure):
        P = pressure / 1000
        T = iapws97._TSat_P(P)
        return convert_K_to_C(T)


#############################
# Lithium Bromide Equations #
#############################


class LiBr_solution:

    def __init__(self, mass_fraction,
                 mass_fraction_min=0.4, mass_fraction_max=0.7,
                 temp_min=15, temp_max=165,
                 Duhring_coefficients={'a0': 0.538, 'a1': 0.845, 'b0': 48.3, 'b1': -35.6}):
        r'''
            Chapter 30: Thermophysical Properties of Refrigerants. ASHRAE (2009)

            Enthalpy-Concentration diagram for water / LiBr Solutions

            Equation is valid for:
                mass fraction range 0.40 < X < 0.70 LiBr
                temperature range 15 < t < 165 deg C

            t:  solution temperature, deg C
            t': refrigerant temperature, deg C
            T': refrigerant temperature, K
            P:  saturation pressure, kPa
        '''
        self.mass_fraction = mass_fraction
        self.mass_fraction_min = mass_fraction_min
        self.mass_fraction_max = mass_fraction_max
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.Duhring_coefficients = Duhring_coefficients

    def enthalpy_LiBr_solution(self, solution_temperature=0):
        r'''
            Chapter 30: Thermophysical Properties of Refrigerants. ASHRAE (2009)

            Enthalpy-Concentration diagram for water / LiBr Solutions

            Equation is valid for:
                concentration range 40 < X < 70 percent LiBr
                temperature range 15 < t < 165 deg C

            t:  solution temperature, deg C
            t': refrigerant temperature, deg C
            T': refrigerant temperature, K
            P:  saturation pressure, kPa

            h = sum(A_n * X^n, 0, 4) + t * sum(B_n * X^n, 0, 4) + t^2 * sum(C_n * X^n, 0, 4)
        '''
        # Coefficients
        A_n = [-2024.33, 163.309, -4.88161,
               6.302948 * 10**-2, -2.913705 * 10**-4]
        B_n = [18.2829, -1.1691757, 3.248041 * 10**-
               2, -4.034184 * 10**-4, 1.8520569 * 10**-6]
        C_n = [-3.7008214 * 10**-2, 2.8877666 * 10**-3, -8.1313015 *
               10**-5, 9.9116628 * 10**-7, -4.4441207 * 10**-9]

        A = self._LiBr_summation(coef=A_n, x=self.mass_fraction)
        B = self._LiBr_summation(coef=B_n, x=self.mass_fraction)
        C = self._LiBr_summation(coef=C_n, x=self.mass_fraction)

        return(A + solution_temperature * B + solution_temperature**2 * C)

    def _LiBr_summation(self, n=5, coef=[], x=0):
        summation = 0
        X = x * 100
        for i in range(n):
            summation += coef[i] * X**i
        return summation

    def solution_temp(self, refrigerant_temp, A_n=[], B_n=[]):
        """
        Incomplete
        """
        A = self._LiBr_summation(n=4, coef=A_n, x=self.mass_fraction)
        B = self._LiBr_summation(n=4, coef=B_n, x=self.mass_fraction)

        return B + refrigerant_temp * A

    def refrigerant_temp(self, solution_temp, A_n=[], B_n=[]):
        """
        Incomplete
        """
        A = self._LiBr_summation(n=4, coef=A_n, x=self.mass_fraction)
        B = self._LiBr_summation(n=4, coef=B_n, x=self.mass_fraction)

        return (solution_temp - B) / A

    def pressure_sat(self, refrigerant_temp, C=7.05, D=-1596.49, E=-104095.5):
        T = convert_C_to_K(refrigerant_temp)
        return 10**(C + D / T + E / (T**2))

    def refrigerant_temp_from_pressure(self, pressure_sat, C, D, E):
        T = (-2 * E) / (D + (D**2 - 4 * E * (C - np.log10(pressure_sat)))**0.5)
        refrigerant_temp = T - 273.15
        if refrigerant_temp < 0:
            print("refrigerant temperature outside possible range")
        else:
            return refrigerant_temp

    def calc_temp_from_enthalpy(self, enthalpy):
        r'''
            Chapter 30: Thermophysical Properties of Refrigerants. ASHRAE (2009)

            Enthalpy-Concentration diagram for water / LiBr Solutions

            Equation is valid for:
                concentration range 40 < X < 70 percent LiBr
                temperature range 15 < t < 165 deg C

            Parameters
            ----------
            h: enthalpy of LiBr solution (kJ / kg)

            Output
            ------
            t:  solution temperature, deg C

            Equation
            --------
            0 = sum(A_n * X^n, 0, 4) + t * sum(B_n * X^n, 0, 4) + t^2 * sum(C_n * X^n, 0, 4) - h
        '''
        from sympy.solvers import solve
        from sympy import Symbol

        # Coefficients
        A_n = [-2024.33, 163.309, -4.88161,
               6.302948 * 10**-2, -2.913705 * 10**-4]
        B_n = [18.2829, -1.1691757, 3.248041 * 10**-
               2, -4.034184 * 10**-4, 1.8520569 * 10**-6]
        C_n = [-3.7008214 * 10**-2, 2.8877666 * 10**-3, -8.1313015 *
               10**-5, 9.9116628 * 10**-7, -4.4441207 * 10**-9]

        A = self._LiBr_summation(coef=A_n, x=self.mass_fraction)
        B = self._LiBr_summation(coef=B_n, x=self.mass_fraction)
        C = self._LiBr_summation(coef=C_n, x=self.mass_fraction)

        t = Symbol('t')

        solutions = solve((A - enthalpy) + (B * t) + (C * t**2), t)

        for i in solutions:
            if isinstance(i, tuple):
                i = i[0]

            if i > 0:
                return i
            else:
                pass

    def Duhring_equilibrium_temperature(self, solvent_dew_point_temp):
        a0 = self.Duhring_coefficients['a0']
        a1 = self.Duhring_coefficients['a1']
        b0 = self.Duhring_coefficients['b0']
        b1 = self.Duhring_coefficients['b1']
        x = self.mass_fraction

        ts_K = convert_C_to_K(solvent_dew_point_temp)
        teq_K = (a0 * x + a1) * ts_K + (b0 * x + b1)
        return convert_K_to_C(teq_K)

    def Duhring_equilibrium_concentration(
            self, solvent_dew_point_temp, solution_temp):
        a0 = self.Duhring_coefficients['a0']
        a1 = self.Duhring_coefficients['a1']
        b0 = self.Duhring_coefficients['b0']
        b1 = self.Duhring_coefficients['b1']
        ts_K = convert_C_to_K(solvent_dew_point_temp)
        t_K = convert_C_to_K(solution_temp)

        return (t_K - (a1 * ts_K + b1)) / (a0 * ts_K + b0)

    def isobaric_heat_capacity(self, solution_temperature, method="Ren"):
        t = solution_temperature
        x = self.mass_fraction

        if method == 'Brak-Cyklis':
            # Coefficients
            N = np.arange(0, 6, 1)
            A_n = [4.124891, -7.643903 * 10**-2,
                   2.589577 - 10**-3, -9.500522 * 10**-5,
                   1.708026 * 10**-6, -1.102363 * 10**-8]
            B_n = [5.743693 * 10**-4, 5.870921 * 10**-4,
                   -7.375319 * 10**-6, 3.277592 * 10**-7,
                   -6.062304 * 10**-9, 3.901897 * 10**-11]

            isobaric_heat_capacity = 0
            for n in N:
                isobaric_heat_capacity += (A_n[n]
                                           * x**n) + (2 * B_n[n] * x**n * t)

            return isobaric_heat_capacity

        elif method == 'Patek-Klomfar':

            # Constants
            cp_t = 76.0226  # J / mol-K
            T_c = 647.096  # K
            T_0 = 221  # K

            # Mass and mole fractions
            x = self.mass_fraction
            mu_LiBr = 0.08685  # kg/mol
            mu_water = 0.01802  # kg/mol
            mol_fraction = ((x / mu_LiBr) / (x / mu_LiBr + (1 - x) / mu_water))

            # Coefficients and exponents
            I = np.arange(0, 8, 1)
            m_i = [2, 3, 3, 3, 3, 2, 1, 1]
            n_i = [0, 0, 1, 2, 3, 0, 3, 2]
            t_i = [0, 0, 0, 0, 0, 2, 3, 4]
            a_i = [-1.42094 * 10, 4.04943 * 10,
                   1.11135 * 10**2, 2.29980 * 10**2,
                   1.34526 * 10**3, -1.41010 * 10**-2,
                   1.24977 * 10**-2, -6.83209 * 10**-4]

        elif method == 'Ren':
            I = np.arange(0, 3, 1)
            A_n = [4.07, -5.123, 2.297]
            B_n = [9.92 * 10**-4, 6.29 * 10**-3, -9.38 * 10**-3]
            C_n = [-1.1988 * 10**-5, 4.3855 * 10**-6, 1.2567 * 10**-5]

            P = []
            for i in I:
                P.insert(i, A_n[i] + B_n[i] * t + C_n[i] * t**2)

            isobaric_heat_capacity = P[0] + P[1] * x + P[2] * x**2

            return isobaric_heat_capacity

    def check_mass_fraction(self):
        if self.mass_fraction > self.mass_fraction_max:
            self.mass_fraction = self.mass_fraction_max
        elif self.mass_fraction < self.mass_fraction_min:
            self.mass_fraction = self.mass_fraction_min
        else:
            pass
####################################
# GENERIC HEAT EXCHANGER EQUATIONS #
####################################

def calculate_Tc_in(Q, UA,
                                  Th_in, Th_out,
                                  Tc_out):
    r'''
        Calculates the refrigerant temperature at inlet (T_3)

                                    ===============
          HOT FLUID IN (T_H1) ====> [  GENERIC   ] ====> HOT FLUID OUT (T_H2)
                                    [  HEAT      ]
        COLD FLUID OUT (T_C2) <==== [  EXCHANGER ] <==== COLD FLUID IN (T_C1)
                                    ==============

        Parameters
        ----------
        Q: the heat transfered through the heat exchanger (kJ)
        Th_in: hot temperature in (C)
        Th_out: hot fluid temperature out (C)
        Tc_out: cold fluid temperature out (C)

        Output
        ------
        Tc_in: cold fluid temperature in

        Assumptions
        -----------
        1. LMTD Method
        2. Chen's approximation for LMTD [1]
        3. Counter-flow heat exchanger

        Equations
        ---------
        Q = (UA) * LMTD
        LMTD ~ {theta_1 * theta_2 * [(theta_1 + theta_2) / 2]} ^ (1/3)
        theta_1 = T_H1 - T_C2
        theta_2 = T_H2 - T_C1

        References
        ----------
        [1] Chen, J. J. J. (2019). Logarithmic mean : Chen ’ s approximation or
            explicit solution ? Computers and Chemical Engineering, 120, 1–3.
            https://doi.org/10.1016/j.compchemeng.2018.10.002
        '''
    from sympy.solvers import solve
    from sympy import Symbol

    # theta_1 = temp diff between chilled water in and refrigerant out
    theta_1 = Th_in - Tc_out

    # x used as an unknown variable for theta_2
    x = Symbol('x')

    solutions = solve((((theta_1)**2) * x) +
                      (theta_1 * x**2) - 2 * (Q / UA)**3, x)

    for i in solutions:
        if isinstance(i, tuple):
            i = i[0]

        if i > 0:
            theta_2 = i
            Tc_in = Th_out - theta_2
            return Tc_in
        else:
            return 4


def calculate_Tc_out(Q, UA,
                                   Th_in, Th_out,
                                   Tc_in):
    r'''
        Calculates the refrigerant temperature at inlet (T_C2)

                                    ===============
          HOT FLUID IN (T_H1) ====> [  GENERIC   ] ====> HOT FLUID OUT (T_H2)
                                    [  HEAT      ]
        COLD FLUID OUT (T_C2) <==== [  EXCHANGER ] <==== COLD FLUID IN (T_C1)
                                    ==============

        Parameters
        ----------
        Q: the heat transfered through the heat exchanger (kJ)
        Th_in: hot temperature in (C)
        Th_out: hot fluid temperature out (C)
        Tc_in: cold fluid temperature in (C)

        Output
        ------
        Tc_out: cold fluid temperature out (C)

        Assumptions
        -----------
        1. LMTD Method
        2. Chen's approximation for LMTD [1]
        3. Counter-flow heat exchanger

        Equations
        ---------
        Q = (UA) * LMTD
        LMTD ~ {theta_1 * theta_2 * [(theta_1 + theta_2) / 2]} ^ (1/3)
        theta_1 = T_H1 - T_C2
        theta_2 = T_H2 - T_C1

        References
        ----------
        [1] Chen, J. J. J. (2019). Logarithmic mean : Chen ’ s approximation or
            explicit solution ? Computers and Chemical Engineering, 120, 1–3.
            https://doi.org/10.1016/j.compchemeng.2018.10.002
        '''
    from sympy.solvers import solve
    from sympy import Symbol

    # theta_1 = temp diff between chilled water in and refrigerant out
    theta_2 = Th_out - Tc_in

    # x used as an unknown variable for theta_2
    x = Symbol('x')

    solutions = solve((((theta_2)**2) * x) +
                      (theta_2 * x**2) - 2 * (Q / UA)**3, x)

    for i in solutions:
        if isinstance(i, tuple):
            i = i[0]
        if i > 0.0:
            theta_1 = i
        else:
            pass

    Tc_out = Th_in - theta_1
    return Tc_out

def calculate_Th_out(Q, UA, Th_in, Tc_in, Tc_out):
    r'''
        Calculates the refrigerant temperature at inlet (T_3)

                                    ===============
          HOT FLUID IN (T_H1) ====> [  GENERIC   ] ====> HOT FLUID OUT (T_H2)
                                    [  HEAT      ]
        COLD FLUID OUT (T_C2) <==== [  EXCHANGER ] <==== COLD FLUID IN (T_C1)
                                    ==============

        Parameters
        ----------
        Q: the heat transfered through the heat exchanger, kJ
        Th_in: hot temperature in, C
        Tc_in: cold fluid temperature in, C
        Tc_out: cold fluid temperature out, C

        Output
        ------
        Th_out: hot fluid temperature out, C

        Assumptions
        -----------
        1. LMTD Method
        2. Chen's approximation for LMTD [1]
        3. Counter-flow heat exchanger

        Equations
        ---------
        Q = (UA) * LMTD
        LMTD ~ {theta_1 * theta_2 * [(theta_1 + theta_2) / 2]} ^ (1/3)
        theta_1 = T_H1 - T_C2
        theta_2 = T_H2 - T_C1

        References
        ----------
        [1] Chen, J. J. J. (2019). Logarithmic mean : Chen ’ s approximation or
            explicit solution ? Computers and Chemical Engineering, 120, 1–3.
            https://doi.org/10.1016/j.compchemeng.2018.10.002
        '''
    from sympy.solvers import solve
    from sympy import Symbol

    # theta_1 = temp diff between chilled water in and refrigerant out
    theta_1 = Th_in - Tc_out

    # x used as an unknown variable for theta_2
    x = Symbol('x')

    solutions = solve((((theta_1)**2) * x) +
                      (theta_1 * x**2) - 2 * (Q / UA)**3, x)

    for i in solutions:
        if i > 0:
            theta_2 = i
        else:
            pass

    Th_out = Tc_in + theta_2
    return Th_out


#################################
# Absorption Chiller Components #
#################################
# TODO
# - Make each component a subclass of a heat-exchanger class

# Absorption Chiller Components and Component Modeling


def calc_circulation_ratio(x_ss=0.6, x_ws=0.57):
    r'''
        Calculates the circulation ratio of the LiBr

        Parameters
        ----------
        x_ss: mass fraction of the strong solution
        x_ws: mass fraction of the weak solution

        Output
        ------
        circulation_ratio: ratio of the strong LiBr solution to the mass flowrate
            of the refrigerant

        Equation
        --------
        CR = x_ws / (x_ss - x_ws)
        CR: Circulation Ratio
        x_ws: concentration of weak solution
        x_ss: concentration of strong solution
    '''
    circulation_ratio = x_ws / \
        (x_ss - x_ws)
    return circulation_ratio


def strong_solution_flowrate(circulation_ratio, refrigerant_mass_flowrate):
    return circulation_ratio * refrigerant_mass_flowrate


def weak_solution_flowrate(circulation_ratio, refrigerant_mass_flowrate):
    return (1 + circulation_ratio) * refrigerant_mass_flowrate


class Generator:
    # TODO
    # Calculate m_S and m_W
    # Calculate enthalpy of strong solution
    # Calculate Q_g
    # Calculate
    r'''
                                    ===============
         WEAK SOLUTION IN (7) ====> [             ] ====> STRONG SOLUTION OUT (8)
                                    [  GENERATOR  ] ====> REFRIGERANT OUT (1)
            HOT WATER IN (20) ====> [             ] ====> HOT WATER OUT (21)
                                    ===============

        Assumptions
        -----------
        1. Hot water temperature in = 90.6 C
        2. Hot water temperature out = 85 C
        3. Refrigerant out is a saturated vapor
        4. Refrigerant temperature out = 80 C
        5. Strong solution temperature out = 80 C
        6. UA is constant = 5.287 kW / K

        Parameters
        ----------
        cp = isobaric heat capacity (kJ / (kg K))
        m = mass_flowrate (kg/s)
        P = pressure (kPa)
        Q_g = Generator heat
        T = Temperature (C)
        x = mass fraction of LiBr in solution

        Subscripts
        ----------
        con = condensation
        dp = dewpoint
        eva = evaporation
        g = generator
        hw = hot water
        in = inflow / influent
        out = outflow / effluent
        ref = refrigerant
        ss = Strong solution
        ws = Weak solution
    '''

    def __init__(self, UA=5.287, T20_max=110, T20_min=70, pressure=10):
        self.UA = UA
        self.hw_temp_max = T20_max
        self.hw_temp_min = T20_min
        self.pressure = pressure

    def refrigerant_enthalpy_out(self, temperature=80):
        r'''
            Calculates the enthalpy of refrigerant vapor out (1)

            Parameters
            ----------
            T_1: refrigerant temperature in (deg C)

            Output
            ------
            h_1: enthalpy of refrigerant in (3), (kJ / kg)

            Assumptions
            -----------
            Refrigerant in a vapor and saturated at T_1
        '''
        enthalpy = saturated_vapor_enthalpy(temperature)
        return enthalpy

    def strong_solution_enthalpy_out(self, mass_fraction=0.60, temperature=80):
        r'''
            Calculates the enthalpy of strong LiBr solution out (8)

            Parameters
            ----------
            temperature: LiBr solution temperature out (C)

            Output
            ------
            enthalpy: enthalpy of LiBr solution (kJ / kg)
        '''
        StrongSolution = LiBr_solution(mass_fraction)
        return StrongSolution.enthalpy_LiBr_solution(temperature)

    def strong_solution_temp_out_LMTD(
            self, Q, HW_temp_in, HW_temp_out, WS_temp_in):
        return calculate_Tc_out(
            Q, self.UA, HW_temp_in, HW_temp_out, WS_temp_in)

    def hot_water_mass_flowrate(self, Q_g, T_20=90.6, T_21=85):
        r'''
            Calculates the hot water flow through the generator

            Parameters
            ----------
            Q_g: the heat transfered through the generator (kJ)
            T_20: hot water temperature in (C)
            T_21: hot water temperature out (C)

            Output
            ------
            m_HW: mass flowrate of hot water (kg/s)
        '''
        delta_T = T_20 - T_21
        return Q_g / (specific_heat_water * delta_T)

    def guess_Q_g(self, Q_e, COP=0.7):
        return Q_e / COP

    def weak_solution_temperature_in(self, Q_g, T_8=80, T_20=90.6, T_21=85):
        return calculate_Tc_in(Q_g, self.UA, T_20, T_21, T_8)

    def generator_pressure(self, evaporation_temperature=80):
        return calculate_P_sat(evaporation_temperature)

    def generator_heat_from_flowrates(self, m_R, CR, h_1, h_8, h_7):
        H_1 = m_R * h_1
        H_8 = CR * m_R * h_8
        H_7 = (1 + CR) * m_R * h_7
        return H_1 + H_8 - H_7

    def hot_water_temp_out_LMTD(self, Q, HW_temp_in, WS_temp_in, SS_temp_out):
        return calculate_Th_out(
            Q, self.UA, HW_temp_in, WS_temp_in, SS_temp_out)

    def hot_water_temp_out(self, Q, m_g, HW_temp_in):
        return HW_temp_in - (Q / (m_g * specific_heat_water))

    def heat_generator_Ren(self, m_ref, m_ws, x_ws, x_ss,
                           Tss_out, Tws_in, Tcon, Tref_out):
        WeakSolution = LiBr_solution(x_ws)
        StrongSolution = LiBr_solution(x_ss)

        T_x = (Tss_out + Tws_in) / 2
        cp_ws = WeakSolution.isobaric_heat_capacity(Tws_in, method="Ren")
        h_con = latent_heat(Tcon)

        Xbar_g = self.mean_isostere_slope(T_dp=Tcon, Tws_in=Tws_in, Tss_out=Tss_out,
                                          WeakSolution=WeakSolution, StrongSolution=StrongSolution)

        Q_ws = m_ws * cp_ws * (Tss_out - Tws_in)

        Q_ref = m_ref * (Xbar_g * h_con +
                         specific_heat_water_vapor * (Tref_out - Tss_out))

        return Q_ws + Q_ref

    def mean_isostere_slope(
            self, T_dp,
            Tws_in, Tss_out,
            WeakSolution, StrongSolution):
        r'''
        Temperatures must be in Kelvin
        WeakSolution = LiBr_solution object
        StrongSolution = LiBr_solution object
        '''
        a0 = WeakSolution.Duhring_coefficients['a0']
        a1 = WeakSolution.Duhring_coefficients['a1']
        b0 = WeakSolution.Duhring_coefficients['b0']
        b1 = WeakSolution.Duhring_coefficients['b1']
        x_ws = WeakSolution.mass_fraction
        x_ss = StrongSolution.mass_fraction

        Tws_eq = WeakSolution.Duhring_equilibrium_temperature(T_dp)

        tws_eq = convert_C_to_K(Tws_eq)
        tss_out = convert_C_to_K(Tss_out)
        t_dp = convert_C_to_K(T_dp)

        first_term = ((tws_eq + tss_out) / (2 * t_dp))**2
        second_term = (((a0 * (x_ws + x_ss)) / 2) + a1)**-1

        return first_term * second_term


class Condenser:
    r'''
                                 ===============
     COOLING WATER IN (11) ====> [             ] ====> COOLING WATER OUT (12)
                                 [  CONDENSER  ]
        REFRIGERANT IN (1) ====> [             ] ====> REFRIGERANT OUT (2)
                                 ================

        Assumptions
        1. Refrigerant in is saturated water vapor at 80 C
        2. Refrigerant out is saturated liquid water
        3. UA is constant and = 10.387 kW/K
    '''

    def __init__(self, UA=10.387, T11_min=10, T11_max=36, pressure=10):
        self.UA = UA
        self.T11_min = T11_min
        self.T11_max = T11_max
        self.pressure = pressure

    def cooling_water_temp_out_LMTD(
            self, Q, refrigerant_temp_in, refrigerant_temp_out, CW_temp_in):
        return calculate_Tc_out(Q, self.UA,
                                              refrigerant_temp_in, refrigerant_temp_out,
                                              CW_temp_in)

    def cooling_water_temp_out(self, Q, m_cw, Tcw_in):
        return (Q / (m_cw * specific_heat_water) + Tcw_in)

    def cooling_water_flowrate(self, Q, CW_temp_in, CW_temp_out):
        return (Q / (specific_heat_water * (CW_temp_out - CW_temp_in)))

    def condenser_temperature(self):
        return sat_liquid_temp_from_pressure(self.pressure)

    def heat_condenser_Ren(self, m_ref, Tr_in, Tr_out):
        h_fg = latent_heat(Tr_out)
        return m_ref * (h_fg * specific_heat_water_vapor * (Tr_in - Tr_out))

    def refrigerant_temp_out_LMTD(self, Q, Tref_in, Tcw_in, Tcw_out):
        return calculate_Th_out(
            Q, self.UA, Tref_in, Tcw_in, Tcw_out)


class Evaporator:
    r'''
                                    ================
        CHILLED WATER IN (18) ====> [              ] ====> CHILLED WATER OUT (19)
                                    [  EVAPORATOR  ]
           REFRIGERANT IN (3) ====> [              ] ====> REFRIGERANT OUT (4)
                                    ================

        Assumptions
        1. Chilled water temperature in = 12 dC
        2. Chilled water temperature out = 6 dC
        3. Refrigerant is water
        4. Refrigerant out is saturated vapor
        5. Refrigerant temperature out = 4 dC
        6. Refrigerant in is saturated liquid
        7. Heat exchange is known
        8. UA is constant and = 12.566 kW/K
    '''

    def __init__(self, UA=12.566, T19_min=3, T19_max=21, pressure=0.814):
        self.UA = UA
        self.T19_min = T19_min
        self.T19_max = T19_max
        self.pressure = pressure

    def chilled_water_flowrate(self, Q_e, T_18=12, T_19=6):
        r'''
            Calculates the chilled water flow through the evaporator

            Parameters
            ----------
            Q_e: the heat transfered through the evaporator (kJ)
            T_18: chilled water temperature in (deg C)
            T_19: chilled water temperature out (deg C)

            Output
            ------
            m_ChW: mass flowrate of chilled water (kg/s)
        '''

        delta_T = T_18 - T_19
        m_ChW = (Q_e / (specific_heat_water * delta_T))
        return m_ChW

    def refrigerant_temp_in(self, Q_e, T_4=4, T_18=12, T_19=6):
        r'''
            Calculates the refrigerant temperature at inlet (T_3)

            Parameters
            ----------
            Q_e: the heat transfered through the evaporator (kJ)
            T_4: refrigerant temperature out (deg C)
            T_18: chilled water temperature in (deg C)
            T_19: chilled water temperature out (deg C)

            Output
            ------
            T_3: refrigerant temperature in

            Assumptions
            -----------
            1. LMTD Method
            2. Chen's approximation for LMTD [1]
            3. Counter-flow heat exchanger

            Equations
            ---------
            Q_e = (UA)_e * LMTD_e
            LMTD_e ~ {theta_1 * theta_2 * [(theta_1 + theta_2) / 2]} ^ (1/3)
            theta_1 = T_18 - T_4
            theta_2 = T_19 - T_3

            References
            ----------
            [1] Chen, J. J. J. (2019). Logarithmic mean : Chen ’ s approximation or
                explicit solution ? Computers and Chemical Engineering, 120, 1–3.
                https://doi.org/10.1016/j.compchemeng.2018.10.002
            '''

        T_3 = calculate_Tc_in(Q_e, self.UA, T_18, T_19, T_4)
        return T_3

    def refrigerant_temp_out_LMTD(self, Q, Tref_in, Tch_in, Tch_out):
        return calculate_Tc_out(
            Q, self.UA, Tch_in, Tch_out, Tref_in)

    def refrigerant_enthalpy_in(self, temperature):
        r'''
            Calculates the enthalpy of liquid refrigerant in (3)

            Parameters
            ----------
            temperature: refrigerant temperature in (deg C)

            Output
            ------
            enthalpy: enthalpy of refrigerant in (3), (kJ / kg)

            Assumptions
            -----------
            Refrigerant in is liquid and saturated at T_3
        '''
        enthalpy = saturated_liquid_enthalpy(temperature)
        return(enthalpy)

    def refrigerant_mass_flowrate(self, Q_e, h_3, h_4=2510.1):
        r'''
            Calculates the chilled water flow through the evaporator

            Parameters
            ----------
            Q_e: the heat transfered through the evaporator (kJ)
            h_3: refrigerant enthalpy at (3), (kJ / kg)
            h_4: refrigerant enthalpy at (4), (kJ / kg)

            Output
            ------
            m_r: mass flowrate of refrigerant (kg/s)

            Equation
            --------
            Q_e = m_r * (h_4 - h_3)

            The evaporator absorbs heat to the chilled water. So h_4 > h_3 to
            keep the Q_e positive
        '''
        m_r = Q_e / (h_4 - h_3)
        return m_r

    def evaporator_pressure(self, evaporation_temperature=4):
        return calculate_P_sat(evaporation_temperature)

    def chilled_water_T_out(self, m_e, Q_e, T_18=12):
        return T_18 - Q_e / (specific_heat_water * m_e)

    def heat_evaporator(self, m_ref, Tref_in, Tref_out):
        h_fg = latent_heat(Tref_out)
        return (m_ref * (h_fg + specific_heat_water * (Tref_out - Tref_in)))


class Absorber:
    # TODO
    # Calculate T_11
    # Assume values for T_17 and T_10

    r'''
                                      ===============
        STRONG SOLUTION IN (10) ====> [             ] ====> WEAK SOLUTION OUT (5)
             REFRIGERANT IN (4) ====> [  ABSORBER   ]
          COOLING WATER IN (17) ====> [             ] ====> COOLING WATER OUT (11)
                                      ===============

        Assumptions
        -----------
        1. Hot water temperature in = 90.6 C
        2. Hot water temperature out = 85 C
        3. Refrigerant out is a saturated vapor
        4. Refrigerant temperature out = 80 C
        5. Strong solution temperature out = 80 C
        6. UA is constant = 6.049 kW/K
    '''

    def __init__(self, UA=6.049,
                 T11_min=10, T11_max=36,
                 T10_min=15, T10_max=165,
                 T5_min=15, T5_max=165,
                 pressure=1):
        self.UA = UA
        self.T11_min = T11_min
        self.T11_max = T11_max
        self.T10_min = T10_min
        self.T10_max = T10_max
        self.T5_min = T5_min
        self.T5_max = T5_max
        self.pressure = pressure

    def heat_from_internal_flows(self, m_R, circulation_ratio, h_4, h_10, h_5):
        H_4 = m_R * h_4
        H_10 = circulation_ratio * m_R * h_10
        H_5 = (1 + circulation_ratio) * m_R * h_5
        return (H_4 + H_10 - H_5)

    def heat_from_external_flows(self, m_CW, h_17, h_11):
        return m_CW * specific_heat_water * (h_17 - h_11)

    def cooling_water_temp_out(self, Q, m_CW, T_17):
        return (Q / (m_CW * specific_heat_water)) + T_17

    def cooling_water_temp_out_LMTD(
            self, Q, SS_temp_in, WS_temp_out, CW_temp_in):
        return calculate_Tc_out(Q, self.UA,
                                              SS_temp_in, WS_temp_out,
                                              CW_temp_in)

    def solution_temperature_out_LMTD(self, Q, SS_temp_in, CW_temp_in, CW_temp_out):
        return calculate_Th_out(
            Q, self.UA, SS_temp_in, CW_temp_in, CW_temp_out)

    def cooling_water_flowrate(self, Q, CW_temp_in, CW_temp_out):
        return (Q / (specific_heat_water * (CW_temp_out - CW_temp_in)))

    def mean_isostere_slope(
            self, T_dp,
            Tws_out,
            WeakSolution, StrongSolution):
        r'''
        WeakSolution = LiBr_solution object
        StrongSolution = LiBr_solution object
        '''
        a0 = WeakSolution.Duhring_coefficients['a0']
        a1 = WeakSolution.Duhring_coefficients['a1']
        x_ws = WeakSolution.mass_fraction
        x_ss = StrongSolution.mass_fraction

        Tss_eq = StrongSolution.Duhring_equilibrium_temperature(T_dp)

        tss_eq = convert_C_to_K(Tss_eq)
        tws_out = convert_C_to_K(Tws_out)
        t_dp = convert_C_to_K(T_dp)

        first_term = ((tss_eq + tws_out) / (2 * t_dp))**2
        second_term = 1 / (((a0 * (x_ws + x_ss)) / 2) + a1)

        return first_term * second_term

    def heat_absorber_Ren(self, m_ref, m_ws, x_ws, x_ss,
                          Tss_in, Tws_out, Tcon, Tref_in):
        WeakSolution = LiBr_solution(x_ws)
        StrongSolution = LiBr_solution(x_ss)

        T_x = (Tss_in + Tws_out) / 2
        Tss_eq = StrongSolution.Duhring_equilibrium_temperature(Tcon)

        cp_ws = WeakSolution.isobaric_heat_capacity(Tws_out, method="Ren")
        cp_ss = StrongSolution.isobaric_heat_capacity(Tss_in, method="Ren")

        h_con = latent_heat(Tcon)

        Xbar_g = self.mean_isostere_slope(T_dp=Tcon, Tws_out=Tws_out,
                                          WeakSolution=WeakSolution, StrongSolution=StrongSolution)

        Q_ws = m_ws * cp_ws * (Tss_in - Tws_out)

        Q_ref = m_ref * (Xbar_g * h_con + specific_heat_water_vapor *
                         (Tref_in - Tss_in) + cp_ss * (Tss_in - Tss_eq))

        return Q_ws + Q_ref


class SolutionHeatExhanger:
    r'''
                                      ===============
           WEAK SOLUTION IN (6) ====> [  SOLUTION   ] ====> WEAK SOLUTION OUT (7)
                                      [  HEAT       ]
        STRONG SOLUTION OUT (9) <==== [  EXCHANGER  ] <==== STRONG SOLUTION IN (8)
                                      ===============

        Assumptions
        -----------
        1. Strong solution temperature in = 80 C
        2. UA is constant = 2.009 kW / K
    '''

    def __init__(self, UA=2.009, effectiveness=0.9):
        self.UA = UA
        self.effectiveness = effectiveness

    def heat_flow(self, mass_flowrate, enthalpy_1, enthalpy_2):
        r'''
        Calculates the heat transfer using Q = m(h_1 - h_2)

        Parameters
        ----------
        mass_flowrate: mass flowrate of the solution (kg / s)
        enthalpy_1: enthalpy of the solution from hot end (kJ / kg)
        enthalpy_2: enthalpy of the solution from cold end (kJ / kg)
        '''
        return mass_flowrate * (enthalpy_1 - enthalpy_2)

    def LMTD_strong_solution_temperature_out(
            self, Q, SS_temp_in, WS_temp_in, WS_temp_out):
        return calculate_Th_out(
            Q, self.UA, SS_temp_in, WS_temp_in, WS_temp_out)

    def weak_solution_temperature_out_Ren(self, m_ws, x_ws, Tws_in, Q_she):
        WeakSolution = LiBr_solution(x_ws)
        cp_ws = WeakSolution.isobaric_heat_capacity(Tws_in)

        return (Q_she / (m_ws * cp_ws)) + Tws_in

    def strong_solution_temperature_out_effectiveness(
            self, SS_temp_in, WS_temp_in):
        return SS_temp_in - self.effectiveness * (SS_temp_in - WS_temp_in)

    def heat_flow_Ren(self, m_ss, Tss_in, Tss_out, x_ss):
        StrongSolution = LiBr_solution(x_ss)
        Tss_avg = np.mean([Tss_in, Tss_out])
        cp_ss = StrongSolution.isobaric_heat_capacity(Tss_avg)

        return (m_ss * cp_ss * (Tss_in - Tss_out))


class CoolingTower:
    r'''
                                     =============
                                     [  COOLING  ] ====> SATURATED AIR OUT (15)
         COOLING WATER IN (12) ====> [  TOWER    ]
                                     [           ]
        COOLING WATER OUT (14) <==== [           ] <==== AMBIENT AIR IN (13)
                                     =============


                                   =========
                                   [       ] ====> COOLING WATER OUT (17)
                                   [ MIXER ]
        MAKEUP WATER IN (16) ====> [       ] <==== COOLING WATER IN (14)
                                   =========

        Assumptions
        -----------
        1. Air exiting cooling tower is saturated (relative humidity = 1)
        2. The heat exchange rate and air mass flowrates are directly proportional to the cooling water mass flowrate
    '''

    def __init__(self, name=''):
        self.name = name

    def mixer_enthalpy_out(self, cooling_water_mass_flowrate, makeup_water_massflowrate,
                           makeup_water_enthalpy, cooling_tower_water_enthalpy):
        H_16 = makeup_water_enthalpy * makeup_water_massflowrate
        m_14 = cooling_water_mass_flowrate = makeup_water_massflowrate
        H_14 = m_14 * cooling_tower_water_enthalpy
        return ((H_16 + H_14) / cooling_water_mass_flowrate)

    def make_up_water_flowrate(self,
                               air_mass_flowrate, humidity_ratio_in, humidity_ratio_out):
        return air_mass_flowrate * (humidity_ratio_out - humidity_ratio_in)

    def water_out_flowrate(self, cooling_water_mass_flowrate,
                           air_mass_flowrate, humidity_ratio_in, humidity_ratio_out):
        return (cooling_water_mass_flowrate + air_mass_flowrate *
                (humidity_ratio_in - humidity_ratio_out))

    def water_out_enthalpy(self, Q,
                           cooling_water_mass_flowrate, cooling_water_enthalpy,
                           water_out_flowrate):
        energy_cooling_water = cooling_water_mass_flowrate * cooling_water_enthalpy
        return (energy_cooling_water - Q) / water_out_flowrate

    def air_mass_flowrate(self, cooling_water_mass_flowrate):
        return 0.58 * cooling_water_mass_flowrate + 7.24

    def heat_exchange(self, cooling_water_mass_flowrate):
        return cooling_water_mass_flowrate / (4.31 * 10**-2)

    def air_enthalpy_out(self, Q, m_air, enthalpy_ambient):
        return ((Q / m_air) + enthalpy_ambient)

    def _air_temp_obj_fcn(self, temperature, pressure,
                          relative_humidity, enthalpy):
        try:
            h = humid_air_enthalpy(temperature, pressure, relative_humidity)
        except (AttributeError, TypeError) as e:
            h = humid_air_enthalpy(
                temperature,
                pressure,
                relative_humidity,
                method='cengel')
        return h - enthalpy

    def temperature_air_out(self, enthalpy, P_atm,
                            cooling_water_temp=40, ambient_air_temp=20,
                            relative_humidity=1,
                            percent_error_allowed=0.01):
        r'''
            Calculate the exit air temperature C

            Parameters
            ----------
            enthalpy: known enthalpy of humid air (kJ / kg)
            P_atm: atmospheric pressure of air (kPa)
            relative_humidity: relative_humidity of the air, as a fraction [0 - 1]

            Output
            ------
            temperature_r: humid air temperature (C)

            Iteration method
            ----------------
            Modified False Position
        '''
        # Initialize Values
        t_L = min(cooling_water_temp, ambient_air_temp)
        t_U = max(cooling_water_temp, ambient_air_temp)
        f_L = self._air_temp_obj_fcn(t_L, P_atm, relative_humidity, enthalpy)
        f_U = self._air_temp_obj_fcn(t_U, P_atm, relative_humidity, enthalpy)

        # First Guess
        t_r = t_U - ((f_U * (t_L - t_U)) / (f_L - f_U))

        # Absolute Percent Relative Error = APRE
        apre = 100

        # Iteration counters
        i_U = 0
        i_L = 0

        iteration = 0
        while apre > percent_error_allowed:

            t_r_old = t_r
            f_r = self._air_temp_obj_fcn(
                t_r, P_atm, relative_humidity, enthalpy)

            test = f_L * f_r
            if test < 0:
                t_U = t_r
                f_U = self._air_temp_obj_fcn(
                    t_U, P_atm, relative_humidity, enthalpy)
                i_U = 0
                i_L += 1
                if i_L >= 2:
                    f_L = f_L / 2
            elif test > 0:
                t_L = t_r
                f_L = self._air_temp_obj_fcn(
                    t_L, P_atm, relative_humidity, enthalpy)
                i_L = 0
                i_U += 1
                if i_U >= 2:
                    f_U = f_U / 2
            else:
                apre = 0

            t_r = t_U - ((f_U * (t_L - t_U)) / (f_L - f_U))

            apre = absolute_percent_relative_error(t_r_old, t_r)

            iteration += 1

        return t_r


def statepoint_df(statepoint_list, method="All"):
    statepoints = []
    flowrates = []
    temperatures = []
    temp_max = []
    temp_min = []
    pressures = []
    enthalpies = []
    fluidtypes = []

    if method == "All":
        for p in statepoint_list:
            statepoints.append(p.name)
            fluidtypes.append(p.fluidtype)
            flowrates.append(p.mass_flowrate)
            temperatures.append(p.temperature)
            temp_min.append(p.temp_min)
            temp_max.append(p.temp_max)
            pressures.append(p.pressure)
            enthalpies.append(p.h)

        points_dict = {
            'statepoint': statepoints,
            'fluidtype': fluidtypes,
            'mass_flowrate_kg_per_s': flowrates,
            'temperature_C': temperatures,
            'min_temperature_C': temp_min,
            'max_temperature_C': temp_max,
            'pressure_kPa': pressures,
            'enthalpy_kJperkg': enthalpies}
    elif method == "Ren":
        for p in statepoint_list:
            statepoints.append(p.name)
            flowrates.append(p.mass_flowrate)
            temperatures.append(p.temperature)

        points_dict = {
            'statepoint': statepoints,
            'mass_flowrate_kg_per_s': flowrates,
            'temperature_C': temperatures}

    df = pd.DataFrame.from_dict(data=points_dict)
    return df


######################
# Workflow Algorithm #
######################
'''
Algorithm from Ren et al. (2019)
Ren, J., Qian, Z., Yao, Z., Gan, N., & Zhang, Y. (2019). Thermodynamic
    evaluation of LiCl-H2O and LiBr-H2O absorption refrigeration systems
    based on a novel model and algorithm. Energies, 14(15), 1–30.
    https://doi.org/10.3390/en12153037


1. X Input external mass flowrates and UA values for heat exchangers
2. X Input influent temp for all external circuits
3. X Assume temp leaving evaporator (T_4)
4. X Calculated h_evap (h_4)
5. X Assume temp leaving condenser (T_2)
6. X Calculate h_cond (h_2)
7. X Assume temp leaving generator (T_1 and T_8)
8. X Calculate X_SS and SS
9. Calculate temperature entering absorber (T_10)
10. Assume value for SS temp leaving SHE (T_9)
11. Calculate cp of SS leaving SHE (h_9)
12. Assume temp of weak solution leaving absorber (T_5)
13. Calculate CR, m_R, temp of WS leaving SHE (T_7), Q_a, and cooling water temp leaving abs (T_11)
14. Determine new temp leaving absorber (T_5, step 12) until error is < 10^-2
15. Calculate Q_she and solve for WS temp leaving the SHE
16. Determine new temprature of SS leaving the SHE (T_9, step 10), until error is < 10^-2
17. Calculate Q_g
18. Solve for hot water temp leaving generator (T_21)
19. Determine new temperature leaving generator (T_8 and T_1, step 8), until error is < 10^-2
20. Calculate Q_c and solve for cooling water temp leaving cond (T_12)
21. Determine new temp of refrigerant leaving cond (T_2, step 5), until error is < 10^-2
22. Calculate Q_E and solve for chilled water temp leaving evaporator (T_4)
23. Determine temp of refrigerant leaving evap (T_4) until error is < 10^-2
END
'''


def log_mean_temperature_difference(theta_1, theta_2):
    r'''
    Calculate the log-mean temperature difference

    Parameters
    ----------
    theta_1: first temperature difference
    theta_2: second temperature difference

    Output
    ------
    LMTD: the log-mean temperature difference

    LMTD = (theta_1 - theta_2) / ln(theta_1 / theta_2)

    *Note:
    The theta values used in the LMTD depend on the type of heat
    exchanger used.

    --------------------------------
    For parallel flow heat exhangers
    --------------------------------
                                  HEAT EXCHANGER:
                                 =================
                                 {[             ]} ====> COLD FLUID OUT (T_C, out)
     HOT FLUID IN (T_H,in) ====> [---------------] ====> HOT FLUID OUT (T_H,out)
    COLD FLUID IN (T_C,in) ====> {[             ]}
                                 =================

    theta_1 = T_H,in - T_C,in
    theta_2 = T_H,out - T_C,out

    -------------------------------
    For counter flow heat exhangers
    -------------------------------
                                   HEAT EXCHANGER:
                                  =================
                                  {[             ]} <==== COLD FLUID IN (T_C, in)
      HOT FLUID IN (T_H,in) ====> [---------------] ====> HOT FLUID OUT (T_H,out)
    COLD FLUID IN (T_C,out) <==== {[             ]}
                                  =================

    theta_1 = T_H,in - T_C,out
    theta_2 = T_H,out - T_C,in
    '''
    return ((theta_1 - theta_2) / (m.log(theta_1 / theta_2)))


def _thermo_dict_to_dataframe(
        statepoint_list, temperature_dict, pressure_dict, enthalpy_dict, massflow_dict):
    temperature_ls = list(temperature_dict.values())
    pressure_ls = list(pressure_dict.values())
    enthalpy_ls = list(enthalpy_dict.values())
    massflow_ls = list(massflow_dict.values())

    # Convert the lists into a pandas dataframe
    data = {'statepoint': statepoint_list,
            'T_degC': temperature_ls,
            'P_kPa': pressure_ls,
            'enthalpy': enthalpy_ls,
            'massflow': massflow_ls}
    df = pd.DataFrame(data=data)
    df.set_index('statepoint', inplace=True, drop=True)

    return df


def absoprtion_chiller_equilibrium(
        Q_e, T_dry_bulb, P_atm, relative_humidity, error_threshold=0.001):
    # Create a dictionary to store all values
    statepoints = [i for i in range(1, 22)]

    # Dictionaries
    temperature_dict = {}
    enthalpy_dict = {}
    pressure_dict = {}
    massflow_dict = {}

    for i in statepoints:
        temperature_dict[F'T{i}'] = None
        enthalpy_dict[F'h{i}'] = None
        pressure_dict[F'P{i}'] = None
        massflow_dict[F'm{i}'] = None

    # 1. Climate data an input for the function

    # 2. Create objects with known UA values
    Evaporator_ = Evaporator()
    Absorber_ = Absorber()
    Generator_ = Generator()
    SHE_ = SolutionHeatExhanger()
    Condenser_ = Condenser()
    CoolingTower_ = CoolingTower()

    # 3 Input known temperatures and pressures

    r'''
        KNOWN VALUES
        ------------
        Q_e
        T_1, h_1 = h_g : 80 C , 2643 kJ/kg
        T_4, h_4 = h_g : 4 C, 2510 kJ/kg
        T_8, h_8, X_8 : 80 C, 193 kJ/kg, 60%
        T_16, h_16 = h_f : 30 C, 125.7 kJ/kg
        T_18, h_18 : 12 C, 50.4 kJ/kg
        T_19, h_19 : 6 C, 25.2 kJ/kg
        T_20, h_20 : 90.6 C, 379.57 kJ/kg
        T_21, h_21 : 85 C, 366.03 kJ/kg
    '''
    # Temperature
    T_1 = 80
    T_4 = 4
    T_8 = T_1
    T_13 = T_dry_bulb
    T_16 = 25
    T_18 = 12
    T_19 = 6
    T_20 = 90.6
    T_21 = 85

    # Insert known temperatures into dictionary
    temperature_dict['T1'] = T_1
    temperature_dict['T4'] = T_4
    temperature_dict['T8'] = T_8
    temperature_dict['T13'] = T_13
    temperature_dict['T16'] = T_16
    temperature_dict['T18'] = T_18
    temperature_dict['T19'] = T_19
    temperature_dict['T20'] = T_20
    temperature_dict['T21'] = T_21

    # Pressure
    P_g = Generator_.generator_pressure(T_1)
    P_c = P_g
    P_e = Evaporator_.evaporator_pressure(T_4)
    P_a = P_e

    # Insert known pressures into dictionary
    upper_vessel = [1, 2, 6, 7, 8, 9]
    lower_vessel = [3, 4, 5, 10]
    for i in range(1, 11):
        if i in upper_vessel:
            pressure_dict[F'P{i}'] = P_g
        else:
            pressure_dict[F'P{i}'] = P_e
    pressure_dict['P13'] = P_atm
    pressure_dict['P15'] = P_atm

    # LiBr Solutions
    x_ss = 0.60
    x_ws = 0.57
    strong_solution = LiBr_solution(x_ss)
    weak_solution = LiBr_solution(x_ws)

    #######################################
    # Calculate enthalpy for known values #
    #######################################

    # H2O
    h_1 = saturated_liquid_enthalpy(T_1)
    h_4 = saturated_vapor_enthalpy(T_4)
    h_16 = saturated_liquid_enthalpy(T_16)
    h_18 = saturated_liquid_enthalpy(T_18)
    h_19 = saturated_liquid_enthalpy(T_19)
    h_20 = saturated_liquid_enthalpy(T_20)
    h_21 = saturated_liquid_enthalpy(T_21)

    # LiBr Solution
    h_8 = strong_solution.enthalpy_LiBr_solution(T_8)

    # Air
    h_13 = humid_air_enthalpy(T_13, P_atm, relative_humidity)
    w_13 = humidity_ratio(T_13, P_atm, relative_humidity)

    # Insert known temperatures into dictionary
    enthalpy_dict['h1'] = h_1
    enthalpy_dict['h4'] = h_4
    enthalpy_dict['h8'] = h_8
    enthalpy_dict['h13'] = h_13
    enthalpy_dict['h16'] = h_16
    enthalpy_dict['h18'] = h_18
    enthalpy_dict['h19'] = h_19
    enthalpy_dict['h20'] = h_20
    enthalpy_dict['h21'] = h_21

    #########
    # START #
    #########
    r'''
        The Evaporator will be the only component with very
        explicit solutions that do not require iteration
        5.  Solve for T_3
        6.  Solve for h_3
        7.  Calculate the refrigerant mass flowrate (m_R)
        8.  Calculate the chilled water mass flowrate (m_ChW)
    '''
    # 4. Solve for m_ChW
    m_ChW = Evaporator_.chilled_water_flowrate(Q_e)
    for i in [18, 19]:
        massflow_dict[F'm{i}'] = m_ChW

    # 5. Solve for T_3 and h_3
    T_3 = Evaporator_.refrigerant_temp_in(Q_e)
    h_3 = Evaporator_.refrigerant_enthalpy_in(T_3)

    # 6 Solve for m_R
    m_R = Evaporator_.refrigerant_mass_flowrate(Q_e, h_3)

    # Calculate weak and strong solution mass flowrates
    r'''
        9.  Calculate the circulation ratio (CR)
        10. Calculate the strong solution mass flowrate (m_SS)
        11. Calculate the weak solution mass flowrate (m_WS)
    '''
    CR = circulation_ratio(
        strong_solution.concentration,
        weak_solution.concentration)
    m_SS = strong_solution_flowrate(CR, m_R)
    m_WS = weak_solution_flowrate(CR, m_R)

    r'''
        12. Solve for h_2 = h_3
        13. Solve for T_2 given h_2 and P_c
        14. Solve for Q_c given h_2, h_1, and m_R
    '''

    # 7 Solve for h_2 and T_2
    h_2 = h_3
    T_2 = convert_K_to_C(iapws97._Backward1_T_Ph(P_c / 1000, h_2))

    # 8 Solve for Q_c
    Q_c = m_R * (h_1 - h_2)

    # Update temperature and enthalpy dict
    temperature_dict['T2'] = T_2
    temperature_dict['T3'] = T_3
    enthalpy_dict['h2'] = h_2
    enthalpy_dict['h3'] = h_3

    # Update massflow dict
    refrigerant = [i for i in range(1, 5)]
    solution_weak = [i for i in range(5, 8)]
    solution_strong = [i for i in range(8, 11)]
    for i in range(1, 11):
        if i in refrigerant:
            massflow_dict[F'm{i}'] = m_R
        elif i in solution_weak:
            massflow_dict[F'm{i}'] = m_WS
        else:
            massflow_dict[F'm{i}'] = m_SS

    pressure_dict['P13'] = P_atm
    pressure_dict['P15'] = P_atm

    def print_dataframe():
        balance_df = _thermo_dict_to_dataframe(statepoints, temperature_dict,
                                               pressure_dict, enthalpy_dict, massflow_dict)

        print(balance_df)

    #######################
    # Iterative Processes #
    #######################

    #######################
    # Assume value of T_9 #
    #######################

    # 9. Assume T_9 = 50.6 C
    T_9 = 50.6
    error_T_9 = 1

    while error_T_9 > error_threshold:

        h_9 = strong_solution.enthalpy_LiBr_solution(T_9)

        # 9.1 Solve for h_10 and T_10
        h_10 = h_9
        T_10 = T_9

        # 9.2 Calculate Q_she
        Q_she = SHE_.heat_flow(m_SS, h_8, h_1)

        #######################
        # Assume value of T_5 #
        #######################

        # 9.3 Assume T_5 = 38 C
        T_5 = 38
        error_T_5 = 1

        while error_T_5 > error_threshold:
            h_5 = weak_solution.enthalpy_LiBr_solution(T_5)

            # 9.3.1 Calculate Q_a
            Q_a = Absorber_.heat_from_internal_flows(m_R, CR, h_4, h_10, h_5)

            # 9.3.2 Calculate h_6
            h_6 = h_5

            # 9.3.3 Calculate h_7 Eq. 85
            h_7 = (Q_she / m_WS) + h_6

            # 9.3.4 Calculate Q_g using Eq. 78
            Q_g = Generator_.generator_heat_from_flowrates(
                m_R, CR, h_1, h_8, h_7)

            # 9.3.5 Evaluate h_13 using Eq. 26
            T_13 = T_dry_bulb
            h_13 = humid_air_enthalpy(T_13, P_atm, relative_humidity)
            w_13 = humidity_ratio(T_13, P_atm, relative_humidity)

            ###############
            # Assume T_17 #
            ###############
            # 9.3.6 Assume value for T_17 = 30 C
            T_17 = 30
            error_T_17 = 1

            while error_T_17 > error_threshold:
                h_17 = saturated_liquid_enthalpy(T_17)

                # 9.3.6.1 Solve for T_11 using Eq. 10
                T_11 = Absorber_.cooling_water_temp_out_LMTD(
                    Q_a, T_10, T_5, T_17)
                print(F'T11 = {T_11}, T17 = {T_17}')
                h_11 = saturated_liquid_enthalpy(T_11)

                # 9.3.6.2 Solve for T_12 using Eq. 16
                T_12 = Condenser_.cooling_water_temp_out_LMTD(
                    Q_c, T_1, T_2, T_11)
                h_12 = saturated_liquid_enthalpy(T_12)

                # 9.3.6.3 Solve for m ̇_CW using Eq. 15
                # m_CW = Condenser_.cooling_water_flowrate(Q_c, T_11, T_12)
                m_CW = Condenser_.cooling_water_flowrate(Q_c, T_11, T_12)

                # 9.3.6.4 Calculate m ̇_air and Q ̇_CT using Eq. 22 and Eq. 23
                m_air = CoolingTower_.air_mass_flowrate(m_CW)
                Q_ct = CoolingTower_.heat_exchange(m_CW)

                # 9.3.6.5 Calculate h_15 using Eq. 25
                h_15 = CoolingTower_.air_enthalpy_out(Q_ct, m_air, h_13)

                # 9.3.6.6 Calculate T_15 using iterative calculations Eq. 26
                # through Eq. 28
                T_15 = CoolingTower_.temperature_air_out(h_15, P_atm,
                                                         T_12, T_13, 1)

                # 9.3.6.7 Evaluate ω_15 using Eq. 29
                w_15 = humidity_ratio(T_15, P_atm, 1)

                # 9.3.6.8 Solve for m_14 using Eq. 21
                m_14 = CoolingTower_.water_out_flowrate(
                    m_CW, m_air, w_13, w_15)

                # 9.3.6.9 Calculate h_14 using Eq. 24
                h_14 = CoolingTower_.water_out_enthalpy(Q_ct, m_CW, h_12, m_14)
                T_14 = saturated_liquid_temperature(h_14)

                # 9.3.6.10 Calculate makeup water
                m_makeup = CoolingTower_.make_up_water_flowrate(
                    m_air, w_13, w_15)

                # 9.3.6.11 Solve for h_17' and T_17' using Eq. 32
                h_17_new = CoolingTower_.mixer_enthalpy_out(
                    m_CW, m_makeup, h_16, h_14)
                print(F'h17_new = {h_17_new}')
                T_17_new = saturated_liquid_temperature(h_17_new)

                # 9.3.6.12 Check error_T_17
                error_T_17 = absolute_total_error(T_17, T_17_new)

                T_17 = T_17_new
                h_17 = h_17_new

                # Update dictionaries:
                temperature_dict['T11'] = T_3
                temperature_dict['T12'] = T_12
                temperature_dict['T14'] = T_14
                temperature_dict['T15'] = T_15
                temperature_dict['T17'] = T_17

                enthalpy_dict['h11'] = h_11
                enthalpy_dict['h12'] = h_12
                enthalpy_dict['h14'] = h_14
                enthalpy_dict['h15'] = h_15
                enthalpy_dict['h17'] = h_17

                for i in [11, 12, 17]:
                    massflow_dict[F'm{i}'] = m_CW
                for i in [13, 15]:
                    massflow_dict[F'm{i}'] = m_air
                massflow_dict['m14'] = m_14
                massflow_dict['m16'] = m_makeup

                print_dataframe()

            # 9.3.7 Calculate Q_a using Eq. 9
            Q_a = Absorber_.heat_from_external_flows(m_CW, h_17, h_11)

            # 9.3.8 Calculate T_5 using Eq. 10
            T_5_new = Absorber_.solution_temperature_out(Q_a, h_10, h_17, h_11)

            # 9.3.9 Check e_5
            error_T_5 = absolute_total_error(T_5, T_5_new)

            T_5 = T_5_new
            h_5 = weak_solution.enthalpy_LiBr_solution(T_5)

        # 9.4 Calculate Q_she
        T_6 = T_5
        h_6 = h_5

        # 9.5 Calculate Q_she
        Q_she = SHE_.heat_flow(m_WS, h_7, h_6)

        # 9.6 Calculate T_7
        T_7 = weak_solution.calc_temp_from_enthalpy(h_7)

        # 9.7 Calculate T_9
        T_9_new = SHE_.strong_solution_temperature_out(Q_she, T_8, T_6, T_7)

        # 9.6 Check e_9
        error_T_9 = absolute_total_error(T_9, T_9_new)

        T_9 = T_9_new
        h_9 = strong_solution.enthalpy_LiBr_solution(T_9)

    # 10. Calcualte Q_g
    Q_g = Generator_.generator_heat_from_flowrates(m_R, CR, h_1, h_8, h_7)

    # 11. Calcualte COP
    COP = Q_e / Q_g

    print(F'Q_g = {Q_g}, m_makeup = {m_makeup}, COP = {COP}')


def absoprtion_chiller_equilibrium_v2(
        Q_e, T_dry_bulb, P_atm, relative_humidity, error_threshold=0.001):
    # Create a dictionary to store all values
    statepoints = [i for i in range(1, 22)]

    p1 = structure('p1')
    p2 = structure('p2')
    p3 = structure('p3')
    p4 = structure('p4')
    p5 = structure('p5')
    p6 = structure('p6')
    p7 = structure('p7')
    p8 = structure('p8')
    p9 = structure('p9')
    p10 = structure('p10')
    p11 = structure('p11')
    p12 = structure('p12')
    p13 = structure('p13')
    p14 = structure('p14')
    p15 = structure('p15')
    p16 = structure('p16')
    p17 = structure('p17')
    p18 = structure('p18')
    p19 = structure('p19')
    p20 = structure('p20')
    p21 = structure('p21')

    # 1. Climate data an input for the function

    # 2. Create objects with known UA values
    Evaporator_ = Evaporator()
    Absorber_ = Absorber()
    Generator_ = Generator()
    SHE_ = SolutionHeatExhanger()
    Condenser_ = Condenser()
    CoolingTower_ = CoolingTower()

    # 3 Input known temperatures and pressures

    r'''
        KNOWN VALUES
        ------------
        Q_e
        T_1, h_1 = h_g : 80 C , 2643 kJ/kg
        T_4, h_4 = h_g : 4 C, 2510 kJ/kg
        T_8, h_8, X_8 : 80 C, 193 kJ/kg, 60%
        T_16, h_16 = h_f : 30 C, 125.7 kJ/kg
        T_18, h_18 : 12 C, 50.4 kJ/kg
        T_19, h_19 : 6 C, 25.2 kJ/kg
        T_20, h_20 : 90.6 C, 379.57 kJ/kg
        T_21, h_21 : 85 C, 366.03 kJ/kg
    '''
    # Temperature
    p1.temperature = 80
    p4.temperature = 4
    p8.temperature = p1.temperature
    p13.temperature = T_dry_bulb
    p16.temperature = 25
    p18.temperature = 12
    p19.temperature = 6
    p20.temperature = 90.6
    p21.temperature = 85

    # Pressure
    P_g = Generator_.generator_pressure(p1.temperature)
    P_c = P_g
    P_e = Evaporator_.evaporator_pressure(p4.temperature)
    P_a = P_e

    # Insert known pressures
    upper_vessel = [p1, p2, p6, p7, p8, p9]
    lower_vessel = [p3, p4, p5, p10]
    for i in upper_vessel:
        i.pressure = P_g
    for i in lower_vessel:
        i.pressure = P_e
    p13.pressure = P_atm
    p15 = P_atm

    ###############
    # CONSTRAINTS #
    ###############
    # Refrigerant
    for i in [p1, p2, p3, p4]:
        if i == p1 or i == p4:
            i.fluidtype = 'saturated vapor - water'
        else:
            i.fluidtype = 'saturated liquid - water'
        i.temp_min = 1
        i.temp_max = 100
    # LiBr solution
    for i in [p5, p6, p7, p8, p9, p10]:
        i.fluidtype = 'LiBr solution'
        i.temp_min = 15
        i.temp_max = 165
    # Cooling Water
    for i in [p11, p12, p14, p16, p17]:
        i.fluidtype = 'saturated liquid - water'
        if i == p11:
            i.temp_min = 7.2
            i.temp_max = 36
        else:
            i.temp_min = 4
            i.temp_max = 99
        i.pressure = 786
    # Hot Water
    for i in [p20, p21]:
        i.fluidtype = 'saturated liquid - water'
        i.temp_min = 70
        i.temp_max = 110
        i.pressure = None
    # Chilled Water
    for i in [p18, p19]:
        i.fluidtype = 'satulrated liquid - water'
        i.pressure = 786
        i.temp_min = 3
        i.temp_max = 21
    for i in [p13, p15]:
        i.pressure = P_atm

    # LiBr Solutions
    X_SS = 60
    X_WS = 57
    strong_solution = LiBr_solution(X_SS)
    weak_solution = LiBr_solution(X_WS)

    #######################################
    # Calculate enthalpy for known values #
    #######################################

    # H2O
    p1.h = saturated_liquid_enthalpy(p1.temperature)
    p4.h = saturated_vapor_enthalpy(p4.temperature)
    p16.h = saturated_liquid_enthalpy(p16.temperature)
    p18.h = saturated_liquid_enthalpy(p18.temperature)
    p19.h = saturated_liquid_enthalpy(p19.temperature)
    p20.h = saturated_liquid_enthalpy(p20.temperature)
    p21.h = saturated_liquid_enthalpy(p21.temperature)

    # LiBr Solution
    p8.h = strong_solution.enthalpy_LiBr_solution(p8.temperature)

    # Air
    p13.h = humid_air_enthalpy(
        p13.temperature,
        p13.pressure,
        relative_humidity)
    p13.w = humidity_ratio(p13.temperature, p13.pressure, relative_humidity)

    #########
    # START #
    #########
    r'''
        The Evaporator will be the only component with very
        explicit solutions that do not require iteration
        5.  Solve for T_3
        6.  Solve for h_3
        7.  Calculate the refrigerant mass flowrate (m_R)
        8.  Calculate the chilled water mass flowrate (m_ChW)
    '''
    # 4. Solve for m_ChW
    m_ChW = Evaporator_.chilled_water_flowrate(Q_e)
    p18.mass_flowrate = m_ChW
    p19.mass_flowrate = m_ChW

    # 5. Solve for T_3 and h_3
    p3.temperature = Evaporator_.refrigerant_temp_in(Q_e)
    p3.h = Evaporator_.refrigerant_enthalpy_in(p3.temperature)

    # 6 Solve for m_R
    m_R = Evaporator_.refrigerant_mass_flowrate(Q_e, p3.h)
    p1.mass_flowrate = m_R
    p2.mass_flowrate = m_R
    p3.mass_flowrate = m_R
    p4.mass_flowrate = m_R

    # Calculate weak and strong solution mass flowrates
    r'''
        9.  Calculate the circulation ratio (CR)
        10. Calculate the strong solution mass flowrate (m_SS)
        11. Calculate the weak solution mass flowrate (m_WS)
    '''
    CR = circulation_ratio(
        strong_solution.concentration,
        weak_solution.concentration)
    m_WS = weak_solution_flowrate(CR, m_R)
    p5.mass_flowrate = m_WS
    p6.mass_flowrate = m_WS
    p7.mass_flowrate = m_WS

    m_SS = strong_solution_flowrate(CR, m_R)
    p8.mass_flowrate = m_SS
    p9.mass_flowrate = m_SS
    p10.mass_flowrate = m_SS

    r'''
        12. Solve for h_2 = h_3
        13. Solve for T_2 given h_2 and P_c
        14. Solve for Q_c given h_2, h_1, and m_R
    '''

    # 7 Solve for h_2 and T_2
    p2.h = p3.h
    p2.temperature = convert_K_to_C(
        iapws97._Backward1_T_Ph(
            p2.pressure / 1000, p2.h))

    # 8 Solve for Q_c
    Q_c = m_R * (p1.h - p2.h)

    #######################
    # Iterative Processes #
    #######################

    #######################
    # Assume value of T_9 #
    #######################

    # 9. Assume T11 is some number between 7.2 and 36 degrees C
    p9.temperature = np.random.uniform(
        size=1, low=p9.temp_min, high=p9.temp_max)
    error_T_9 = 1

    while error_T_9 > error_threshold:

        p9.h = strong_solution.enthalpy_LiBr_solution(p9.temperature)

        # 9.1 Solve for h_10 and T_10
        p10.h = p9.h
        p10.temperature = p9.temperature

        # 9.2 Calculate Q_she
        Q_she = SHE_.heat_flow(m_SS, p8.h, p1.h)

        #######################
        # Assume value of T_5 #
        #######################

        # 9.3 Assume T_5 = 38 C
        p5.temperature = 38
        error_T_5 = 1

        while error_T_5 > error_threshold:
            p5.h = weak_solution.enthalpy_LiBr_solution(p5.temperature)

            # 9.3.1 Calculate Q_a
            Q_a = Absorber_.heat_from_internal_flows(
                m_R, CR, p4.h, p10.h, p5.h)

            # 9.3.2 Calculate h_6
            p6.h = p5.h

            # 9.3.3 Calculate h_7 Eq. 85
            p7.h = (Q_she / m_WS) + p6.h

            # 9.3.4 Calculate Q_g using Eq. 78
            Q_g = Generator_.generator_heat_from_flowrates(
                m_R, CR, p1.h, p8.h, p7.h)

            # 9.3.5 Evaluate h_13 using Eq. 26
            p13.temperature = T_dry_bulb
            p13.h = humid_air_enthalpy(
                p13.temperature, P_atm, relative_humidity)
            p13.w = humidity_ratio(p13.temperature, P_atm, relative_humidity)

            ###############
            # Assume T_17 #
            ###############
            # 9.3.6 Assume value for T_17 = 30 C
            p17.temperature = 30
            error_T_17 = 1

            while error_T_17 > error_threshold:
                p17.h = saturated_liquid_enthalpy(p17.temperature)

                # 9.3.6.1 Solve for T_11 using Eq. 10
                p11.temperature = Absorber_.cooling_water_temp_out_LMTD(
                    Q_a, p10.temperature, p5.temperature, p17.temperature)
                # ADD A CHECK FOR MAX AND MIN TEMPERATURE
                print(F'T11 = {T_11}, T17 = {T_17}')
                p11.h = saturated_liquid_enthalpy(p11.temperature)

                # 9.3.6.2 Solve for T_12 using Eq. 16
                p12.temperature = Condenser_.cooling_water_temp_out_LMTD(
                    Q_c, p1.temperature, p2.temperature, p11.temperature)
                p12.h = saturated_liquid_enthalpy(p12.temperature)

                # 9.3.6.3 Solve for m ̇_CW using Eq. 15
                # m_CW = Condenser_.cooling_water_flowrate(Q_c, T_11, T_12)
                m_CW = Condenser_.cooling_water_flowrate(
                    Q_c, p11.temperature, p12.temperature)

                # 9.3.6.4 Calculate m ̇_air and Q ̇_CT using Eq. 22 and Eq. 23
                m_air = CoolingTower_.air_mass_flowrate(m_CW)
                Q_ct = CoolingTower_.heat_exchange(m_CW)

                # 9.3.6.5 Calculate h_15 using Eq. 25
                p15.h = CoolingTower_.air_enthalpy_out(Q_ct, m_air, p13.h)

                # 9.3.6.6 Calculate T_15 using iterative calculations Eq. 26
                # through Eq. 28
                p15.temperature = CoolingTower_.temperature_air_out(p15.h, P_atm,
                                                                    p12.h, p13.h, 1)

                # 9.3.6.7 Evaluate ω_15 using Eq. 29
                p15.w = humidity_ratio(p15.temperature, P_atm, 1)

                # 9.3.6.8 Solve for m_14 using Eq. 21
                p14.mass_flowrate = CoolingTower_.water_out_flowrate(
                    m_CW, m_air, p13.w, p15.w)

                # 9.3.6.9 Calculate h_14 using Eq. 24
                p14.h = CoolingTower_.water_out_enthalpy(
                    Q_ct, m_CW, p12.h, p14.mass_flowrate)
                p14.temperature = saturated_liquid_temperature(p14.h)

                # 9.3.6.10 Calculate makeup water
                m_makeup = CoolingTower_.make_up_water_flowrate(
                    m_air, p13.w, p15.w)

                # 9.3.6.11 Solve for h_17' and T_17' using Eq. 32
                h_17_new = CoolingTower_.mixer_enthalpy_out(
                    m_CW, m_makeup, p16.h, p14.h)
                print(F'h17_new = {h_17_new}')
                T_17_new = saturated_liquid_temperature(h_17_new)

                # 9.3.6.12 Check error_T_17
                error_T_17 = absolute_total_error(p17.temperature, T_17_new)

                p17.temperature = T_17_new
                p17.h = h_17_new

                # Update dictionaries:
                temperature_dict['T11'] = T_3
                temperature_dict['T12'] = T_12
                temperature_dict['T14'] = T_14
                temperature_dict['T15'] = T_15
                temperature_dict['T17'] = T_17

                enthalpy_dict['h11'] = h_11
                enthalpy_dict['h12'] = h_12
                enthalpy_dict['h14'] = h_14
                enthalpy_dict['h15'] = h_15
                enthalpy_dict['h17'] = h_17

                for i in [11, 12, 17]:
                    massflow_dict[F'm{i}'] = m_CW
                for i in [13, 15]:
                    massflow_dict[F'm{i}'] = m_air
                massflow_dict['m14'] = p14.mass_flowrate
                massflow_dict['m16'] = m_makeup

                print_dataframe()

            # 9.3.7 Calculate Q_a using Eq. 9
            Q_a = Absorber_.heat_from_external_flows(m_CW, p17.h, p11.h)

            # 9.3.8 Calculate T_5 using Eq. 10
            T_5_new = Absorber_.solution_temperature_out(
                Q_a, p10.h, p17.h, p11.h)

            # 9.3.9 Check e_5
            error_T_5 = absolute_total_error(p5.temperature, T_5_new)

            p5.temperature = T_5_new
            p5.h = weak_solution.enthalpy_LiBr_solution(p5.temperature)

        # 9.4 Calculate Q_she
        T_6 = T_5
        h_6 = h_5

        # 9.5 Calculate Q_she
        Q_she = SHE_.heat_flow(m_WS, h_7, h_6)

        # 9.6 Calculate T_7
        T_7 = weak_solution.calc_temp_from_enthalpy(h_7)

        # 9.7 Calculate T_9
        T_9_new = SHE_.strong_solution_temperature_out(Q_she, T_8, T_6, T_7)

        # 9.6 Check e_9
        error_T_9 = absolute_total_error(T_9, T_9_new)

        T_9 = T_9_new
        h_9 = strong_solution.enthalpy_LiBr_solution(T_9)

    # 10. Calcualte Q_g
    Q_g = Generator_.generator_heat_from_flowrates(m_R, CR, h_1, h_8, h_7)

    # 11. Calcualte COP
    COP = Q_e / Q_g

    print(F'Q_g = {Q_g}, m_makeup = {m_makeup}, COP = {COP}')


def abs_eq_GA():
    pass


def simple_absorption_chiller_model(
        Q_e, T_dry_bulb, P_atm, relative_humidity, error_threshold=0.01,
        closest_approach_temp_CW=8,
        COP_max=0.8):

    p1 = structure()
    p2 = structure()
    p3 = structure()
    p4 = structure()
    p5 = structure()
    p6 = structure()
    p7 = structure()
    p8 = structure()
    p9 = structure()
    p10 = structure()
    p11 = structure()
    p12 = structure()
    p13 = structure()
    p14 = structure()
    p15 = structure()
    p16 = structure()
    p17 = structure()
    p18 = structure()
    p19 = structure()
    p20 = structure()
    p21 = structure()

    p1.name = 'p1'
    p2.name = 'p2'
    p3.name = 'p3'
    p4.name = 'p4'
    p5.name = 'p5'
    p6.name = 'p6'
    p7.name = 'p7'
    p8.name = 'p8'
    p9.name = 'p9'
    p10.name = 'p10'
    p11.name = 'p11'
    p12.name = 'p12'
    p13.name = 'p13'
    p14.name = 'p14'
    p15.name = 'p15'
    p16.name = 'p16'
    p17.name = 'p17'
    p18.name = 'p18'
    p19.name = 'p19'
    p20.name = 'p20'
    p21.name = 'p21'

    # 1. Climate data an input for the function

    # 2. Create objects with known UA values
    Evaporator_ = Evaporator()
    Absorber_ = Absorber()
    Generator_ = Generator()
    Condenser_ = Condenser()

    # 3 Input known temperatures and pressures
    # Insert known pressures
    upper_vessel = [p1, p2, p6, p7, p8, p9]
    lower_vessel = [p3, p4, p5, p10]
    for i in upper_vessel:
        i.pressure = Generator_.pressure
    for i in lower_vessel:
        i.pressure = Evaporator_.pressure
    p13.pressure = P_atm
    p15 = P_atm

    r'''
        KNOWN VALUES
        ------------
        Q_e
        T_1, h_1 = h_g : 80 C , 2643 kJ/kg
        T_4, h_4 = h_g : 4 C, 2510 kJ/kg
        T_8, h_8, X_8 : 80 C, 193 kJ/kg, 60%
        T_16, h_16 = h_f : 30 C, 125.7 kJ/kg
        T_18, h_18 : 12 C, 50.4 kJ/kg
        T_19, h_19 : 6 C, 25.2 kJ/kg
        T_20, h_20 : 90.6 C, 379.57 kJ/kg
        T_21, h_21 : 85 C, 366.03 kJ/kg
    '''
    # Temperature
    p1.temperature = 80
    p4.temperature = convert_K_to_C(iapws97._TSat_P(p4.pressure * 10**-3))
    # Set absorber temperature
    p5.temperature = 25
    p8.temperature = p1.temperature
    p17.temperature = 20
    p18.temperature = 12
    p19.temperature = 6
    p20.temperature = 90.6
    # p21.temperature = 85 Make variable

    ###############
    # CONSTRAINTS #
    ###############
    # Refrigerant
    for i in [p1, p2, p3, p4]:
        if i == p1 or i == p4:
            i.fluidtype = 'saturated vapor - water'
        else:
            i.fluidtype = 'saturated liquid - water'
        i.temp_min = 1
        i.temp_max = 100
    # LiBr solution
    for i in [p5, p6, p7, p8, p9, p10]:
        i.fluidtype = 'LiBr solution'
        i.temp_min = 15
        i.temp_max = 165
    # Cooling Water
    for i in [p11, p12, p14, p16, p17]:
        i.fluidtype = 'saturated liquid - water'
        if i == p11:
            i.temp_min = 7.2
            i.temp_max = 36
        else:
            i.temp_min = 4
            i.temp_max = 99
        i.pressure = 786
    # Hot Water
    for i in [p20, p21]:
        i.fluidtype = 'saturated liquid - water'
        i.temp_min = 70
        i.temp_max = 110
        i.pressure = None
    # Chilled Water
    for i in [p18, p19]:
        i.fluidtype = 'satulrated liquid - water'
        i.pressure = 786
        i.temp_min = 3
        i.temp_max = 21
        i.pressure = None
    '''for i in [p13, p15]:
        i.pressure = P_atm
        i.fluidtype = 'humid air'
        i.temp_min = None
        i.temp_max = None'''

    # LiBr Solutions
    X_SS = 60
    X_WS = 57
    strong_solution = LiBr_solution(X_SS)
    weak_solution = LiBr_solution(X_WS)

    #######################################
    # Calculate enthalpy for known values #
    #######################################

    # H2O
    p1.h = superheated_steam_enthalpy(p1.temperature, p1.pressure)
    p4.h = saturated_vapor_enthalpy(p4.temperature)
    p17.h = saturated_liquid_enthalpy(p17.temperature)
    p18.h = saturated_liquid_enthalpy(p18.temperature)
    p19.h = saturated_liquid_enthalpy(p19.temperature)
    p20.h = saturated_liquid_enthalpy(p20.temperature)
    # p21.h = saturated_liquid_enthalpy(p21.temperature)

    # LiBr Solution
    p5.h = weak_solution.enthalpy_LiBr_solution(p5.temperature)
    p8.h = strong_solution.enthalpy_LiBr_solution(p8.temperature)

    # Air
    '''p13.h = humid_air_enthalpy(p13.temperature, p13.pressure, relative_humidity)
    p13.w = humidity_ratio(p13.temperature, p13.pressure, relative_humidity)'''

    #########
    # START #
    #########
    r'''
    EVAPORATOR
    ----------
        The Evaporator will be the only component with very
        explicit solutions that do not require iteration
        1.  Calculate the chilled water mass flowrate (m_ChW)
        2.  Solve for T_3
        3.  Solve for h_3
        4.  Calculate the refrigerant mass flowrate (m_R)
    '''
    # 1. Solve for m_ChW
    m_ChW = Evaporator_.chilled_water_flowrate(Q_e)
    for p in [p18, p19]:
        p.mass_flowrate = m_ChW

    # 2. Solve for T_3 and h_3
    p3.temperature = Evaporator_.refrigerant_temp_in(Q_e)
    p3.h = Evaporator_.refrigerant_enthalpy_in(p3.temperature)

    # 4 Solve for m_R
    m_R = Evaporator_.refrigerant_mass_flowrate(Q_e, p3.h)
    for p in [p1, p2, p3, p4]:
        p.mass_flowrate = m_R

    r'''
    CONDENSER
    ---------
        5. Solve for h2
        6. Solve for Q_c
    '''
    # 5 Solve for h_2 and T_2
    p2.h = p3.h
    p2.temperature = sat_liquid_temp_from_pressure(p2.pressure)

    # 6 Solve for Q_c
    Q_c = m_R * (p1.h - p2.h)

    r'''
    ABSORBER
    ---------
        7.  Calculate the circulation ratio (CR)
        8.  Calculate the strong solution mass flowrate (m_SS)
        9.  Calculate the weak solution mass flowrate (m_WS)
        10. Solve for h10 and T10
        11. Assume value for T5
    '''
    # 7. Calculate CR
    circulation_ratio = calc_circulation_ratio(
        strong_solution.concentration,
        weak_solution.concentration)

    # 8.    Calculate m_SS
    m_SS = strong_solution_flowrate(circulation_ratio, m_R)
    for p in [p8, p10]:
        p.mass_flowrate = m_SS

    # 9. Calculate m_WS
    m_WS = weak_solution_flowrate(circulation_ratio, m_R)
    for p in [p5, p7]:
        p.mass_flowrate = m_WS

    # 10. Solve for h10 and T10
    p10.h = p8.h
    p10.temperature = strong_solution.calc_temp_from_enthalpy(p10.h)

    # 11. Calculate Q_a
    Q_a = Absorber_.heat_from_internal_flows(
        m_R, circulation_ratio, p4.h, p10.h, p5.h)

    # 12. Calculate T11
    p11.temperature = Absorber_.cooling_water_temp_out_LMTD(
        Q_a, p10.temperature, p5.temperature, p17.temperature)
    p11.h = saturated_liquid_enthalpy(p11.temperature)

    # 12. Calculate mCW
    m_CW = Absorber_.cooling_water_flowrate(
        Q_a, p17.temperature, p11.temperature)
    for p in [p11, p12, p17]:
        p.mass_flowrate = m_CW

    # 13. Calculate h7 and T7
    p7.h = p5.h
    p7.temperature = weak_solution.calc_temp_from_enthalpy(p5.h)

    # 14. Calculate Q_g
    Q_g = Generator_.generator_heat_from_flowrates(
        m_R, circulation_ratio, p1.h, p8.h, p7.h)

    # 15. Calculate T21
    p21.temperature = Generator_.hot_water_temp_out(
        Q_g, p20.temperature, p7.temperature, p8.temperature)
    p21.h = saturated_liquid_enthalpy(p21.temperature)

    # 16. Calculate m_HW
    m_HW = Generator_.hot_water_mass_flowrate(
        Q_g, p20.temperature, p21.temperature)

    # 17. Calculate T12
    p12.temperature = (Q_c / (specific_heat_water * m_CW)) + p11.temperature
    p12.h = saturated_liquid_enthalpy(p12.temperature)

    # 11. Calcualte COP
    COP = Q_e / Q_g

    # Function to print statepoints, and all values as a pd dataframe
    statepoints = [
        p1, p2, p3, p4,
        p5, p6, p7, p8,
        p9, p10, p11, p12,
        p17, p18, p19, p20, p21]

    df = statepoint_df(statepoints)
    df.set_index('statepoint', inplace=True)

    print(f"COP: {COP}")
    print(df)


def spare_code():
    '''########################
    # Assume value of T_11 #
    ########################
    # 11. Get T11 and h11
    p11.temperature = np.random.uniform(size=1, low=p17.temperature, high=p11.temp_max)[0]
    if p11.temperature <= p17.temperature:
        p11.temperature = p17.temperature + closest_approach_temp_CW
    p11.h = saturated_liquid_enthalpy(p11.temperature)

    error_T_11 = 1

    iterations = 0
    while (error_T_11 > error_threshold) and (iterations < 100):


        # CONDENSER
        # ---------
        #    12. Calculate T_12
        #    13. Calculate T_11a and h11a
        #    14. Calculate m_CWa

        # 12. Calculate T12 and h12
        p12.temperature = Condenser_.cooling_water_temp_out_LMTD(Q_c, p1.temperature, p2.temperature, p11.temperature)
        p12.h = saturated_liquid_enthalpy(p12.temperature)

        # 13. Calculate m_CW
        m_CW = Condenser_.cooling_water_flowrate(Q_c, p11.temperature, p12.temperature)
        for p in [p11, p12, p17]:
            p.mass_flowrate = m_CW

        # 14. Calculate Q_a
        Q_a = Absorber_.heat_from_external_flows(m_CW, p17.h, p11.h)

        # 15. Calculate Q_g
        Q_g = Q_c + Q_a - Q_e

        #if Q_g < 0  or Q_e / Q_g > 0.8:
        # Q_g = Q_e / COP_max
        # Q_a = Q_g + Q_e - Q_c


        # 16. Solve for T7 and h7
        p7.temperature = Generator_.weak_solution_temperature_in(Q_g, p8.temperature, p20.temperature, p21.temperature)
        p7.h = weak_solution.enthalpy_LiBr_solution(p7.temperature)

        # 17. Solve for h5 and T5
        p5.h = p7.h
        p5.temperature = weak_solution.calc_temp_from_enthalpy(p5.h)

        # 18. Calculate Q_a new
        Q_a_new = Absorber_.heat_from_internal_flows(m_R, circulation_ratio, p4.h, p10.h, p5.h)

        # 19. Calculate T11 new
        T11_new = Absorber_.cooling_water_temp_out_LMTD(Q_a_new, p10.temperature, p5.temperature, p17.temperature)

        if T11_new < p11.temp_min:
            T11_new = p11.temp_min
        if T11_new > p11.temp_max:
            T11_new = p11.temp_max

        error_T_11 = absolute_total_error(p11.temperature, T11_new)

        p11.temperature = T11_new
        p12.temperature = Q_c / specific_heat_water + p11.temperature

        iterations += 1

    # 11. Calcualte COP
    COP = Q_e / Q_g

    # Function to print statepoints, and all values as a pd dataframe
    statepoints = [
        p1, p2, p3, p4,
        p5, p6, p7, p8,
        p9, p10, p11, p12]

    df = statepoint_df(statepoints)
    df.set_index('statepoint', inplace=True)

    print(f"Iterations: {iterations}")
    print(f"COP: {COP}")
    print(df)

    print(F'Qe={Q_e} Qc={Q_c} Qa={Q_a} Qg={Q_g}')
    print(F'Error T11 = {error_T_11}')'''
    pass


def Ren_et_al_AbsCh_model_sansSHX(
        Q_e, T_dry_bulb, P_atm, relative_humidity,
        m_e=2.39232, m_g=2.43054, m_a=3.29257, m_c=2.53054,
        T_g=80, T_a=38,
        single_CW_circuit=True, absorber_condenser_temp_same=True,
        error_threshold=0.01, closest_approach_temp_CW=8,
        COP_max=0.8):

    p1 = structure()
    p2 = structure()
    p3 = structure()
    p4 = structure()
    p5 = structure()
    p6 = structure()
    p7 = structure()
    p8 = structure()
    p9 = structure()
    p10 = structure()
    p11 = structure()
    p12 = structure()
    p13 = structure()
    p14 = structure()
    p15 = structure()
    p16 = structure()
    p17 = structure()
    p18 = structure()
    p19 = structure()
    p20 = structure()
    p21 = structure()
    p22 = structure()

    p1.name = 'p1'
    p2.name = 'p2'
    p3.name = 'p3'
    p4.name = 'p4'
    p5.name = 'p5'
    p6.name = 'p6'
    p7.name = 'p7'
    p8.name = 'p8'
    p9.name = 'p9'
    p10.name = 'p10'
    p11.name = 'p11'
    p12.name = 'p12'
    p13.name = 'p13'
    p14.name = 'p14'
    p15.name = 'p15'
    p16.name = 'p16'
    p17.name = 'p17'
    p18.name = 'p18'
    p19.name = 'p19'
    p20.name = 'p20'
    p21.name = 'p21'
    p22.name = 'p22'

    # 1. Climate data an input for the function

    # 2. Create objects with known UA values
    Evaporator_ = Evaporator()
    Absorber_ = Absorber()
    Generator_ = Generator()
    Condenser_ = Condenser()

    # 3 Input known temperatures and pressures
    # Insert known pressures
    upper_vessel = [p1, p2, p6, p7, p8, p9]
    lower_vessel = [p3, p4, p5, p10]
    for i in upper_vessel:
        i.pressure = Generator_.pressure
    for i in lower_vessel:
        i.pressure = Evaporator_.pressure
    p13.pressure = P_atm
    p15 = P_atm

    r'''
        KNOWN VALUES
        ------------
        Q_e
        T_1, h_1 = h_g : 80 C , 2643 kJ/kg
        T_4, h_4 = h_g : 4 C, 2510 kJ/kg
        T_8, h_8, X_8 : 80 C, 193 kJ/kg, 60%
        T_16, h_16 = h_f : 30 C, 125.7 kJ/kg
        T_18, h_18 : 12 C, 50.4 kJ/kg
        T_19, h_19 : 6 C, 25.2 kJ/kg
        T_20, h_20 : 90.6 C, 379.57 kJ/kg
        T_21, h_21 : 85 C, 366.03 kJ/kg
    '''
    # Temperature
    p1.temperature = T_g
    p4.temperature = convert_K_to_C(iapws97._TSat_P(p4.pressure * 10**-3))
    # Set absorber temperature
    if absorber_condenser_temp_same is True:
        p5.temperature = Condenser_.condenser_temperature()
    else:
        p5.temperature = T_a
    p8.temperature = p1.temperature
    p17.temperature = 20
    p18.temperature = 12
    p20.temperature = 90.6
    # p21.temperature = 85 Make variable

    ###############
    # CONSTRAINTS #
    ###############
    # Refrigerant
    for i in [p1, p2, p3, p4]:
        if i == p1 or i == p4:
            i.fluidtype = 'saturated vapor - water'
        else:
            i.fluidtype = 'saturated liquid - water'
        i.temp_min = 1
        i.temp_max = 100
    # LiBr solution
    for i in [p5, p6, p7, p8, p9, p10]:
        i.fluidtype = 'LiBr solution'
        i.temp_min = 15
        i.temp_max = 165
    # Cooling Water
    for i in [p11, p12, p14, p16, p17]:
        i.fluidtype = 'saturated liquid - water'
        if i == p11:
            i.temp_min = 7.2
            i.temp_max = 36
        else:
            i.temp_min = 4
            i.temp_max = 99
        i.pressure = 786
    # Hot Water
    for i in [p20, p21]:
        i.fluidtype = 'saturated liquid - water'
        i.temp_min = 70
        i.temp_max = 110
        i.pressure = None
    # Chilled Water
    for i in [p18, p19]:
        i.fluidtype = 'satulrated liquid - water'
        i.pressure = 786
        i.temp_min = 3
        i.temp_max = 21
        i.pressure = None
    '''for i in [p13, p15]:
        i.pressure = P_atm
        i.fluidtype = 'humid air'
        i.temp_min = None
        i.temp_max = None'''

    # LiBr Solutions
    X_SS = 60
    X_WS = 57
    strong_solution = LiBr_solution(X_SS)
    weak_solution = LiBr_solution(X_WS)

    #######################################
    # Calculate enthalpy for known values #
    #######################################

    # H2O
    p1.h = superheated_steam_enthalpy(p1.temperature, p1.pressure)
    p4.h = saturated_vapor_enthalpy(p4.temperature)
    p17.h = saturated_liquid_enthalpy(p17.temperature)
    p18.h = saturated_liquid_enthalpy(p18.temperature)
    p20.h = saturated_liquid_enthalpy(p20.temperature)
    # p21.h = saturated_liquid_enthalpy(p21.temperature)

    # LiBr Solution
    p5.h = weak_solution.enthalpy_LiBr_solution(p5.temperature)
    p8.h = strong_solution.enthalpy_LiBr_solution(p8.temperature)

    # Air
    '''p13.h = humid_air_enthalpy(p13.temperature, p13.pressure, relative_humidity)
    p13.w = humidity_ratio(p13.temperature, p13.pressure, relative_humidity)'''

    #########
    # START #
    #########
    # 1. Calculate T19
    p19.temperature = Evaporator_.chilled_water_T_out(
        m_e, Q_e, p18.temperature)
    p19.h = saturated_liquid_enthalpy(p19.temperature)

    for p in [p18, p19]:
        p.mass_flowrate = m_e

    # 2. Solve for T_3 and h_3
    p3.temperature = Evaporator_.refrigerant_temp_in(
        Q_e, p4.temperature, p18.temperature, p19.temperature)
    p3.h = saturated_liquid_enthalpy(p3.temperature)

    # 4 Solve for m_R
    m_R = Evaporator_.refrigerant_mass_flowrate(Q_e, p3.h, p4.h)
    for p in [p1, p2, p3, p4]:
        p.mass_flowrate = m_R

    r'''
    CONDENSER
    ---------
        5. Solve for h2
        6. Solve for Q_c
    '''
    # 5 Solve for h_2 and T_2
    p2.h = p3.h
    p2.temperature = sat_liquid_temp_from_pressure(p2.pressure)

    # 6 Solve for Q_c
    Q_c = m_R * (p1.h - p2.h)

    r'''
    ABSORBER
    ---------
        7.  Calculate the circulation ratio (CR)
        8.  Calculate the strong solution mass flowrate (m_SS)
        9.  Calculate the weak solution mass flowrate (m_WS)
        10. Solve for h10 and T10
        11. Assume value for T5
    '''
    # 7. Calculate CR
    circulation_ratio = calc_circulation_ratio(
        strong_solution.concentration,
        weak_solution.concentration)

    # 8.    Calculate m_SS
    m_SS = strong_solution_flowrate(circulation_ratio, m_R)
    for p in [p8, p10]:
        p.mass_flowrate = m_SS

    # 9. Calculate m_WS
    m_WS = weak_solution_flowrate(circulation_ratio, m_R)
    for p in [p5, p7]:
        p.mass_flowrate = m_WS

    # 10. Solve for h10 and T10
    p10.h = p8.h
    p10.temperature = strong_solution.calc_temp_from_enthalpy(p10.h)

    # 11. Calculate Q_a
    Q_a = Absorber_.heat_from_internal_flows(
        m_R, circulation_ratio, p4.h, p10.h, p5.h)

    # 12. Calculate T22
    p22.temperature = Absorber_.cooling_water_temp_out(
        Q_a, m_a, p17.temperature)
    p22.h = saturated_liquid_enthalpy(p22.temperature)

    # 12. Calculate mCW
    for p in [p17, p22]:
        p.mass_flowrate = m_a
    for p in [p11, p12]:
        p.mass_flowrate = m_a  # m_c

    # 13. Calculate h7 and T7
    p7.h = p5.h
    p7.temperature = weak_solution.calc_temp_from_enthalpy(p5.h)

    # 14. Calculate Q_g
    Q_g = Generator_.generator_heat_from_flowrates(
        m_R, circulation_ratio, p1.h, p8.h, p7.h)

    # 15. Calculate T21
    p21.temperature = Generator_.hot_water_temp_out(Q_g, m_g, p20.temperature)
    p21.h = saturated_liquid_enthalpy(p21.temperature)

    # 16. Calculate T12
    if single_CW_circuit:
        p11.temperature = p22.temperature  # 30
        p11.h = saturated_liquid_enthalpy(p11.temperature)
    else:
        p11.temperature = 30
        p11.h = saturated_liquid_enthalpy(p11.temperature)
    p12.temperature = (Q_c / (specific_heat_water *
                       p11.mass_flowrate)) + p11.temperature
    p12.h = saturated_liquid_enthalpy(p12.temperature)

    # 11. Calcualte COP
    COP = Q_e / Q_g

    # Function to print statepoints, and all values as a pd dataframe
    statepoints = [
        p1, p2, p3, p4,
        p5, p6, p7, p8,
        p9, p10, p11, p12,
        p17, p18, p19, p20,
        p21, p22]

    df = statepoint_df(statepoints)
    df.set_index('statepoint', inplace=True)

    print(f"COP: {COP}")
    print(df)


def Ren_et_al_AbsCh_model_wSHX(
        Q_e, T_dry_bulb, P_atm, relative_humidity,
        m_e=2.39232, m_g=2.43054, m_a=3.29257, m_c=2.53054,
        T_g=80, T_a=38,
        eff_shx=0.9,
        single_CW_circuit=True, absorber_condenser_temp_same=True,
        error_threshold=0.01, closest_approach_temp_CW=8,
        COP_max=0.8):

    p1 = structure()
    p2 = structure()
    p3 = structure()
    p4 = structure()
    p5 = structure()
    p6 = structure()
    p7 = structure()
    p8 = structure()
    p9 = structure()
    p10 = structure()
    p11 = structure()
    p12 = structure()
    p13 = structure()
    p14 = structure()
    p15 = structure()
    p16 = structure()
    p17 = structure()
    p18 = structure()
    p19 = structure()
    p20 = structure()
    p21 = structure()
    p22 = structure()

    p1.name = 'p1'
    p2.name = 'p2'
    p3.name = 'p3'
    p4.name = 'p4'
    p5.name = 'p5'
    p6.name = 'p6'
    p7.name = 'p7'
    p8.name = 'p8'
    p9.name = 'p9'
    p10.name = 'p10'
    p11.name = 'p11'
    p12.name = 'p12'
    p13.name = 'p13'
    p14.name = 'p14'
    p15.name = 'p15'
    p16.name = 'p16'
    p17.name = 'p17'
    p18.name = 'p18'
    p19.name = 'p19'
    p20.name = 'p20'
    p21.name = 'p21'
    p22.name = 'p22'

    # 1. Climate data an input for the function

    # 2. Create objects with known UA values
    Evaporator_ = Evaporator()
    Absorber_ = Absorber()
    Generator_ = Generator()
    Condenser_ = Condenser()
    SHX_ = SolutionHeatExhanger(effectiveness=eff_shx)

    # 3 Input known temperatures and pressures
    # Insert known pressures
    upper_vessel = [p1, p2, p6, p7, p8, p9]
    lower_vessel = [p3, p4, p5, p10]
    for i in upper_vessel:
        i.pressure = Generator_.pressure
    for i in lower_vessel:
        i.pressure = Evaporator_.pressure
    p13.pressure = P_atm
    p15 = P_atm

    r'''
        KNOWN VALUES
        ------------
        Q_e
        T_1, h_1 = h_g : 80 C , 2643 kJ/kg
        T_4, h_4 = h_g : 4 C, 2510 kJ/kg
        T_8, h_8, X_8 : 80 C, 193 kJ/kg, 60%
        T_16, h_16 = h_f : 30 C, 125.7 kJ/kg
        T_18, h_18 : 12 C, 50.4 kJ/kg
        T_19, h_19 : 6 C, 25.2 kJ/kg
        T_20, h_20 : 90.6 C, 379.57 kJ/kg
        T_21, h_21 : 85 C, 366.03 kJ/kg
    '''
    # Temperature
    p1.temperature = T_g
    p4.temperature = convert_K_to_C(iapws97._TSat_P(p4.pressure * 10**-3))
    # Set absorber temperature
    if absorber_condenser_temp_same is True:
        p5.temperature = Condenser_.condenser_temperature()
    else:
        p5.temperature = T_a
    p8.temperature = p1.temperature
    p17.temperature = 30
    p18.temperature = 12
    p20.temperature = 90.6
    # p21.temperature = 85 Make variable

    ###############
    # CONSTRAINTS #
    ###############
    # Refrigerant
    for i in [p1, p2, p3, p4]:
        if i == p1 or i == p4:
            i.fluidtype = 'saturated vapor - water'
        else:
            i.fluidtype = 'saturated liquid - water'
        i.temp_min = 1
        i.temp_max = 100
    # LiBr solution
    for i in [p5, p6, p7, p8, p9, p10]:
        i.fluidtype = 'LiBr solution'
        i.temp_min = 15
        i.temp_max = 165
    # Cooling Water
    for i in [p11, p12, p14, p16, p17]:
        i.fluidtype = 'saturated liquid - water'
        if i == p11:
            i.temp_min = 7.2
            i.temp_max = 36
        else:
            i.temp_min = 4
            i.temp_max = 99
        i.pressure = 786
    # Hot Water
    for i in [p20, p21]:
        i.fluidtype = 'saturated liquid - water'
        i.temp_min = 70
        i.temp_max = 110
        i.pressure = None
    # Chilled Water
    for i in [p18, p19]:
        i.fluidtype = 'satulrated liquid - water'
        i.pressure = 786
        i.temp_min = 3
        i.temp_max = 21
        i.pressure = None
    '''for i in [p13, p15]:
        i.pressure = P_atm
        i.fluidtype = 'humid air'
        i.temp_min = None
        i.temp_max = None'''

    # LiBr Solutions
    X_SS = 60
    X_WS = 57
    strong_solution = LiBr_solution(X_SS)
    weak_solution = LiBr_solution(X_WS)

    #######################################
    # Calculate enthalpy for known values #
    #######################################

    # H2O
    p1.h = superheated_steam_enthalpy(p1.temperature, p1.pressure)
    p4.h = saturated_vapor_enthalpy(p4.temperature)
    p17.h = saturated_liquid_enthalpy(p17.temperature)
    p18.h = saturated_liquid_enthalpy(p18.temperature)
    p20.h = saturated_liquid_enthalpy(p20.temperature)
    # p21.h = saturated_liquid_enthalpy(p21.temperature)

    # LiBr Solution
    p5.h = weak_solution.enthalpy_LiBr_solution(p5.temperature)
    p8.h = strong_solution.enthalpy_LiBr_solution(p8.temperature)

    # Air
    '''p13.h = humid_air_enthalpy(p13.temperature, p13.pressure, relative_humidity)
    p13.w = humidity_ratio(p13.temperature, p13.pressure, relative_humidity)'''

    #########
    # START #
    #########
    # 1. Calculate T19
    p19.temperature = Evaporator_.chilled_water_T_out(
        m_e, Q_e, p18.temperature)
    p19.h = saturated_liquid_enthalpy(p19.temperature)

    for p in [p18, p19]:
        p.mass_flowrate = m_e

    # 2. Solve for T_3 and h_3
    p3.temperature = Evaporator_.refrigerant_temp_in(
        Q_e, p4.temperature, p18.temperature, p19.temperature)
    p3.h = saturated_liquid_enthalpy(p3.temperature)

    # 4 Solve for m_R
    m_R = Evaporator_.refrigerant_mass_flowrate(Q_e, p3.h, p4.h)
    for p in [p1, p2, p3, p4]:
        p.mass_flowrate = m_R

    # 5 Solve for h_2 and T_2
    p2.h = p3.h
    p2.temperature = sat_liquid_temp_from_pressure(p2.pressure)

    # 6 Solve for Q_c
    Q_c = m_R * (p1.h - p2.h)

    # 7. Calculate CR
    circulation_ratio = calc_circulation_ratio(
        strong_solution.concentration,
        weak_solution.concentration)

    # 8.    Calculate m_SS
    m_SS = strong_solution_flowrate(circulation_ratio, m_R)
    for p in [p8, p9, p10]:
        p.mass_flowrate = m_SS

    # 9. Calculate m_WS
    m_WS = weak_solution_flowrate(circulation_ratio, m_R)
    for p in [p5, p6, p7]:
        p.mass_flowrate = m_WS

    # 10. Calculate h6, T6
    p6.h = p5.h
    p6.temperature = weak_solution.calc_temp_from_enthalpy(p6.h)

    # 11. Calculate T9 using SHX
    p9.temperature = SHX_.strong_solution_temperature_out_effectiveness(
        p8.temperature, p6.temperature)
    p9.h = weak_solution.enthalpy_LiBr_solution(p9.temperature)

    # 12. Calculate Q_shx
    Q_shx = SHX_.heat_flow(m_SS, p8.h, p9.h)

    # 13. Calculate T7
    p7.h = (Q_shx / m_SS) + p6.h
    p7.temperature = weak_solution.calc_temp_from_enthalpy(p7.h)

    # 14. Calculate Q_g
    Q_g = Generator_.generator_heat_from_flowrates(
        m_R, circulation_ratio, p1.h, p8.h, p7.h)

    # 15. Solve for h10 and T10
    p10.h = p9.h
    p10.temperature = strong_solution.calc_temp_from_enthalpy(p9.h)

    # 16. Calculate Q_a
    Q_a = Absorber_.heat_from_internal_flows(
        m_R, circulation_ratio, p4.h, p10.h, p5.h)

    # 17. Calculate T21
    p21.temperature = Generator_.hot_water_temp_out(Q_g, m_g, p20.temperature)
    p21.h = saturated_liquid_enthalpy(p21.temperature)

    # 18. Calculate T22
    p22.temperature = Absorber_.cooling_water_temp_out(
        Q_a, m_a, p17.temperature)
    p22.h = saturated_liquid_enthalpy(p22.temperature)

    # 19. mCW
    for p in [p17, p22]:
        p.mass_flowrate = m_a
    for p in [p11, p12]:
        p.mass_flowrate = m_a  # m_c

    # 20. Calculate T12
    if single_CW_circuit:
        p11.temperature = p22.temperature  # 30
        p11.h = saturated_liquid_enthalpy(p11.temperature)
    else:
        p11.temperature = 30
        p11.h = saturated_liquid_enthalpy(p11.temperature)
    p12.temperature = (Q_c / (specific_heat_water *
                       p11.mass_flowrate)) + p11.temperature
    p12.h = saturated_liquid_enthalpy(p12.temperature)

    # 11. Calcualte COP
    COP = Q_e / Q_g

    # Function to print statepoints, and all values as a pd dataframe
    statepoints = [
        p1, p2, p3, p4,
        p5, p6, p7, p8,
        p9, p10, p11, p12,
        p17, p18, p19, p20,
        p21, p22]

    df = statepoint_df(statepoints)
    df.set_index('statepoint', inplace=True)

    print(f"COP: {COP}")
    print(df)


###########
# TESTING #
###########
# Evap_ = Evaporator()
# t_3 = Evap_.refrigerant_temp_in(50)
# print(t_3)
# print(calculate_P_sat(4))
###############
# Test Values #
###############
Q_e_test = 49.5
T_drybulb_test = 30.6
RH_test = 0.57
P_test = convert_mbar_to_kPa(978)

# absoprtion_chiller_equilibrium(Q_e_test, T_drybulb_test, P_test, RH_test)


'''simple_absorption_chiller_model(Q_e=50,
    T_dry_bulb=T_drybulb_test, P_atm=P_test, relative_humidity=RH_test)'''

'''Ren_et_al_AbsCh_model_wSHX(
    Q_e=50,
    T_dry_bulb=T_drybulb_test, P_atm=P_test, relative_humidity=RH_test,
    T_g = 80, T_a = 38, eff_shx=0.7,
    absorber_condenser_temp_same=False,
    single_CW_circuit=True)'''


def T_a(T11=36, T17=25, T_gen=80):
    A = Absorber()
    G = Generator()
    E = Evaporator()

    p1 = structure()
    p4 = structure()
    p5 = structure()
    p7 = structure()
    p8 = structure()
    p10 = structure()
    p11 = structure()
    p17 = structure()
    p20 = structure()
    p21 = structure()

    # Refrigerant - H2O
    p1.t = 80
    p1.p = G.pressure
    p1.h = superheated_steam_enthalpy(p1.t, p1.p)

    p4.p = E.pressure
    p4.t = convert_K_to_C(iapws97._TSat_P(p4.p * 10**-3))
    p4.h = saturated_vapor_enthalpy(p4.t)

    p17.t = T17
    p17.h = saturated_liquid_enthalpy(p17.t)

    p20.t = T_gen
    p20.h = saturated_liquid_enthalpy(p20.t)

    # LiBr
    strong_solution = LiBr_solution(60)
    weak_solution = LiBr_solution(57)

    p8.t = p1.t
    p8.h = strong_solution.enthalpy_LiBr_solution(p8.t)

    p10.t = p8.t
    p10.h = p8.h

    m_r = 0.02
    m_w = 0.4
    m_s = 0.38

    T_a = np.arange(30, 50, 0.5)
    h5 = weak_solution.enthalpy_LiBr_solution(T_a)
    print(p4.t)

    Q_a = m_r * p4.h + m_s * p10.h - m_w * h5
    Q_g = m_r * p1.h + m_s * p8.h - m_w * h5

    t11 = A.cooling_water_temp_out_LMTD(Q_a, p10.t, T_a, p17.t)

    print(t11)


def check_Qg(T_g, X_s=60, X_w=57):
    strong_solution = LiBr_solution(X_s)
    weak_solution = LiBr_solution(X_w)

    E = Evaporator()
    T4 = convert_K_to_C(iapws97._TSat_P(E.pressure * 10**-3))
    h4 = saturated_vapor_enthalpy(T4)
    T18 = 12
    h18 = saturated_liquid_enthalpy(T18)
    m_ch = 2.39232

    Q_e = np.arange(5, 51, 1)

    T19 = E.chilled_water_T_out(m_ch, Q_e, T18)

    T3_array = []
    h3_array = []
    mr_array = []
    for (i, j) in zip(Q_e, T19):

        T3 = E.refrigerant_temp_in(i, T4, T18, j)
        h3 = saturated_liquid_enthalpy(T3)
        mr = E.refrigerant_mass_flowrate(i, h3, h4)

        T3_array.append(T3)
        h3_array.append(h3)
        mr_array.append(mr)

    T3_array = np.asarray(T3_array)
    h3_array = np.asarray(h3_array)
    mr_array = np.asarray(mr_array)

    CR = calc_circulation_ratio(X_s, X_w)

    mss_array = CR * mr_array
    mws_array = (1 + CR) * mr_array

    Q_g = Q_e / 0.8

    h1 = superheated_steam_enthalpy(T_g, 10)
    h8 = strong_solution.enthalpy_LiBr_solution(T_g)

    print(mr_array)
    print(mss_array)
    print(mws_array)
    print(Q_g)
    print(h1, h8)
    exit()

    h7_max = (mr_array * h1 + mss_array * h8 - Q_g) / mws_array

    print(h7_max)

    exit()
    T7_max = weak_solution.calc_temp_from_enthalpy(h7_max)

    for i in Q_e:
        print(i, h7_max)

# check_Qg(T_g = 80)


def Rens_algorithm(
        m_e=2.39232, m_g=2.43054, m_a=3.29257, m_c=2.53054,
        T_g=80, T_a=38,
        eff_shx=0.9,
        single_CW_circuit=True, absorber_condenser_temp_same=True,
        error_threshold=0.01):

    p1 = structure()
    p2 = structure()
    p3 = structure()
    p4 = structure()
    p5 = structure()
    p6 = structure()
    p7 = structure()
    p8 = structure()
    p9 = structure()
    p10 = structure()
    p11 = structure()
    p12 = structure()
    p13 = structure()
    p14 = structure()
    p15 = structure()
    p16 = structure()
    p17 = structure()
    p18 = structure()

    p1.name = 'p1'
    p2.name = 'p2'
    p3.name = 'p3'
    p4.name = 'p4'
    p5.name = 'p5'
    p6.name = 'p6'
    p7.name = 'p7'
    p8.name = 'p8'
    p9.name = 'p9'
    p10.name = 'p10'
    p11.name = 'p11'
    p12.name = 'p12'
    p13.name = 'p13'
    p14.name = 'p14'
    p15.name = 'p15'
    p16.name = 'p16'
    p17.name = 'p17'
    p18.name = 'p18'

    # 1. Create objects with known UA values
    Evaporator_ = Evaporator()
    Absorber_ = Absorber()
    Generator_ = Generator()
    Condenser_ = Condenser()
    SHX_ = SolutionHeatExhanger(effectiveness=eff_shx)

    m_ss = 0.5004

    # 3 Input known temperatures and pressures
    # Insert known pressures
    upper_vessel = [p2, p3, p4, p5, p7, p8]
    lower_vessel = [p1, p6, p9, p10]
    for i in upper_vessel:
        i.pressure = Generator_.pressure
    for i in lower_vessel:
        i.pressure = Evaporator_.pressure

    # Temperature
    p11.temperature = 91  # Hot Water in
    p13.temperature = 30  # A Cooling Water in
    p15.temperature = 30  # C Cooling Water in
    p17.temperature = 13  # Chilled Water in

    specific_heat_vapor = 1.86

    StrongSolution = LiBr_solution(0.60, mass_fraction_max=0.64)
    WeakSolution = LiBr_solution(0.57, mass_fraction_max=0.57)

    #####################
    # Calculation Start #
    #####################
    # Assume T10 (Evaporator Temp)
    p10.temperature = 5
    error_10 = 1

    while error_10 > error_threshold:
        p10.h = saturated_vapor_enthalpy(p10.temperature)

        # Assume T8 (Condenser Temp)
        p8.temperature = 38

        error_8 = 1
        while error_8 > error_threshold:
            p8.h = saturated_liquid_enthalpy(p8.temperature)

            # Assume value of T4 (Generator Temp)
            p4.temperature = 80
            error_4 = 1
            while error_4 > error_threshold:
                p4.h = StrongSolution.enthalpy_LiBr_solution(p4.temperature)

                StrongSolution.mass_fraction = StrongSolution.Duhring_equilibrium_concentration(
                    p8.temperature, p4.temperature)
                StrongSolution.check_mass_fraction()

                p6.temperature_eq = StrongSolution.Duhring_equilibrium_temperature(
                    p10.temperature)

                # Assume value of T5
                p5.temperature = 50.6
                error_5 = 1
                while error_5 > error_threshold:
                    cp_ss = StrongSolution.isobaric_heat_capacity(
                        np.mean([p5.temperature, p4.temperature]))

                    # Assume value of T1
                    p1.temperature = 38
                    error_1 = 1
                    
                    calcs = 0
                    while error_1 > error_threshold:
                        WeakSolution.mass_fraction = WeakSolution.Duhring_equilibrium_concentration(
                            p10.temperature, p1.temperature)
                        WeakSolution.check_mass_fraction()

                        CR = calc_circulation_ratio(
                            StrongSolution.mass_fraction, WeakSolution.mass_fraction)

                        m_r = m_ss / CR
                        print(F'{calcs}:')
                        print(F'xs = {StrongSolution.mass_fraction}, xws: {WeakSolution.mass_fraction}') 
                        print(F'mr: {m_r}, {CR}')
                        m_ws = m_ss + m_r

                        p3.temperature_eq = WeakSolution.Duhring_equilibrium_temperature(
                            p8.temperature)

                        # Absorber
                        Q_a = Absorber_.heat_absorber_Ren(m_ref=m_r,
                                                          m_ws=m_ws,
                                                          x_ws=WeakSolution.mass_fraction,
                                                          x_ss=StrongSolution.mass_fraction,
                                                          Tss_in=p5.temperature,
                                                          Tws_out=p1.temperature,
                                                          Tcon=p10.temperature,
                                                          Tref_in=p10.temperature)
                        
                        # print(F'Calcs: {calcs}, Mass Flows: ({m_r}, {m_ws}), x: ({WeakSolution.mass_fraction}, {StrongSolution.mass_fraction}), temp ({p5.temperature}, {p1.temperature}, {p10.temperature}')
                        calcs += 1
                        # Solve for T14
                        p14.temperature = Absorber_.cooling_water_temp_out(
                            Q_a, m_a, p13.temperature)

                        # Calculate new T1 according to Qa = UA LMTD
                        # ERROR
                        
                        T1_new = Absorber_.solution_temperature_out_LMTD(
                            Q_a, p5.temperature, p13.temperature, p14.temperature)

                        # Check error
                        error_1 = absolute_total_error(p1.temperature, T1_new)
                        p1.temperature = T1_new
                        p2.temperature = p1.temperature

                    # Calculate QSHE
                    Q_shx = SHX_.heat_flow_Ren(
                        m_ss, p4.temperature, p5.temperature, StrongSolution.mass_fraction)

                    # Solve for T3
                    p3.temperature = SHX_.weak_solution_temperature_out_Ren(
                        m_ws, WeakSolution.mass_fraction, p2.temperature, Q_shx)

                    # Calculate new T5
                    T5_new = SHX_.LMTD_strong_solution_temperature_out(
                        Q_shx, p4.temperature, p2.temperature, p3.temperature)

                    # Check error
                    error_5 = absolute_total_error(p5.temperature, T5_new)

                    p5.temperature = T5_new

                # Generator
                # Calculate QG
                Q_g = Generator_.heat_generator_Ren(m_r, m_ws, WeakSolution.mass_fraction,
                                                    StrongSolution.mass_fraction, p4.temperature, p3.temperature, p8.temperature, p7.temperature)

                # Solve for T12
                p12.temperature = Generator_.hot_water_temp_out(
                    Q_g, m_g, p11.temperature)

                # Determinte new T4
                T4_new = Generator_.strong_solution_temp_out_LMTD(
                    Q_g, p11.temperature, p12.temperature, p3.temperature)

                # Check Error
                error_4 = absolute_total_error(p4.temperature, T4_new)
                p4.temperature = T4_new

            # Condenser
            # Calculate Qc
            Q_c = Condenser_.heat_condenser_Ren(
                m_r, p7.temperature, p8.temperature)

            # Solve for T16
            p16.temperature = Condenser_.cooling_water_temp_out(
                Q_c, m_cw, p15.temperature)

            # Determine new T8
            T8_new = Condenser_.refrigerant_temp_out_LMTD(
                Q_c, p7.temperature, p15.temperature, p16.temperature)

            # Check Error
            error_8 = absolute_total_error(p8.temperature, T8_new)
            p8.temperature = T8_new
            p9.temperature = p8.temperature

        # Calculate Qe
        Q_e = Evaporator_.heat_evaporator(m_r, p9.temperature, p10.temperature)
        # Solve for T18
        p18.temperature = Evaporator_.chilled_water_T_out(
            m_e, Q_e, p17.temperature)

        # Calculate new T10
        T10_new = Evaporator_.refrigerant_temp_out_LMTD(
            Q_e, p9.temperature, p17.temperature, p18.temperature)

        # Check error
        error_10 = absolute_total_error(p10.temperature, T10_new)
        p10.temperature = T10_new

    # 11. Calcualte COP
    COP = Q_e / Q_g

    # Function to print statepoints, and all values as a pd dataframe
    for i in [p1, p2, p3]:
        i.mass_flowrate = m_ws
    for i in [p4, p5, p6]:
        i.mass_flowrate = m_ss
    for i in [p7, p8, p9, p10]:
        i.mass_flowrate = m_r
    for i in [p11, p12]:
        i.mass_flowrate = m_g
    for i in [p13, p14]:
        i.mass_flowrate = m_a
    for i in [p15, p16]:
        i.mass_flowrate = m_c
    for i in [p17, p18]:
        i.mass_flowrate = m_e

    statepoints = [
        p1, p2, p3, p4,
        p5, p6, p7, p8,
        p9, p10, p11, p12,
        p17, p18]

    df = statepoint_df(statepoints, method="Ren")
    df.set_index('statepoint', inplace=True)

    print(f"COP: {COP}")
    print(df)


# Rens_algorithm()
