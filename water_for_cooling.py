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

from sysClasses import *
##########################################################################

####################
# GLOBAL VARIABLES #
####################

specific_heat_water = 4.186  # kJ / (kg K)

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


def absolute_total_error(old_value, new_value):
    difference = old_value - new_value
    return abs(difference)


def absolute_percent_relative_error(old_value, new_value):
    absolute_relative_error = abs(
        absolute_total_error(
            old_value,
            new_value) / new_value)
    return absolute_relative_error * 100

#############################
# Lithium Bromide Equations #
#############################


class LiBr_solution:

    def __init__(self, concentration):
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
        '''
        self.concentration = concentration

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

        A = self._Li_Br_summation(coef=A_n, x=self.concentration)
        B = self._Li_Br_summation(coef=B_n, x=self.concentration)
        C = self._Li_Br_summation(coef=C_n, x=self.concentration)

        return(A + solution_temperature * B + solution_temperature**2 * C)

    def _Li_Br_summation(self, n=5, coef=[], x=0):
        summation = 0
        for i in range(n):
            summation += coef[i] * x**i
        return summation

    def solution_temp(self, refrigerant_temp, A_n=[], B_n=[]):
        A = self._Li_Br_summation(n=4, coef=A_n, x=self.concentration)
        B = self._Li_Br_summation(n=4, coef=B_n, x=self.concentration)

        return B + refrigerant_temp * A

    def refrigerant_temp(self, solution_temp, A_n=[], B_n=[]):
        A = self._Li_Br_summation(n=4, coef=A_n, x=self.concentration)
        B = self._Li_Br_summation(n=4, coef=B_n, x=self.concentration)

        return (solution_temp - B) / A

    def pressure_sat(self, refrigerant_temp, C, D, E):
        T = refrigerant_temp + 273.15
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

        A = self._Li_Br_summation(coef=A_n, x=self.concentration)
        B = self._Li_Br_summation(coef=B_n, x=self.concentration)
        C = self._Li_Br_summation(coef=C_n, x=self.concentration)

        t = Symbol('t')

        solutions = solve((A - enthalpy) + (B * t) + (C * t**2), t)

        for i in solutions:
            if i > 0:
                return i
            else:
                pass

####################################
# GENERIC HEAT EXCHANGER EQUATIONS #
####################################


def calculate_cold_temperature_in(Q, UA,
                                  hot_temperature_in, hot_temperature_out,
                                  cold_temperature_out):
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
        hot_temperature_in: hot temperature in (C)
        hot_temperature_out: hot fluid temperature out (C)
        cold_temperature_out: cold fluid temperature out (C)

        Output
        ------
        cold_temperature_in: cold fluid temperature in

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
    theta_1 = hot_temperature_in - cold_temperature_out

    # x used as an unknown variable for theta_2
    x = Symbol('x')

    solutions = solve((((theta_1)**2) * x) +
                      (theta_1 * x**2) - 2 * (Q / UA)**3, x)

    for i in solutions:
        if i > 0:
            theta_2 = i
        else:
            pass

    cold_temperature_in = hot_temperature_out - theta_2
    return cold_temperature_in


def calculate_cold_temperature_out(Q, UA,
                                   hot_temperature_in, hot_temperature_out,
                                   cold_temperature_in):
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
        hot_temperature_in: hot temperature in (C)
        hot_temperature_out: hot fluid temperature out (C)
        cold_temperature_in: cold fluid temperature in (C)

        Output
        ------
        cold_temperature_out: cold fluid temperature out (C)

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
    theta_2 = hot_temperature_out - cold_temperature_in

    # x used as an unknown variable for theta_2
    x = Symbol('x')

    solutions = solve((((theta_2)**2) * x) +
                      (theta_2 * x**2) - 2 * (Q / UA)**3, x)

    for i in solutions:
        if i > 0:
            theta_1 = i
        else:
            pass

    cold_temperature_out = hot_temperature_in - theta_1
    return cold_temperature_out


def calculate_hot_temperature_out(Q, UA,
                                  hot_temperature_in, cold_temperature_in, cold_temperature_out):
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
        hot_temperature_in: hot temperature in, C
        cold_temperature_in: cold fluid temperature in, C
        cold_temperature_out: cold fluid temperature out, C

        Output
        ------
        hot_temperature_out: hot fluid temperature out, C

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
    theta_1 = hot_temperature_in - cold_temperature_out

    # x used as an unknown variable for theta_2
    x = Symbol('x')

    solutions = solve((((theta_1)**2) * x) +
                      (theta_1 * x**2) - 2 * (Q / UA)**3, x)

    for i in solutions:
        if i > 0:
            theta_2 = i
        else:
            pass

    hot_temperature_out = cold_temperature_in + theta_2
    return hot_temperature_out


#################################
# Absorption Chiller Components #
#################################
# TODO
# - Make each component a subclass of a heat-exchanger class

# Absorption Chiller Components and Component Modeling


def circulation_ratio(concentration_SS=60, concentration_WS=57):
    r'''
        Calculates the circulation ratio of the LiBr

        Parameters
        ----------
        concentration_SS: mass concentration of the strong solution (%)
        concentration_WS: mass conceentration of the weak solution (%)

        Output
        ------
        circulation_ratio: ratio of the strong LiBr solution to the mass flowrate
            of the refrigerant

        Equation
        --------
        CR = X_WS / (X_SS - X_WS)
        CR: Circulation Ratio
        X_WS: concentration of weak solution
        X_SS: concentration of strong solution
    '''
    circulation_ratio = concentration_WS / \
        (concentration_SS - concentration_WS)
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
    '''

    def __init__(self, UA=5.287):
        self.UA = UA

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

    def strong_solution_enthalpy_out(self, concentration=60, temperature=80):
        r'''
            Calculates the enthalpy of strong LiBr solution out (8)

            Parameters
            ----------
            temperature: LiBr solution temperature out (C)

            Output
            ------
            enthalpy: enthalpy of LiBr solution (kJ / kg)
        '''
        strong_solution = LiBr_solution(concentration)
        return strong_solution.enthalpy_LiBr_solution(temperature)

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
        return calculate_cold_temperature_in(Q_g, self.UA, T_20, T_21, T_8)

    def generator_pressure(self, evaporation_temperature=80):
        return calculate_P_sat(evaporation_temperature)

    def generator_heat_from_flowrates(self, m_R, CR, h_1, h_8, h_7):
        H_1 = m_R * h_1
        H_8 = CR * m_R * h_8
        H_7 = (1 + CR) * m_R * h_7
        return H_1 + H_8 - H_7


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

    def __init__(self, UA=10.387):
        self.UA = UA

    def cooling_water_temp_out(
            self, Q, refrigerant_temp_in, refrigerant_temp_out, CW_temp_in):
        return calculate_cold_temperature_out(Q, self.UA,
                                              refrigerant_temp_in, refrigerant_temp_out,
                                              CW_temp_in)

    def cooling_water_flowrate(self, Q, CW_temp_in, CW_temp_out):
        return (Q / (specific_heat_water * (CW_temp_out - CW_temp_in)))


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

    def __init__(self, UA=12.566):
        self.UA = UA

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
        from sympy.solvers import solve
        from sympy import Symbol

        T_3 = calculate_cold_temperature_in(Q_e, self.UA, T_18, T_19, T_4)
        return T_3

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

    def __init__(self, UA=6.049):
        self.UA = UA

    def heat_from_internal_flows(self, m_R, circulation_ratio, h_4, h_10, h_5):
        H_4 = m_R * h_4
        H_10 = circulation_ratio * m_R * h_10
        H_5 = (1 + circulation_ratio) * m_R * h_5
        return (H_4 + H_10 - H_5)

    def heat_from_external_flows(self, m_CW, h_17, h_11):
        return m_CW * specific_heat_water * (h_17 - h_11)

    def cooling_water_temp_out(self, Q, SS_temp_in, SS_temp_out, CW_temp_in):
        return calculate_cold_temperature_out(Q, self.UA,
                                              SS_temp_in, SS_temp_out,
                                              CW_temp_in)

    def solution_temperature_out(self, Q, SS_temp_in, CW_temp_in, CW_temp_out):
        return calculate_hot_temperature_out(
            Q, self.UA, SS_temp_in, CW_temp_in, CW_temp_out)

    def cooling_water_flowrate(self, Q, CW_temp_in, CW_temp_out):
        return (Q / (specific_heat_water * (CW_temp_out - CW_temp_in)))


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

    def __init__(self, UA=2.009):
        self.UA = UA

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

    def strong_solution_temperature_out(
            self, Q, SS_temp_in, WS_temp_in, WS_temp_out):
        return calculate_cold_temperature_out(
            Q, self.UA, SS_temp_in, WS_temp_in, WS_temp_out)


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
    X_SS = 60
    X_WS = 57
    strong_solution = LiBr_solution(X_SS)
    weak_solution = LiBr_solution(X_WS)

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
                T_11 = Absorber_.cooling_water_temp_out(Q_a, T_10, T_5, T_17)
                print(F'T11 = {T_11}, T17 = {T_17}')
                h_11 = saturated_liquid_enthalpy(T_11)

                # 9.3.6.2 Solve for T_12 using Eq. 16
                T_12 = Condenser_.cooling_water_temp_out(Q_c, T_1, T_2, T_11)
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


absoprtion_chiller_equilibrium(Q_e_test, T_drybulb_test, P_test, RH_test)


'''CT = CoolingTower()
T_15 = CT.temperature_air_out(72.724, 97.8, 46.908, 30.6)
print(T_15)

print(humid_air_enthalpy(24.04, 97.8, 1))'''

'''strong_solution = LiBr_solution(60)
weak_solution = LiBr_solution(57)

h5 = weak_solution.enthalpy_LiBr_solution(38)
print(h5)'''

'''absorb = Absorber()
T11 = absorb.cooling_water_temp_out(62.59, 50.6, 38, 30)
print(T11)

cond = Condenser()
T12 = cond.cooling_water_temp_out(6.29, 80, 4.4, 37.48)
print(T12)'''

statepoints = [i for i in range(1, 22)]
temperature_dict = {}
enthalpy_dict = {}
pressure_dict = {}


'''temperature_list = list(temperature_dict.values())
enthalpy_list = list(enthalpy_dict.values())

# Convert the lists into a pandas dataframe
data = {'statepoint': statepoints, 'T_degC': temperature_list, 'enthalpy':enthalpy_list}
df = pd.DataFrame(data=data)
df.set_index('statepoint', inplace=True, drop=True)
print(df)'''
