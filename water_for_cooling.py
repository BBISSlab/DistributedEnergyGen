####################################
# LIBRARIES NEEDED TO RUN THE TOOL #
####################################
from logging import error
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

###########################
# PROPERTIES OF HUMID AIR #
###########################

# Unit Conversions


def convert_mbar_to_kPa(pressure):
    # 10 millibar = 1 kPa
    return pressure / 10


def convert_C_to_K(temperature):
    return temperature + 273.15

# Relative humidity calculations


def calculate_humidity_ratio(
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

    if (relative_humidity > 1) or (relative_humidity < 0):
        print('Enter value between 0 and 1')
        exit()

    P_sat = calculate_P_sat(temperature, 'C')

    humidity_ratio = (0.622 * relative_humidity * P_sat) / (P_atm - P_sat)
    return humidity_ratio


def mass_fraction_of_water_in_humid_air(humidity_ratio):
    return (1 / (1 + (1 / humidity_ratio)))

# Thermodynamic properties of humid air


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


def humid_air_enthalpy(temperature, pressure, relative_humidity):
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
    humidity_ratio = calculate_humidity_ratio(temperature=temperature,
                                              P_atm=pressure,
                                              relative_humidity=relative_humidity)

    # Calcualte the mass fraction of water in humid air
    W = mass_fraction_of_water_in_humid_air(humidity_ratio)

    # Convert pressure from kPa to MPA
    P_MPa = pressure / 1000
    # Convert temperature from C to K
    temp_K = convert_C_to_K(temperature)

    # Create Humid Air Class from iawps
    Humid_Air = humidAir.HumidAir(T=temp_K, P=P_MPa, W=W)

    return Humid_Air.h


#############################
# Lithium Bromide Equations #
#############################

def enthalpy_LiBr_solution(concentration=0., solution_temperature=0):
    r'''
        Chapter 30: Thermophysical Properties of Refrigerants. ASHRAE (2009)

        Enthalpy-Concentration diagram for water / LiBr Solutions

        Equation is valid for:
            concentration range 40 < X < 70% LiBr
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

    A = Li_Br_summation(coef=A_n, x=concentration)
    B = Li_Br_summation(coef=B_n, x=concentration)
    C = Li_Br_summation(coef=C_n, x=concentration)

    return(A + solution_temperature * B + solution_temperature**2 * C)


def Li_Br_summation(n=5, coef=[], x=0):
    summation = 0
    for i in range(n):
        summation += coef[i] * x**i
    return summation


def solution_temp(refrigerant_temp, A_n=[], B_n=[], concentration=0):
    A = Li_Br_summation(n=4, coef=A_n, x=concentration)
    B = Li_Br_summation(n=4, coef=B_n, x=concentration)

    return B + refrigerant_temp * A


def refrigerant_temp(solution_temp, A_n=[], B_n=[], concentration=0):
    A = Li_Br_summation(n=4, coef=A_n, x=concentration)
    B = Li_Br_summation(n=4, coef=B_n, x=concentration)

    return (solution_temp - B) / A


def pressure_sat(refreigerant_temp, C, D, E):
    T = refrigerant_temp + 273.15
    return 10**(C + D / T + E / (T**2))


def refrigerant_temp_from_pressure(pressure_sat, C, D, E):
    T = (-2 * E) / (D + (D**2 - 4 * E * (C - np.log10(pressure_sat)))**0.5)
    refrigerant_temp = T - 273.15
    if refrigerant_temp < 0:
        print("refrigerant temperature outside possible range")
    else:
        return refrigerant_temp


def LiBr_equilibrium(concentration=0,
                     solution_temp=0, refrigerant_temp=0,
                     pressure_sat=0,
                     method='refrigerant to solution'
                     ):
    r'''
        WHAT WAS THIS EQUATION FOR?
        Chapter 30: Thermophysical Properties of Refrigerants. ASHRAE (2009)

        Equilibrium Chart for Aqueous LiBr Solutions

        Equation is valid for:
            concentration range 40 < X < 70% LiBr
            solution temperature range 5 < t' < 165 deg C
            refrigerant temperature range -15 < t < 175 deg C

        In aqueous LiBr solutions, water acts as the refrigerant.

        t:  solution temperature, deg C
        t': refrigerant temperature, deg C
        T': refrigerant temperature, K
        P:  saturation pressure, kPa

        h = sum(A_n * X^n, 0, 4) + t * sum(B_n * X^n, 0, 4) + t^2 * sum(C_n * X^n, 0, 4)
    '''
    # Coefficients
    A_n = [-2.00755, 0.16976, -3.133362 * 10**-3, 1.97668 * 10**-5]
    B_n = [124.937, -7.71649, 0.152286, -7.9590 * 10**-4]
    C = 7.05
    D = -1596.49
    E = -104095.5

#################################
# Absorption Chiller Components #
#################################

# Absorption Chiller Components and Component Modeling



class Generator:

    def __init__(self,
                 heat_input=0,
                 hot_water_temp_in=0, hot_water_temp_out=0,
                 solution_temp_in=0, solution_temp_out=0,
                 refrigerant_temp_out=0,
                 generator_temp=0
                 ):
        '''
                                ================
        WEAK SOLUTION IN (1) ====> [              ] ====> STRONG SOLUTION OUT (2)
                                [  GENERATOR   ] ====> REFRIGERANT OUT (3)
            HOT WATER IN (4) ====> [              ] ====> HOT WATER OUT (5)
                                ================

        Assumptions
        '''
        import iapws.iapws97 as iapws
        # refrigerant_temp_out =

        if heat_input > 0:
            pass

        pass

    def refrigerant_temp_out(self):
        pass

    def heat_demand(self, var1):
        pass
    


class Condenser:

    def __init__(self, UA):
        self.UA = UA


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
    3. Heat exchange is known
    4. UA is constant and = 12.566 kW/K
    '''
    def __init__(self, UA=12.566):
        self.UA = UA

    def get_chiller_water_flow(Q_e, T_18=12, T_19=6):
        pass



class Absorber:

    def __init__(self, UA):
        self.UA = UA


class SolutionHeatExhanger:

    def __init__(self, UA):
        self.UA = UA

'''H = enthalpy_LiBr_solution(concentration=40., solution_temperature=60)
print(H)
exit()'''

#################################
# Water Thermodynamic Equations #
#################################


######################
# Workflow Algorithm #
######################
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

def LMTD_guess(theta_1, theta_2 ):
    
    pass

def absoprtion_chiller_equilibrium(Q_e = 0):

    # Evaporator

    # 1. Solve for T_3
    UA_e = 12.566 # kJ / kg
    Constant_e = Q_e / UA_e

    pass

def test_solver(theta_1, Q, UA):
    from sympy.solvers import solve
    from sympy import Symbol

    x = Symbol('x')

    solutions = solve((((theta_1)**2) * x) + (theta_1 * x**2) - 2 * (Q / UA)**3, x)

    for i in solutions:
        if i > 0:
            return i
        else:
            pass

value = test_solver(theta_1=2, Q=50, UA=12.566)
print(value)