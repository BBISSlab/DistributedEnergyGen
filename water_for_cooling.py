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

####################
# GLOBAL VARIABLES #
####################

specific_heat_water = 4.186  # kJ / (kg K)

###########################
# PROPERTIES OF HUMID AIR #
###########################
# Unit Conversions


def convert_mbar_to_kPa(pressure):
    # 10 millibar = 1 kPa
    return pressure / 10


def convert_C_to_K(temperature):
    return temperature + 273.15


def convert_K_to_C(temperature):
    return temperature - 273.15

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


# Generic thermodynamic property functions
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
    P_MPa = calculate_P_sat(temperature)
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
    P_MPa = calculate_P_sat(temperature)
    saturated_vapor = iapws97._Region4(temp_K, P_MPa)

    enthalpy = saturated_vapor['h']

    return enthalpy

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


#################################
# Absorption Chiller Components #
#################################

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
    # Calculate Q_G
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

    def heat_demand(self, var1):
        pass


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

    # Need


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


class CoolingTower:

    def __init__(self):
        pass

    def mixer_temp_out(self):

        pass

    def make_up_water_flowrate(self, cooling_water_mass_flowrate):

        pass

    def calculate_air_mass_flowrate(self, cooling_water_mass_flowrate):
        return 34.87 * cooling_water_mass_flowrate + 26062.54

    pass


#################################
# Water Thermodynamic Equations #
#################################


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
13. Calculate CR, m_R, temp of WS leaving SHE (T_7), Q_A, and cooling water temp leaving abs (T_11)
14. Determine new temp leaving absorber (T_5, step 12) until error is < 10^-2
15. Calculate Q_SHE and solve for WS temp leaving the SHE
16. Determine new temprature of SS leaving the SHE (T_9, step 10), until error is < 10^-2
17. Calculate Q_G
18. Solve for hot water temp leaving generator (T_21)
19. Determine new temperature leaving generator (T_8 and T_1, step 8), until error is < 10^-2
20. Calculate Q_C and solve for cooling water temp leaving cond (T_12)
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


def LMTD_guess(theta_1, theta_2):

    pass


def absoprtion_chiller_equilibrium(Q_e, T_air, RH, error_threshold=0.001):
    # Create objects with known UA values
    Evaporator_ = Evaporator()
    Absorber_ = Absorber()
    Generator_ = Generator()
    SHE_ = SolutionHeatExhanger()
    Condenser_ = Condenser()
    CoolingTower_ = CoolingTower()

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
    ####################################
    # Known Temperatures and Pressures #
    ####################################
    # Temperature
    T_1 = 80
    T_4 = 4
    T_8 = T_1
    T_16 = 30
    T_18 = 12
    T_19 = 6
    T_20 = 90.6
    T_21 = 85

    # Pressure
    P_g = Generator_.generator_pressure(T_1)
    P_c = P_g
    P_e = Evaporator_.evaporator_pressure(T_4)
    P_a = P_e

    # LiBr Solutions
    X_SS = 60
    X_WS = 57
    strong_solution = LiBr_solution(X_SS)
    weak_solution = LiBr_solution(X_WS)

    ############################
    # Calculate initial knowns #
    ############################
    r'''
        1. Calculate h_1
        2. Calculate h_4
        3. Calculate h_16, h_18, h_19, h_20, h_21
        4. Calculate h_8
    '''
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

    ##############
    # Evaporator #
    ##############
    r'''
        The Evaporator will be the only component with very
        explicit solutions that do not require iteration
        5.  Solve for T_3
        6.  Solve for h_3
        7.  Calculate the refrigerant mass flowrate (m_R)
        8.  Calculate the chilled water mass flowrate (m_ChW)
    '''
    T_3 = Evaporator_.refrigerant_temp_in(Q_e)
    h_3 = Evaporator_.refrigerant_enthalpy_in(T_3)
    m_R = Evaporator_.refrigerant_mass_flowrate(Q_e, h_3)
    m_ChW = Evaporator_.chilled_water_flowrate(Q_e)

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
    h_2 = h_3
    T_2 = convert_K_to_C(iapws97._Backward1_T_Ph(P_c / 1000, h_2))
    Q_c = m_R * (h_1 - h_2)
    #######################
    # Iterative Processes #
    #######################
    r'''
        Values that must be determined iteratively
        T_11
        T_10
        T_5
        T_17

    '''
    #######################
    # Assume value of Q_g #
    #######################
    Q_g = Generator_.guess_Q_g(Q_e)
    error_Q_g = 1
    while error_Q_g > error_threshold:
        # Calculate m_G
        m_G = Generator_.hot_water_mass_flowrate(Q_g, T_20, T_21)
        T_7 = Generator_.weak_solution_temperature_in(Q_g)
        
        ###############
        # Assume T_12 #
        ###############
        error_T_12 = 1
        while error_T_12 > error_threshold:
            T_12 = 35 # from Ren 2019
            
            '''
            You need to specify which equations you are using to solve each value
            in the iterative process.
            
            Calculate T_11
            Calculate m_CW
            Calculate m_air
            Calculate Point 14 and 15
            Calculate Point 17
            Calculate Q_a from m_CW, T_11 and T_17
                Assume T_10
                Calculate T_5
                Calculate T_6
                Calculate T_9
            '''
                
    r'''
        CONDENSER
        GENERATOR
        ABSORBER
        10. Calculate T_10
        11. Assume T_5
        12. Calculate h_5
        13.
    '''

    pass


###########
# TESTING #
###########
# Evap_ = Evaporator()
# t_3 = Evap_.refrigerant_temp_in(50)
# print(t_3)
# print(calculate_P_sat(4))

# Test Values
Q_e_test = 49.5
T_drybulb_test = 30.6
RH_test = 57

