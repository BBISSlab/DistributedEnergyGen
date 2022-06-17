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

# Thermofluids modules
from iapws import iapws97
from iapws import humidAir
from sympy import QQ_gmpy

from ypstruct import structure
##########################################################################

r'''
Symbols and Abbreviations
==================================================================

a       Duhring gradient
b       Duhring intercept
COP     coefficient of performance
c_p     isobaric heat capacity [kJ / (kg K)]
eff     efficiency
h       specific_enthalpy [kJ / kg]
H       specific_enthalpy [kJ]
K       Kelving [K]
LMTD    log-mean temperature difference [C]
m       mass flowrate [kg / s]
P       pressure [kPa]
Q       heat [kW]
T       temperature [C]
theta   temperature difference between two fluids [C]
UA      overall heat transfer coefficient - area product [kW / K]
x       mass fraction of LiBr in solution

Subscripts
===================================================================

A       absorber
atm     atmospheric
C       condenser
c       cold
con     condensation
cw      cooling water
chw     chilled water
dp      dew point
E       evaporator
eva     evaporation
G       generator
h       hot
hw      hot water
in/out  incoming / outgoing
r       refrigerant
SHX     solution heat exchanger
ss      strong solution (concentrated solution)
ws      weak solution (dilute solution)

'''


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


def absolute_total_error(old_value, new_value):
    difference = old_value - new_value
    return abs(difference)


def absolute_percent_relative_error(old_value, new_value):
    absolute_relative_error = abs(
        absolute_total_error(
            old_value,
            new_value) / new_value)
    return absolute_relative_error * 100


class Water:

    def __init__(self, 
        name=None,
        state=None,
        mass_flowrate=None,
        temperature=None,
        pressure=None,
        specific_enthalpy=None):
        r'''
            t:  solution temperature, deg C
            t': refrigerant temperature, deg C
            T': refrigerant temperature, K
            P:  saturation pressure, kPa
        '''
        
        self.name = name
        self.state = state
        self.mass_flowrate = mass_flowrate
        self.temperature = temperature
        self.pressure = pressure
        self.specific_enthalpy = specific_enthalpy

        # Constants
        self.isobaric_heat_capacity_L = 4.186  # kJ / (kg K)
        self.isobaric_heat_capacity_G = 1.86  # kJ / (kg K)

    def __repr__(self):
        attrs = ['name', 'state', 'mass_flowrate', 'temperature', 'pressure', 'specific_enthalpy']

        return ('Water:\n' + '\n'.join('       {}: {}'.format(attr, getattr(self, attr)) for attr in attrs))

    # Saturation State
    def heat_of_vaporization(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        else:
            temperature = temperature
        h_f = self.saturated_liquid_specific_enthalpy(temperature)
        h_g = self.saturated_vapor_specific_enthalpy(temperature)
        return h_g - h_f

    def Psat_T(self, temperature=None, units="C"):
        r'''
        Calculate the saturation pressure of water.

        Parameters
        ----------
        temperature: temperature of the air, in C or K

        Output
        ------
        P_sat_kPa: saturation pressure of water (kPa)
        '''
        if temperature is None:
            temperature = self.temperature

        if units == "K":
            P_sat_MPa = iapws97._PSat_T(temperature)
            P_sat_kPa = P_sat_MPa * 1000
            return P_sat_kPa
        elif units == "C":
            temp_K = convert_C_to_K(temperature)
            return self.Psat_T(temp_K, "K")
        else:
            print("Unit Error - enter temperature in C or K")
            exit()
 
    # Saturated Liquid
    def saturated_liquid_specific_enthalpy(self, temperature=None):
        r'''
        Calculates the specific_enthalpy of liquid water

        Parameters
        ----------
        temperature: liquid water temperature (C)

        Output
        ------
        specific_enthalpy: specific_enthalpy of saturated liquid water (kJ / kg)
        '''
        if temperature is None:
            temperature = self.temperature
        
        temp_K = convert_C_to_K(temperature)
        P_MPa = self.Psat_T(temperature) / 1000
        saturated_liquid = iapws97._Region1(temp_K, P_MPa)

        specific_enthalpy = saturated_liquid['h']

        return specific_enthalpy

    def _sat_liquid_obj_fcn(self, specific_enthalpy=None):
        if specific_enthalpy is None:
            specific_enthalpy = self.specific_enthalpy

        return self.saturated_liquid_specific_enthalpy(self.temperature) - specific_enthalpy

    def saturated_liquid_temperature(self, specific_enthalpy=None,
                                     percent_error_allowed=0.01):
        if specific_enthalpy is None:
            specific_enthalpy = self.specific_enthalpy
        
        # Initialize Values using upper and lower bounds for
        # the current system
        t_U = convert_C_to_K(97)
        t_L = convert_C_to_K(4)

        f_U = self._sat_liquid_obj_fcn(t_U, specific_enthalpy)
        f_L = self._sat_liquid_obj_fcn(t_L, specific_enthalpy)

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
            f_r = self._sat_liquid_obj_fcn(t_r, specific_enthalpy)

            test = f_L * f_r
            if test < 0:
                t_U = t_r
                f_U = self._sat_liquid_obj_fcn(t_U, specific_enthalpy)
                i_U = 0
                i_L += 1
                if i_L >= 2:
                    f_L = f_L / 2
            elif test > 0:
                t_L = t_r
                f_L = self._sat_liquid_obj_fcn(t_L, specific_enthalpy)
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

    def sat_liquid_temp_from_pressure(self, pressure=None):
        if pressure is None:
            pressure = self.pressure
        P = pressure / 1000
        T = iapws97._TSat_P(P)
        return convert_K_to_C(T)

    # Saturated Vapor
    def saturated_vapor_specific_enthalpy(self, temperature=None):
        r'''
            Calculates the specific_enthalpy of saturated water vapor

            Parameters
            ----------
            temperature: saturated water vapor temperature (C)

            Output
            ------
            specific_enthalpy: specific_enthalpy of saturated water vapor (kJ / kg)
        '''
        if temperature is None:
            temperature = self.temperature
        temp_K = convert_C_to_K(temperature)
        P_MPa = self.Psat_T(temperature) / 1000
        saturated_vapor = iapws97._Region2(temp_K, P_MPa)

        specific_enthalpy = saturated_vapor['h']

        return specific_enthalpy

    # Superheater State
    def superheated_steam_specific_enthalpy(self, temperature=None, pressure=None):
        if (temperature is None) and (pressure is None):
            temperature = self.temperature
            pressure = self.pressure
        
        T = convert_C_to_K(temperature)
        P = pressure / 1000
        superheated_steam = iapws97._Region2(T, P)

        return superheated_steam['h']

    def enthalpy(self):
        try:
            return self.mass_flowrate * self.specific_enthalpy
        except TypeError:
            if self.state == 'l':
                self.specific_enthalpy = self.saturated_liquid_specific_enthalpy(self.temperature)
            elif self.state == 'g':
                self.specific_enthalpy = self.saturated_vapor_specific_enthalpy(self.temperature)
            elif self.state == 'shg':
                self.specific_enthalpy = self.superheated_steam_specific_enthalpy(self.temperature, self.pressure)
            else:
                raise TypeError('specify the specific enthalpy or phase state of water')
            return self.mass_flowrate * self.specific_enthalpy

class Air:
    
    def __init__(self, temperature=20, pressure=101.325, state=None,
                 specific_enthalpy=None):
        r'''
            t:  solution temperature, deg C
            t': refrigerant temperature, deg C
            T': refrigerant temperature, K
            P:  saturation pressure, kPa
        '''
        self.temperature = temperature
        self.pressure = pressure
        self.state = state
        self.specific_enthalpy = specific_enthalpy

    # Relative humidity calculations
    def humidity_ratio(self,
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
            P_sat = Water().Psat_T(temperature, 'C')
            humidity_ratio = (0.622 * relative_humidity * P_sat) / (P_atm - P_sat)
            return humidity_ratio


    def mass_fraction_of_water_in_humid_air(self, humidity_ratio):
        return (1 / (1 + (1 / humidity_ratio)))

    # Thermodynamic properties of humid air
    def humid_air_specific_enthalpy(self, temperature, pressure, relative_humidity,
                        method='iawps'):
        r'''
        Calculate the specific_enthalpy (kJ/kg) of the humid air.

        Parameters
        ----------
        temperature: temperature of the air (C)
        pressure: pressure of the air (kPa)
        relative_humidity: relative_humidity of the air, as a fraction [0 - 1]

        Output
        ------
        specific_enthalpy: specific_enthalpy (kJ/kg) of humid air

        '''
        # Calculate the  humidity ratio
        HR = self.humidity_ratio(temperature=temperature,
                            P_atm=pressure,
                            relative_humidity=relative_humidity)

        if method == 'iawps':
            # Calcualte the mass fraction of water in humid air
            W = self.mass_fraction_of_water_in_humid_air(HR)

            # Convert pressure from kPa to MPA
            P_MPa = pressure / 1000
            # Convert temperature from C to K
            temp_K = convert_C_to_K(temperature)

            # Create Humid Air Class from iawps
            Humid_Air = humidAir.HumidAir(T=temp_K, P=P_MPa, W=W)

            return Humid_Air.h
        elif method == 'cengel':
            specific_enthalpy_dry_air = 1.005 * temperature
            specific_enthalpy_water_vapor = saturated_vapor_specific_enthalpy(temperature)
            return specific_enthalpy_dry_air + HR * specific_enthalpy_water_vapor

        else:
            print('Choose iawps or cengel')


class LiBr_solution:
    
    def __init__(self, name=None, 
                mass_fraction=0.5,
                mass_flowrate=None,
                temperature=None,
                pressure=None,
                specific_enthalpy=None,
                mass_fraction_min=0.4, mass_fraction_max=0.65,
                temp_min=15, temp_max=165,
                Duhring_coefficients={'a0': 0.538, 'a1': 0.845, 'b0': 48.3, 'b1': -35.6}):
        r'''
            Chapter 30: Thermophysical Properties of Refrigerants. ASHRAE (2009)

            specific_Enthalpy-Concentration diagram for water / LiBr Solutions

            Equation is valid for:
                mass fraction range 0.40 < X < 0.70 LiBr
                temperature range 15 < t < 165 deg C

            t:  solution temperature, deg C
            t': refrigerant temperature, deg C
            T': refrigerant temperature, K
            P:  saturation pressure, kPa
        '''
        self.name = name
        self.mass_fraction = mass_fraction
        self.mass_flowrate = mass_flowrate
        self.temperature = temperature
        self.pressure = pressure
        self.specific_enthalpy = specific_enthalpy
        self.mass_fraction_min = mass_fraction_min
        self.mass_fraction_max = mass_fraction_max
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.Duhring_coefficients = Duhring_coefficients

    def __repr__(self):
        attrs = ['name', 'mass_fraction', 'mass_flowrate', 'temperature', 'pressure', 'specific_enthalpy']
        return ('LiBr solution:\n' + '\n'.join('       {}: {}'.format(attr, getattr(self, attr)) for attr in attrs))

    def specific_enthalpy_LiBr_solution(self, solution_temperature):
        r'''
            Chapter 30: Thermophysical Properties of Refrigerants. ASHRAE (2009)

            specific_Enthalpy-Concentration diagram for water / LiBr Solutions

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

    def calc_temp_from_specific_enthalpy(self, specific_enthalpy):
        r'''
            Chapter 30: Thermophysical Properties of Refrigerants. ASHRAE (2009)

            specific_Enthalpy-Concentration diagram for water / LiBr Solutions

            Equation is valid for:
                concentration range 40 < X < 70 percent LiBr
                temperature range 15 < t < 165 deg C

            Parameters
            ----------
            h: specific_enthalpy of LiBr solution (kJ / kg)

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

        solutions = solve((A - specific_enthalpy) + (B * t) + (C * t**2), t)

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
            self, solvent_dew_point_temp):
        
        temperature_solution = self.temperature
        
        a0 = self.Duhring_coefficients['a0']
        a1 = self.Duhring_coefficients['a1']
        b0 = self.Duhring_coefficients['b0']
        b1 = self.Duhring_coefficients['b1']
        ts_K = convert_C_to_K(solvent_dew_point_temp)
        t_K = convert_C_to_K(temperature_solution)

        mass_fraction = (t_K - (a1 * ts_K + b1)) / (a0 * ts_K + b0)

        if mass_fraction > self.mass_fraction_max:
            mass_fraction = self.mass_fraction_max
        elif mass_fraction < self.mass_fraction_min:
            mass_fraction = self.mass_fraction_min 

        return mass_fraction

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

    def enthalpy(self):
        try:
            return self.mass_flowrate * self.specific_enthalpy
        except TypeError:
            self.specific_enthalpy = self.specific_enthalpy_LiBr_solution(self.temperature)
            return self.mass_flowrate * self.specific_enthalpy

class HeatExhanger:

    def __init__(self, name='Heat exchanger', 
        UA=0, effectiveness=None,
        m_h=0, m_c=0,
        Th_in=0, Th_out=0,
        Th_min=0, Th_max=0, 
        Tc_in=0, Tc_out=0,
        Tc_min=0, Tc_max=0,
        Q=0,
        hx_type='counter-flow'):

        r'''                        ===============
              HOT FLUID IN () ====> [  GENERIC   ] ====> HOT FLUID OUT (T_H2)
                                    [  HEAT      ]
        COLD FLUID OUT (T_C2) <==== [  EXCHANGER ] <==== COLD FLUID IN (T_C1)
                                    ==============

        Parameters
        ----------
        UA: the overall heat-transfer coefficient - area product (kJ / K)
        Q: the heat transfered through the heat exchanger (kJ)
        Th_in: hot temperature in (C)
        Th_out: hot fluid temperature out (C)
        Tc_in: cold fluid temperature in (C)
        effectiveness: Heat exchanger effectiveness (unitless)
        cp_h: isobaric heat capacity of the hot fluid
        cp_c: isobaric heat capacity of the cold fluid
        '''
        self.name = name
        
        # Heat Transfer Coefficients
        self.UA=UA
        self.effectiveness=effectiveness

        # Hot Fluid
        self.Th_in=Th_in
        self.Th_out=Th_out
        self.Th_min=Th_min
        self.Th_max=Th_max

        # Cold Fluid
        self.Tc_in=Tc_in
        self.Tc_out=Tc_out
        self.Tc_min=Tc_min
        self.Tc_max=Tc_max

        self.Q = Q

    # Methods for determining temperatures at inlet or outlet
    def Tc_out_Ebalance(self, Q, mass_flowrate, cp_c, Tc_in):
        return (Q / (mass_flowrate * cp_c) + Tc_in)
    
    def Tc_in_LMTD(self, 
        Q,
        Th_in, Th_out,
        Tc_out):
        
        # WRONG VALUE
        r'''
        Calculates the cold fluid temperature at inlet (Tc_in) using the LMTD method

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
        1. Chen's approximation for LMTD [1]
        2. Counter-flow heat exchanger

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
                        (theta_1 * x**2) - 2 * (Q / self.UA)**3, x)


        for i in solutions:
            if isinstance(i, tuple):
                i = i[0]

            if i > 0:
                theta_2 = i
                Tc_in = Th_out - theta_2
                return Tc_in
            else:
                pass

    def Tc_out_LMTD(self,
        Q,
        Th_in, Th_out,
        Tc_in):
        r'''
        Calculates the fluid temperature at outlet (Tc_out)

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
                        (theta_2 * x**2) - 2 * (Q / self.UA)**3, x)

        for i in solutions:
            
            if isinstance(i, tuple):
                i = i[0]
            # theta_1 = i
            # print(Th_in - theta_1)
            if i > 0.0:
                theta_1 = i
            else:
                pass

        Tc_out = Th_in - theta_1
        return Tc_out

    def Th_out_Ebalance(self, Q, mass_flowrate, cp_c, Th_in):
        return Th_in - (Q / (mass_flowrate * cp_c))

    def Th_out_LMTD(self, Q, Th_in, Tc_in, Tc_out):
        r'''
        Calculates the refrigerant temperature at inlet (Th_in)
        
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
        1. Chen's approximation for LMTD [1]
        2. Counter-flow heat exchanger

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
                        (theta_1 * x**2) - 2 * (Q / self.UA)**3, x)

        for i in solutions:
            if i > 0:
                theta_2 = i
            else:
                pass

        Th_out = Tc_in + theta_2
        return Th_out

    def LMTD_counterflow(self, Th_in, Th_out, Tc_in, Tc_out, method="LMTD"):
        if method == 'LMTD':
            theta1 = Th_in - Tc_out
            theta2 = Th_out - Tc_in

            return (theta1 - theta2) / m.log(theta1/theta2)

        if method == 'Chen':
            theta1 = Th_in - Tc_out
            theta2 = Th_out - Tc_in

            return (theta1 * theta2 * (theta1 + theta2)/2) ** (1/3)

    def Q_hot_fluid_flow(self):
        pass

    def Q_cold_fluid_flow(self):
        pass

    def Q_LMTD(self):
        pass

# Create HeatExchanger Subclasses for

#########################################################
# Thermal Compressor: Absorber and Generator (Desorber) #
#########################################################
# Absorber 
class Absorber(HeatExhanger):
    r'''
                                    ===============
    STRONG SOLUTION IN (h_in) ====> [             ] ====> WEAK SOLUTION OUT (h_out)
           REFRIGERANT IN (r) ====> [  ABSORBER   ]
      COOLING WATER IN (c_in) ====> [             ] ====> COOLING WATER OUT (c_out)
                                    ===============

    Assumptions
    -----------
    1. Cooling water temperature in = 30 C
    2. Refrigerant in is a saturated vapor
    3. UA is constant = 6.049 kW/K
    '''
    
    def __init__(self, name='Absorber', 
        UA=6.049, effectiveness=None,
        Refrigerant=None,
        CoolingWater_in=None,
        CoolingWater_out=None,
        StrongSolution=None,
        WeakSolution=None,
        Q=None, 
        hx_type='counter-flow'):

        super().__init__(name=name, 
            UA=UA, effectiveness=effectiveness, Q=Q)

        self.name = name
        # Shell-side fluids
        self.Refrigerant = Refrigerant
        self.StrongSolution = StrongSolution
        self.WeakSolution = WeakSolution
        # Tube-side fluids
        self.CoolingWater_in = CoolingWater_in
        self.CoolingWater_out = CoolingWater_out

        self.hx_type = hx_type

    def __repr__(self):
        attrs = ['name', 'Q', 'UA', 'effectiveness', 
            'Refrigerant', 'StrongSolution', 'WeakSolution',
            'CoolingWater_in', 'CoolingWater_out']

        return ('Absorber:\n' + '\n'.join('     {}: {}'.format(attr, getattr(self, attr)) for attr in attrs))

    def Q_from_specific_enthalpy(self):
        H_r = self.Refrigerant.enthalpy()
        H_ss =  self.StrongSolution.enthalpy()
        H_ws = self.WeakSolution.enthalpy()

        return (H_r + H_ss - H_ws)

    def _mean_isostere_slope(self, T_dp):
        r'''
        Temperatures must be in Kelvin
        '''
        a0 = self.WeakSolution.Duhring_coefficients['a0']
        a1 = self.WeakSolution.Duhring_coefficients['a1']
        
        x_ws = self.WeakSolution.mass_fraction
        x_ss = self.StrongSolution.mass_fraction
        Tws_out = self.WeakSolution.temperature

        Tss_eq = self.StrongSolution.Duhring_equilibrium_temperature(T_dp)

        tss_eq = convert_C_to_K(Tss_eq)
        tws_out = convert_C_to_K(Tws_out)
        t_dp = convert_C_to_K(T_dp)

        first_term = ((tss_eq + tws_out) / (2 * t_dp))**2
        second_term = 1 / (((a0 * (x_ws + x_ss)) / 2) + a1)

        return first_term * second_term

    def Q_Ren(self, T_dp):
        m_r = self.Refrigerant.mass_flowrate
        m_ws = self.WeakSolution.mass_flowrate
        m_ss = self.StrongSolution.mass_flowrate

        Tr_in = self.Refrigerant.temperature
        Tss_in = self.StrongSolution.temperature
        Tws_out = self.WeakSolution.temperature


        T_x = (Tss_in + Tws_out) / 2
        Tss_eq = self.StrongSolution.Duhring_equilibrium_temperature(T_dp)

        cp_ws = self.WeakSolution.isobaric_heat_capacity(T_x, method="Ren")
        cp_ss = self.StrongSolution.isobaric_heat_capacity(T_x, method="Ren")
        cp_r = self.Refrigerant.isobaric_heat_capacity_G

        h_con = self.Refrigerant.heat_of_vaporization()

        Xbar_g = self._mean_isostere_slope(T_dp=T_dp)

        Q_ws = m_ws * cp_ws * (Tss_in - Tws_out)

        Q_ref = m_r * (Xbar_g * h_con + cp_r *
                         (Tr_in - Tss_in) + cp_ss * (Tss_in - Tss_eq))

        return Q_ws + Q_ref

    # Cooling Water Properties
    def CW_temperature_out(self, method='LMTD'):
        CW = self.CoolingWater_in
        # One using energy balance
        if method == 'Ebalance':
            CW = self.CoolingWater_in
            return (self.Tc_out_Ebalance(self.Q, CW.mass_flowrate, 
                    CW.isobaric_heat_capacity_L, CW.temperature))
            
        # Method 2 using LMTD
        if method == 'LMTD':
            WS = self.WeakSolution
            SS = self.StrongSolution
            return self.Tc_out_LMTD(self.Q, SS.temperature, WS.temperature, CW.temperature)

    def CW_mass_flowrate(self):
        Q = self.Q
        H_in = self.CoolingWater_in.specific_enthalpy
        H_out = self.CoolingWater_out.specific_enthalpy
        return Q / (H_out - H_in)
    
    # Weak Solution 
    def WS_temperature_out(self, method ='LMTD'):
        SS = self.StrongSolution
        CW_i = self.CoolingWater_in
        CW_o = self.CoolingWater_out
        return self.Th_out_LMTD(self.Q, SS.temperature, CW_i.temperature, CW_o.temperature)    

# Generator
class Generator(HeatExhanger):
    r'''
                                       ================
        WEAK SOLUTION IN (ws_in) ====> [              ] ====> STRONG SOLUTION OUT (ss_out)
                                       [  GENERATOR   ] ====> REFRIGERANT OUT (r)
            HOT WATER IN (hw_in) ====> [              ] ====> HOT WATER OUT (hw_out)
                                       ===============

    Assumptions
    -----------
    1. Hot water temperature in = 90.6 C
    2. Refrigerant out is a superheated vapor
    3. UA is constant = 5.287 kW/K
    '''
    
    def __init__(self, name='Generator', 
        UA=5.287, effectiveness=None,
        Refrigerant=None,
        StrongSolution=None,
        WeakSolution=None,
        HotWater_in=None,
        HotWater_out=None,
        Q=None, 
        hx_type='counter-flow'):

        super().__init__(name=name, 
            UA=UA, effectiveness=effectiveness, Q=Q)

        self.name = name
        # Shell-side fluids
        self.Refrigerant = Refrigerant
        self.StrongSolution = StrongSolution
        self.WeakSolution = WeakSolution
        # Tube-side fluids
        self.HotWater_in = HotWater_in
        self.HotWater_out = HotWater_out

        self.hx_type = hx_type

    def __repr__(self):
        attrs = ['name', 'Q', 'UA', 'effectiveness', 
            'Refrigerant', 'StrongSolution', 'WeakSolution',
            'HotWater_in', 'HotWater_out']

        return ('Generator:\n' + '\n'.join('     {}: {}'.format(attr, getattr(self, attr)) for attr in attrs))
    

    # Hot Water Properties
    def HW_temperature_out(self, method='LMTD'):
        HW = self.HotWater_in
        # One using energy balance
        if method == 'Ebalance':
            return (self.Th_out_Ebalance(self.Q, HW.mass_flowrate, 
                    HW.isobaric_heat_capacity_L, HW.temperature))
            
        # Method 2 using LMTD
        if method == 'LMTD':
            WS = self.WeakSolution
            SS = self.StrongSolution
            return self.Th_out_LMTD(self.Q, HW.temperature, WS.temperature, SS.temperature)
    
    def HW_mass_flowrate(self):
        Thw_in = self.HotWater_in.temperature
        Thw_out = self.HotWater_out.temperature
        Q = self.Q
        cp_l = self.HotWater_in.isobaric_heat_capacity_L

        return (Q/(cp_l * (Thw_in - Thw_out)))

    def SS_temperature_out(self):
        Q = self.Q
        Th_in = self.HotWater_in.temperature
        Th_out = self.HotWater_out.temperature
        Tc_in = self.WeakSolution.temperature
        return self.Tc_out_LMTD(Q, Th_in, Th_out, Tc_in)

    def WS_temperature_in(self):
        Q = self.Q
        Th_in = self.HotWater_in.temperature
        Th_out = self.HotWater_out.temperature
        Tc_out = self.StrongSolution.temperature
        return self.Tc_in_LMTD(Q, Th_in, Th_out, Tc_out)

    # Heat Output
    def Q_from_specific_enthalpy(self):
        H_r = self.Refrigerant.enthalpy()
        H_ss =  self.StrongSolution.enthalpy()
        H_ws = self.WeakSolution.enthalpy()

        return (H_r + H_ss - H_ws)

    def _mean_isostere_slope(
            self, T_dp):
        r'''
        Temperatures must be in Kelvin
        '''
        a0 = self.WeakSolution.Duhring_coefficients['a0']
        a1 = self.WeakSolution.Duhring_coefficients['a1']
        
        x_ws = self.WeakSolution.mass_fraction
        x_ss = self.StrongSolution.mass_fraction
        Tss_out = self.StrongSolution.temperature

        Tws_eq = self.WeakSolution.Duhring_equilibrium_temperature(T_dp)

        tws_eq = convert_C_to_K(Tws_eq)
        tss_out = convert_C_to_K(Tss_out)
        t_dp = convert_C_to_K(T_dp)

        first_term = ((tws_eq + tss_out) / (2 * t_dp))**2
        second_term = (((a0 * (x_ws + x_ss)) / 2) + a1)**-1

        return first_term * second_term       

    def Q_Ren(self, T_dp):
        # T_dp = condenser temperature
        m_r = self.Refrigerant.mass_flowrate
        Tr = self.Refrigerant.temperature

        m_ws = self.WeakSolution.mass_flowrate
        Tws_in = self.WeakSolution.temperature
        Tss_out = self.StrongSolution.temperature

        T_avg = (Tss_out + Tws_in) / 2
        cp_ws = self.WeakSolution.isobaric_heat_capacity(T_avg, method="Ren")
        
        h_con = Water().heat_of_vaporization(T_dp)
        cp_v = Water().isobaric_heat_capacity_G

        Xbar_g = self._mean_isostere_slope(T_dp=T_dp)

        Q_ws = m_ws * cp_ws * (Tss_out - Tws_in)

        Q_ref = m_r * (Xbar_g * h_con +
                         cp_v * (Tr - Tss_out))

        return Q_ws + Q_ref


# SolutionHeatExchanger
class SHX(HeatExhanger):
    r'''
                                           ================
            WEAK SOLUTION IN (ws_in) ====> [  SOLUTION    ] ====> WEAK SOLUTION OUT (ws_out)
                                           [  HEAT        ] 
        STRONG SOLUTION OUT (ss_out) <==== [  EXCHANGER   ] <==== STRONG SOLUTION IN (ss_in)
                                            ===============

    Assumptions
    -----------
    1. UA is constant = 2.009 kW/K
    '''
    
    def __init__(self, name='Solution Heat Exchanger', 
        UA=2.009, effectiveness=None,
        StrongSolution_in=None,
        StrongSolution_out=None,
        WeakSolution_in=None,
        WeakSolution_out=None,
        Q=None, 
        hx_type='counter-flow'):

        super().__init__(name=name, 
            UA=UA, effectiveness=effectiveness, Q=Q)

        self.name = name
        # StrongSolution
        self.StrongSolution_in = StrongSolution_in
        self.StrongSolution_out = StrongSolution_out
        # Tube-side fluids
        self.WeakSolution_in = WeakSolution_in
        self.WeakSolution_out = WeakSolution_out

        self.hx_type = hx_type

    def __repr__(self):
        attrs = ['name', 'Q', 'UA', 'effectiveness', 
            'StrongSolution_in', 'StrongSolution_out',
            'WeakSolution_in', 'WeakSolution_out']

        return ('Generator:\n' + '\n'.join('     {}: {}'.format(attr, getattr(self, attr)) for attr in attrs))
    
    def SS_temp_out(self, method='LMTD'):
        Tss_in = self.StrongSolution_in.temperature
        if method == 'LMTD':
            Tss_in = self.StrongSolution_in.temperature
            Tws_in = self.WeakSolution_in.temperature
            Tws_out = self.WeakSolution_out.temperature
            return self.Th_out_LMTD(self.Q, Tss_in, Tws_in, Tws_out)
        elif method == 'Ebalance':
            cp_ss = self.StrongSolution_in.isobaric_heat_capacity(Tss_in)
            m_ss = self.StrongSolution_in.mass_flowrate
            return Tss_in - (self.Q / (m_ss * cp_ss))

    def WS_temp_out(self, method='LMTD'):
        Tws_in = self.WeakSolution_in.temperature
        if method == 'LMTD':
            Tss_in = self.StrongSolution_in.temperature
            Tss_out = self.StrongSolution_out.temperature            
            return self.Tc_out_LMTD(self.Q, Tss_in, Tss_out, Tws_in)
        elif method == 'Ebalance':
            cp_ws = self.WeakSolution_in.isobaric_heat_capacity(Tws_in)
            m_ws = self.WeakSolution_in.mass_flowrate
            return Tws_in + (self.Q / (m_ws * cp_ws))

    # Heat Transfer
    def Q_WS_side(self):
        H_in = self.WeakSolution_in.enthalpy()
        H_out = self.WeakSolution_out.enthalpy()
        return H_out - H_in

    def Q_SS_side(self):
        H_in = self.StrongSolution_in.enthalpy()
        H_out = self.StrongSolution_out.enthalpy()
        return H_in - H_out

    def Q_Ren(self):
        T_in = self.StrongSolution_in.temperature
        T_out = self.StrongSolution_out.temperature
        Tss_avg = (T_in + T_out) / 2
        cp_ss = self.StrongSolution_in.isobaric_heat_capacity(Tss_avg)
        m_ss = self.StrongSolution_in.mass_flowrate
        return (m_ss * cp_ss * (T_in - T_out))

    
# Condenser
class Condenser(HeatExhanger):
    r'''
                                         ================
             REFRIGERANT IN (r_in) ====> [              ] ====> REFRIGERANT OUT (r_out)
                                         [  CONDENSER   ] 
        COOLING WATER OUT (cw_out) <==== [              ] <==== COOLING WATER IN (cw_in)
                                         ================

    Assumptions
    -----------
    1. UA is constant = 2.469 kW/K

    -   Ren et al (2019) uses a UA value of 10.387 kW/K; however, this value does not match
        the equilibrium temperatures and heat transfer values.
    -   We calculated the UA using Q_c = 52.89 kW and LMTD_c = 21.42 C
    '''
    
    def __init__(self, name='Condenser', 
        UA=2.469, effectiveness=None,
        Refrigerant_in=None,
        Refrigerant_out=None,
        CoolingWater_in=None,
        CoolingWater_out=None,
        Q=None,
        hx_type='counter-flow'):

        super().__init__(name=name, 
            UA=UA, effectiveness=effectiveness, 
            Q=Q, hx_type=hx_type)

        self.Refrigerant_in = Refrigerant_in
        self.Refrigerant_out = Refrigerant_out
        self.CoolingWater_in = CoolingWater_in
        self.CoolingWater_out = CoolingWater_out

    def __repr__(self):
        attrs = ['name', 'Q', 'UA', 'effectiveness', 
            'Refrigerant_in', 'Refrigerant_out',
            'CoolingWater_in', 'CoolingWater_out']

        return ('Condenser:\n' + '\n'.join('     {}: {}'.format(attr, getattr(self, attr)) for attr in attrs))

    def CW_temperature_out(self, method="LMTD"):
        CW = self.CoolingWater_in
        # One using energy balance
        if method == 'Ebalance':
            return (self.Tc_out_Ebalance(self.Q, CW.mass_flowrate, 
                    CW.isobaric_heat_capacity_L, CW.temperature))
        
        # Method 2 using LMTD
        if method == 'LMTD':
            Ref_in = self.Refrigerant_in
            Ref_out = self.Refrigerant_out
            return self.Tc_out_LMTD(self.Q, Ref_in.temperature, Ref_out.temperature, CW.temperature)

    def CW_mass_flowrate(self):
        Q = self.Q
        h_in = self.CoolingWater_in.specific_enthalpy
        h_out = self.CoolingWater_out.specific_enthalpy
        return Q / (h_out - h_in)

    def R_temperature_out(self, method='LMTD'):
        
        if method == 'LMTD':
            Ref_in = self.Refrigerant_in
            CW_i = self.CoolingWater_in
            CW_o = self.CoolingWater_out
            return self.Th_out_LMTD(self.Q, Ref_in.temperature, CW_i.temperature, CW_o.temperature)    

    def Q_from_specific_enthalpy(self, side='shellside'):
        if side == 'tubeside':
            H_in = self.CoolingWater_in.enthalpy()
            H_out = self.CoolingWater_out.enthalpy()
        if side == 'shellside':
            H_in = self.Refrigerant_in.enthalpy()
            H_out =  self.Refrigerant_out.enthalpy()
        return abs(H_in - H_out)

    def Q_Ren(self):
        pass

    def LMTD(self):
        return self.LMTD_counterflow(self.Refrigerant_in.temperature,
                                self.Refrigerant_out.temperature,
                                self.CoolingWater_in.temperature,
                                self.CoolingWater_out.temperature)

# Evaporator
class Evaporator(HeatExhanger):
    r'''
                                          ================
              REFRIGERANT IN (r_in) ====> [              ] ====> REFRIGERANT OUT (r_out)
                                          [  EVAPORATOR  ] 
        CHILLED WATER OUT (chw_out) <==== [              ] <==== CHILLED WATER IN (chw_in)
                                          ================

    Assumptions
    -----------
    1. UA is constant = 12.566 kW/K

    '''
    
    def __init__(self, name='Evaporator', 
        UA=12.566, effectiveness=None,
        Refrigerant_in=None,
        Refrigerant_out=None,
        ChilledWater_in=None,
        ChilledWater_out=None,
        Q=None,
        hx_type='counter-flow'):

        super().__init__(name=name, 
            UA=UA, effectiveness=effectiveness, 
            Q=Q, hx_type=hx_type)

        self.Refrigerant_in = Refrigerant_in
        self.Refrigerant_out = Refrigerant_out
        self.ChilledWater_in = ChilledWater_in
        self.ChilledWater_out = ChilledWater_out

    def __repr__(self):
        attrs = ['name', 'Q', 'UA', 'effectiveness', 
            'Refrigerant_in', 'Refrigerant_out',
            'ChilledWater_in', 'ChilledWater_out']

        return ('Evaporator:\n' + '\n'.join('     {}: {}'.format(attr, getattr(self, attr)) for attr in attrs))

    # Chilled Water Methods
    def ChW_mass_flowrate(self):
        Q = self.Q
        ChW_in = self.ChilledWater_in
        ChW_out = self.ChilledWater_out

        try:
            h_in = ChW_in.specific_enthalpy
            h_out = ChW_out.specific_enthalpy
        except TypeError:
            h_in = ChW_in.saturated_liquid_specific_enthalpy(ChW_in.temperature)
            h_in = ChW_out.saturated_liquid_specific_enthalpy(ChW_out.temperature)
        return Q / (h_in - h_in)

    def ChW_temperature_out(self, method='LMTD'):
        ChW = self.ChilledWater_out
        # One using energy balance
        if method == 'Ebalance':
            return (self.Tc_out_Ebalance(self.Q, ChW.mass_flowrate, 
                    ChW.isobaric_heat_capacity_L, ChW.temperature))
        
        # Method 2 using LMTD
        if method == 'LMTD':
            Ref_in = self.Refrigerant_in
            Ref_out = self.Refrigerant_out
            return self.Tc_out_LMTD(self.Q, Ref_in.temperature, Ref_out.temperature, ChW.temperature)

    # Refrigerant_Methods
    def R_mass_flowrate(self, method='enthalpy'):
        Ref_in = self.Refrigerant_in
        Ref_out = self.Refrigerant_out

        if method == 'enthalpy':
            h_in = Ref_in.specific_enthalpy
            h_out = Ref_out.specific_enthalpy
            return self.Q / (h_out - h_in)

    def R_temperature_in(self, method='LMTD'):
        
        if method == 'LMTD':
            Ref_out = self.Refrigerant_out
            CW_i = self.CoolingWater_in
            CW_o = self.CoolingWater_out
            return self.Tc_in_LMTD(self.Q, CW_i.temperature, CW_o.temperature, Ref_out.temperature)

    def R_enthalpy_in(self):
        Ref_out = self.Refrigerant_out
        m_r = Ref_out.mass_flowrate
        h_out = Ref_out.specific_enthalpy
        Q = self.Q
        return h_out - (Q / m_r)


        pass

    # Heat Transfer Methods
    def Q_from_specific_enthalpy(self, side='shellside'):
        if side == 'tubeside':
            H_in = self.ChilledWater_in.enthalpy()
            H_out = self.ChilledWater_out.enthalpy()
        if side == 'shellside':
            H_in = self.Refrigerant_in.enthalpy()
            H_out =  self.Refrigerant_out.enthalpy()
        return abs(H_in - H_out)
    
    def Q_Ren(self):
        Ref_in = self.Refrigerant_in
        Ref_out = self.Refrigerant_out
        
        # Physical Properties
        m_r = Ref_in.mass_flowrate
        T_in = Ref_in.temperature
        T_out = Ref_out.temperature

        # Thermodynamic Properties
        h_fg = Ref_out.heat_of_vaporization(Ref_out.temperature)
        cp_l = Ref_out.isobaric_heat_capacity_L

        return (m_r * (h_fg + cp_l * (T_out - T_in)))


#################
# Cooling Tower #
#################
# Make a function that calculates design NTU
# The summation of points between the range

class CoolingTower:
    pass

def design_NTU(cold_water_temperature, hot_water_temperature, 
                wet_bulb_temperature, LG_ratio):
    cooling_range = hot_water_temperature - cold_water_temperature

    bulk_water_enthalpy = []
    wet_bulb_enthalpy = []

    enthalpy_diff = []
    NTU_n = []

    h_wb =  Air().humid_air_specific_enthalpy(temperature = wet_bulb_temperature, pressure=101.325, relative_humidity=1)

    N = [0.1, 0.4, 0.6, 0.9]

    for n in N:
        tw_n = cold_water_temperature + n * cooling_range
        hw_n = Water().saturated_liquid_specific_enthalpy(temperature = tw_n)

        bulk_water_enthalpy.append(hw_n)

        ha_n = h_wb + n * LG_ratio * cooling_range
        wet_bulb_enthalpy.append(ha_n)
        
        h_diff = hw_n - ha_n
        enthalpy_diff.append(h_diff)

        NTU_n.append(1 / h_diff)


    print(enthalpy_diff)
    print(NTU_n)

    NTU = (sum(NTU_n)/4 * cooling_range)

    print(NTU)


design_NTU(39.67, 40, 26, 1.6492)



# Testing
def testing_absorber():
    p10 = Water('p10', 'g', 0.02125, 6.)
    p10.specific_enthalpy = p10.superheated_steam_specific_enthalpy(p10.temperature, 1)

    p6 = LiBr_solution('p6', 0.58335, 0.47914, 50.6)
    p6.specific_enthalpy = p6.specific_enthalpy_LiBr_solution(p6.temperature)

    p1 = LiBr_solution('p7', 0.55858, 0.50039, 38)
    p1.specific_enthalpy = p1.specific_enthalpy_LiBr_solution(p1.temperature)

    p13 = Water('p13', 'l', 3.29257, 30.)
    p14 = Water('p14', 'l', 3.29257, 35.)

    for i in [p13, p14]:
        i.specific_enthalpy = i.saturated_liquid_specific_enthalpy(i.temperature)


    A = Absorber(Refrigerant=p10, 
        CoolingWater_in=p13, CoolingWater_out=p14,
        StrongSolution=p6, WeakSolution=p1)
    '''x = A.LMTD_counterflow(Th_in=50.6, Th_out = 38, Tc_in=30, Tc_out=35)
    y = A.LMTD_counterflow(Th_in=50.6, Th_out = 38, Tc_in=30, Tc_out=35, method='Chen')
    print(x, y)'''

    A.Q = A.Q_from_specific_enthalpy()
    p14.temperature = A.CW_temperature_out(method='LMTD')
    A.CoolingWater_out = p14

    print('Absorber')
    print(p14.temperature)

def testing_generator():
    p3 = LiBr_solution('p3', 0.55858, 0.50039, 65.4, 10)
    p4 = LiBr_solution('p4', 0.58335, 0.47914, 80, 10)
    p7 = Water('p7', 'shg', 0.02125, 80, 10)
    p11 = Water('p11', 'l', 2.43054, 90.)
    p12 = Water('p12', 'l', p11.mass_flowrate, 83.)

    G = Generator('Generator', Refrigerant=p7, 
        StrongSolution=p4, WeakSolution=p3,
        HotWater_in=p11, HotWater_out=p12)

    G.Q = G.Q_Ren(38)

    print("Generator")
    print(G.SS_temperature_out())

def testing_shx():
    p2 = LiBr_solution('p2', 0.55858, 0.50039, 38, 10)
    p3 = LiBr_solution('p3', 0.55858, 0.50039, 65.4, 10)
    p4 = LiBr_solution('p4', 0.58335, 0.47914, 80, 10)
    p5 = LiBr_solution('p5', 0.58335, 0.47914, 50.6, 10)

    S = SHX('shx', StrongSolution_in=p4, StrongSolution_out=p5,
            WeakSolution_in=p2, WeakSolution_out=p3)

    S.Q = S.Q_Ren()

    print('SHX')
    print(S.WS_temp_out(method="LMTD"))
    
def testing_condenser():
    p7 = Water('p7', 'shg', 0.02125, 80, 10)
    p8 = Water('p8', 'l', 0.02125, 38., 10)
    p15 = Water('p15', 'l', 2.53054, 30)
    p16 = Water('p16', 'l', 2.53054, 35)

    C = Condenser(Refrigerant_in=p7, Refrigerant_out=p8, CoolingWater_in=p15, CoolingWater_out=p16)

    C.Q = C.Q_from_specific_enthalpy(side='tubeside')

    print(C.R_temperature_out())

def testing_evaporator():
    p9 = Water('p9', 'l', 0.02125, 6., 1)
    p10 = Water('p10', 'g', 0.02125, 6., 1)
    p17 = Water('p17', 'l', 2.39232, 13.)
    p18 = Water('p18', 'l', 2.39232, 8.)

    E = Evaporator(Refrigerant_in=p9, Refrigerant_out=p10,
        ChilledWater_in=p17, ChilledWater_out=p18)

    m_chw = E.ChW_mass_flowrate()
    Tchw_out = E.ChW_temperature_out("LMTD")
    Tchw_out2 = E.ChW_temperature_out("Ebalance")
    print('Chilled Water')
    print(F'm_chw: {m_chw}' + '\n' + F'Tchw_out LMTD: {Tchw_out}' +
            F'Tchw_out Ebal: {Tchw_out2}')

    m_r = E.R_mass_flowrate()
    Tr_LMTD = E.R_temperature_in()
    Tr_Ebal = E.R_temperature_in('Ebalance')
    hr = E.R_enthalpy_in()
    print('Refrigerant')
    print(F'm_r: {m_r}' + '\n' + F'Tr_LMTD: {Tr_LMTD}' +
            F'Tr_Ebal: {Tr_Ebal}' + '\n' + F'hr: {hr}')


# testing_absorber()
# testing_generator()
# testing_shx()
# testing_condenser()
# testing_evaporator()


def LMTD_solver(TH_in=None, TH_out=None, TC_in=None, TC_out=None, method='fsolve'):
    if method == 'fsolve':
        from scipy.optimize import fsolve as fx

    
    pass

def func(x, TH_in, TH_out, TC_in, Q, UA):
    LMTD = Q/UA

    return[]
     