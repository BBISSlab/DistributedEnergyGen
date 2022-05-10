from AbsorptionChiller import *

from logging import error
from msilib.schema import Error
from openpyxl import load_workbook
import math as m

# Scientific python add-ons
import pandas as pd     # To install: pip install pandas
import numpy as np      # To install: pip install numpy

# Data Storage and Reading
from pyarrow import feather  # To install: pip install pyarrow

# Thermofluids modules
from iapws import iapws97
from iapws import humidAir
from sympy import QQ_gmpy

def Absorption_Chiller_eq_Ren(
        T_g=80,
        T_a=32,
        T_hw=90.6,
        T_cw_a=30,
        T_cw_c=30,
        T_chw = 13, 
        m_ss=0.5004,
        m_hw=2.43054, 
        m_cw_a = 3.29257,
        m_cw_c = 2.53054,
        m_chw = 2.39232,
        error_threshold=0.01):
    
    ##########################
    # Initialize Statepoints #
    ##########################
    # Thermal Compressor Circuit
    p1 = LiBr_solution(name='p1')
    p2 = LiBr_solution(name='p2')
    p3 = LiBr_solution(name='p3')
    p4 = LiBr_solution(name='p4')
    p5 = LiBr_solution(name='p5')
    p6 = LiBr_solution(name='p6')
    # Refrigerant Circuit
    p7 = Water('p7', 'shg', pressure=10)
    p8 = Water('p8', 'l', pressure=10)
    p9 = Water('p9', 'l', pressure=1)
    p10 = Water('p10', 'g', pressure=1)

    # External Circuits
    p11 = Water('p11', 'l', m_hw, T_hw)
    p12 = Water('p12', 'l', m_hw)
    p13 = Water('p13', 'l', m_cw_a, T_cw_a)
    p14 = Water('p14', 'l', m_cw_a)
    p15 = Water('p15', 'l', m_cw_c, T_cw_c)
    p16 = Water('p16', 'l', m_cw_c)
    p17 = Water('p17', 'l', m_chw, T_chw)
    p18 = Water('p18', 'l', m_chw)

    #########
    # START #
    #########
    '''LOOP 1'''
    # Assume Evaporator Temp (T10)
    p10.temperature = 5
    error_L1 = 1

    while error_L1 > error_threshold:        
        '''LOOP 2'''
        # Assume Condenser Temp (T8)
        p8.temperature = 38
        error_L2 = 1

        while error_L2 > error_threshold:
            '''LOOP 3'''
            # Assume Generator Temp (T4 and T7)
            p4.temperature = T_g
            p7.temperature = T_g
            error_L3 = 1

            while error_L3 > error_threshold:

                # Calculate the strong solution mass fraction
                x_ss = p4.Duhring_equilibrium_concentration(p8.temperature)

                for p in [p4, p5, p6]:
                    p.mass_fraction = x_ss
                
                T6_eq = p4.Duhring_equilibrium_temperature(p10.temperature)

                '''LOOP 4'''
                # Assume value of T5
                p5.temperature = 50.6
                error_L4 = 1

                while error_L4 > error_threshold:
                    
                    Tss_avg = (p4.temperature + p5.temperature)/2
                    cp_ss = p5.isobaric_heat_capacity(Tss_avg)
                    
                    '''LOOP 5'''
                    p1.temperature = T_a
                    error_L5 = 1

                    L5_counter = 0
                    while error_L5 > error_threshold:
                        x_ws = p1.Duhring_equilibrium_concentration(p10.temperature)

                        for p in [p1, p2, p3]:
                            p.mass_fraction = x_ws
                        
                        circulation_ratio = x_ss / (x_ss - x_ws)

                        m_r = m_ss / circulation_ratio
                        m_ws = m_r + m_ss

                        # Update mass flowrates
                        for p in [p1, p2, p3]:
                            p.mass_flowrate = m_ws
                        for p in [p4, p5, p6]:
                            p.mass_flowrate = m_ss
                        for p in [p7, p8, p9, p10]:
                            p.mass_flowrate = m_r

                        p6.temperature = p5.temperature

                        Absorber_ = Absorber(Refrigerant=p10, WeakSolution=p1, StrongSolution=p6,
                            CoolingWater_in=p13, CoolingWater_out=p14)

                        Absorber_.Q = Absorber_.Q_Ren(p10.temperature)

                        p14.temperature = Absorber_.CW_temperature_out(method='Ebalance')
                        Absorber_.CoolingWater_out = p14

                        print(L5_counter)
                        print(Absorber_)
                        
                        try:
                            T1_new = Absorber_.WS_temperature_out()
                        except TypeError:
                            
                            print(F'count: {L5_counter}')
                            print(Absorber_)
                            print(F'T1 new: {T1_new}')
                            exit()
                        
                        error_L5 = absolute_total_error(p1.temperature, T1_new)

                        p1.temperature = T1_new
                        L5_counter += 1
                    
                    print(p1.temperature)
                    exit()



Absorption_Chiller_eq_Ren()