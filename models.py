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

# TO DO
# Review the model
# Verify that demand and supply are working well
# Add distribution losses for the Absorption Chiller and the CHP


def model(Building_, City_,
          Grid_=None, efficiency_Grid=0.533,
          Furnace_=None, efficiency_Furnace=0.95,
          COP_AC=3.8, beta_AC=1,
          COP_ABC=0.7, beta_ABC=0,
          has_PVSystem=False, PVSystem_=None, BES_=None,
          alpha_CHP=0, PrimeMover_=None, HPR_CHP=1, efficiency_CHP=0.73,
          memory={}):
    """
    This function/model is discussed in detail in [CITE ES&T PAPER]

    This function simulates the performance of the energy systems. It begins by simulating the PV system,
    followed by the CCHP system to meet the remaining demand, and finally both the Grid and Storage systems
    to deal with any excess or deficit.

    PARAMETERS
    ----------
    Building_:
    City_:
    Grid_object:
    efficiency_Grid:
    Furnace_object:
    efficiency_Furnace:
    COP_AC:
    beta_AC:
    COP_ABC:
    beta_ABC:
    has_PVSystem:
    PVSystem_:
    BES_object:
    alpha_CHP:
    PrimeMover_object:
    HPR_CHP:
    efficiency_CHP:
    memory:
    """
    # Making copies of the objects so they aren't modified for the following
    # runs
    current_Building = Building_
    City_ = City_
    # Current object is unused because we assume that all Grid electricity
    current_Grid = Grid_
    # is supplied by a Natural Gas Combined Cycle Plant. Future versions
    # will allow the user to customize their own gridmix.
    current_CHP = PrimeMover_
    current_Furnace = Furnace_
    CHP_PrimeMover = 'unspecified'

    # Turnedoff SettingWithCopyWarning for now.
    pd.set_option('mode.chained_assignment', None)

    if current_CHP is not None:
        HPR_CHP = current_CHP.hpr
        efficiency_CHP = max(current_CHP.chp_efficiency_LHV,
                             current_CHP.chp_efficiency_HHV)
        CHP_PrimeMover = current_CHP.PM_id
    if current_Furnace is not None:
        efficiency_Furnace = current_Furnace.efficiency

    # All of the data is stored in a pandas dataframe
    df = pd.DataFrame()

    ###########################
    # 1. SETTING UP DATAFRAME #
    ###########################

    # 1.1 ENERGY DEMANDS #
    ######################

    # Check cooling technologies:
    beta_AC = 1 - beta_ABC

    # Initializing Demands. All demands are in kWh
    df['cooling_demand'] = Building_.cooling_demand
    df['base_electricity_demand'] = Building_.electricity_demand

    # The electricity and heat demands depend on the presence of an absorption
    # chiller
    df['electricity_demand'] = Building_.electricity_demand + \
        beta_AC * df.cooling_demand / COP_AC
    df['heat_demand'] = Building_.heat_demand + \
        beta_ABC * df.cooling_demand / COP_ABC

    # Setting Seasons
    df['Season'] = ''
    df['Season'].loc['2020-01':'2020-03'] = 'Winter'
    df['Season'].loc['2020-12-01':'2021'] = 'Winter'
    df['Season'].loc['2020-03':'2020-06'] = 'Spring'
    df['Season'].loc['2020-06':'2020-09'] = 'Summer'
    df['Season'].loc['2020-09':'2020-12'] = 'Autumn'

    # Metadata
    df['City'] = City_.name
    df['Building'] = Building_.building_type
    df['DryBulb_C'] = City_.tmy_data.DryBulb
    df['Prime_Mover'] = CHP_PrimeMover

    # The smallest microturbine can power 5 houses, see [1]
    if Building_ == 'single_family_residential':
        df.electricity_demand = df.electricity_demand.apply(lambda x: x * 5)
        df.heat_demand = df.heat_demand.apply(lambda x: x * 5)
        df.cooling_demand = df.cooling_demand.apply(lambda x: x * 5)
        Building_.floor_area = Building_.floor_area * 5

    # Intensities of Demand
    df['electricity_demand_int'] = df.electricity_demand / \
        Building_.floor_area
    df['heat_demand_int'] = df.heat_demand / Building_.floor_area
    df['cooling_demand_int'] = df.cooling_demand / Building_.floor_area

    # The DEFICIT column is initially equal to the total demand. Each module (CHP, PV, Storage, and Grid)
    # modifies this column at the end before feeding into the next module.
    df['electricity_deficit'] = df.electricity_demand
    df['heat_deficit'] = df.heat_demand
    df['cooling_deficit'] = df.cooling_demand

    # 1.2 INITIALIZE COLUMNS #
    ##########################
    # Initialize SURPLUS columns:
    # Each surplus column indicates the amount of energy generated that is greater than
    # the demand for each time-step.
    system_states_ls = ['alpha_Grid', 'alpha_PV', 'alpha_CHP',
                        'beta_AC', 'beta_ABC', 'HPR_CHP']

    surplus_ls = ['electricity_surplus', 'heat_surplus', 'cooling_surplus',
                  'electricity_surplus_CHP', 'electricity_surplus_PV',
                  'heat_surplus_Furnace', 'heat_surplus_CHP',
                  'cooling_surplus_AC', 'cooling_surplus_ABC']

    energy_ls = ['electricity_Grid', 'electricity_PV', 'electricity_CHP',
                 'heat_Furnace', 'heat_CHP',
                 'cooling_AC', 'cooling_ABC',
                 'PrimaryEnergy_electricity', 'PrimaryEnergy_heat']

    wasteHeat_ls = ['wasteHeat_Grid', 'wasteHeat_Furnace',
                    'wasteHeat_CHP', 'wasteHeat_ABC', 'wasteHeat']

    emissions_ls = ['co', 'co2', 'ch4', 'voc',
                    'n2o', 'nox', 'pm', 'so2', 'w4e']

    efficiency_metrics_ls = ['TFCE']

    to_intensity = [energy_ls, wasteHeat_ls, emissions_ls]
    intensities = []

    for i in to_intensity:
        for j in i:
            j = j + '_int'
            intensities.append(j)

    df['max_alpha_CHP'] = 1
    df['alpha_CHP'] = alpha_CHP
    df['HPR_CHP'] = HPR_CHP
    df['beta_AC'] = beta_AC
    df['cooling_AC'] = df.cooling_demand * df.beta_AC
    df['beta_ABC'] = beta_ABC
    df['cooling_ABC'] = df.cooling_demand * df.beta_ABC
    df['TFCE'] = 1

    # empty_df = df.copy()
    # Placed here in case we loop through

    def _balance_electricity(dataframe, electricity_source):
        # This function will output 3 variables
        #   deficit:    how much of the demand is still unmet
        #   surplus:    how much of the generated energy exceeds the demand
        #   consumed:   how much of the energy generated was consumed at the
        #               specified timestep
        energy_generated = 'electricity_' + electricity_source + '_gen'
        energy_consumed = 'electricity_' + electricity_source
        dataframe[energy_consumed] = 0
        for index, row in dataframe.iterrows():
            # First check if we have a deficit.
            generated = row[energy_generated]
            deficit = row['electricity_deficit']
            difference = deficit - generated
            if difference >= 0:
                # If a deficit exists, then surplus = 0 and all of the energy
                # generated was consumed
                surplus = 0
                consumed = generated
            else:
                # If the deficit is less than zero, then we have surplus, and only energy equal to
                # the deficit is consumed
                consumed = deficit
                surplus = generated - consumed
            dataframe.loc[index, energy_consumed] = consumed
            dataframe.loc[index, 'electricity_surplus_' +
                          electricity_source] = surplus
            # dataframe.loc[index, 'electricity_deficit'] += deficit
        dataframe['electricity_deficit'] -= dataframe[energy_consumed]
        dataframe.electricity_surplus += dataframe['electricity_surplus_' +
                                                   electricity_source]

    #################
    # 2. Simulation #
    #################

    if has_PVSystem is True:
        # print('Sizing PV')

        try:
            ac_out, num_inverter, PV_age, PV_DEGRADATION_RATE = memory[
                City_.name + '_' + Building_.building_type]
            df['electricity_PV'] = (
                1 - PV_age * PV_DEGRADATION_RATE) * ((ac_out['p_ac'] * num_inverter) / 1000)
            print('PV from memory. alpha CHP={}, beta ABC={}, HPR={}'.format(
                alpha_CHP, beta_ABC, HPR_CHP), end='\r')
        except KeyError:
            memory = {}
            current_PVSystem, num_inverter, capitalCost_PV, omCost_PV = size_pv(Building_=Building_,
                                                                                City_=City_,
                                                                                PVSystem_=PVSystem_,
                                                                                percent_roof_cover=100,
                                                                                percent_energy_demand=0,
                                                                                method='roof')
            # print('Simulating PV outputs')
            dc_out, ac_out = pv_simulation(PVSystem_=current_PVSystem,
                                           City_=City_)
            PV_age = 1
            # Avg degradation rate between 0.5-0.6%, see [X]
            PV_DEGRADATION_RATE = 0.006
            # [X] Jordan, D. C.; Kurtz, S. R.; VanSant, K.; Newmiller, J. Compendium of photovoltaic degradation rates. Prog. Photovoltaics Res. Appl. 2016, 24 (7), 978–989.

            df['electricity_PV'] = (
                1 - PV_age * PV_DEGRADATION_RATE) * ((ac_out['p_ac'] * num_inverter) / 1000)

            # Need to make a dictionary for memory systems so that it runs
            # faster.
            memory[City_.name + '_' + Building_.building_type] = (
                ac_out, num_inverter, PV_age, PV_DEGRADATION_RATE)

        if Building_ == 'single_family_residential':
            df.electricity_PV = df.electricity_PV.apply(lambda x: x * 5)

            df['electricityInt_PV'] = df.electricity_PV / \
                Building_.floor_area

        df.alpha_PV = df.electricity_PV / df.electricity_demand

        # For now assume no net metering
        df.alpha_PV = np.where((df.alpha_PV > 1),
                               1, (df.alpha_PV))

        df.electricity_deficit = df.electricity_deficit - \
            (df.alpha_PV * df.electricity_demand)

        df.max_alpha_CHP = 1 - df.alpha_PV

        df.alpha_CHP = np.where((df.alpha_CHP <= df.max_alpha_CHP),
                                (df.alpha_CHP),
                                (df.max_alpha_CHP))
    else:
        df['alpha_PV'] = 0
        df['alpha_PV_int'] = 0
        df['electricity_PV'] = 0
        df['electricity_PV_int'] = 0

    # CHP supply
    df['electricity_CHP'] = df.alpha_CHP * df.electricity_demand
    df['heat_CHP'] = HPR_CHP * df.electricity_CHP
    df['electricity_CHP_int'] = df.electricity_CHP / Building_.floor_area
    df['heat_CHP_int'] = df.heat_CHP / Building_.floor_area
    # Adjust deficits from CHP supply
    df['electricity_deficit'] = np.where(
        df.electricity_CHP >= df.electricity_demand, 0, df.electricity_deficit - df.electricity_CHP)
    df['heat_deficit'] = np.where(
        df.heat_CHP >= df.heat_demand, 0, df.heat_deficit - df.heat_CHP)
    df['cooling_deficit'] = df.cooling_deficit - df.cooling_AC - df.cooling_ABC

    # Grid Outputs
    df['alpha_Grid'] = 1 - df.alpha_CHP - df.alpha_PV
    df['electricity_Grid'] = df.alpha_Grid * df.electricity_demand
    df['electricity_Grid_int'] = df.electricity_Grid / Building_.floor_area

    # Furnace Outputs
    df['heat_Furnace'] = df.heat_deficit  # Any Heat not supplied by CHP
    df['heat_Furnace_int'] = df.heat_Furnace / \
        Building_.floor_area

    # Adjust Deficits (Should all be 0 at this point)
    '''df.electricity_deficit = df.electricity_deficit - \
        df.electricity_PV - df.electricity_Grid - df.electricity_CHP'''
    df['electricity_deficit'] = np.where(
        df.electricity_Grid >= df.electricity_demand, 0, df.electricity_deficit - df.electricity_Grid)

    # 2.1 EMISSIONS #
    #################
    impacts = ['co2_int', 'ch4_int', 'n2o_int',
               'co_int', 'nox_int', 'so2_int', 'pm_int', 'voc_int',
               'w4e_int']
    # lb/MMBtu to kg/MWh or g/kWh
    # conversion_factor = (3412 / (10**9 * 2.20462))
    GLF = 0.049  # Average Grid Loss Factor (GLF) from [2]

    for impact in impacts:
        # All air emissions are in kg/MWh

        # Emission Factors from the Grid:
        #   We assume that the grid is powered by an NGCC.
        #   The emission factors for the NGCC are compiled from [3] and [4]
        # Emission Factors from the CHP:
        #   We take the CHP performance metrics from [5]
        #   Only co, co2, nox, and voc's are reported as significant
        # Emission Factors from the Furnace:
        #   We take the Furnace operating metrics from [6]
        # ng_fraction is the fraction of the compound (e.g., ch4, co2, voc, etc.)
        # that is present in natural gas. We use the natural gas composition from [7]
        #   Note that only methane is currently considered in this model.
        if impact == 'ch4_int':
            emission_factor_Grid = 2.50 * (10**-2)
            emission_factor_CHP = 0
            emission_factor_Furnace = current_Furnace.ch4
            ng_fraction = 0.9
        if impact == 'co_int':
            emission_factor_Grid = 8.71 * (10**-2)
            emission_factor_CHP = current_CHP.co
            emission_factor_Furnace = current_Furnace.co
            ng_fraction = 0
        if impact == 'co2_int':
            emission_factor_Grid = 170.24
            emission_factor_CHP = current_CHP.co2
            emission_factor_Furnace = current_Furnace.co2
            ng_fraction = 0.02
        if impact == 'n2o_int':
            emission_factor_Grid = 8.71 * (10**-3)
            emission_factor_CHP = 0
            emission_factor_Furnace = current_Furnace.n2o
            ng_fraction = 0
        if impact == 'nox_int':
            emission_factor_Grid = 3.1 * (10**-2)
            emission_factor_CHP = current_CHP.nox
            emission_factor_Furnace = current_Furnace.nox  # kg/MWh
            ng_fraction = 0
        if impact == 'so2_int':
            emission_factor_Grid = 5.11 * (10**-3)
            emission_factor_CHP = 0
            emission_factor_Furnace = current_Furnace.so2  # kg/MWh
            ng_fraction = 0
        if impact == 'pm_int':
            emission_factor_Grid = 1.92 * (10**-2)
            emission_factor_CHP = 0
            emission_factor_Furnace = current_Furnace.pm
            ng_fraction = 0
        if impact == 'voc_int':
            emission_factor_Grid = 6.1 * (10**-3)
            emission_factor_CHP = current_CHP.voc
            emission_factor_Furnace = current_Furnace.voc
            ng_fraction = 0.01
        if impact == 'w4e_int':
            # James et al. (2016) 0.2 gal/kWh converted to L/kWh
            emission_factor_Grid = 0.757
            emission_factor_CHP = current_CHP.w4e
            emission_factor_Furnace = 0
            ng_fraction = 0

        # 2.1.1 NATURAL GAS LEAKAGE #
        # ========================= #

        # We take the natural gas leakage values from [8]. It is understood that governmental
        # inventories often underestimate fugitive emissions. We take the leakage values from
        # Alvarez et al (2018) [9] as our baseline value.

        # These values are expressed in [8] as the fraction of losses of gross methane withdrawals.
        # Since natural gas is ~ 90% methane, we divide these leakage rates to obtain the
        # percentage of natural gas leakages.

        # SPECIFIC ENERGY FROM NATURAL GAS
        # We take the LHV (48,252 J/g of natural gas) from [10]
        spec_energy_ng = 13.4 * (10**-3)  # MWh/kg

        if impact == 'ch4_int':  # We only consider methane (CH4) emissions
            # The natural gas power plant does not include the distribution
            # phase in the leakage
            leakage_ng = [0.027,    # Total leakage from production through transmission & storage
                          0.028]    # Total leakage from production through distribution
        else:
            leakage_ng = [0, 0]

        Grid_leakage_rate = leakage_ng[0]
        Total_leakage_rate = leakage_ng[1]
        K_g = (Grid_leakage_rate * ng_fraction) / (spec_energy_ng *
                                                   efficiency_Grid * (1 - Grid_leakage_rate)) + emission_factor_Grid
        K_CHP = ((Total_leakage_rate * ng_fraction * (1 + HPR_CHP)) /
                 (spec_energy_ng * efficiency_CHP * (1 - Total_leakage_rate))) + emission_factor_CHP
        K_f = (Total_leakage_rate * ng_fraction) / (spec_energy_ng *
                                                    efficiency_Furnace * (1 - Total_leakage_rate)) + emission_factor_Furnace

        df[impact] = (np.where((df.heat_CHP >= df.heat_demand),
                               (((df.alpha_Grid) / (1 - GLF)) * K_g +
                                df.alpha_CHP * K_CHP) * df.electricity_demand,
                               ((((df.alpha_Grid) / (1 - GLF)) * K_g + df.alpha_CHP * (K_CHP - HPR_CHP * K_f)) * df.electricity_demand
                                + df.heat_demand * K_f))) / Building_.floor_area

    # 2.1.2 Greenhouse Gas Emissions #
    # ============================== #
    # We use the Global Warming Potential values from [11]
    #   Global warming potential values are taken from IPCC AR5, pg 87
    df['GHG_int'] = df.co2_int + \
        (df.ch4_int * 28) + \
        (df.n2o_int * 265)

    # 2.1.3 CHP Offsets #
    # ================= #
    # In cases where the CHP produces more heat than is demanded, we assume
    # that the waste heat can be used to offset the emissions of a conventional
    # heat generator (i.e., furnace or boiler)

    # Calculate CHP heat credits
    df['CHP_heat_surplus'] = np.where(df.heat_CHP_int > df.heat_demand_int,
                                      df.heat_CHP_int - df.heat_demand_int, 0)
    # GHG credits
    df['CHP_co2_credit'] = df.CHP_heat_surplus * Furnace_.co2
    df['CHP_n2o_credit'] = df.CHP_heat_surplus * Furnace_.n2o
    df['CHP_ch4_credit'] = df.CHP_heat_surplus * Furnace_.ch4
    # Conventional Air Pollutant Credits
    df['CHP_co_credit'] = df.CHP_heat_surplus * Furnace_.co
    df['CHP_nox_credit'] = df.CHP_heat_surplus * Furnace_.nox
    df['CHP_so2_credit'] = df.CHP_heat_surplus * Furnace_.so2
    df['CHP_pm_credit'] = df.CHP_heat_surplus * Furnace_.pm
    df['CHP_voc_credit'] = df.CHP_heat_surplus * Furnace_.voc

    # 2.2 WASTE HEAT, FUEL CONSUMPTION, AND EFFICIENCY #
    ####################################################

    # 2.2.1 WASTE HEAT #
    # ================ #
    df['wasteHeat'] = np.where((df.heat_CHP < df.heat_demand),
                               (df.alpha_Grid) / (1 - GLF) * (1 / efficiency_Grid - 1) *
                               df.electricity_demand,
                               (((df.alpha_Grid) / (1 - GLF) * (1 / efficiency_Grid - 1) + df.alpha_CHP * HPR_CHP)
                                * df.electricity_demand - df.heat_demand)
                               )
    df['wasteHeat_int'] = df.wasteHeat / Building_.floor_area

    # 2.2.2 FUEL CONSUMPTION #
    # ====================== #

    # Fuel Consumption
    df['Fuel_Grid'] = df.alpha_Grid / \
        (1 - GLF) * (1 / efficiency_Grid) * df.electricity_demand
    df['Fuel_CHP'] = df.alpha_CHP * \
        (1 + HPR_CHP) * df.electricity_demand / efficiency_CHP
    # In the System Classes file, is modified to become a decimal. But is
    # loaded as a percent (i.e., as 95 rather than 0.95)
    df['Fuel_Furnace'] = df.heat_Furnace / efficiency_Furnace

    # Fuel Consumption Intensity
    df['Fuel_Grid_int'] = df.Fuel_Grid / Building_.floor_area
    df['Fuel_CHP_int'] = df.Fuel_CHP / Building_.floor_area
    df['Fuel_Furnace_int'] = df.Fuel_Furnace / Building_.floor_area
    df['Fuel_int_total'] = df.Fuel_Grid_int + \
        df.Fuel_CHP_int + df.Fuel_Furnace_int

    # CHP Fuel consumption Credit
    df['CHP_Fuel_credit'] = df.CHP_heat_surplus / \
        Furnace_.efficiency

    # 2.2.3 TOTAL FUEL CYCLE EFFICIENCY #
    # ================================= #
    df['TFCE'] = ((df.electricity_demand_int +
                   df.heat_demand_int) / df.Fuel_int_total) * 100

    # 2.3 AGGREGATE TO DAILY VALUES #
    #################################
    # Saving the data by hourly values can be memory intensive. We can aggregate these values by day.
    # For some reason, the index is not being recognized as datetime before
    # resampling.
    df.index = pd.to_datetime(df.index)
    agg_df = df.groupby(['HPR_CHP', 'Prime_Mover', 'alpha_CHP', 'beta_ABC', 'City', 'Season', 'Building']).resample('d').agg({'DryBulb_C': ['mean'],
                                                                                                                              'energy_useful_int': ['sum'],
                                                                                                                              'electricity_demand_int': ['sum'],
                                                                                                                              'cooling_demand_int': ['sum'],
                                                                                                                              'heat_demand_int': ['sum'],
                                                                                                                              'electricity_Grid_int': ['sum'],
                                                                                                                              'electricity_PV_int': ['sum'],
                                                                                                                              'electricity_CHP_int': ['sum'],
                                                                                                                              'heat_CHP_int': ['sum'],
                                                                                                                              'Fuel_Grid_int': ['sum'],
                                                                                                                              'Fuel_CHP_int': ['sum'],
                                                                                                                              'Fuel_Furnace_int': ['sum'],
                                                                                                                              'wasteHeat_int': ['sum'],
                                                                                                                              # Emissions
                                                                                                                              # Green
                                                                                                                              # House
                                                                                                                              # Gases
                                                                                                                              'co2_int': ['sum'],
                                                                                                                              'ch4_int': ['sum'],
                                                                                                                              'n2o_int': ['sum'],
                                                                                                                              # Criteria
                                                                                                                              # Air
                                                                                                                              # Pollutants
                                                                                                                              'co_int': ['sum'],
                                                                                                                              'nox_int': ['sum'],
                                                                                                                              'pm_int': ['sum'],
                                                                                                                              'so2_int': ['sum'],
                                                                                                                              'voc_int': ['sum'],
                                                                                                                              # Water
                                                                                                                              # for
                                                                                                                              # Energy
                                                                                                                              # Consumption
                                                                                                                              'w4e_int': ['sum']})

    agg_df.columns = agg_df.columns.map('_'.join)
    # agg_df.columns = agg_df.columns.droplevel(1)
    agg_df.rename(columns={'energy_useful_int_sum': 'energy_useful_int',
                           'electricity_demand_int_sum': 'electricity_demand_int', 'cooling_demand_int_sum': 'cooling_demand_int', 'heat_demand_int_sum': 'heat_demand_int',
                           'electricity_Grid_int_sum': 'electricity_Grid_int', 'electricity_PV_int_sum': 'electricity_PV_int',
                           'electricity_CHP_int_sum': 'electricity_CHP_int', 'heat_CHP_int_sum': 'heat_CHP_int',
                           'Fuel_Grid_int_sum': 'Fuel_Grid_int', 'Fuel_CHP_int_sum': 'Fuel_CHP_int', 'Fuel_Furnace_int_sum': 'Fuel_Furnace_int',
                           'wasteHeat_int_sum': 'wasteHeat_int',
                           # Emissions
                           'co2_int_sum': 'co2_int', 'ch4_int_sum': 'ch4_int', 'n2o_int_sum': 'n2o_int',
                           'co_int_sum': 'co_int', 'nox_int_sum': 'nox_int', 'pm_int_sum': 'pm_int',
                           'so2_int_sum': 'so2_int', 'voc_int_sum': 'voc_int', 'w4e_int_sum': 'w4e_int'}, inplace=True)

    # PV calcs unused ATM
    #  agg_df['alpha_PV'] = agg_df.electricity_PV_int / agg_df.electricity_demand_int
    # print(agg_df.index)

    agg_df = agg_df.reset_index().set_index('datetime')
    agg_df.dropna(inplace=True)
    agg_df.index = agg_df.index.date

    return df, agg_df


def energy_demand_sim(Building_, City_, AC_,
                      ABC_, beta_ABC=0,
                      thermal_distribution_loss_rate=0.1,
                      thermal_distribution_loss_factor=1.0,
                      memory={}):
    """
    This function simulates the electric, heating, and cooling demands based
    on the types of technologies adopted.
    """

    # Turnedoff SettingWithCopyWarning for now.
    pd.set_option('mode.chained_assignment', None)

    # All of the data is stored in a pandas dataframe
    df = pd.DataFrame()

    # Check cooling technologies:
    beta_AC = 1 - beta_ABC

    # Initializing Demands. All demands are in kWh
    df['electricity_demand'] = Building_.electricity_demand
    df['heat_demand'] = Building_.heat_demand
    df['cooling_demand'] = Building_.cooling_demand

    # Setting Seasons
    # df['Season'] = ''
    # df['Season'].loc['2020-01':'2020-03'] = 'Winter'
    # df['Season'].loc['2020-12-01':'2021'] = 'Winter'
    # df['Season'].loc['2020-03':'2020-06'] = 'Spring'
    # df['Season'].loc['2020-06':'2020-09'] = 'Summer'
    # df['Season'].loc['2020-09':'2020-12'] = 'Autumn'

    # Metadata
    df['City'] = City_.name
    df['Building'] = Building_.building_type
    # df['DryBulb_C'] = City_.tmy_data.DryBulb
    df['beta_ABC'] = beta_ABC

    if beta_AC > 0:
        df['AC_id'] = AC_.AC_id
    else:
        df['AC_id'] = 'None'
    if beta_ABC > 0:
        df['ABC_id'] = ABC_.ABC_id
    else:
        df['ABC_id'] = 'None'

    '''
    The smallest microturbine can power 5 houses, see [1]
    [1] James, J.-A.; Thomas, V. M.; Pandit, A.; Li, D.; Crittenden, J. C. Water, Air Emissions,
        and Cost Impacts of Air-Cooled Microturbines for Combined Cooling, Heating, and Power Systems:
        A Case Study in the Atlanta Region. Engineering 2016, 2 (4), 470–480.
        https://doi.org/10.1016/J.ENG.2016.04.008.
    '''
    if Building_ == 'single_family_residential':
        df['electricity_demand'] = df.electricity_demand.apply(lambda x: x * 5)
        df['heat_demand'] = df.heat_demand.apply(lambda x: x * 5)
        df['cooling_demand'] = df.cooling_demand.apply(lambda x: x * 5)
        Building_.floor_area = Building_.floor_area * 5

    df['electricity_cooling'] = beta_AC * df.cooling_demand / AC_.COP
    df['heat_cooling'] = beta_ABC * df.cooling_demand / \
        (ABC_.COP * (1 - (thermal_distribution_loss_rate * thermal_distribution_loss_factor)))

    df['total_electricity_demand'] = df.electricity_demand + df.electricity_cooling
    df['total_heat_demand'] = df.heat_demand + df.heat_cooling

    # Intensities of Demand
    df['electricity_demand_int'] = df.electricity_demand / \
        Building_.floor_area
    df['heat_demand_int'] = df.heat_demand / Building_.floor_area
    df['cooling_demand_int'] = df.cooling_demand / Building_.floor_area
    df['total_electricity_demand_int'] = df.total_electricity_demand / \
        Building_.floor_area
    df['total_heat_demand_int'] = df.total_heat_demand / Building_.floor_area

    return df


def energy_supply_sim(Building_, City_,
                      energy_demand_df=None,
                      Furnace_=None, efficiency_Furnace=0.95,
                      pv_energy_sim=None, has_PVSystem=False, PVSystem_=None,
                      BES_=None,
                      alpha_CHP=0, PrimeMover_=None, HPR_CHP=1, efficiency_CHP=0.73,
                      thermal_distribution_loss_rate=0.1,
                      thermal_distribution_loss_factor=1.0,
                      aggregate='A',
                      memory={}):

    # Read the energy demand data
    if thermal_distribution_loss_factor == 1:
        file_path = r'model_outputs\energy_demands'
        file_name = F'Hourly_{City_.name}_{Building_.building_type}_energy_dem.feather'
    else:
        file_path = r'model_outputs\distribution_sensitivity'
        file_name = F'Hourly_{City_.name}_{Building_.building_type}_energy_dem_dist_sens.feather'
    df = pd.read_feather(F'{file_path}\\{file_name}')

    # Initialize Energy Supply Columns
    if alpha_CHP == 0:
        df['PM_id'] = 'None'
    else:
        df['PM_id'] = PrimeMover_.PM_id
    df['max_alpha_CHP'] = 1
    df['alpha_CHP'] = alpha_CHP
    df['HPR_CHP'] = HPR_CHP

    df['Furnace_id'] = Furnace_.Furnace_id

    """
    ENERGY SUPPLY SIMULATION
    """
    df['electricity_deficit'] = df.total_electricity_demand
    df['heat_deficit'] = df.total_heat_demand

    # Check for Compatibility of Prime Mover and Absorption Chiller
    if PrimeMover_.abc_compatibility == 0:
        df.drop(df[(df.ABC_id == 'ABC_TS1') | (
            df.ABC_id == 'ABC_TS2')].index, inplace=True)

    # PV Energy Simulation
    if (pv_energy_sim is not None):
        pass

    if Furnace_.Furnace_id in ['F1', 'F2', 'F4', 'B2']:
        #######
        # CHP #
        #######
        df['electricity_CHP'] = df.alpha_CHP * df.electricity_deficit
        df['heat_CHP_0'] = PrimeMover_.hpr * df.electricity_CHP
        df['heat_CHP'] = df['heat_CHP_0'] * \
            (1 - (thermal_distribution_loss_rate * thermal_distribution_loss_factor))
        # CHP Intensities
        df['electricity_CHP_int'] = df.electricity_CHP / Building_.floor_area
        df['heat_CHP_int'] = df.heat_CHP / Building_.floor_area

        # Adjust deficits from CHP supply
        df['electricity_deficit'] = calculate_energy_deficit(
            df.electricity_deficit, df.electricity_CHP)
        df['heat_deficit'] = calculate_energy_deficit(
            df.heat_deficit, df.heat_CHP)

        ###########
        # Furnace #
        ###########
        df['electricity_Furnace'] = 0
        df['heat_Furnace'] = df.heat_deficit  # Any Heat not supplied by CHP
        df['heat_Furnace_int'] = df.heat_Furnace / \
            Building_.floor_area

        # Adjust heat deficit
        df['heat_deficit'] = calculate_energy_deficit(
            df.heat_deficit, df.heat_Furnace)

        ########
        # Grid #
        ########
        df['electricity_Grid'] = df.electricity_deficit
        df['electricity_Grid_int'] = df.electricity_Grid / Building_.floor_area

        # Adjust electricity deficit
        df['electricity_deficit'] = calculate_energy_deficit(
            df.electricity_deficit, df.electricity_Grid)

    else:
        ###########
        # Furnace #
        ###########
        # Calculate the electricity demand of the furnace
        df['electricity_Furnace'] = (df.heat_demand - df.alpha_CHP * PrimeMover_.hpr * (df.electricity_demand + df.electricity_cooling)) \
            / (df.alpha_CHP * PrimeMover_.hpr + Furnace_.efficiency)
        # If the Furnace electricity is negative, the CHP is supplying more heat than what is needed,
        # therefore, the Furnace won't be used and you will have waste heat
        # from the CHP
        df['electricity_Furnace'] = np.where(
            df.electricity_Furnace < 0, 0, df.electricity_Furnace)
        df['electricity_Furnace_int'] = df.electricity_Furnace / Building_.floor_area

        # Heat supply of the furnace
        df['heat_Furnace'] = df.electricity_Furnace * Furnace_.efficiency
        df['heat_Furnace_int'] = df.heat_Furnace / \
            Building_.floor_area

        # Adjust Electricity Demand for Electric Furnace
        df['total_electricity_demand'] = df.electricity_demand + \
            df.electricity_cooling + df.electricity_Furnace
        df['total_electricity_demand_int'] = df.total_electricity_demand / \
            Building_.floor_area

        # Adjust deficits
        df['electricity_deficit'] = df.total_electricity_demand
        df['heat_deficit'] = calculate_energy_deficit(
            df.heat_deficit, df.heat_Furnace)

        #######
        # CHP #
        #######
        df['electricity_CHP'] = df.alpha_CHP * df.electricity_deficit
        df['heat_CHP_0'] = PrimeMover_.hpr * df.electricity_CHP
        df['heat_CHP'] = df['heat_CHP_0'] * \
            (1 - (thermal_distribution_loss_rate * thermal_distribution_loss_factor))
        # CHP Intensities
        df['electricity_CHP_int'] = df.electricity_CHP / Building_.floor_area
        df['heat_CHP_int'] = df.heat_CHP / Building_.floor_area

        # Adjust deficits from CHP supply
        df['electricity_deficit'] = calculate_energy_deficit(
            df.electricity_deficit, df.electricity_CHP)
        df['heat_deficit'] = calculate_energy_deficit(
            df.heat_deficit, df.heat_CHP)

        ########
        # Grid #
        ########
        df['electricity_Grid'] = df.electricity_deficit
        df['electricity_Grid_int'] = df.electricity_Grid / Building_.floor_area

        # Adjust electricity deficit
        df['electricity_deficit'] = calculate_energy_deficit(
            df.electricity_deficit, df.electricity_Grid)

    ####################
    # Aggregate Values #
    ####################
    df.index = pd.to_datetime(df.index)
    agg_df = df.groupby(['City', 'Building',
                         'PM_id', 'alpha_CHP',
                         'AC_id', 'ABC_id', 'beta_ABC',
                         'Furnace_id'
                         ]).resample(F'{aggregate}').agg({
                             # DEMANDS
                             'electricity_demand_int': ['sum'],
                             'cooling_demand_int': ['sum'],
                             'heat_demand_int': ['sum'],
                             # Adjusted total electricity and heat
                             'total_electricity_demand_int': ['sum'],
                             'total_heat_demand_int': ['sum'],
                             # SUPPLY
                             'electricity_Grid_int': ['sum'],
                             # 'electricity_PV_int': ['sum'],
                             'electricity_CHP_int': ['sum'],
                             'heat_CHP_int': ['sum'],
                             'heat_Furnace_int': ['sum']})

    agg_df.columns = agg_df.columns.map('_'.join)
    # agg_df.columns = agg_df.columns.droplevel(1)
    agg_df.rename(columns={'electricity_demand_int_sum': 'electricity_demand_int', 'cooling_demand_int_sum': 'cooling_demand_int', 'heat_demand_int_sum': 'heat_demand_int',
                           'electricity_Grid_int_sum': 'electricity_Grid_int',
                           'total_electricity_demand_int_sum': 'total_electricity_demand_int', 'total_heat_demand_int_sum': 'total_heat_demand_int',
                           # 'electricity_PV_int_sum': 'electricity_PV_int',
                           'electricity_CHP_int_sum': 'electricity_CHP_int', 'heat_CHP_int_sum': 'heat_CHP_int',
                           'heat_Furnace_int_sum': 'heat_Furnace_int'}, inplace=True)

    agg_df.reset_index(inplace=True)

    return agg_df


def impacts_sim(data,
                Building_=None, City_=None,
                Furnace_=None,
                PrimeMover_=None,
                Grid_type='NGCC',
                thermal_distribution_loss_rate=0.1,
                leakage_factor=1.0,
                GWP_factor=1.0,
                GLF=0.049):

    # Read the energy demand data
    if data is None:
        '''file_path = r'model_outputs\\energy_supply'
        file_name = F'Annual_{City_.name}_{Building_.building_type}_energy_sup.feather'
        data = pd.read_feather(F'{file_path}\\{file_name}')'''
        pass

    # Load Furnace Dataframe
    furnace_df = pd.read_csv('data\\Tech_specs\\Furnace_specs.csv', header=1)
    furnace_df.rename(columns={'carbon_dioxide': 'Furnace_co2',
                               'nitrous_oxide': 'Furnace_n2o',
                               'PM': 'Furnace_pm',
                               'sulfur_dioxide': 'Furnace_so2',
                               'methane': 'Furnace_ch4',
                               'voc': 'Furnace_voc',
                               'nox': 'Furnace_nox',
                               'carbon_monoxide': 'Furnace_co',
                               'efficiency': 'Furnace_efficiency'}, inplace=True)

    # Load CHP Dataframe
    pm_df = pd.read_csv('data\\Tech_specs\\PrimeMover_specs.csv', header=2)
    pm_df.rename(columns={'carbon_monoxide': 'CHP_co',
                          'carbon_dioxide': 'CHP_co2',
                          'voc': 'CHP_voc',
                          'nox': 'CHP_nox'}, inplace=True)

    pm_df['CHP_efficiency'] = pm_df[['chp_EFF_LHV', 'chp_EFF_HHV']].max(axis=1)

    # Create Grid Dataframe
    # Ou, L., & Cai, H. (2020). ANL-20/41: Update of Emission Factors of 
    # Greenhouse Gases and Criteria Air Pollutants, and Generation 
    # Efficiencies of the U.S. Electricity Generation Sector Energy 
    # Systems Division. www.anl.gov.
    NGCC_dict = {'ch4': 9. * 10**-3,
                 'co': 3.4 * 10**-2,
                 'co2': 340.,
                 'nox': 5.81 * 10**-2,
                 'n2o': 1. * 10**-3,
                 'pm': 1.7 * 10**-2,
                 'so2': 9.58 * 10**-3,
                 'voc': 4.0 * 10**-3,
                 'efficiency': 0.533}

    # Merge the emission factors of each
    df = pd.merge(data, furnace_df[['Furnace_id',
                                   'Furnace_co2', 'Furnace_n2o', 'Furnace_ch4',
                                    'Furnace_co', 'Furnace_nox', 'Furnace_pm',
                                    'Furnace_so2', 'Furnace_voc']], on='Furnace_id', how='left').fillna(0)
    df = pd.merge(df, furnace_df[['Furnace_id',
                                  'Furnace_efficiency']], on='Furnace_id', how='left').fillna(1)
    df = pd.merge(df, pm_df[['PM_id', 'CHP_co', 'CHP_co2',
                            'CHP_voc', 'CHP_nox']], on='PM_id', how='left').fillna(0)
    df = pd.merge(df, pm_df[['PM_id', 'CHP_efficiency']],
                  on='PM_id', how='left').fillna(1)

    ###################################
    # Calculate Operational Emissions #
    ###################################
    column_list = ['City', 'Building', 'PM_id', 'alpha_CHP',
                   'AC_id', 'ABC_id', 'beta_ABC', 'Furnace_id',
                   'electricity_demand_int', 'cooling_demand_int', 'heat_demand_int',
                   'total_electricity_demand_int', 'total_heat_demand_int',
                   'electricity_Grid_int', 'electricity_CHP_int', 'heat_CHP_int', 'heat_Furnace_int']

    impacts = ['co2', 'n2o', 'ch4', 'co', 'nox', 'pm', 'so2', 'voc']

    # All impacts are in g per sq m of floor area
    for impact in impacts:
        # Furnace Emissions
        df[F'Furnace_{impact}_int'] = df.heat_Furnace_int * \
            df[F'Furnace_{impact}']
        column_list.append(F'Furnace_{impact}_int')

        # CHP Emissions
        if impact in ['n2o', 'ch4', 'pm', 'so2']:
            pass
        else:
            df[F'CHP_{impact}_int'] = df.electricity_CHP_int * \
                df[F'CHP_{impact}']
            column_list.append(F'CHP_{impact}_int')

        # Grid Emissions
        if Grid_type == 'NGCC':
            df[F'Grid_{impact}_int'] = NGCC_dict[impact] * \
                df.electricity_Grid_int
            column_list.append(F'Grid_{impact}_int')

    ####################
    # Fuel Consumption #
    ####################
    # Fuel consumption is in kWh per m^2 of floor area
    df['Grid_NG_int'] = (df.electricity_Grid_int /
                         ((1 - GLF) * NGCC_dict['efficiency']))
    df['Furnace_NG_int'] = df.heat_Furnace_int / \
        (df.Furnace_efficiency / 100)  # Furnace efficiency is as % in the tech file
    df['CHP_NG_int'] = (df.electricity_CHP_int +
                        (df.heat_CHP_int) / (1 - thermal_distribution_loss_rate)) / df.CHP_efficiency

    # Append columns
    column_list.extend(['Grid_NG_int', 'Furnace_NG_int', 'CHP_NG_int'])

    ###############################
    # Calculate Leakage Emissions #
    ###############################
    grid_leakage_rate = 0.027 * leakage_factor  # Excludes distribution
    total_leakage_rate = 0.028 * leakage_factor  # Includes distribution

    df['Grid_ch4_leak_int'] = calculate_leakage(
        grid_leakage_rate, df.Grid_NG_int)
    df['Furnace_ch4_leak_int'] = calculate_leakage(
        total_leakage_rate, df.Furnace_NG_int)
    df['CHP_ch4_leak_int'] = calculate_leakage(
        total_leakage_rate, df.CHP_NG_int)

    df['ch4_leak_int'] = df.Grid_ch4_leak_int + \
        df.Furnace_ch4_leak_int + df.CHP_ch4_leak_int

    column_list.extend(['Grid_ch4_leak_int', 'Furnace_ch4_leak_int',
                        'CHP_ch4_leak_int', 'ch4_leak_int'])
    ###############################
    # Calculate Avoided Emissions #
    ###############################
    surplus_heat_int = calculate_energy_surplus(
        df.heat_demand_int, df.heat_CHP_int)
    for impact in impacts:
        Furnace_emission_factor = df[F'Furnace_{impact}'].mean()
        Furnace_efficiency = df['Furnace_efficiency'].mean()

        if impact == 'ch4':
            df[F'avoided_{impact}_int'] = calculate_avoided_emisions(
                surplus_heat_int, Furnace_emission_factor, Furnace_efficiency)
        else:
            df[F'avoided_{impact}_int'] = calculate_avoided_emisions(
                surplus_heat_int, Furnace_emission_factor, Furnace_efficiency, leakage_rate=0)
        column_list.append(F'avoided_{impact}_int')

    ###########################
    # Calculate GHG Emissions #
    ###########################
    # Grid and Furnace
    for system in ['Grid', 'Furnace']:
        df[F'{system}_GHG_int_100'] = calculate_GHG(co2=df[F'{system}_co2_int'],
                                                    ch4=df[F'{system}_ch4_int'] +
                                                    df[F'{system}_ch4_leak_int'],
                                                    n2o=df[F'{system}_n2o_int'])

        df[F'{system}_GHG_int_20'] = calculate_GHG(co2=df[F'{system}_co2_int'],
                                                   ch4=df[F'{system}_ch4_int'] +
                                                   df[F'{system}_ch4_leak_int'],
                                                   n2o=df[F'{system}_n2o_int'],
                                                   GWP_year=20)
        column_list.extend([F'{system}_GHG_int_100', F'{system}_GHG_int_20'])
    # CHP
    df[F'CHP_GHG_int_100'] = calculate_GHG(co2=df[F'CHP_co2_int'],
                                           ch4=df[F'CHP_ch4_leak_int'] -
                                           df[F'avoided_ch4_int'],
                                           n2o=0)

    df[F'CHP_GHG_int_20'] = calculate_GHG(co2=df[F'CHP_co2_int'],
                                          ch4=df[F'CHP_ch4_leak_int'] -
                                          df[F'avoided_ch4_int'],
                                          n2o=0, GWP_year=20)
    column_list.extend([F'CHP_GHG_int_100', F'CHP_GHG_int_20'])

    ##############################
    # Calculate Totals Emissions #
    ##############################
    impacts.extend(['GHG', 'NG'])
    for impact in impacts:
        if impact == 'GHG':
            df['GHG_int_100'], df['GHG_int_20'] = aggregate_impacts(df, impact)
            column_list.extend(['GHG_int_100', 'GHG_int_20'])
        else:
            df[F'{impact}_int'] = aggregate_impacts(df, impact)
            column_list.append(F'{impact}_int')

    ##########################
    # Calculate Total Energy #
    ##########################
    total_electricity_output = df['electricity_Grid_int'] + \
        df['electricity_CHP_int']
    total_heat_output = df['heat_Furnace_int'] + df['heat_CHP_int']
    total_cooling_output = df['cooling_demand_int']
    df['energy_demand_int'] = df.electricity_demand_int + df.heat_demand_int
    column_list.append('energy_demand_int')

    df['NG_int'] = df['Grid_NG_int'] + df['Furnace_NG_int'] + df['CHP_NG_int']

    ##################
    # Calculate TFCE #
    ##################
    df['TFCE'] = (total_electricity_output + total_heat_output) / df['NG_int']
    column_list.append('TFCE')

    ######################################
    # Calculate Trigeneration Efficiency #
    ######################################
    df['trigen_efficiency'] = (
        total_electricity_output + total_heat_output + total_cooling_output) / df['NG_int']
    column_list.append('trigen_efficiency')

    # Copy only emissions data
    impacts_df = df[column_list]

    return impacts_df


def calculate_leakage(leakage_rate, fuel_consumption):
    if leakage_rate == 0:
        return 0
    specific_energy_NG = 13.4 * (10**-3)  # MWh/kg = kWh/g

    system_leakage = ((leakage_rate * fuel_consumption) /
                      (specific_energy_NG * (1 - leakage_rate)))

    return system_leakage


def calculate_energy_deficit(energy_demand, energy_supply):
    deficit = np.where(energy_supply >= energy_demand,
                       0,
                       energy_demand - energy_supply)
    return deficit


def calculate_energy_surplus(energy_demand, energy_supply):
    surplus_energy = np.where(energy_supply > energy_demand,
                              energy_supply - energy_demand, 0)
    return surplus_energy


def calculate_avoided_emisions(
        surplus_energy,
        system_emission_factor, system_efficiency,
        leakage_rate=0.028,
        leakage_calculation=False):
    # Avoided natural gas consumption and leakage
    avoided_NG_consumption = surplus_energy / system_efficiency
    avoided_leakage_emissions = calculate_leakage(
        leakage_rate, avoided_NG_consumption)

    # Avoided emisions
    avoided_emissions = (surplus_energy *
                         system_emission_factor) + avoided_leakage_emissions

    return avoided_emissions


def calculate_GHG(co2=0, ch4=0, n2o=0, GWP_year=100, GWP_factor=1,
                  feedbacks=False):
    '''
    This function calculates the total greenhouse gas
    emissions as co2_eq. The co2, ch4, and n2o must be in
    the same units (e.g., grams, kg)

    The Global Warming Potential Values were taken from [1]

    [1]     IPCC. Climate Change 2014: Synthesis Report;
                Pachauri, R. K., Meyer, L., Eds.;
                Intergovernmental Panel on Climate Change:
                Geneva, Switzerland, 2014.
    '''
    if feedbacks is False:
        if GWP_year == 100:
            GWP_ch4 = 28
            GWP_n2o = 265
        if GWP_year == 20:
            GWP_ch4 = 84
            GWP_n2o = 264

    if feedbacks is True:
        if GWP_year == 100:
            GWP_ch4 = 34
            GWP_n2o = 298
        if GWP_year == 20:
            GWP_ch4 = 86
            GWP_n2o = 268

    co2_eq = co2 + (GWP_ch4 * ch4 + GWP_n2o * n2o) * GWP_factor

    return co2_eq


def aggregate_impacts(dataframe, impact):
    if impact == 'GHG':
        GHG_100 = calculate_GHG(co2=dataframe.co2_int,
                ch4=dataframe.ch4_int,
                n2o=dataframe.n2o_int,
                GWP_year=100)
        GHG_20 = calculate_GHG(co2=dataframe.co2_int,
                ch4=dataframe.ch4_int,
                n2o=dataframe.n2o_int,
                GWP_year=20)
        return GHG_100, GHG_20
    else:
        if impact in ['n2o', 'pm', 'so2']:
            op_impacts = dataframe[F'Grid_{impact}_int'] + \
                dataframe[F'Furnace_{impact}_int']
        elif impact in ['ch4']:
            op_impacts = dataframe[F'Grid_{impact}_int'] + \
                dataframe[F'Grid_{impact}_leak_int'] + \
                dataframe[F'Furnace_{impact}_int'] + \
                dataframe[F'Furnace_{impact}_leak_int'] + \
                dataframe[F'CHP_{impact}_leak_int']
        else:
            op_impacts = dataframe[F'Grid_{impact}_int'] + \
                dataframe[F'Furnace_{impact}_int'] + \
                dataframe[F'CHP_{impact}_int']

        try:
            avoided_impact = dataframe[F'avoided_{impact}_int']
        except KeyError:
            avoided_impact = 0

        total_impact = op_impacts - avoided_impact

    return total_impact


def getDuplicateColumns(df):

    # Create an empty set
    duplicateColumnNames = set()

    # Iterate through all the columns
    # of dataframe
    for x in range(df.shape[1]):

        # Take column at xth index.
        col = df.iloc[:, x]

        # Iterate through all the columns in
        # DataFrame from (x + 1)th index to
        # last index
        for y in range(x + 1, df.shape[1]):

            # Take column at yth index.
            otherCol = df.iloc[:, y]

            # Check if two columns at x & y
            # index are equal or not,
            # if equal then adding
            # to the set
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])

    # Return list of unique column names
    # whose contents are duplicates.
    return list(duplicateColumnNames)


"""
REFERENCES
------------------------------------------------------------------------------------------------------
[1]     James, J.-A.; Thomas, V. M.; Pandit, A.; Li, D.; Crittenden, J. C. Water, Air Emissions,
        and Cost Impacts of Air-Cooled Microturbines for Combined Cooling, Heating, and Power Systems:
        A Case Study in the Atlanta Region. Engineering 2016, 2 (4), 470–480.
        https://doi.org/10.1016/J.ENG.2016.04.008.
[2]     U.S. Environmental Protection Agency. eGRID2018
        https://www.epa.gov/energy/emissions-generation-resource-integrated-database-egrid
        (accessed Apr 13, 2020).
[3]     NREL. 2020 Annual Technology Baseline; Golden, CO, 2020.
[4]     U.S. Environmental Protection Agency. 3.1 Stationary Gas Turbines. In AP-42: Compilation of
        Air Emissions Factors; Washington D.C., 2000.
[5]     US Environmental Protection Agency. Catalog of CHP Technologies. 2015.
[6]     U.S. Environmental Protection Agency. 1.4 Natural Gas Combustion. In AP-42: Compilation of
        Air Emissions Factors; Washington D.C., 1998.
[7]     Spath, P. L.; Mann, M. K. Life Cycle Assessment of a Natural Gas Combined-Cycle Power
        Generation System - National Renewable Energy Laboratory - NREL/TP-570-27715; Golden, CO, 2000.

[8]     Grubert, E. A.; Brandt, A. R. Three Considerations for Modeling Natural Gas System
        Methane Emissions in Life Cycle Assessment. J. Clean. Prod. 2019, 222, 760–767.
        https://doi.org/10.1016/j.jclepro.2019.03.096.
[9]     Alvarez, R. A.; Zavala-Araiza, D.; Lyon, D. R.; Allen, D. T.; Barkley, Z. R.; Brandt, A. R.;
        Davis, K. J.; Herndon, S. C.; Jacob, D. J.; Karion, A.; et al. Assessment of Methane Emissions
        from the U.S. Oil and Gas Supply Chain. Science (80-. ). 2018, eaar7204.
        https://doi.org/10.1126/science.aar7204.
[10]    Mann, M. K.; Whitaker, M.; Driver, T. PIER Project Report: Life Cycle
        Assessment of Existing and Emerging Distributed Generation Technologies in
        California; Golden, CO, 2011.
[11]    IPCC. Climate Change 2014: Synthesis Report; Pachauri, R. K., Meyer, L., Eds.;
        Intergovernmental Panel on Climate Change: Geneva, Switzerland, 2014.

"""


def building_pv(Building_, 
                City_, 
                PVSystem_, pv_deg_rate=0,
                Furnace_=None,
                AC_=None,
                year=0):
    df = pd.DataFrame()
    
    # Read building demands. All demands are in kWh
    df['electricity_demand'] = Building_.electricity_demand
    df['heat_demand'] = Building_.heat_demand
    df['cooling_demand'] = Building_.cooling_demand

    # Electrify building loads
    # df['heat_electricity'] = Building_.thermal_to_electricity(df.heat_demand, efficiency=Furnace_.efficiency)
    # df['cooling_electricity'] = Building_.thermal_to_electricity(df.cooling_demand, efficiency=AC_.efficiency)

    # Calculate Net electricity demand
    # df['net_electricity_demand'] = df.electricity_demand + df.heat_electricity + df.cooling_electricity
    
    # Design PV system for Peak Load
    PVSystem_ = size_pv(PVSystem_, 
                        peak_electricity=df.electricity_demand.max(),
                        method='peak')

    # Run PV supply
    pv_energy_output = pv_simulation(PVSystem_=PVSystem_, City_=City_) 
    df.index = pv_energy_output.index
    pv_energy_output['electricity_demand'] = df.electricity_demand
    pv_energy_output.to_csv(r'model_outputs\testing\pv_energy_output.csv')
    # Move to following year
    # Works up to here

    return pv_energy_output
