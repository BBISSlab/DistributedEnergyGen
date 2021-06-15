#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Osvaldo A. Broesicke'
__copyright__ = 'Copyright 2021. Osvaldo A. Broesicke. All Rights Reserved.'
__credits__ = ['Osvaldo A. Broesicke']
__license__ = ''
__maintainer__ = ['']
__email__ = ['obroesicke3@gatech.edu']
__version__ = '0.1'

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
############################################################################################

##########################
# DICTIONARIES AND LISTS #
##########################
# Below are lists and dictionaries used in the code. For a description of each. See Table 1.

'''
Table 1. Description of dictionaries and lists used by each of the classes. The data contained in each of these lists is used in class specific modules or functions
=============================       ======================================================================================================================================================
List or Dictionary Name             description
=============================       ======================================================================================================================================================
building_type_list                  List of representative building types simulated by the U.S. Department of Energy. The simulated energy demands and attributes for each of these
                                    buildings within its representative climate zone can be obtained via the Open Energy Information portal, see [2]
city_list                           List of the DoE's 16 representative cities for the United States, see [2]
city_building_dictionary            Dictionary linking each of the representative cities to the dictionary containing the simulated energy demands for each representative building. Each
                                    city dictionary links to a CSV file supplied by the OpenEI Database.
climate_zone_dictionary             Dictionary linking data for the climate zone of each representative city, see [3]
emm_region_dictionary               Dictionary linking each of the representative cities to their corresponding Electric Market Module (EMM) region. The projected energy-mix, fuel and
                                    electricity prices, levelized costs or electricity, and emissions are simulated for each EMM region by the Energy Information Administration (EIA),
                                    see [4]. While there is much overlap between the NERC subregions and the EMM regions, they do not always encompass the same geographic region.
floor_area_dictionary               Dictionary linking each of the representative buildings to their corresponging floor area, see [2]. Floor area is in square meters.                                    
nerc_region_dictionary              Dictionary linking each of the representative cities to their corresponding North American Electric Reliability Council (NERC) subregion. The
                                    grid-level emissions factors and grid-level losses are modeled by the Environmental Protection Agency (EPA) for each NERC subregion, see [5]
processed_tmy3_dictionary           Processed TMY3 files for use in pvlib. These files have been modified separately to avoid errors in reading data when using pvlib functions.
roof_dimensions_dictionary          Dictionary linking each of the representative buildings to their approximate roof dimensions, see [2]. Dimensions (L x W) are in meters.
roof_area_dictionary                Dictionary linking each of the representative buildings to their estimated roof areas, see [2]. Roof area is in square meters.
tmy3_city_dictionary                Dictionary of cities and their corresponding climate (Typical Meteorological Year, a.k.a. tmy3) CSV files, see [2]
=============================       ======================================================================================================================================================
'''
# Buildings and Building attributes
building_type_list = [  # 'single_family_residential',
    'full_service_restaurant', 'hospital', 'large_hotel', 'large_office',
    'medium_office', 'midrise_apartment', 'outpatient_healthcare', 'primary_school',
    'quick_service_restaurant', 'secondary_school', 'small_hotel', 'small_office',
    'stand_alone_retail', 'strip_mall', 'supermarket', 'warehouse']
floor_area_dictionary = {'midrise_apartment': 3135, 'small_office': 511,
                         'medium_office': 4982, 'large_office': 46320,
                         'small_hotel': 4014, 'large_hotel': 11345, 'supermarket': 4181,
                         'strip_mall': 2090, 'quick_service_restaurant': 232,
                         'warehouse': 4835, 'secondary_school': 19592,
                         'stand_alone_retail': 2294, 'hospital': 22422,
                         'full_service_restaurant': 569.52, 'outpatient_healthcare': 3804,
                         'primary_school': 6871, 'single_family_residential': 2271}
roof_dimensions_dictionary = {'midrise_apartment': [152, 55.5], 'small_office': [90.8, 60.5],
                              'medium_office': [163.8, 109.2], 'large_office': [240, 160],
                              'small_hotel': [54.9, 18.3], 'large_hotel': [284, 75], 'supermarket': [79.2, 52.8],
                              'strip_mall': [91.4, 22.9], 'quick_service_restaurant': [15.24, 15.24],
                              'warehouse': [100.6, 45.7], 'secondary_school': [140.2, 103.6],
                              'stand_alone_retail': [54.3, 42.4], 'hospital': [70.1, 53.3],
                              'full_service_restaurant': [22.6, 22.6], 'outpatient_healthcare': [73.0, 52.1],
                              'primary_school': [103.6, 82.3], 'single_family_residential': [36.3, 20.1]}
roof_area_dictionary = {'midrise_apartment': 784.66, 'small_office': 598.8,
                        'medium_office': 1661, 'large_office': 3563,
                        'small_hotel': 1003, 'large_hotel': 1477.55, 'supermarket': 4181,
                        'strip_mall': 2090, 'quick_service_restaurant': 258.86,
                        'warehouse': 4598.25, 'secondary_school': 11798.70,
                        'stand_alone_retail': 2294, 'hospital': 3739,
                        'full_service_restaurant': 511, 'outpatient_healthcare': 1373.29,
                        'primary_school': 6871, 'single_family_residential': 2271}

# City dictionaries and attributes
city_list = ['albuquerque', 'atlanta',  'baltimore',
             'chicago',
             'denver', 'duluth', 'fairbanks', 'helena',
             'houston', 'las_vegas', 'los_angeles', 'miami',
             'minneapolis', 'phoenix', 'san_francisco', 'seattle']
# Typical Meteorological Year 3 CSV files (Climate)
tmy3_city_dictionary = {'albuquerque': ['data\Tmy3_files\TMY3Albuquerque.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/723650TYA.CSV'],
                        'atlanta': ['data\Tmy3_files\TMY3Atlanta.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/722190TYA.CSV'],
                        'baltimore': ['data\Tmy3_files\TMY3Baltimore.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/724060TYA.CSV'],
                        'chicago': ['data\Tmy3_files\TMY3Chicago.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/725300TYA.CSV'],
                        'denver': ['data\Tmy3_files\TMY3Denver.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/725650TYA.CSV'],
                        'duluth': ['data\Tmy3_files\TMY3Duluth.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/727450TYA.CSV'],
                        'fairbanks': ['data\Tmy3_files\TMY3Fairbanks.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/702610TYA.CSV'],
                        'helena': ['data\Tmy3_files\TMY3Helena.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/727720TYA.CSV'],
                        'houston': ['data\Tmy3_files\TMY3Houston.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/722430TYA.CSV'],
                        'las_vegas': ['data\Tmy3_files\TMY3LasVegas.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/723860TYA.CSV'],
                        'los_angeles': ['data\Tmy3_files\TMY3LosAngeles.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/722950TYA.CSV'],
                        'miami': ['data\Tmy3_files\TMY3Miami.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/722020TYA.CSV'],
                        'minneapolis': ['data\Tmy3_files\TMY3Minneapolis.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/726580TYA.CSV'],
                        'phoenix': ['data\Tmy3_files\TMY3Phoenix.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/722780TYA.CSV'],
                        'san_francisco': ['data\Tmy3_files\TMY3SanFrancisco.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/724940TYA.CSV'],
                        'seattle': ['data\Tmy3_files\TMY3Seattle.csv', 'https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/727930TYA.CSV']}
processed_tmy3_dictionary = {'albuquerque': 'data\Tmy3_files\TMY3Albuquerque_Processed.csv',
                             'atlanta': 'data\Tmy3_files\TMY3Atlanta_Processed.csv',
                             'baltimore': 'data\Tmy3_files\TMY3Baltimore_Processed.csv',
                             'chicago': 'data\Tmy3_files\TMY3Chicago_Processed.csv',
                             'denver': 'data\Tmy3_files\TMY3Denver_Processed.csv',
                             'duluth': 'data\Tmy3_files\TMY3Duluth_Processed.csv',
                             'fairbanks': 'data\Tmy3_files\TMY3Fairbanks_Processed.csv',
                             'helena': 'data\Tmy3_files\TMY3Helena_Processed.csv',
                             'houston': 'data\Tmy3_files\TMY3Houston_Processed.csv',
                             'las_vegas': 'data\Tmy3_files\TMY3LasVegas_Processed.csv',
                             'los_angeles': 'data\Tmy3_files\TMY3LosAngeles_Processed.csv',
                             'miami': 'data\Tmy3_files\TMY3Miami_Processed.csv',
                             'minneapolis': 'data\Tmy3_files\TMY3Minneapolis_Processed.csv',
                             'phoenix': 'data\Tmy3_files\TMY3Phoenix_Processed.csv',
                             'san_francisco': 'data\Tmy3_files\TMY3SanFrancisco_Processed.csv',
                             'seattle': 'data\Tmy3_files\TMY3Seattle_Processed.csv'}
climate_zone_dictionary = {'albuquerque': '4B',
                           'atlanta': '3A',
                           'baltimore': '4A',
                           'chicago': '5A',
                           'denver': '5B',  # Denver is a proxy for Boulder
                           'duluth': '7',
                           'fairbanks': '8',
                           'helena': '6B',
                           'houston': '2A',
                           'las_vegas': '3B',
                           'los_angeles': '3B-CA',
                           'miami': '1A',
                           'minneapolis': '6A',
                           'phoenix': '2B',
                           'san_francisco': '3C',
                           'seattle': '4C'}
nerc_region_dictionary = {'albuquerque': 'AZNM',
                          'atlanta': 'SRSE',
                          'baltimore': 'RFCE',
                          'chicago': 'RFCW',
                          'denver': 'RMPA',
                          'duluth': 'MROW',
                          'fairbanks': 'AKGD',
                          'helena': 'NWPP',
                          'houston': 'ERCT',
                          'las_vegas': 'NWPP',
                          'los_angeles': 'CAMX',
                          'miami': 'FRCC',
                          'minneapolis': 'MROW',
                          'phoenix': 'AZNM',
                          'san_francisco': 'CAMX',
                          'seattle': 'NWPP'}
emm_region_dictionary = {'albuquerque': 'SRSG',
                         'atlanta': 'SRSE',
                         'baltimore': 'PJMD',
                         'chicago': 'PJMC',
                         'denver': 'RMRG',
                         'duluth': 'SPPN',
                         'fairbanks': 'NONE',
                         'helena': 'NWPP',
                         'houston': 'TRE',
                         'las_vegas': 'BASN',
                         'los_angeles': 'CASO',
                         'miami': 'FRCC',
                         'minneapolis': 'MISW',
                         'phoenix': 'SRSG',
                         'san_francisco': 'CANO',
                         'seattle': 'NWPP'}
hdd_dictionary = {'albuquerque': 2318,
                  'atlanta': 1511,
                  'baltimore': 2498,
                  'chicago': 3448,
                  'denver': 3143,
                  'duluth': 9818,
                  'fairbanks': 7167,
                  'helena': 4171,
                  'houston': 774,
                  'las_vegas': 1239,
                  'los_angeles': 652,
                  'miami': 68,
                  'minneapolis': 4224,
                  'phoenix': 543,
                  'san_francisco': 1566,
                  'seattle': 2555}
cdd_dictionary = {'albuquerque': 125,
                  'atlanta': 207,
                  'baltimore': 125,
                  'chicago': 61,
                  'denver': 56,
                  'duluth': 180,
                  'fairbanks': 0,
                  'helena': 6,
                  'houston': 475,
                  'las_vegas': 904,
                  'los_angeles': 0,
                  'miami': 667,
                  'minneapolis': 555,
                  'phoenix': 1383,
                  'san_francisco': 0,
                  'seattle': 0}

# Building files for each city
albuquerque_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_NM_Albuquerque.Intl.AP.723650_TMY3_BASE.csv',
                                   'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgHospitalNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgStripMallNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv',
                                   'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NM_Albuquerque.Intl.AP.723650_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_4B_USA_NM_ALBUQUERQUE.csv'
                                   }
atlanta_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3_BASE.csv',
                               'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgHospitalNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgStripMallNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv',
                               'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_3A_USA_GA_ATLANTA.csv'
                               }
baltimore_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3_BASE.csv',
                                 'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgHospitalNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgStripMallNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv',
                                 'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv'
                                 }
chicago_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3_BASE.csv',
                               'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgHospitalNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgStripMallNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv',
                               'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv'
                               }
denver_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_CO_Denver.Intl.AP.725650_TMY3_BASE.csv',
                              'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgHospitalNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgStripMallNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv',
                              'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CO_Denver.Intl.AP.725650_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv'
                              }
duluth_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_MN_Duluth.Intl.AP.727450_TMY3_BASE.csv',
                              'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgHospitalNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgStripMallNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv',
                              'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Duluth.Intl.AP.727450_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_7A_USA_MN_DULUTH.csv'
                              }
fairbanks_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_AK_Fairbanks.Intl.AP.702610_TMY3_BASE.csv',
                                 'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgHospitalNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgStripMallNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv',
                                 'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AK_Fairbanks.Intl.AP.702610_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_8A_USA_AK_FAIRBANKS.csv'
                                 }
helena_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_MT_Helena.Rgnl.AP.727720_TMY3_BASE.csv',
                              'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgHospitalNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgStripMallNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv',
                              'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MT_Helena.Rgnl.AP.727720_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_6B_USA_MT_HELENA.csv'
                              }
houston_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3_BASE.csv',
                               'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgHospitalNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgStripMallNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv',
                               'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv'
                               }
las_vegas_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3_BASE.csv',
                                 'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgFullServiceRestaurantNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgHospitalNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgLargeHotelNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgLargeOfficeNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgMediumOfficeNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgMidriseApartmentNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgOutPatientNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgPrimarySchoolNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgQuickServiceRestaurantNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgSecondarySchoolNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgSmallHotelNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgSmallOfficeNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgStand-aloneRetailNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgStripMallNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgSuperMarketNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv',
                                 'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3/RefBldgWarehouseNew2004_7.1_5.0_3B_USA_NV_LAS_VEGAS.csv'
                                 }
los_angeles_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_CA_Los.Angeles.Intl.AP.722950_TMY3_BASE.csv',
                                   'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgFullServiceRestaurantNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgHospitalNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgLargeHotelNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgLargeOfficeNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgMediumOfficeNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgMidriseApartmentNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgOutPatientNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgPrimarySchoolNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgQuickServiceRestaurantNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgSecondarySchoolNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgSmallHotelNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgSmallOfficeNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgStand-aloneRetailNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgStripMallNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgSuperMarketNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv',
                                   'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/RefBldgWarehouseNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv'
                                   }
miami_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_FL_Miami.Intl.AP.722020_TMY3_BASE.csv',
                             'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgHospitalNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgStripMallNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv',
                             'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_FL_Miami.Intl.AP.722020_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_1A_USA_FL_MIAMI.csv'
                             }
minneapolis_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3_BASE.csv',
                                   'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgHospitalNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgStripMallNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv',
                                   'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv'
                                   }
phoenix_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3_BASE.csv',
                               'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgHospitalNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgStripMallNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv',
                               'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_2B_USA_AZ_PHOENIX.csv'
                               }
san_francisco_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_CA_San.Francisco.Intl.AP.724940_TMY3_BASE.csv',
                                     'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgFullServiceRestaurantNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgHospitalNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgLargeHotelNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgLargeOfficeNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgMediumOfficeNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgMidriseApartmentNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgOutPatientNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgPrimarySchoolNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgQuickServiceRestaurantNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgSecondarySchoolNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgSmallHotelNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgSmallOfficeNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgStand-aloneRetailNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgStripMallNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgSuperMarketNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv',
                                     'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_CA_San.Francisco.Intl.AP.724940_TMY3/RefBldgWarehouseNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv'
                                     }
seattle_building_dictionary = {'single_family_residential': 'https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3_BASE.csv',
                               'full_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'hospital': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgHospitalNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'large_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgLargeHotelNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'large_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgLargeOfficeNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'medium_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgMediumOfficeNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'midrise_apartment': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgMidriseApartmentNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'outpatient_healthcare': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgOutPatientNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'primary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgPrimarySchoolNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'quick_service_restaurant': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgQuickServiceRestaurantNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'secondary_school': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'small_hotel': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgSmallHotelNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'small_office': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgSmallOfficeNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'stand_alone_retail': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgStand-aloneRetailNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'strip_mall': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgStripMallNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'supermarket': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgSuperMarketNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv',
                               'warehouse': 'https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3/RefBldgWarehouseNew2004_v1.3_7.1_4C_USA_WA_SEATTLE.csv'
                               }

city_building_dictionary = {'albuquerque': albuquerque_building_dictionary,
                            'atlanta': atlanta_building_dictionary,
                            'baltimore': baltimore_building_dictionary,
                            'chicago': chicago_building_dictionary,
                            'denver': denver_building_dictionary,  # Denver is a proxy for Boulder
                            'duluth': duluth_building_dictionary,
                            'fairbanks': fairbanks_building_dictionary,
                            'helena': helena_building_dictionary,
                            'houston': houston_building_dictionary,
                            'las_vegas': las_vegas_building_dictionary,
                            'los_angeles': los_angeles_building_dictionary,
                            'miami': miami_building_dictionary,
                            'minneapolis': minneapolis_building_dictionary,
                            'phoenix': phoenix_building_dictionary,
                            'san_francisco': san_francisco_building_dictionary,
                            'seattle': seattle_building_dictionary}

'''
------------------------------------
CLASSES AND CLASS SPECIFIC FUNCTIONS
------------------------------------

=============================       ===================================================================
Classes                             description
=============================       ===================================================================
City                                The City class is a subclass of pvlib's Location class. It contains 
                                    weather and location data and uses functions that are built into 
                                    pvlib.
Building                            The Building class contains general building attributes (e.g., floor
                                    area) and energy demand data. Each building object contains the 
                                    energy demands simulated for the DoE's reference buildings. These 
                                    demands are not simulated within this program.  
PrimeMover                          
AbsorptionChiller
AirConditioner
BatteryStorage
Grid                                This panel contains the Grid class. Each Grid object is meant to represent the electric grid of each city. 
Furnace                             The emission factors were gathered from the US EPA AP-42: Compilation of Air Emissions Factors. 
                                    The NOx, CO, and N2O emissions factors are dependent on existing NOx controls and/or system size. 
                                    By default, the system has NOx control systems.
===============================     ======================================================================================================================================================

===============================       ======================================================================================================================================================
Function                              description
===============================       ======================================================================================================================================================
list_pv_modules()                     List of the DoE's 16 representative cities for the United States, see [1]
list_inverters()
select_pv_system()
pv_simulation()
pivot_heat_map()
graph_heatmap()
size_pv()
generate_Cities()
tmy_read()
_generate_PrimeMover_dataframe()
size_cchpSys()
generate_AbsorptionChiller_dataframe
generate_AbsorptionChillers
size_ABC
_generate_AirConditioner_dataframe
generate_AirConditioner
size_AC
_generate_BES_dataframe
generate_BatteryStorage
generate_Grid_dataframes
generate_Grid
_generate_Furnace_dataframe
generate_Furnace
size_Furnace
===============================       ======================================================================================================================================================
'''

################
# PV Funcitons #
################
# Functions for Photovoltaic systems are not currently used in this model

def list_pv_modules():
    """
    This function lists pv modules in an interactive selction system.
    """
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    for module in sandia_modules:
        print('Module: {}'.format(module))
        print('Voltage, V: {:f}'.format(module['Vmpo']))
        print('Current, A: {:f}'.format(module['Impo']))
        print('Power, W: {:f}'.format(module['Vmpo'] * module['Impo']))


def list_inverters():
    """
    This function lists inverters in an interactive selction system.
    """
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    for inverter in sapm_inverters:
        print('Inverter: {}'.format(inverter))
        print('Voltage, V: {:f}'.format(inverter['Vdcmax']))
        print('Current, A: {:f}'.format(inverter['Idcmax']))
        print('Power, W: {:f}'.format(inverter['Vdcmax'] * inverter['Idcmax']))


def select_pv_system(module=None, inverter=None, surface_azimuth=180, name=None):
    """This function allows you to select a module and inverter for your system."""
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    if module is None:
        see_modules = input('Do you want to see a list of modules?: y/n')
        if see_modules == 'y':
            module_dict = list(sandia_modules.keys())
            for key in module_dict:
                print(key)
            # selected_module = input('Select module: ')
            module = sandia_modules[input('Select module: ')]
    else:
        module = sandia_modules[module]

    if inverter is None:
        see_inverters = input('Do you want to see a list of inverters?: y/n')
        if see_inverters == 'y':
            inverter_dict = list(sapm_inverters.keys())
            for key in inverter_dict:
                print(key)
            # selected_inverter = input('Select inverter: ')
            inverter = sapm_inverters[input('Select inverter: ')]
    else:
        inverter = sapm_inverters[inverter]

    if surface_azimuth == 0:
        surface_azimuth = float(
            input('Enter surface azimuth (degrees from N): '))

    pv_system = PVSystem(module=module, inverter=inverter,
                         surface_azimuth=surface_azimuth, name=name)

    return pv_system


def pv_simulation(PVSystem_, City_, tmy_filename=None):
    """
    This function runs the simulation for the energy produced from a PV system.
    """

    """
    1) READ CLIMATE DATA
    ====================
    Reads the tmy3 file from the City object. The metadata is also extracted and used to identify the timezone, latitute,
    longitute, and altitude of the site.
    """
    try:
        tmy_data = City_.tmy_data
        metadata = City_.metadata
    except ValueError:
        tmy_data, metadata = tmy_read(tmy_filename)
        City_ = City(name=metadata['Name'],
                     tmy_data=tmy_data,
                     metadata=metadata)

    # First define location and array parameters
    location = City_  # You may want to just use City_ instead of location

    """
    2) SET UP PV SYSTEM
    ===================
    The following functions set up the PV system. The functions take a PVSystem_ which contains the module
    parameters, inverter parameters, and the surface azimuth.
    Other parameters (e.g., albedo and surface type) are not currently functional.
    """
    # the PVSystem_ contains data on the module, inverter, and azimuth.
    pv_system = PVSystem_

    print('Running PV Simulation for {}'.format(location.name.upper()))

    # The surface_type_list is for a future iteration.
    # It will be used to calculate the ground albedo and subsequent reflected radiation.
    surface_type_list = ['urban', 'grass', 'fresh grass',
                         'snow', 'fresh snow', 'asphalt', 'concrete',
                         'aluminum', 'copper', 'fresh steel', 'dirty steel', 'sea']

    pv_system.surface_tilt = location.latitude  # Tilt angle of array

    # system['albedo'] = input('Input albedo (default is 0.25; typically 0.1-0.4 for surfaces on Earth)')
    # system['surface_type'] = input('To overwrite albedo, input surface type from {}'.format(surface_type_list))

    """
    3) IRRADIANCE CALCULATIONS
    ===========================
    The following functions simulate the solar position, extraterrastrial radiation, airmass, sky diffuse irradiance,
    ground diffuse irradiance, and the angle of incidence which are input into the following step to simulate electricity
    outputs.
    """

    # Calculate the solar position for all the times in the TMY file
    solpos = location.get_solarposition(tmy_data.index,
                                        temperature=tmy_data.DryBulb,
                                        pressure=tmy_data.Pressure)

    # Calculate the extraterrestrial radiation
    dni_extra = pvlib.irradiance.get_extra_radiation(tmy_data.index)
    dni_extra = pd.Series(dni_extra, index=tmy_data.index)

    # Calculate the airmass, makes dataframe with 'airmass_absolute'
    # and 'airmass_relative'
    airmass = location.get_airmass(times=tmy_data.index,
                                   solar_position=solpos)

    # Calculate the plane of array radiation. First need the angle of incidence, and the  sky diffuse and
    # ground diffuse radiation
    poa_sky_diffuse = pvlib.irradiance.haydavies(pv_system.surface_tilt,
                                                 pv_system.surface_azimuth,
                                                 tmy_data.DHI,
                                                 tmy_data.DNI,
                                                 dni_extra,
                                                 solpos['apparent_zenith'],
                                                 solpos['azimuth'])
    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(pv_system.surface_tilt,
                                                             tmy_data.GHI)  # ,albedo = system['albedo'], surface_type = system['surface_type'])
    aoi = pvlib.irradiance.aoi(pv_system.surface_tilt,
                               pv_system.surface_azimuth,
                               solpos['apparent_zenith'],
                               solpos['azimuth'])

    poa_irrad = pvlib.irradiance.poa_components(aoi,
                                                tmy_data.DNI,
                                                poa_sky_diffuse,
                                                poa_ground_diffuse)

    """
    4) ENERGY SIMULATION
    ====================
    The following functions calculate the energy output of the PV system. The simulation incorporates the efficiency losses
    from temperature increase, PV module efficiency, and the inverter efficiency (dc-ac conversion)
    """

    # Calculate the PV cell and module temperature
    pvtemps = pvlib.pvsystem.sapm_celltemp(poa_irrad['poa_global'],
                                           tmy_data.Wspd,
                                           tmy_data.DryBulb)

    # DC power generation
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_irrad.poa_direct,
                                                                    poa_irrad.poa_diffuse,
                                                                    airmass.airmass_absolute,  # change to absolute once you debug
                                                                    aoi,
                                                                    pv_system.module)

    # SAPM = Sandia PV Array Performance Model, generates a dataframe with short-circuit current,
    # current at the maximum-power point, open-circuit voltage, maximum-power point voltage and power
    dc_out = pvlib.pvsystem.sapm(effective_irradiance,
                                 pvtemps.temp_cell,
                                 pv_system.module)  # This will calculate the DC power output for a module

    ac_out = pd.DataFrame()
    # ac_out['p_ac'] is the AC power output in W from the DC power input.
    ac_out['p_ac'] = pvlib.pvsystem.snlinverter(
        dc_out.v_mp, dc_out.p_mp, pv_system.inverter)

    # p_ac/sqm is the AC power generated per square meter of module (W/m^2)
    ac_out['p_ac/sqm'] = ac_out.p_ac.apply(lambda x: x/pv_system.module.Area)

    print('PV simulation completed for {}'.format(location.name.upper()))

    return dc_out, ac_out


def size_pv(Building_, City_, PVSystem_,
            percent_roof_cover=100, percent_energy_demand=0,
            method='roof'):
    """
    This function adjusts the PVSystem object by calculating the number of PV modules, number of inverters, and number of
    strings required for the given building object. Future iterations will output area needed based on the % demand met.
    """
    module_area = PVSystem_.module['Area']
    # print('Module area: {}'.format(module_area))
    covered_roof_area = Building_.roof_area * percent_roof_cover/100

    num_modules = covered_roof_area // module_area
    # print('number of modules: {}'.format(num_modules))
    modules_per_string = int(PVSystem_.inverter.Vdcmax
                             / PVSystem_.module.Vmpo)

    PVSystem_.modules_per_string = modules_per_string
    PVSystem_.strings_per_inverter = 1

    num_inverter = num_modules // modules_per_string
    total_modules = num_inverter * modules_per_string

    # Nominal Power Output in W
    nominal_P_out = PVSystem_.module.Vmpo * PVSystem_.module.Impo

    # Solar PV Prices from:
    # NREL (National Renewable Energy Laboratory). 2019. 2019 Annual Technology
    # Baseline. Golden, CO: National Renewable Energy Laboratory.
    # https://atb.nrel.gov/electricity/2019
    if Building_.building_type == 'single_family_residential':
        # CAPEX includes construction and overnight capital cost in $/kW
        CAPEX = 2770
        # Annual OM cost in $/kW/yr
        annual_om_cost = 24
    else:
        # CAPEX includes construction and overnight capital cost in $/kW
        CAPEX = 1857
        # Annual OM cost in $/kW/yr
        annual_om_cost = 18

    # Capital Cost in $
    capital_cost_PV = CAPEX * nominal_P_out/1000
    # O&M cost in $/kWh
    om_cost_PV = annual_om_cost/(24*365)

    return PVSystem_, num_inverter, capital_cost_PV, om_cost_PV

##############
# City Class #
##############


class City(Location):
    '''
    City class is a subclass of pvlib's Location. Moreover, this class uses functions
    that are built into pvlib.

    Parameters
    ----------
    name: String

    nerc_region: None or string, default None
        If supplied, grid-level data will be extracted from the corresponding file

    tmy3_file: None or string, default None
        If supplied, attaches and extracts climate data using pv_lib functions

    tmy_data: None or dataframe, default None

    Location Parameters
    -------------------

    '''

    # Location class comes from pvlib and takes in (
    # latitude, longitude, tz ='UTC', altitude = 0, name = Non, **kwargs)
    def __init__(self, name, HDD=0, CDD=0, avg_RHum=0, nerc_region=None, tmy3_file=None, tmy_data=None, metadata=None):
        self.name = name
        self.nerc_region = nerc_region
        self.tmy3_file = tmy3_file
        self.climate_zone = climate_zone_dictionary[self.name]
        self.tmy_data = tmy_data
        self.metadata = metadata

    def _get_data(self, tmy3_file, how='processed'):

        self.tmy3_file = tmy3_file
        if how == 'processed':
            self.tmy_data = pd.read_csv(
                processed_tmy3_dictionary[self.name], index_col='datetime')
            metadata = pd.read_csv(
                'data\Tmy3_files\TMY3Metadata.csv', index_col='city')
            latitude = float(metadata.loc[self.name]['latitude'])
            longitude = float(metadata.loc[self.name]['longitude'])
            timezone = float(metadata.loc[self.name]['TZ'])
            altitude = float(metadata.loc[self.name]['altitude'])
            USAF = int(metadata.loc[self.name]['USAF'])

            Location.__init__(self, latitude, longitude,
                              timezone, altitude, self.name)
        elif how == 'tmy_read':
            try:
                self.tmy_data, self.metadata = tmy_read(self.tmy3_file)
                Location.__init__(self, self.metadata['latitude'],
                                  self.metadata['longitude'],
                                  self.metadata['TZ'],
                                  self.metadata['altitude'],
                                  self.name)
            except KeyError:
                print("Skipped {} \n".format(self.tmy3_file))
                pass
        self.HDD, self.CDD, self.avg_RHum = self._calc_degreedays()

    def _calc_degreedays(self):
        df = self.tmy_data
        df = df[['DryBulb', 'RHum']].copy()
        df.index = pd.to_datetime(df.index)
        df = df.resample('d').mean()
        df['HDD'] = 0
        df['CDD'] = 0
        df.HDD = np.where((df.DryBulb < 18), 18 - df.DryBulb, 0)
        df.CDD = np.where((df.DryBulb > 24), df.DryBulb - 24, 0)
        HDD = df.HDD.sum()
        CDD = df.CDD.sum()
        avg_RHum = df.RHum.mean()
        return HDD, CDD, avg_RHum


def tmy_read(tmy_city_name):
    """
    This function was initially generated to read raw TMY3 files from a CSV file or directly from the URL database.
    The NREL webpage has been down (July 6, 2020), and the built in parser in pvlib.iotools.read_tmy3() seems to have
    issues reading some of the files.

    For the moment (July 6, 2020) This function will not be used, and all of the data will be input from processed
    tmy3 files and the metadata will be
    """

    # This function reads and processes the tmy3 data.
    tmy_filename = tmy_city_name[0]
    tmy_url = tmy_city_name[1]

    try:
        print('Attempt 1: Read from filename')
        pvlib_abspath = os.path.dirname(
            os.path.abspath(inspect.getfile(pvlib)))
        datapath = os.path.join(pvlib_abspath, 'data', tmy_filename)

        # read tmy data with year values coerced to a single year
        tmy_data, metadata = pvlib.iotools.read_tmy3(
            datapath, coerce_year=2020)

    except (OSError, ValueError):
        print('Attempt 2: Reading from URL')
        tmy_data, metadata = pvlib.iotools.read_tmy3(
            filename=tmy_url, coerce_year=2020)

    except KeyError:
        # print("Skipped {}".format(tmy_city_name))
        pass

    tmy_data = tmy_data.shift(freq='-30Min')  # ['2015']
    tmy_data.index.name = 'Time'

    """
    The following lines of code are a "fix" the final row of the read tmy file. When the date is coerced to 2015, the
    final row is 2015-01-01. When the time is shifted back 30 minutes, the row is in the previous year, which causes issues
    in pandas, since the dates are out of order. Accordingly, these following lines:
    1) Copy the data from the last row
    2) Drop the final row of the tmy_data DataFrame with the incorrect index
    3) Create a new dataframe with the shifted date and copied data, and
    4) Append the row to the tmy_data DataFrame

    This correction was done in this way because the nature of the index, with the inclusion of the timezone, makes the
    index immutable. The process is tedius, but will hopefully be fixed in the following PVlib iteration.
    """
    # Identify the last row, and copy the data
    last_date = tmy_data.iloc[[-1]].index
    copied_data = tmy_data.iloc[[-1]]
    copied_data = copied_data.to_dict(orient='list')
    # Drop the row with the incorrect index
    tmy_data.drop([last_date][0], inplace=True)
    # Shift the date of the last date
    # had n = 1, inside shift function
    last_date = last_date.shift(periods=1, freq='A')
    # Create a new DataFrame with the shifted date and copied data
    last_row = pd.DataFrame(data=copied_data,
                            index=[last_date][0])

    tmy_data = tmy_data.append(last_row)

    return tmy_data, metadata


def _generate_Cities(all_cities=True, selected_cities=[], how='processed'):
    """
    This function generates all of my City objects from the selected tmy3 files.  It returns a dictionary with the city
    ames as the key and the value as the City object.
    """
    City_dictionary = {}

    if all_cities is True:
        for city in tmy3_city_dictionary:
            City_dictionary[city] = City(name=city,
                                         nerc_region=nerc_region_dictionary[city],
                                         tmy3_file=tmy3_city_dictionary[city])
            City_dictionary[city]._get_data(
                City_dictionary[city].tmy3_file, how)
    else:
        for city in selected_cities:
            print(selected_cities)
            print(city)
            City_dictionary[city] = City(name=city,
                                         nerc_region=nerc_region_dictionary[city],
                                         tmy3_file=tmy3_city_dictionary[city])
            City_dictionary[city]._get_data(
                City_dictionary[city].tmy3_file, how)
    ''' City_dictionary[selected_cities] = City(name=selected_cities, nerc_region=nerc_region_dictionary[selected_cities], tmy3_file=tmy3_city_dictionary[selected_cities]) City_dictionary[selected_cities]._get_data(City_dictionary[selected_cities].tmy3_file, how)'''
    return City_dictionary

################################
# End City Class and Functions #
################################

##################
# Building Class #
##################


class Building:
    def __init__(self, name, building_type, city_name='', City_=None,
                 demand_file=None, _get_data=True):
        self.name = name
        self.building_type = building_type
        self.City_ = City_

        try:
            self.city_name = self.City_.name
            self.demand_file = city_building_dictionary[self.city_name][building_type]
            # print('Read from object')
        except KeyError:
            self.city_name = city_name

        # Areas are in square meters
        self.floor_area = floor_area_dictionary[self.building_type]
        self.roof_area = roof_area_dictionary[self.building_type]

        if demand_file is None:
            self.demand_file = city_building_dictionary[self.city_name][building_type]
        else:
            self.demand_file = demand_file

        if _get_data:
            self._get_data()

    def __repr__(self):
        attrs = ['name', 'building_type',
                 'city_name', 'floor_area', 'roof_area']
        return ('Building: \n ' + ' \n '.join('{}: {}'.format(attr, getattr(self, attr))
                                              for attr in attrs))

    def _get_data(self):
        """
        This is a Building method to read the EnergyPlus files and extract the data as attributes of the Building class.
        """
        df = pd.read_csv(self.demand_file)

        if not self.City_ is None:
            df.index = self.City_.tmy_data.index

        # Single family residential files have mismatched column labels
        if self.building_type == 'single_family_residential':
            self.electricity_demand = (df['Electricity:Facility [kW](Hourly)']
                                       - df['Heating:Electricity [kW](Hourly)']
                                       - df['Cooling:Electricity [kW](Hourly)']
                                       - df['Water Heater:WaterSystems:Electricity [kW](Hourly) '])
            # - df['HVACFan:Fans:Electricity [kW](Hourly)'])
            self.heat_demand = (df['Heating:Electricity [kW](Hourly)']
                                + df['Heating:Gas [kW](Hourly)']
                                + df['Water Heater:WaterSystems:Electricity [kW](Hourly) '])
        else:  # If not residential, its a commercial file.
            self.electricity_demand = (df['Electricity:Facility [kW](Hourly)']
                                       - df['Heating:Electricity [kW](Hourly)']
                                       - df['Cooling:Electricity [kW](Hourly)'])
            # Strip mall, warehouse, and secondary school do not have a water heater load.
            try:
                self.heat_demand = (df['Heating:Electricity [kW](Hourly)']
                                    + df['Heating:Gas [kW](Hourly)']
                                    + df['Water Heater:WaterSystems:Gas [kW](Hourly)'])
            except KeyError:
                self.heat_demand = (df['Heating:Electricity [kW](Hourly)']
                                    + df['Heating:Gas [kW](Hourly)'])

        # Multiply the electric cooling demand by the COP of the AC system to get the cooling demand.
        self.cooling_demand = (df['Cooling:Electricity [kW](Hourly)'].apply(
            lambda x: x*3.8))  # need to do some algebra here

        # You cannot calculate the thermal demand until you have a sized abs chiller.
        # self.thermalDemand = self.heat_demand + self.cooling_demand

        df.fillna(value=0, inplace=True)
        # Add the demand onto the dataframe just to have data aggregated.
        df['aggregated_electricity_demand_kW'] = self.electricity_demand
        df['aggregated_heat_demand_kW'] = self.heat_demand
        df['aggregated_cooling_demand_kW'] = self.cooling_demand

        # Calculated demand intensities are in kW/sq m
        df['electricity_demand_intensity_kW_m-2'] = self.electricity_demand / \
            self.floor_area
        df['heat_demand_intensity_kW_m-2'] = self.heat_demand / self.floor_area
        df['cooling_demand_intensity_kW_m-2'] = self.cooling_demand / self.floor_area

        self.dataframe = df

        return self.dataframe

    def get_thermal_demand(self, COP_absCh=0, absCh_cap=0):
        """
        This function checks if we have an absorption chiller to meet the cooling load. If one is present, the cooling 
        demand (currently electric) is converted into a heating demand. The heat and thermo-cooling demand are then added 
        for a net heating demand.
        """
        if COP_absCh == 0:
            self.thermalDemand = self.heat_demand
            self.electricity_demand = self.electricity_demand + self.cooling_demand
        elif COP_absCh > 0:
            self.thermalDemand = self.heat_demand + self.cooling_demand/COP_absCh
        else:
            raise ValueError
        return self.thermalDemand

        # if absCh_cap is None:
        # heat_demand_absCh = self.cooling_demand / COP_absCh
        # self.thermalDemand = self.heat_demand + heat_demand_absCh
        # self.dataframe['aggregated_thermalDemand_kW'] = self.thermalDemand
        # self.dataframe['thermalDemand_intensity_kW_m-2'] = self.thermalDemand / self.floor_area

        # else:
        # num_absCh = max(self.cooling_demand) / absCh_cap
        # You will need to adjust this later to assume the variability of COP

    def peak_electricity(self):
        return self.electricity_demand.max()

    def peak_heat(self):
        return self.heat_demand.max()

    def peak_cooling(self):
        return self.cooling_demand.max()

    def peak_thermal(self):
        return self.thermalDemand.max()

    def graph_demands(self, method=None, electricity=True, heat=True, cooling=True, thermal=False):
        """
        This method graphs the various demands of the building. Upcoming updates to the method include:
        - total vs the intensity (per sq m)
        - combined
        - bar vs line
        """

        df = self.dataframe

        sns.set_context('talk', font_scale=8)

        if electricity:
            # sns.
            # ax = sns.lineplot(x=df.index, y=self.electricity_demand, data=df)
            pass

        if heat:
            # ax1 = sns.lineplot(x = 'Time', y = 'heat_demand_kW', data = df)
            pass

        if cooling:
            # ax2 = sns.lineplot(x = 'Time', y = 'cooling_demand_kW', data = df)
            pass

        if thermal:
            # ax3 = sns.lineplot(x = 'Time', y = 'thermalDemand_kW', data = df)
            pass

    def thermal_to_electricity(self, thermal_energy_demand, efficiency=1):
        '''Returns the equivalent electricity required to supply the input energy demand'''
        return thermal_energy_demand / efficiency

####################################
# End Building Class and Functions #
####################################

####################
# PrimeMover Class #
##################


class PrimeMover:
    def __init__(self, PM_id, model='', technology=None, power_nom=0., heat_nom=0, fuel_nom=0,
                 capital_cost=0., om_cost=0, carbon_monoxide=0, carbon_dioxide=0, nox=0, voc=0, water_for_energy=0,
                 embedded_ch4=0, embedded_co2=0, embedded_n2o=0,
                 electric_efficiency_LHV=1,  electric_efficiency_HHV=1, thermal_efficiency=1,
                 chp_efficiency_LHV=1, chp_efficiency_HHV=1, effective_efficiency=1,
                 heat_rate=1, phr=1, exhaust_temperature=20, heat_recovery_type='hot_water', abc_compatibility=0,
                 lifetime=20, age=0):
        self.PM_id = PM_id
        self.model = model
        self.technology = technology

        # Lifetime and age are in years.
        # USDoE and NRDC state that CHPs have up to 20 years of service life.
        self.lifetime = lifetime
        self.age = age

        # System nominal capacity is in kW
        self.power_nom = power_nom
        self.fuel_nom = fuel_nom
        self.phr = phr
        self.hpr = 1/self.phr

        if heat_nom == 0:
            self.heat_nom = power_nom * self.hpr
        else:
            self.heat_nom = heat_nom

        # All costs are in 2013$ / kW
        self.capital_cost = capital_cost
        self.om_cost = om_cost

        # All emissions are in kg/MWh = g/kWh
        self.co = float(carbon_monoxide)
        self.co2 = float(carbon_dioxide)
        self.nox = float(nox)
        self.voc = float(voc)

        # Embedded emissions are in g
        self.embedded_ch4 = float(embedded_ch4)
        self.embedded_co2 = float(embedded_co2)
        self.embedded_n2o = float(embedded_n2o)

        # Water for Energy is in L/kWh
        self.w4e = water_for_energy

        # All efficiencies and ratios are unitless
        self.electric_efficiency_LHV = electric_efficiency_LHV
        self.electric_efficiency_HHV = electric_efficiency_HHV
        self.thermal_efficiency = thermal_efficiency
        self.effective_efficiency = effective_efficiency
        self.chp_efficiency_LHV = chp_efficiency_LHV
        self.chp_efficiency_HHV = chp_efficiency_HHV
        self.heat_rate = heat_rate

        # Exhaust Characteristics
        self.exhaust_temperature = exhaust_temperature  # deg C
        self.heat_recovery_type = heat_recovery_type
        # 0: Only compatible with single-stage absorption chiller; 1: Compatible with single-stage and two-stage absorption chiller
        self.abc_compatibility = abc_compatibility

    def __repr__(self):
        attrs = ['PM_id', 'model', 'technology', 'lifetime', 'age',
                 'power_nom', 'heat_nom', 'fuel_nom', 'capital_cost', 'om_cost',
                 'co', 'co2', 'nox', 'voc', 'w4e',
                 'electric_efficiency_LHV', 'electric_efficiency_HHV', 'thermal_efficiency', 'chp_efficiency_LHV', 'chp_efficiency_HHV',
                 'heat_rate', 'phr', 'hpr',
                 'exhaust_temperature', 'heat_recovery_type', 'abc_compatibility']
        return ('Prime Mover: \n ' + ' \n '.join('{}: {}'.format(attr, getattr(self, attr)) for attr in attrs))

    def _get_data(self, dataframe, sheet_name=None, index=0):
        """
        This method extracts data from a dataframe or an csv file to populate the attributes of each prime mover.

        The Current CSV File has impact units in kg/MWh. 
        """

        self.PM_id = dataframe.iloc[index]['PM_id']
        self.model = dataframe.iloc[index]['model']
        self.technology = dataframe.iloc[index]['technology']

        # System capacity is in kW
        self.power_nom = dataframe.iloc[index]['capacity_kW']

        # All costs are in 2013$ / kW
        self.capital_cost = dataframe.iloc[index]['capital_cost']
        self.om_cost = dataframe.iloc[index]['om_cost']

        # All emissions from the specs sheet are in are in kg/MWh, or g/kWh
        self.co = dataframe.iloc[index]['carbon_monoxide']
        self.co2 = dataframe.iloc[index]['carbon_dioxide']
        self.nox = dataframe.iloc[index]['nox']
        self.voc = dataframe.iloc[index]['voc']

        # Embedded emissions from specs sheet are in g
        # The embodied emissions for the CHP system were obtained from the Ecoinvent 3 database
        self.embedded_ch4 = dataframe.iloc[index]['embedded_ch4']
        self.embedded_co2 = dataframe.iloc[index]['embedded_co2']
        self.embedded_n2o = dataframe.iloc[index]['embedded_n2o']

        # Water for Energy is in L/MWh
        # self.w4e = water_for_energy

        # All efficiencies and ratios are unitless
        self.electric_efficiency_LHV = dataframe.iloc[index]['electric_EFF_LHV']
        self.electric_efficiency_HHV = dataframe.iloc[index]['electric_EFF_HHV']
        self.thermal_efficiency = dataframe.iloc[index]['thermal_EFF']
        self.effective_efficiency = dataframe.iloc[index]['effective_electric_EFF']
        self.chp_efficiency_LHV = dataframe.iloc[index]['chp_EFF_LHV']
        self.chp_efficiency_HHV = dataframe.iloc[index]['chp_EFF_HHV']
        self.heat_rate = dataframe.iloc[index]['thermal_out_fuel_in_ratio']
        self.phr = dataframe.iloc[index]['phr']
        self.hpr = 1/self.phr

        # converted to kW from mmBtu/hr
        self.heat_nom = dataframe.iloc[index]['heat_output']*(10**6/3412.14)
        self.fuel_nom = dataframe.iloc[index]['fuel_input']*(10**6/3412.14)

        # Converted to deg C from F
        self.exhaust_temperature = (
            dataframe.iloc[index]['exhaust_temp'] - 32.)*(5./9.)
        # Heat Recovery Characteristics
        self.heat_recovery_type = dataframe.iloc[index]['heat_recovery_type']
        self.abc_compatibility = dataframe.iloc[index]['abc_compatibility']

        # These two are added so other functions are still working even without the derate
        self.min_capacity = self.power_nom.min()
        self.min_heatoutput = self.heat_nom.min()

    def derate(self, City_, altitude=0, dry_bulb_temp=20, pressure=1, relative_humidity=0):
        """
        This method calculates the capacity and efficiency of the prime mover due to the variation of altitude (air 
        pressure), temperature, and humidity. The capacity derate from variation in temperature and altitude are based on 
        linear regression performed by R.E. Best et al (2015). Humidity is incorporated by adjusting the heat released in 
        the combustion process. Humidity portion is pending. According to S.F. Al Fahed et al (2009), the effect of relative
        humidity on cogeneration is negligible.
        """

        # Check if tmy3 data has been read. If not, it will read it.
        altitude = City_.altitude  # meters
        dry_bulb_T = City_.tmy_data.DryBulb  # deg C
        pressure = City_.tmy_data.Pressure  # bar
        relative_humidity = City_.tmy_data.RHum
        # tmy_data, tmy_metadata = tmy_read(tmy3_file)

        derated_df = pd.DataFrame()

        # CF = correction factor
        CF_altitude = ((-1 / 300.) * altitude + 100) / (100)

        # print(type(self.power_nom))
        # print(type(CF_altitude))

        derated_df['capacityCF_temperature'] = dry_bulb_T.apply(lambda x: ((-1 / 6.) * x
                                                                           + 100) / 100)
        derated_df['efficiencyCF_temperature'] = dry_bulb_T.apply(lambda x: ((-9 / 250) * x
                                                                             + 110) / 100)
        derated_df['power_nom'] = derated_df.capacityCF_temperature.apply(lambda x:
                                                                          x * self.power_nom * CF_altitude
                                                                          )

        self.derated_capacity_df = derated_df['power_nom']

        # Will add to ouput later
        derated_df['heat_nom'] = derated_df['power_nom'].apply(
            lambda x: x * self.hpr)

        derated_df.index = City_.tmy_data.index

        self.minimum_capacity = derated_df.power_nom.min()
        self.min_heatoutput = self.minimum_capacity * self.hpr
        minimum_electrical_capacity = derated_df.power_nom.min()

        return minimum_electrical_capacity

    def size_system(self, peak_electricity_demand, peak_thermal_demand, prime_mover, city=None, operation_mode='FTL'):
        """
        This method returns the number of CHP units that will be required to meet the peak electric or thermal load. For 
        UPDATE TASK: Make the sizing system able to incorporate different engine size.
        """
        if operation_mode == 'FEL':
            num_engines = peak_electricity_demand/derated_pm_capacity

        else:
            derated_pm_heat = derated_pm_capacity * prime_mover.hpr
            num_engines = peak_thermal_demand/derated_pm_heat

    def get_impacts(self, electricity_demand):
        """
        Incomplete method that will output the CO, CO2, NOx, VOC, and Water for Energy (w4e) impacts
        """
        try:
            carbon_monoxide = electricity_demand * self.co2
        except AttributeError:
            carbon_monoxide = 9999
        try:
            carbon_dioxide = electricity_demand * self.co2
        except AttributeError:
            carbon_dioxide = 9999
        try:
            nox = electricity_demand * self.nox
        except AttributeError:
            nox = 9999
        try:
            voc = electricity_demand * self.voc
        except AttributeError:
            voc = 9999
        # try:
            # water_for_energy = electricity_demand * self.water_for_energy
        # except KeyError: #Verify the type of error you get
            # water_for_energy =  9999

        return carbon_monoxide, carbon_dioxide, nox, voc  # , water_for_energy


def _generate_PrimeMover_dataframe(csv_file, sheet_name=None, header=0):
    """
    This function reads my CSV file which documents typical parameters for each prime mover. It corrects the type of data 
    read (i.e., string, integer, float), and inserts them into a pandas dataframe.    
    This function is used on the following function which reads the processed data to generate Prime Mover objects.
    """

    dataframe = pd.read_csv(filepath_or_buffer=csv_file,
                            header=header,
                            dtype={'PM_id': 'object',
                                   'model': 'object',
                                   'technology': 'object',
                                   'technology_2': 'object',
                                   'capacity_kW': 'float64',
                                   'capital_cost': 'float64',
                                   'om_cost': 'float64',
                                   'carbon_monoxide': 'float64',
                                   'carbon_dioxide': 'float64',
                                   'nox': 'float64',
                                   'voc': 'float64',
                                   'embedded_ch4': 'float',
                                   'embedded_co2': 'float',
                                   'embedded_n2o': 'float',
                                   'electric_EFF_LHV': 'float64',
                                   'electric_EFF_HHV': 'float64',
                                   'effective_electric_EFF': 'float64',
                                   'chp_EFF_LHV': 'float64',
                                   'chp_EFF_HHV': 'float64',
                                   'thermal_out_fuel_in_ratio': 'float64',
                                   'phr': 'float64',
                                   'fuel_input': 'float64',
                                   'heat_output': 'float64',
                                   'exhaust_temp': 'float64'})
    # dataframe.fillna(value = 0.)

    return dataframe


def _generate_PrimeMovers(csv_file, sheet_name=None, header=0):
    """
    This function generates Prime Mover Objects from the CSV file. 
    """

    dataframe = _generate_PrimeMover_dataframe(csv_file=csv_file,
                                               sheet_name=sheet_name,
                                               header=header)

    dataframe.drop(dataframe[dataframe['technology'] ==
                             'Steam Turbine'].index, inplace=True)
    dataframe.reset_index(inplace=True)
    # Need to add an If statement indicating what type of prime mover I have
    i = 0
    PrimeMover_dictionary = {}
    while i < dataframe.PM_id.count():

        # Create new PrimeMover object
        PrimeMover_ = PrimeMover(dataframe.loc[i].PM_id)
        PrimeMover_._get_data(dataframe=dataframe, index=i)
        PrimeMover_dictionary[PrimeMover_.PM_id] = PrimeMover_
        i += 1
    return PrimeMover_dictionary

# Consider making a new object which takes in a pm object and an abs. chiller


def size_chp(PrimeMover_,
             peak_electricity_load,
             peak_thermal_load,
             City_=None,
             operation_mode='FTL',
             alpha=0):
    """
    The purpose of this function is to size the prime mover system to the building demand. In other words, how many 
    absportion chillers or air conditioning units do we need to meet the cooling demand; and, subsequently, how many prime 
    movers do we need to meet the electric or thermal demand?    
    These systems are sized based on the operation mode (i.e., Follow the Thermal Load, Follow the Electric Load, or 
    Follow the Base Load).
    - 'FTL' = Follow the Thermal Load (sized to match the peak thermal demand)
    - 'alpha' = The CHP is sized to meet a fraction of the electric load [0-1]. If alpha=1, CHP can meet the full electric load.    
    """
    # Because we may modify the Prime Mover, we make a copy, that will be derated for the city.
    CHP_ = PrimeMover_

    if City_ is None:
        minimum_electrical_capacity = CHP_.power_nom
    else:
        minimum_electrical_capacity = CHP_.derate(City_)

    if operation_mode == 'FTL':
        design_peak = peak_thermal_load
        # design capacity in this case is heat output
        design_capacity = CHP_.min_heatoutput

    if operation_mode == 'alpha':
        design_peak = alpha * peak_electricity_load
        # design capacity in this case is electricity output
        design_capacity = minimum_electrical_capacity

    number_PM = m.ceil(design_peak / design_capacity)

    capital_cost_PM = number_PM*CHP_.power_nom * \
        CHP_.capital_cost

    return CHP_, number_PM, capital_cost_PM


######################################
# End PrimeMover Class and Functions #
######################################

###########################
# AbsorptionChiller Class #
###########################

class AbsorptionChiller:
    def __init__(self, ABC_id, technology='single_stage', heat_input_type='hot water',
                 cooling_capacity=175.95, COP=0.7, capital_cost=0, om_cost=0,
                 lifetime=20, age=0):
        self.ABC_id = ABC_id
        self.COP = COP
        self.technology = technology

        # Capacity rating is in kW
        self.capacity = cooling_capacity

        # Capital cost in $, variable cost is size dependent
        self.capital_cost = capital_cost
        self.om_cost = om_cost

        # Embodied impacts are in g
        self.embedded_ch4 = self.capacity * 20.4
        self.embedded_co2 = self.capacity * 265226
        self.embedded_n2o = self.capacity * 11.2

        # Lifetime is in years
        self.lifetime = lifetime
        self.age = age
        # self.fixedMaintenance = fixedMaintenance

    def __repr__(self):
        attrs = ['ABC_id', 'technology', 'COP', 'capacity',
                 'capital_cost', 'om_cost', 'lifetime']
        return ('Absorption Chiller: \n ' + ' \n '.join('{}: {}'.format(attr, getattr(self, attr)) for attr in attrs))

    def cooling_output(self, heat_input):
        return heat_input * self.COP

    def get_heat_demand(self, cooling_demand):
        return cooling_demand / self.COP

    def _get_data(self, dataframe, sheet_name=None, index=0):
        """
        This method extracts data from a dataframe or an csv file to populate the attributes of each absorption chiller.
        """

        self.ABC_id = dataframe.iloc[index]['ABC_id']
        self.technology = dataframe.iloc[index]['technology']
        self.heat_input_type = dataframe.iloc[index]['heat_input_type']

        # System capacity is in kW
        self.capacity = dataframe.iloc[index]['capacity_kW']

        # No part-load COP data found
        self.COP = dataframe.iloc[index]['COP_full_load']

        # Lifetime is in years
        self.lifetime = dataframe.iloc[index]['avg_life']

        # All costs are in 2013$/kW
        self.capital_cost = dataframe.iloc[index]['capital_cost']
        self.om_cost = dataframe.iloc[index]['om_cost']


def _generate_AbsorptionChiller_dataframe(csv_file, sheet_name=None, header=0):
    """
    This function reads my CSV file which documents typical parameters for each prime mover. It corrects the type of data 
    read (i.e., string, integer, float), and inserts them into a pandas dataframe. This function is used on the following 
    function which reads the processed data to generate Absorption Chiller objects.
    """

    dataframe = pd.read_csv(filepath_or_buffer=csv_file,
                            header=header,
                            dtype={'ABC_id': 'object',
                                   'technology': 'object',
                                   'energy_input_type': 'object',
                                   'capacity_kW': 'float64',
                                   'COP_full_load': 'float64',
                                   'avg_life': 'Int64',
                                   'capital_cost': 'float64',
                                   'om_cost': 'float64'})
    dataframe.fillna(value=0.)
    return dataframe


# End AbsorptionChiller Methods #
#################################


def _generate_AbsorptionChillers(csv_file, sheet_name=None, header=1):
    """
    This function generates Absorption Chiller Objects from the CSV file. 
    """
    dataframe = _generate_AbsorptionChiller_dataframe(csv_file=csv_file,
                                                      sheet_name=sheet_name,
                                                      header=header)
    i = 0
    AbsorptionChiller_dictionary = {}
    while i < dataframe.ABC_id.count():
        # Create new Absorption Chiller object
        current_ABC = AbsorptionChiller(dataframe.ABC_id[i])
        current_ABC._get_data(dataframe=dataframe, index=i)
        AbsorptionChiller_dictionary[current_ABC.ABC_id] = current_ABC
        i += 1
    return AbsorptionChiller_dictionary


def size_ABC(peak_demand=0., AbsorptionChiller_=None, beta=0):
    """
    The purpose of this function is to size the absorption chiller system to the building demand. In other words, how many 
    absportion chillers do we need to meet the cooling demand? I assume that the system is sized to meet the coling demand.
    """
    design_peak = beta * peak_demand
    number_ABC = m.ceil(design_peak/AbsorptionChiller_.capacity)
    # ABC capital cost given in $/kW
    capital_cost_ABC = number_ABC*AbsorptionChiller_.capacity * \
        AbsorptionChiller_.capital_cost

    return number_ABC, capital_cost_ABC

#############################################
# End AbsorptionChiller Class and Functions #
#############################################

########################
# AirConditioner Class #
########################


class AirConditioner:
    def __init__(self, AC_id, technology=None, building_type='all', climate_zone='all',
                 lifetime=20, age=0, cooling_capacity=0,
                 SEER=13, COP=3.8, COP_PLR=0,
                 capital_cost=0, om_cost=0):
        # Basic data
        self.AC_id = AC_id
        self.technology = technology
        self.building_type = building_type
        self.climate_zone = climate_zone
        # Average service life is in years
        self.lifetime = lifetime
        self.age = age
        # Capacity rating is in kW
        self.capacity = cooling_capacity
        # Efficiency ratings
        self.SEER = SEER
        self.COP = COP
        self.COP_PLR = COP_PLR
        # Capital cost in $, variable cost is size dependent
        self.capital_cost = capital_cost
        self.om_cost = om_cost
        # Embedded emissions are in g
        self.embedded_ch4 = 0
        self.embedded_co2 = 0
        self.embedded_n2o = 0

    def cooling_output(self, electricity_input):
        return electricity_input * self.COP

    def get_electricity_demand(self, cooling_demand):
        return cooling_demand / self.COP

    def _get_data(self, dataframe, sheet_name=None, index=0):
        """
        This method extracts data from a dataframe or an csv file to populate the attributes of each air conditioning or 
        chiller system.
        """
        self.AC_id = dataframe.iloc[index]['AC_id']
        self.technology = dataframe.iloc[index]['technology']
        self.building_type = dataframe.iloc[index]['building']
        self.climate_zone = dataframe.iloc[index]['climate']
        self.lifetime = dataframe.iloc[index]['avg_life']
        # System capacity is in kW
        self.capacity = dataframe.iloc[index]['capacity_kW']
        # Efficiency metrics
        self.SEER = dataframe.iloc[index]['SEER']
        self.COP = dataframe.iloc[index]['COP_full_load']
        self.COP_PLR = dataframe.iloc[index]['COP_PLR']
        # Capital cost is in 2017$/kW; O&M costs are in 2017$/kWh
        self.capital_cost = dataframe.iloc[index]['capital_cost']
        self.om_cost = dataframe.iloc[index]['om_cost']


def _generate_AirConditioner_dataframe(csv_file, sheet_name=None, header=1):
    """
    This function reads my CSV file which documents typical parameters for each prime mover. It corrects the type of data 
    read (i.e., string, integer, float), and inserts them into a pandas dataframe. This function is used on the following 
    function which reads the processed data to generate Air Conditioner or Chiller objects.
    """

    dataframe = pd.read_csv(filepath_or_buffer=csv_file,
                            header=header,
                            dtype={'AC_id': 'object',
                                   'technology': 'object',
                                   'building': 'object',
                                   'climate': 'object',
                                   'avg_life': 'Int64',
                                   'capacity_kW': 'float64',
                                   'SEER': 'float64',
                                   'COP_full_load': 'float64',
                                   'COP_PLR': 'float64',
                                   'capital_cost': 'float64',
                                   'om_cost': 'float64'})
    dataframe.fillna(value=0.)
    return dataframe


# End AirConditioner Methods #
##############################


def _generate_AirConditioner(csv_file, sheet_name=None, header=1):
    """
    This function generates Prime Mover Objects from the CSV file. 
    """

    dataframe = _generate_AirConditioner_dataframe(csv_file=csv_file,
                                                   sheet_name=sheet_name,
                                                   header=header)
    i = 0
    AirConditioner_dictionary = {}
    while i < dataframe.AC_id.count():

        # Create new PrimeMover object
        current_AC = AirConditioner(dataframe.AC_id[i])
        current_AC._get_data(dataframe=dataframe, index=i)
        AirConditioner_dictionary[current_AC.AC_id] = current_AC
        i += 1
    return AirConditioner_dictionary


def size_AC(peak_demand=0., AirConditioner_=None, beta=0):
    """
    The purpose of this function is to size the air chiller system to the building demand. In other words, how many 
    chillers do we need to meet the cooling demand? I assume that the system is sized to meet the cooling demand.
    """
    design_peak = beta * peak_demand

    number_AC = m.ceil(design_peak/AirConditioner_.capacity)
    # ABC capital cost given in $/kW
    capital_cost_AC = number_AC*AirConditioner_.capacity * \
        AirConditioner_.capital_cost

    return number_AC, capital_cost_AC

##########################################
# End AirConditioner Class and Functions #
##########################################

########################
# BatteryStorage Class #
########################


class BatteryStorage:
    """
    Because we are typically given the roundtrip efficiencies, this method assumes that we can store all the electricity 
    input. The loss of energy is assumed to occur during discharge. This assumption holds if the maximum charging and 
    discharging are equal.        
    """

    def __init__(self, BES_id, model='', technology='', manufacturer='', chemistry='', application='any',
                 nom_capacity=0, depth_of_discharge=0, peak_power=0, power_cont=0,
                 degradation_rate=0, roundtripEfficiency=1,
                 lifetime=10, age=0, cycling_times=0, end_of_life_capacity=1, warranty=0,
                 volt_nom=0, battery_cost=0, install_cost=0, total_cost=0, specific_cost=0,
                 stateOfCharge=0, num_units=1):
        # BASIC INFORMATION
        self.BES_id = BES_id
        self.model = model
        self.manufacturer = manufacturer
        self.chemistry = chemistry
        self.application = application
        # PERFORMANCE METRICS
        # Capacity, DoD in kWh
        self.nom_capacity = nom_capacity
        self.DoD = depth_of_discharge
        # Power in kW
        self.peak_power = peak_power
        self.power_cont = power_cont
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
        self.specific_cost = specific_cost
        # OPERATIONAL PARAMETERS
        # SoC and all other parameters in kWh; number of units is integer
        self.SoC = stateOfCharge
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
        return ('Battery Storage: \n ' + ' \n '.join('{}: {}'.format(attr, getattr(self, attr)) for attr in attrs))

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

    def size_BES(self, Building_, storage_hours=24, method='mean'):
        """
        This method determines the number of storage units required to supply energy for a consecutive number of 
        'storage_hours' to the Building. This is determined by looking at the minimum, maximum, and mean sum of electricity
        used within the consecutive hours for the time specified.

        NEXT UPDATE MUST CHECK FOR THE NUMBER OF UNITS IN PARALLEL AND IN SERIES. CONNECTING BATTERIES IN PARALLEL DOUBLES
        THE POWER OUTPUT BUT MAINTAINS THE SAME VOLTAGE. CONNECTING THEM IN SERIES DOUBLES THE VOLTAGE, BUT MAINTAINS THE
        SAME POWER OUTPUT. We can call batteries in series a train.
        """
        electricity_demand = Building_.electricity_demand.tolist()

        # The storage list contains all of the required storage values for a consecutive number of storage hours.
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

    def _get_data(self, dataframe, index=0):
        """
        This method extracts data from a dataframe or an csv file to populate the attributes of each BES.
        """
        self.BES_id = dataframe.iloc[index]['BES_id']
        self.model = dataframe.iloc[index]['model']
        self.manufacturer = dataframe.iloc[index]['manufacturer']
        self.chemistry = dataframe.iloc[index]['chemistry']
        self.application = dataframe.iloc[index]['application']
        self.nom_capacity = dataframe.iloc[index]['capacity_kWh']
        self.DoD = dataframe.iloc[index]['depth_of_discharge_kWh']
        self.peak_power = dataframe.iloc[index]['peak_power_kW']
        self.power_cont = dataframe.iloc[index]['power_continuous_kW']
        self.degradation_rate = dataframe.iloc[index]['degradation_rate']
        self.RTE = dataframe.iloc[index]['roundtrip_efficiency']
        self.volt_nom = dataframe.iloc[index]['volt_nominal_V']
        self.EoL_cap = dataframe.iloc[index]['end_of_life_capacity']
        self.lifetime = dataframe.iloc[index]['lifetime_yr']
        self.warranty = dataframe.iloc[index]['warranty']
        self.cycling_times = dataframe.iloc[index]['cycling_times']
        self.battery_cost = dataframe.iloc[index]['battery_cost']
        self.install_cost = dataframe.iloc[index]['install_cost']
        self.total_cost = dataframe.iloc[index]['total_cost']
        self.specific_cost = dataframe.iloc[index]['specific_cost']
        return dataframe


def _generate_BES_dataframe(csv_file, sheet_name=None, header=1):
    dataframe = pd.read_csv(filepath_or_buffer=csv_file,
                            header=header,
                            dtype={'BES_id': 'object',
                                   'model': 'object',
                                   'manufacturer': 'object',
                                   'chemistry': 'object',
                                   'application': 'object',
                                   'capacity_kWh': 'float64',
                                   'peak_power_kW': 'float64',
                                   'power_continuous_kW': 'float64',
                                   'depth_of_discharge_kWh': 'float64',
                                   'degradation_rate': 'float64',
                                   'roundtrip_efficiency': 'float64',
                                   'lifetime_yr': 'Int64',
                                   'cycling_times': 'Int64',
                                   'end_of_life_capacity': 'float64',
                                   'volt_nominal_V': 'float64',
                                   'battery_cost': 'float64',
                                   'install_cost': 'float64',
                                   'total_cost': 'float64',
                                   'specific_cost': 'float64',
                                   'warranty': 'Int64'})
    dataframe.fillna(value=0.)
    return dataframe


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

##############
# Grid Class #
##############


class Grid:
    def __init__(self, Grid_id, name='', emm_region='', iso_name='', grid_loss=0, efficiency=1,
                 co=0, co2=0, ch4=0, n2o=0, nox=0, so2=0, voc=0, pm25=0, pm10=0, heat_co2=0, heat_nox=0, w4e=0,
                 lcoe=0, electricity_price_residential=0, electricity_price_commercial=0, natural_gas_price=0,
                 projections_df=None):
        self.Grid_id = Grid_id
        self.name = name
        self.emm_region = emm_region

        # Grid loss and efficiency are in decimals
        self.grid_loss = grid_loss
        self.efficiency = efficiency

        # All emissions are in kg/MWh
        self.co = co
        self.co2 = co2
        self.ch4 = ch4
        self.n2o = n2o
        self.nox = nox
        self.so2 = so2
        self.voc = voc
        self.pm25 = pm25  # pm 2.5
        self.pm10 = pm10

        # Currently adding values from furnace to grid. These can be a district system
        self.heat_co2 = heat_co2
        self.heat_nox = heat_nox

        # Water for Energy is in L/MWh
        self.w4e = w4e

        # Levelized cost of Electricity is in $/kWh
        self.lcoe = lcoe

        # Prices $/kWh
        self.electricity_price_residential = electricity_price_residential
        self.electricity_price_commercial = electricity_price_commercial

        # Fuel Prices $/kWh
        self.natural_gas_price = natural_gas_price
        '''self.CoalPrice = CoalPrice
        self.DistillateFuelPrice = DistillateFuelPrice
        self.ResidualFuelOilPrice = ResidualFuelOilPrice'''

        # Projection Data
        if projections_df is None:
            self.projections = None
        else:
            self.projections = projections_df[self.emm_region]

    def __repr__(self):
        attrs = ['Grid_id', 'name', 'emm_region', 'iso_name', 'grid_loss',
                 'co', 'co2', 'ch4', 'n2o', 'nox', 'so2', 'voc', 'pm25', 'pm10', 'heat_co2', 'heat_nox', 'w4e', 'lcoe']
        return ('Grid: \n ' + ' \n '.join('{}: {}'.format(attr, getattr(self, attr))
                                          for attr in attrs))

    def get_Grid_data(self, impacts_df, projections_df=None, index=0):
        """
        This method extracts data from a dataframe or an csv file to populate the attributes of each prime mover.
        """

        self.Grid_id = impacts_df.iloc[index]['Grid_id']  # NERC subregion name
        self.name = impacts_df.iloc[index]['eGRID_subregion_name']
        self.emm_region = impacts_df.iloc[index]['ElectricityMarketModule_region']
        self.iso_name = impacts_df.iloc[index]['ISO_subregion_name']
        self.grid_loss = impacts_df.iloc[index]['grid_loss_factor']
        self.ch4 = impacts_df.iloc[index]['CH4_kg_MWh-1']
        self.n2o = impacts_df.iloc[index]['N2O_kg_MWh-1']

        if self.emm_region == 'Not Applicable':
            # If the Electricity Market Module is None, the prices will be obtained
            # from a dataframe with averaged prices
            self.projections = pd.read_csv(
                'data\Tech_specs\Avg_prices.csv', index_col='year')
            self.co2 = impacts_df.iloc[index]['CO2_kg_MWh-1']
            self.nox = impacts_df.iloc[index]['NOx_kg_MWh-1']
            self.so2 = impacts_df.iloc[index]['SO2_kg_MWh-1']
            self.electricity_price_residential = self.projections.loc[
                2020]['ElectricityPrice_Residential_Dollars_kWh-1']
            self.electricity_price_commercial = self.projections.loc[
                2020]['ElectricityPrice_Commercial_Dollars_kWh-1']
            self.natural_gas_price = self.projections.loc[2020]['NaturalGasPrice_Dollars_kWh-1']
        else:
            self.projections = projections_df  # .loc[self.emm_region]
            self.electricity_price_residential = projections_df.loc[self.emm_region,
                                                                    2020]['ElectricityPrice_Residential_Dollars_kWh-1']
            self.electricity_price_commercial = projections_df.loc[self.emm_region,
                                                                   2020]['ElectricityPrice_Commercial_Dollars_kWh-1']
            self.natural_gas_price = projections_df.loc[self.emm_region,
                                                        2020]['NaturalGasPrice_Dollars_kWh-1']
            self.co2 = projections_df.loc[self.emm_region,
                                          2020]['CO2_kg_MWh-1']
            self.nox = projections_df.loc[self.emm_region,
                                          2020]['NOx_kg_MWh-1']
            self.so2 = projections_df.loc[self.emm_region,
                                          2020]['SO2_kg_MWh-1']

        return impacts_df, projections_df

# End Grid Class Methods #
##########################


def _generate_Grid_dataframes(impacts_csv_file, projections_csv_file, header=0):
    impacts_df = pd.read_csv(filepath_or_buffer=impacts_csv_file,
                             header=header,
                             dtype={'Grid_id': 'object',
                                    'NERC_subregion_name': 'object',
                                    'ElectricityMarketModule_region': 'object',
                                    'ISO_subregion_name': 'object',
                                    'grid_loss_factor': 'float64',
                                    'CO2_kg_MWh-1': 'float64',
                                    'CH4_kg_MWh-1': 'float64',
                                    'N2O_kg_MWh-1': 'float64',
                                    'NOx_kg_MWh-1': 'float64',
                                    'SO2_kg_MWh-1': 'float64'})

    impacts_df.fillna(value=0.)

    projections_df = pd.read_csv(projections_csv_file, index_col=[
        0, 1], skipinitialspace=True)

    return impacts_df, projections_df


def _generate_Grid(impacts_csv_file, projections_csv_file, header=0):
    """
    Similar to the PrimeMover class, this function generaes a series of Grid classes from a CSV file. The CSV file is read 
    and processed by the generate_storage_dataframe function.
    """
    impacts_df, projections_df = _generate_Grid_dataframes(impacts_csv_file=impacts_csv_file,
                                                           projections_csv_file=projections_csv_file,
                                                           header=header)
    i = 0
    Grid_dictionary = {}
    while i < impacts_df.Grid_id.count():
        # Create new Grid object
        current_Grid = Grid(impacts_df.Grid_id[i])
        current_Grid.get_Grid_data(
            impacts_df=impacts_df, projections_df=projections_df, index=i)
        Grid_dictionary[current_Grid.Grid_id] = current_Grid
        i += 1

    return Grid_dictionary

################################
# End Grid Class and Functions #
################################

#################
# Furnace Class #
#################


class Furnace:
    def __init__(self, Furnace_id, technology='furnace', electric=False, building='', climate='any',
                 capacity=234, efficiency=1, electric_consumption=0, lifetime=20, age=0,
                 retail_equipment_cost=0, total_installed_cost=0, annual_maintenance_cost=0,
                 equipment_cost=0, capital_cost=0, om_cost=0,
                 co2=182, n2o=9.71*10**-4, pm=2.88**10-3, ch4=3.49*10**-3, so2=9.10*10**-4,
                 voc=8.35*10**-3, nox=0.152, co=6.07**10-2):
        self.Furnace_id = Furnace_id
        self.technology = technology
        self.electric = electric  # Boolean
        self.building = building
        self.climate = climate
        # Capacity is in kW
        self.capacity = capacity
        # Efficiency in spec sheet is as a percentage, so must divide by 100.
        self.efficiency = efficiency
        # Electric Consumption is in kW. This is the fans and other equipment in the heater
        self.electric_consumption = electric_consumption
        # Lifetime in years
        self.lifetime = lifetime
        self.age = age
        # Costs in 2017$
        self.retail_equipment_cost = retail_equipment_cost
        self.total_installed_cost = total_installed_cost
        self.annual_maintenance_cost = annual_maintenance_cost
        # Costs in $/kW
        self.equipment_cost = equipment_cost
        self.capital_cost = capital_cost
        # OM Costs in $/kWh. Does not include fuel or electricity consumption
        self.om_cost = om_cost
        # Impact factors in kg/MWh or g/kWh
        self.co2 = co2
        self.n2o = n2o
        self.pm = pm
        self.so2 = so2
        self.ch4 = ch4
        self.voc = voc
        self.nox = nox
        self.co = co
        # Embedded emissions are in g
        self.embedded_ch4 = 0.356 * self.capacity
        self.embedded_co2 = 12023 * self.capacity
        self.embedded_n2o = 0.262 * self.capacity

    def __repr__(self):
        attrs = ['Furnace_id', 'technology', 'electric', 'building', 'climate',
                 'capacity', 'efficiency', 'electric_consumption', 'lifetime', 'age',
                 'retail_equipment_cost', 'total_installed_cost', 'annual_maintenance_cost',
                 'capital_cost', 'om_cost', 'co2', 'n2o', 'pm', 'so2', 'ch4', 'voc', 'nox', 'co']
        return ('Furnace: \n ' + ' \n '.join('{}: {}'.format(attr, getattr(self, attr))
                                             for attr in attrs))

    def _get_data(self, dataframe, index=0):
        """
        This method extracts data from a dataframe or an csv file to populate the attributes
        of each furnace.
        """
        self.Furnace_id = dataframe.iloc[index]['Furnace_id']
        self.technology = dataframe.iloc[index]['technology']
        self.electric = dataframe.iloc[index]['electric']
        self.building = dataframe.iloc[index]['building']
        self.climate = dataframe.iloc[index]['climate']
        self.capacity = dataframe.iloc[index]['capacity_kW']
        # Furnace efficiency must be divided by 100 to make a decimal. Will correct in future iteration
        self.efficiency = dataframe.iloc[index]['efficiency']/100
        self.electric_consumption = dataframe.iloc[index]['electric_consumption']
        self.lifetime = dataframe.iloc[index]['average_life']
        self.retail_equipment_cost = dataframe.iloc[index]['retail_equipment_cost']
        self.total_installed_cost = dataframe.iloc[index]['total_installed_cost']
        self.annual_maintenance_cost = dataframe.iloc[index]['annual_maintenance_cost']
        self.equipment_cost = dataframe.iloc[index]['equipment_cost']
        self.capital_cost = dataframe.iloc[index]['capital_cost']
        self.om_cost = dataframe.iloc[index]['om_cost']
        self.co2 = dataframe.iloc[index]['carbon_dioxide']
        self.n2o = dataframe.iloc[index]['nitrous_oxide']
        self.pm = dataframe.iloc[index]['PM']
        self.so2 = dataframe.iloc[index]['sulfur_dioxide']
        self.ch4 = dataframe.iloc[index]['methane']
        self.voc = dataframe.iloc[index]['voc']
        self.nox = dataframe.iloc[index]['nox']
        self.co = dataframe.iloc[index]['carbon_monoxide']
        return dataframe


def _generate_Furnace_dataframe(csv_file, sheet_name=None, header=1):
    dataframe = pd.read_csv(filepath_or_buffer=csv_file,
                            header=header,
                            dtype={'Furnace_id': 'object',
                                   'technology': 'object',
                                   'primary_energy': 'object',
                                   'electric': 'bool',
                                   'building': 'object',
                                   'capacity_kW': 'float64',
                                   'efficiency': 'float64',
                                   'electric_consumption': 'float64',
                                   'average_life': 'Int64',
                                   'retail_equipment_cost': 'float64',
                                   'total_installed_cost': 'float64',
                                   'annual_maintenance_cost': 'float64',
                                   'equipment_cost': 'float64',
                                   'capital_cost': 'float64',
                                   'om-cost': 'float64',
                                   'carbon_dioxide': 'float64',
                                   'nitrous_oxide': 'float64',
                                   'PM': 'float64',
                                   'sulfur_dioxide': 'float64',
                                   'methane': 'float64',
                                   'voc': 'float64',
                                   'nox': 'float64',
                                   'carbon_monoxide': 'float64'})
    dataframe.fillna(value=0.)
    return dataframe


def _generate_Furnace(csv_file, sheet_name=None, header=1):
    """
    Similar to the PrimeMover class, this function generaes a series of 
    Furnace Storage classes from a CSV file. The CSV file is read and 
    processed by the _generate_Furnace_dataframe function.
    """
    dataframe = _generate_Furnace_dataframe(csv_file=csv_file,
                                            sheet_name=sheet_name,
                                            header=header)
    i = 0
    Furnace_dictionary = {}
    while i < dataframe.Furnace_id.count():

        # Create new Furnace object
        current_Furnace = Furnace(dataframe.Furnace_id[i])
        current_Furnace._get_data(dataframe=dataframe, index=i)
        Furnace_dictionary[current_Furnace.Furnace_id] = current_Furnace
        i += 1

    return Furnace_dictionary


def size_Furnace(peak_heat_load, Furnace_=None):
    """
    The purpose of this function is to size the furnace to the building demand. 
    In other words, how many absportion chillers do we need to meet the heating demand?

    I assume that the system is sized to meet the heating demand.
    """
    # This may need some reworking because we are assuming that ABC energy wont be in here
    design_peak = peak_heat_load

    number_Furnace = m.ceil(design_peak / Furnace_.capacity)
    # ABC capital cost given in $/kW
    capital_cost_Furnace = number_Furnace * \
        Furnace_.capacity*Furnace_.capital_cost

    return number_Furnace, capital_cost_Furnace

###################################
# End Furnace Class and Functions #
###################################


def generate_objects(all_cities=True, selected_cities=[]):

    City_dictionary = _generate_Cities(
        all_cities=all_cities, selected_cities=selected_cities, how='processed')
    print('Cities generated for {}'.format(selected_cities))

    Grid_dictionary = _generate_Grid(
        impacts_csv_file='data\Tech_specs\Grid_emission_factors.csv', projections_csv_file='data\Tech_specs\power_projections.csv')
    print('Grid generated')

    PrimeMover_dictionary = _generate_PrimeMovers(
        csv_file='data\Tech_specs\PrimeMover_specs.csv', header=2)
    print('Prime Movers generated')

    BatteryStorage_dictionary = _generate_BatteryStorage(
        csv_file='data\Tech_specs\Battery_specs.csv')
    print('Batteries generated')

    Furnace_dictionary = _generate_Furnace(
        csv_file='data\Tech_specs\Furnace_specs.csv')
    print('Furnaces generated')

    AC_dictionary = _generate_AirConditioner(
        csv_file='data\Tech_specs\AC_specs.csv')
    print('Air Conditioners generated')

    ABC_dictionary = _generate_AbsorptionChillers(
        csv_file='data\Tech_specs\ABC_specs.csv')
    print('Absorption Chillers generated')

    system_dict = {'City_dict': City_dictionary,
                   'Grid_dict': Grid_dictionary,
                   'PrimeMover_dict': PrimeMover_dictionary,
                   'BatteryStorage_dict': BatteryStorage_dictionary,
                   'Furnace_dict': Furnace_dictionary,
                   'AC_dict': AC_dictionary,
                   'ABC_dict': ABC_dictionary}
    return (system_dict)


"""
REFERENCES

    [1]     William F. Holmgren, Clifford W. Hansen, and Mark A. Mikofski. "pvlib python: a python package 
            for modeling solar energy systems." Journal of Open Source Software, 3(29), 884, (2018). 
            https://doi.org/10.21105/joss.00884
    [2]     Deru, M.; Field, K.; Studer, D.; Benne, K.; Griffith, B.; Torcellini, P.; Bing, L.; Halverson, 
            M.; Winiarski, D.; Rosenberg, M.; et al. U.S. Department of Energy Commercial Reference Building 
            Models of the National Building Stock; Golden, Colorado, 2011.
    [3]     Baechler, M. C.; Gilbride, T. L.; Cole, P. C.; Hefty, M. G.; Ruiz, K. Guide to Determining 
            Climate Regions by County; Benton County, WA, 2015.
    [4]     U.S. Energy Information Administration. Annual Energy Outlook 2020; Washington D.C., 2020.
    [5]     U.S. Environmental Protection Agency. eGRID2018 
            https://www.epa.gov/energy/emissions-generation-resource-integrated-database-egrid (accessed Apr 13, 2020).
"""
