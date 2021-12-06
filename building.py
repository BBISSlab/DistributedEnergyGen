
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

##################
# Building Class #
##################
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
        return thermal_energy_demand * efficiency
        
####################################
# End Building Class and Functions #
####################################