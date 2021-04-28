def convert_feather_to_csv(file=r'model_outputs\energy_supply\Annual_atlanta_full_service_restaurant_energy_sup.feather'):
    import pandas as pd
    import pyarrow
    
    df = pd.read_feather(file)
    print(df.head())

    df.to_csv(r'model_outputs\testing\TEST_FILE.csv')

import pandas as pd

NGCC_dict = {'NGCC_ch4': [2.5 * 10**-2],
                 'NGCC_co': [8.71 * 10**-2],
                 'NGCC_co2': [170.24],
                 'NGCC_nox': [3.1 * 10**-2],
                 'NGCC_n2o': [8.71 * 10**-3],
                 'NGCC_pm': [1.92 * 10**-2],
                 'NGCC_so2': [5.11 * 10**-3],
                 'NGCC_voc': [6.10 * 10**-3],
                 'NGCC_efficiency': [0.533]}

df = pd.DataFrame.from_dict(NGCC_dict)

print(df)