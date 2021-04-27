def convert_feather_to_csv(file=r'model_outputs\energy_supply\Annual_atlanta_full_service_restaurant_energy_sup.feather'):
    import pandas as pd
    import pyarrow
    
    df = pd.read_feather(file)
    print(df.head())

    df.to_csv(r'model_outputs\testing\TEST_FILE.csv')


convert_feather_to_csv()