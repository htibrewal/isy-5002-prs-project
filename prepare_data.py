import pandas as pd
import glob
import os

from setup import base_folder


def prepare_historical_parking_df():
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    # read csv in a dataframe and put all the DataFrames in a list
    dfs = []
    for subfolder in subfolders:
        all_files = glob.glob(os.path.join(subfolder, '*.csv'))
        all_files.sort(key=lambda x: os.path.basename(x))

        for file in all_files:
            df = pd.read_csv(file)
            dfs.append(df)

    # concat all the dataframes
    historical_parking_df = pd.concat(dfs, ignore_index=True)

    # compute the occupied lots for each parking space
    historical_parking_df['occupied_lots'] = historical_parking_df['total_lots'] - historical_parking_df['available_lots']

    print("Historical parking data shape = ", historical_parking_df.shape)
    print("Historical parking data top 5")
    print(historical_parking_df.head())

    return historical_parking_df


def prepare_parking_info_df(folder_path = None):
    info_base_folder = folder_path if folder_path is not None else '/Users/harsh/Desktop/Pattern Recognition Systems/Project/Data/HDB'
    parking_info_df = pd.read_csv(os.path.join(info_base_folder, 'HDBCarparkInformation.csv'))

    parking_info_df['car_park_type'] = pd.Categorical(parking_info_df['car_park_type']).codes
    parking_info_df['type_of_parking_system'] = pd.Categorical(parking_info_df['type_of_parking_system']).codes
    parking_info_df['short_term_parking'] = pd.Categorical(parking_info_df['short_term_parking']).codes
    parking_info_df['free_parking'] = pd.Categorical(parking_info_df['free_parking']).codes
    parking_info_df['night_parking'] = pd.Categorical(parking_info_df['night_parking']).codes
    parking_info_df['car_park_decks'] = pd.Categorical(parking_info_df['car_park_decks']).codes
    parking_info_df['car_park_basement'] = pd.Categorical(parking_info_df['car_park_basement']).codes

    parking_info_df.drop(labels=['gantry_height'], axis=1, inplace=True)
    parking_info_df.rename(columns={'car_park_no': 'car_park_number'}, inplace=True)

    print("Car park static info shape = ", parking_info_df.shape)
    print("Car park static info top 5")
    print(parking_info_df.head())

    return parking_info_df


def prepare_resultant_df():
    # fetch prepared car lot info (static)
    parking_info_df = prepare_parking_info_df()

    # fetch prepared car parking data (historical)
    historical_parking_df = prepare_historical_parking_df()

    # prepare a resultant DataFrame
    resultant_df = pd.merge(historical_parking_df, parking_info_df, on='car_park_number', how='inner')
    resultant_df.drop(['fetch_timestamp', 'lot_type', 'address'], axis=1, inplace=True)
    resultant_df.head()

    # separate out update timestamp into different features (year, month, day, hour, minute, second)
    resultant_df['update_timestamp'] = pd.to_datetime(resultant_df['update_timestamp'])

    resultant_df['update_year'] = resultant_df['update_timestamp'].dt.year
    resultant_df['update_month'] = resultant_df['update_timestamp'].dt.month
    resultant_df['update_day'] = resultant_df['update_timestamp'].dt.day
    resultant_df['update_hour'] = resultant_df['update_timestamp'].dt.hour
    resultant_df['update_minute'] = resultant_df['update_timestamp'].dt.minute
    resultant_df['update_second'] = resultant_df['update_timestamp'].dt.second
    resultant_df.drop('update_timestamp', axis=1, inplace=True)

    print("Resultant dataframe shape = ", resultant_df.shape)
    print("Resultant dataframe top 5")
    print(resultant_df.head())

    return resultant_df
