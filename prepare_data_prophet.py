import os

import pandas as pd
from setup import fetch_data_from_subfolders, SAMPLED_30_MIN_FOLDER

car_park_key = 'car_park_number'
timestamp_key = 'update_timestamp'

def prepare_data_prophet(car_park_number, historical_5_min_df):
    # pick historical data for only one car park
    car_park_5_min_df = historical_5_min_df[historical_5_min_df[car_park_key] == car_park_number]

    # drop not required features
    data_5_min_df = car_park_5_min_df.drop(columns=[car_park_key, 'fetch_timestamp', 'lot_type', 'total_lots'])

    # convert data that has been sampled at 5 mins to 30 mins interval
    data_reduced_30_min_df = convert_5_min_to_30_min_sampled(data_5_min_df)

    # fetch data that has been sampled at an interval of 30 mins
    data_30_min_df = prepare_data_30_min_sampled(car_park_number)

    # combine 2 dataframes - 5 min converted to 30 min and 30 min sampled
    combined_df = pd.concat([data_reduced_30_min_df, data_30_min_df], axis=0)
    combined_df[timestamp_key] = pd.to_datetime(combined_df[timestamp_key])

    return (combined_df
            .sort_values(by=timestamp_key)
            .rename(columns={timestamp_key: 'ds', 'available_lots': 'y'}))


# prepares data from the 30 min sampled folder
def prepare_data_30_min_sampled(car_park_number):
    data_df = fetch_data_from_subfolders(SAMPLED_30_MIN_FOLDER)

    # pick historical data for only one car park
    car_park_df = data_df[data_df[car_park_key] == car_park_number].drop(columns=[car_park_key])

    # reorganise columns
    return car_park_df[[timestamp_key, 'available_lots']]


def convert_5_min_to_30_min_sampled(parking_df):
    parking_df[timestamp_key] = pd.to_datetime(parking_df[timestamp_key])
    parking_df.set_index(timestamp_key, inplace=True)

    parking_reduced_df = (parking_df
                          .sort_values(by=[timestamp_key])
                          .resample('30min')
                          .agg({'available_lots': 'mean'})
                          .reset_index())
    return parking_reduced_df.dropna()


def get_car_park_no_list():
    info_base_folder = os.getenv('PARKING_METADATA_BASE_FOLDER')
    parking_info_df = pd.read_csv(os.path.join(info_base_folder, 'HDBCarparkInformation.csv'))

    return list(parking_info_df['car_park_no'])
