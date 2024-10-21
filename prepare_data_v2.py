import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os

from setup import fetch_data_from_subfolders, categorical_features


load_dotenv()

numerical_features = ['total_lots', 'available_lots', 'x_coord', 'y_coord']

# function to convert the date and time features into cyclic with the help of sin and cos
def replace_timestamp_with_cyclic_features(target_df, timestamp_key):
    # feature engineering - updated timestamp
    target_df[timestamp_key] = pd.to_datetime(target_df[timestamp_key])

    target_df['month'] = target_df[timestamp_key].dt.month
    target_df['day_of_week'] = target_df[timestamp_key].dt.weekday
    target_df['hour'] = target_df[timestamp_key].dt.hour

    # create cyclic features from month, day_of_week, hour
    target_df['sin_hour'] = np.sin(2 * np.pi * target_df['hour'] / 24)
    target_df['cos_hour'] = np.cos(2 * np.pi * target_df['hour'] / 24)

    target_df['sin_day_of_week'] = np.sin(2 * np.pi * target_df['day_of_week'] / 7)
    target_df['cos_day_of_week'] = np.cos(2 * np.pi * target_df['day_of_week'] / 7)

    target_df['sin_month'] = np.sin(2 * np.pi * target_df['month'] / 12)
    target_df['cos_month'] = np.cos(2 * np.pi * target_df['month'] / 12)

    return target_df.drop(columns = [timestamp_key, 'month', 'day_of_week', 'hour'])

# function to sample every car_park_number by 30 mins frequency
# for available_lots and apply mean when sampling
def get_timestamp_mean_sampled_df(target_df, timestamp_key):
    target_df[timestamp_key] = pd.to_datetime(target_df[timestamp_key])
    target_df.set_index(timestamp_key, inplace=True)

    target_reduced_df = (target_df
                         .groupby('car_park_number')
                         .resample('30T')
                         .agg({'available_lots': 'mean'})
                         .reset_index())
    return target_reduced_df.dropna()

# function to retain the data points closest to every half an hour
# and the logic is to check 5 mins before and after the half an hour mark
def get_timestamp_filtered_df(target_df, timestamp_key):
    ts_rounded_key = 'timestamp_rounded'

    target_df[timestamp_key] = pd.to_datetime(target_df[timestamp_key])
    target_df[ts_rounded_key] = target_df[timestamp_key].dt.round('30T')

    target_df_filtered = target_df[
        (target_df[timestamp_key] >= target_df[ts_rounded_key] - pd.Timedelta(minutes=5)) &
        (target_df[timestamp_key] <= target_df[ts_rounded_key] + pd.Timedelta(minutes=5))
    ]

    target_df_reduced = target_df_filtered.groupby(['car_park_number', ts_rounded_key]).first().reset_index()

    return target_df_reduced.drop(columns = [ts_rounded_key])


# please ensure that only one of the flags is True at one time
def prepare_historical_parking_df_v2(use_mean_sampling = False, use_time_difference = False):
    historical_parking_df = fetch_data_from_subfolders()

    # drop not required features
    historical_parking_df = historical_parking_df.drop(columns=['fetch_timestamp', 'lot_type'])

    if use_time_difference:
        historical_parking_df = get_timestamp_filtered_df(historical_parking_df, 'update_timestamp')

    elif use_mean_sampling:
        parking_lot_capacity_df = historical_parking_df.drop(columns=['available_lots', 'update_timestamp'])
        parking_lot_capacity_df = parking_lot_capacity_df.groupby('car_park_number')['total_lots'].last().reset_index()

        historical_parking_reduced_df = get_timestamp_mean_sampled_df(historical_parking_df, 'update_timestamp')

        historical_parking_df = pd.merge(historical_parking_reduced_df, parking_lot_capacity_df, on='car_park_number', how='inner')

    # replace timestamp with cyclic values for its features
    historical_parking_df = replace_timestamp_with_cyclic_features(
        historical_parking_df.copy(),
        'update_timestamp'
    )

    print("Historical parking data shape = ", historical_parking_df.shape)
    print("Historical parking data top 5")
    print(historical_parking_df.head())

    return historical_parking_df


def prepare_parking_info_df_v2(folder_path = None):
    info_base_folder = folder_path if folder_path is not None else os.getenv('PARKING_METADATA_BASE_FOLDER')
    parking_info_df = pd.read_csv(os.path.join(info_base_folder, 'HDBCarparkInformation.csv'))

    parking_info_df = (parking_info_df
           .drop(columns=['address', 'gantry_height'])
           .rename(columns={'car_park_no': 'car_park_number'}))

    encoder = OneHotEncoder()
    encoded_features = pd.DataFrame(encoder.fit_transform(parking_info_df[categorical_features]).toarray(),
            columns=encoder.get_feature_names_out())
    parking_info_df = parking_info_df.drop(columns=categorical_features).reset_index(drop=True)

    print("Car park static info shape = ", parking_info_df.shape)
    print("Car park static info top 5")
    print(parking_info_df.head())

    return pd.concat([parking_info_df, encoded_features], axis=1)


def prepare_resultant_df_v2(use_mean_sampling = False, use_time_difference = False):
    # fetch prepared car lot info (static)
    parking_info_df = prepare_parking_info_df_v2()

    # fetch prepared car parking data (historical)
    historical_parking_df = prepare_historical_parking_df_v2(use_mean_sampling, use_time_difference)

    # prepare a resultant DataFrame
    merged_df = pd.merge(historical_parking_df, parking_info_df, on='car_park_number', how='inner')

    scaler = MinMaxScaler()
    merged_df[numerical_features] = scaler.fit_transform(merged_df[numerical_features])

    print("Resultant dataframe shape = ", merged_df.shape)
    print("Resultant dataframe top 5")
    print(merged_df.head())

    return merged_df


if __name__ == '__main__':
    print(prepare_resultant_df_v2())