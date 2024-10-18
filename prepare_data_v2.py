import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os

from setup import fetch_data_from_subfolders, categorical_features


load_dotenv()

numerical_features = ['total_lots', 'available_lots', 'x_coord', 'y_coord']

def prepare_historical_parking_df_v2():
    historical_parking_df = fetch_data_from_subfolders()

    # feature engineering - updated timestamp
    historical_parking_df['update_timestamp'] = pd.to_datetime(historical_parking_df['update_timestamp'])

    # get month, day_of_week, hour
    historical_parking_df['month'] = historical_parking_df['update_timestamp'].dt.month
    historical_parking_df['day_of_week'] = historical_parking_df['update_timestamp'].dt.weekday
    historical_parking_df['hour'] = historical_parking_df['update_timestamp'].dt.hour
    historical_parking_df['minute'] = historical_parking_df['update_timestamp'].dt.minute

    # create cyclic features from month, day_of_week, hour
    historical_parking_df['sin_minute'] = np.sin(2 * np.pi * historical_parking_df['minute'] / 60)
    historical_parking_df['cos_minute'] = np.cos(2 * np.pi * historical_parking_df['minute'] / 60)

    historical_parking_df['sin_hour'] = np.sin(2 * np.pi * historical_parking_df['hour'] / 24)
    historical_parking_df['cos_hour'] = np.cos(2 * np.pi * historical_parking_df['hour'] / 24)

    historical_parking_df['sin_day_of_week'] = np.sin(2 * np.pi * historical_parking_df['day_of_week'] / 7)
    historical_parking_df['cos_day_of_week'] = np.cos(2 * np.pi * historical_parking_df['day_of_week'] / 7)

    historical_parking_df['sin_month'] = np.sin(2 * np.pi * historical_parking_df['month'] / 12)
    historical_parking_df['cos_month'] = np.cos(2 * np.pi * historical_parking_df['month'] / 12)

    # drop not required features
    historical_parking_df = historical_parking_df.drop(
        columns=['fetch_timestamp', 'lot_type', 'update_timestamp', 'month', 'day_of_week', 'hour', 'minute']
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


def prepare_resultant_df_v2():
    # fetch prepared car lot info (static)
    parking_info_df = prepare_parking_info_df_v2()

    # fetch prepared car parking data (historical)
    historical_parking_df = prepare_historical_parking_df_v2()

    # prepare a resultant DataFrame
    resultant_df = pd.merge(historical_parking_df, parking_info_df, on='car_park_number', how='inner')

    scaler = MinMaxScaler()
    resultant_df[numerical_features] = scaler.fit_transform(resultant_df[numerical_features])

    print("Resultant dataframe shape = ", resultant_df.shape)
    print("Resultant dataframe top 5")
    print(resultant_df.head())

    return resultant_df


if __name__ == '__main__':
    print(prepare_resultant_df_v2())