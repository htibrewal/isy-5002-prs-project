import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

from prepare_data_v2 import prepare_parking_info_df_v2, prepare_historical_parking_df_v2

load_dotenv()

static_numerical_features = ['total_lots', 'x_coord', 'y_coord',]

numerical_features = [
    'sin_hour',
    'cos_hour',
    'sin_day_of_week',
    'cos_day_of_week'
]

def prepare_resultant_df_v3(output_scalar, use_static_features=True):
    # fetch prepared car parking data (historical)
    historical_parking_df = prepare_historical_parking_df_v2(use_mean_sampling=True)
    historical_parking_df = historical_parking_df.drop(columns=['sin_month', 'cos_month'])

    # get a smaller set of historical parking dataframe to work upon
    historical_parking_df = historical_parking_df[:100000]

    # prepare a resultant DataFrame
    if use_static_features:
        # fetch prepared car lot info (static)
        parking_info_df = prepare_parking_info_df_v2()
        resultant_df = pd.merge(historical_parking_df, parking_info_df, on='car_park_number', how='inner')

        # also extend the numerical features list
        # when using static features of parking lot
        numerical_features.extend(static_numerical_features)
    else:
        resultant_df = historical_parking_df.drop(columns=['total_lots'])


    scaler = MinMaxScaler()
    resultant_df[numerical_features] = scaler.fit_transform(resultant_df[numerical_features])

    # fit and transform available_lots column
    resultant_df['available_lots'] = output_scalar.fit_transform(resultant_df[['available_lots']])

    print("Resultant dataframe shape = ", resultant_df.shape)
    print("Resultant dataframe top 5")
    print(resultant_df.head())

    return resultant_df

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])  # Past n_steps
        y.append(data[i + n_steps, 0])  # Target is the next available_spaces
    return np.array(X), np.array(y)
