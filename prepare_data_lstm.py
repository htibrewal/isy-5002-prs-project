import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

from prepare_data_v2 import prepare_parking_info_df_v2, prepare_historical_parking_df_v2

load_dotenv()

numerical_features = [
    'total_lots',
    'x_coord',
    'y_coord',
    'sin_hour',
    'cos_hour',
    'sin_day_of_week',
    'cos_day_of_week'
]

def prepare_resultant_df_v3(output_scalar):
    # fetch prepared car lot info (static)
    parking_info_df = prepare_parking_info_df_v2()

    # fetch prepared car parking data (historical)
    historical_parking_df = prepare_historical_parking_df_v2(use_mean_sampling=True)
    historical_parking_df = historical_parking_df.drop(columns=['sin_month', 'cos_month'])

    # get a smaller set of historical parking dataframe to work upon
    historical_parking_df = historical_parking_df[:100000]

    # prepare a resultant DataFrame
    resultant_df = pd.merge(historical_parking_df, parking_info_df, on='car_park_number', how='inner')

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
