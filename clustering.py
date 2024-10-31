import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from constants import DATA_PATH, ANY_VALUE, CAR_PARK_NO


numerical_cols = ['x_coord', 'y_coord']
categorical_cols = ['car_park_type', 'free_parking', 'night_parking', 'car_park_basement']
default_categorical_values = {
    'car_park_type': 'SURFACE CAR PARK',
    'free_parking': 'YES',
    'night_parking': 'YES',
    'car_park_basement': 'N'
}

def perform_hierarchical_clustering(x, y, filters):
    user_df = create_user_df(x, y, filters)
    scaler = StandardScaler()

    fitted_data = prepare_hierarchical_data(user_df, scaler)

    clustering_columns = numerical_cols + categorical_cols
    linked = linkage(np.array(fitted_data[clustering_columns]), method='ward')

    max_distance = 3
    cluster_labels = fcluster(linked, max_distance, criterion='distance')
    fitted_data['cluster'] = cluster_labels

    # Identify the user's cluster by taking the cluster value of last entry
    user_cluster = fitted_data.iloc[-1]['cluster']
    car_parks_cluster = fitted_data[(fitted_data['cluster'] == user_cluster) & (fitted_data['car_park_no'] != CAR_PARK_NO)]

    if car_parks_cluster.shape[0] != 0:
        car_parks_cluster[numerical_cols] = scaler.inverse_transform(car_parks_cluster[numerical_cols])
        return car_parks_cluster[['car_park_no'] + numerical_cols]

    return None


def prepare_hierarchical_data(user_df, scaler):
    parking_info_data = load_parking_data()

    # # should we filter the dataframe based on selected categorical values???
    # cols_to_drop = []
    # for key, value in filters.items():
    #     if value != ANY_VALUE:
    #         parking_info_data = parking_info_data[parking_info_data[key] == value]
    #         cols_to_drop.append(key)
    #
    # parking_info_data = parking_info_data.drop(columns=cols_to_drop)
    #
    # for key in list(filters.keys()):
    #     if key not in cols_to_drop:
    #         categorical_cols.append(key)


    required_columns = ['car_park_no'] + numerical_cols + categorical_cols
    filtered_data = parking_info_data[required_columns]

    concatenated_data = pd.concat([filtered_data, user_df], axis=0, ignore_index=True)

    # Perform scaler on the numerical values
    concatenated_data[numerical_cols] = scaler.fit_transform(concatenated_data[numerical_cols])

    # Perform categorical encoding on categorical values
    encoder = OrdinalEncoder()
    concatenated_data[categorical_cols] = encoder.fit_transform(concatenated_data[categorical_cols])

    return concatenated_data


def load_parking_data():
    parking_info_data = pd.read_csv(os.path.join(DATA_PATH, 'HDBCarparkInformation.csv'))
    parking_info_data['free_parking'] = parking_info_data['free_parking'].apply(lambda x: 'NO' if x == 'NO' else 'YES')
    return parking_info_data


def create_user_df(x, y, filters):
    user_data = {
        'car_park_no': [CAR_PARK_NO],
        'x_coord': [x],
        'y_coord': [y],
    }

    for key, value in filters.items():
        if value != ANY_VALUE:
            user_data[key] = [value]
        else:
            # If no preference given by user then take value with more representation
            user_data[key] = [default_categorical_values[key]]

    return pd.DataFrame.from_dict(user_data)
