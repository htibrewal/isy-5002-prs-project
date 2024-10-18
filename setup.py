import glob
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

load_dotenv()

historical_parking_base_folder = os.getenv('PARKING_HISTORY_BASE_FOLDER')

categorical_features = [
    'car_park_type',
    'type_of_parking_system',
    'short_term_parking',
    'free_parking',
    'night_parking',
    'car_park_basement'
]

def fetch_data_from_subfolders():
    subfolders = [f.path for f in os.scandir(historical_parking_base_folder) if f.is_dir()]

    # read csv in a dataframe and put all the DataFrames in a list
    dfs = []
    for subfolder in subfolders:
        all_files = glob.glob(os.path.join(subfolder, '*.csv'))
        all_files.sort(key=lambda x: os.path.basename(x))

        for file in all_files:
            df = pd.read_csv(file)
            dfs.append(df)

    # concat all the dataframes
    return pd.concat(dfs, ignore_index=True)


def get_train_test_X_y(resultant_df, target=None, test_size=0.2):
    # prepare X & y (classification)
    X, y_encoded = get_X_y_encoded(resultant_df, target)

    # train and test split for X & y
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)

    print("Shape of X_train = ", X_train.shape)
    print("Shape of X_test = ", X_test.shape)
    print("Shape of y_train = ", y_train.shape)
    print("Shape of y_test = ", y_test.shape)

    return X_train, X_test, y_train, y_test


def get_X_y_encoded(resultant_df, target=None):
    if target is None:
        target = ['car_park_number']

    X = resultant_df.drop(columns=target).to_numpy()
    y = resultant_df[target]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded
