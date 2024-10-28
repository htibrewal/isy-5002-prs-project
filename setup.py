import glob
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

historical_parking_base_folder = os.getenv('PARKING_HISTORY_BASE_FOLDER')
SAMPLED_5_MIN_FOLDER = '5-Min-Sampled'
SAMPLED_30_MIN_FOLDER = '30-Min-Sampled'

categorical_features = [
    'car_park_type',
    'type_of_parking_system',
    'short_term_parking',
    'free_parking',
    'night_parking',
    'car_park_basement'
]

# this fetches data from the folders (5-min sampled/30-min sampled)
def fetch_data_from_subfolders(sampling = SAMPLED_5_MIN_FOLDER):
    subfolders = [
        f.path
        for f in os.scandir(os.path.join(historical_parking_base_folder, sampling))
        if f.is_dir()
    ]

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
