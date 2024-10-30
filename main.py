import multiprocessing as mp
import os
from datetime import datetime
from functools import partial
from tqdm import tqdm
from prophet_model import train_single_model
from setup import fetch_data_from_subfolders, get_time_taken


# Iterative Approach
def train_models_iterative(car_park_list, historical_df):
    results = []
    for car_park in car_park_list:
        result = train_single_model(car_park, historical_df)
        results.append(result)

        if result:
            if result['status'] == 'error':
                print(f"\nError training model for {result['car_park']}: {result['error_message']}")

            elif result['status'] == 'success':
                print(f"\nCross-Validation Metrics for {result['car_park']}")
                print(f"Mean MSE: {result['mse']:.2f}")
                print(f"Mean RMSE: {result['rmse']:.2f}")
                print(f"Mean MAE: {result['mae']:.2f}")

    return results


# Multiprocessing Approach
def train_models_parallel(car_park_list, historical_df, n_cores = None):

    if n_cores is None:
        n_cores = mp.cpu_count() - 1

    print(f"Training {len(car_park_list)} models using {n_cores} cores")

    train_func = partial(train_single_model, historical_5_min_df=historical_df)

    results = []
    with mp.Pool(processes=n_cores) as pool:
        for result in tqdm(pool.imap_unordered(train_func, car_park_list), total=len(car_park_list), desc="Training models"):
            results.append(result)

            if result and result['status'] == 'error':
                print(f"\nError training model for {result['car_park']}: {result['error_message']}")

    return results


if __name__ == "__main__":
    # create the "models" directory and don't raise an error if already existing
    os.makedirs("models", exist_ok=True)

    historical_5_min_df = fetch_data_from_subfolders()
    car_park_no_list = list(historical_5_min_df['car_park_number'].unique())
    car_park_no_list = car_park_no_list[:10]

    print("Car park list")
    print(car_park_no_list)

    start_time = datetime.now()
    results = train_models_iterative(car_park_no_list, historical_5_min_df)

    print(f"Time taken to process all car parking lots = {get_time_taken(start_time)}")

    successful = sum(1 for r in results if r and r['status'] == 'success')
    failed = sum(1 for r in results if r and r['status'] == 'error')
    skipped = len(car_park_no_list) - successful - failed

    print("\nTraining Summary:")
    print(f"Successfully trained: {successful} models")
    print(f"Failed: {failed} models")
    print(f"Skipped (insufficient data): {skipped} models")
