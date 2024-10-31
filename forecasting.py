import os.path

import numpy as np
import pandas as pd
from prophet.serialize import model_from_json
from constants import MODELS_PATH


def calculate_forecast(car_parks: pd.DataFrame, datetime) -> pd.DataFrame:
    car_parks['availability'] = car_parks.apply(
        lambda row: get_forecasted_availability(row['car_park_no'], datetime), axis=1
    )

    car_parks = car_parks[~car_parks['availability'].isna()]
    return car_parks


def get_forecasted_availability(car_park_no, datetime):
    model_name = f"model_{car_park_no}.json"

    try:
        with open(os.path.join(MODELS_PATH, model_name), 'r') as file:
            model = model_from_json(file.read())

            future = model.make_future_dataframe(periods=8760, freq='h', include_history=False)
            forecast = model.predict(future)

            availability_forecast = forecast.loc[(forecast['ds'] - pd.to_datetime(datetime)).abs().idxmin()]
            print(f"Predicted availability for car park {car_park_no} = {availability_forecast['yhat']}")

            # when fetch for a given date
            # availability_forecast = forecast[forecast['ds'].apply(lambda dt: pd.to_datetime(dt).date() == pd.to_datetime(datetime).date())]

            return availability_forecast['yhat']
    except FileNotFoundError:
        print(f"No model found for car park {car_park_no}")

    return np.NaN
