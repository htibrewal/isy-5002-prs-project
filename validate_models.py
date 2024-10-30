import gc
import glob
import os
import re

import numpy as np
from prophet.serialize import model_from_json
from prophet_model import model_cross_validation


all_models = glob.glob(os.path.join("models", "*.json"))

output = {
    'mse': list(),
    'rmse': list(),
    'mae': list()
}
car_parks = list()

for model_name in all_models:
    match = re.search(r'model_([A-Z]+[0-9]*)\.json', model_name)

    car_park = None
    if match:
        car_park = match.group(1)

    # add car_park to the list of car_parks
    if car_park is not None:
        car_parks.append(car_park)

    with open(model_name, 'r') as file:
        m = model_from_json(file.read())

        metrics = model_cross_validation(m, parallel="threads")

        for metric in ['mse', 'rmse', 'mae']:
            if metric in metrics:
                output[metric].append(metrics[metric].mean())

        del file, m, metrics
        gc.collect()


print(f"Collected metrics for {len(car_parks)} car parks")
print(f"Mean MSE: {np.mean(output['mse']):.2f}")
print(f"Mean RMSE: {np.mean(output['rmse']):.2f}")
print(f"Mean MAE: {np.mean(output['mae']):.2f}")
