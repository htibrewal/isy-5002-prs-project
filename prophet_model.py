import gc
import os.path

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_to_json
from prepare_data_prophet import prepare_data_prophet
from prepare_holidays import sg_holidays


def train_single_model(car_park, historical_5_min_df):
    try:
        print(f"Processing car parking lot {car_park}")
        prepared_data = prepare_data_prophet(car_park, historical_5_min_df)

        if prepared_data.shape[0] <= 1000:
            print(f"Warning: Not enough parking data found for car parking lot {car_park}")

            del prepared_data
            gc.collect()

            return None

        print(f"Prepared data shape = {prepared_data.shape}")

        # initialise model and fit the prepared data on the model specific to car_park
        model = Prophet(holidays=sg_holidays)
        model.fit(prepared_data)

        metrics = model_cross_validation(model)

        model_name = f"model_{car_park}.json"
        with open(os.path.join("models", model_name), 'w') as f:
            f.write(model_to_json(model))

        future = model.make_future_dataframe(periods=365, include_history=False)
        forecast = model.predict(future)

        del prepared_data, model, future
        gc.collect()

        output = {
            'car_park': car_park,
            'status': 'success',
            'forecast': forecast,
        }

        for metric in ['mse', 'rmse', 'mae']:
            if metric in metrics:
                output[metric] = metrics[metric].mean()

        return output

    except Exception as e:
        return {
            'car_park': car_park,
            'status': 'error',
            'error_message': str(e)
        }


def model_cross_validation(model, parallel='processes'):
    df_cv = cross_validation(model, initial='240 days', horizon='30 days', parallel=parallel)
    df_metrics = performance_metrics(df_cv, metrics=['mse', 'rmse', 'mae'])

    del df_cv

    return df_metrics
