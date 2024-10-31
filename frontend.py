import gradio as gr

from forecasting import calculate_forecast
from geospatial_features import convert_lat_long_to_xy
from helper import get_input_elements, get_datetime_element, get_output_element, calculate_distance
from clustering import perform_hierarchical_clustering

location_dropdown, car_park_type_dropdown, free_parking_dropdown, night_parking_dropdown, car_park_basement_dropdown = get_input_elements()
datetime_element = get_datetime_element()
output_element = get_output_element()

def get_best_parking_location(location, datetime, car_park_type, free_parking, night_parking, basement_parking):
    x, y, filters = preprocess_input(location, car_park_type, free_parking, night_parking, basement_parking)

    car_parks = perform_hierarchical_clustering(x, y, filters)
    if car_parks is not None:
        car_parks = calculate_forecast(car_parks, datetime)
        car_parks['dist'] = car_parks.apply(lambda df: calculate_distance(df, x, y), axis=1)

        sorted_car_parks = car_parks.sort_values(by=['availability', 'dist'], ascending=[False, True])

        if sorted_car_parks.empty:
            print("No models found for forecasting of availability")

        elif sorted_car_parks.shape[0] > 3:
            top_car_parks = sorted_car_parks[:3]['car_park_no'].tolist()
            return f"Top recommended car parks are {top_car_parks}"

        else:
            top_car_parks = sorted_car_parks['car_park_no'].tolist()
            return f"Top recommended car parks are {top_car_parks}"


    return "No suitable car park found"


def preprocess_input(location, car_park_type, free_parking, night_parking, basement_parking):
    lat, long = location.split(",")
    x, y = convert_lat_long_to_xy(lat, long)

    filters = {
        'car_park_type': car_park_type,
        'free_parking': free_parking,
        'night_parking': night_parking,
        'car_park_basement': basement_parking
    }

    return x, y, filters


gr.Interface(
    fn=get_best_parking_location,
    inputs=[location_dropdown, datetime_element, car_park_type_dropdown, free_parking_dropdown, car_park_basement_dropdown, night_parking_dropdown],
    outputs=output_element,
    title="Best Parking Location Finder",
    description="Select a location and enter a time to find the best parking option available.",
).launch()
