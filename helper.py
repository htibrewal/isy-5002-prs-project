import os
from datetime import datetime
import gradio as gr
import numpy as np
import pandas as pd

from constants import DATA_PATH, ANY_VALUE


def load_singapore_locations():
    sg_locations = pd.read_csv(os.path.join(DATA_PATH, "sg_locations.csv"), index_col=0)
    filtered_sg_locations = sg_locations[sg_locations['address'].apply(lambda address: 'Singapore' in address)].sort_values(by='address')
    # print(filtered_sg_locations.shape) -> 692 locations

    locations = []
    for _, row in filtered_sg_locations.iterrows():
        lat_long = f"{row['latitude']},{row['longitude']}"
        locations.append((row['address'], lat_long))

    return locations


def get_input_elements():
    any_choice = ('Any', ANY_VALUE)
    yes_choice = ('Yes', 'YES')
    no_choice = ('No', 'NO')

    location_dropdown = gr.Dropdown(choices=load_singapore_locations(), label="Select Preferred Location", interactive=True)
    car_park_type_dropdown = gr.Dropdown(choices=[
        any_choice, ("Surface Car Park", 'SURFACE CAR PARK'), ("Multi-Storey Car Park", 'MULTI-STOREY CAR PARK')
    ], label="Car Park Type")
    free_parking_dropdown = gr.Dropdown(choices=[any_choice, yes_choice, no_choice], label="Free Parking")
    night_parking_dropdown = gr.Dropdown(choices=[any_choice, yes_choice, no_choice], label="Night Parking")
    car_park_basement_dropdown = gr.Dropdown(choices=[any_choice, ('Yes', 'Y'), ('No', 'N')], label="Basement Parking")

    return location_dropdown, car_park_type_dropdown, free_parking_dropdown, night_parking_dropdown, car_park_basement_dropdown

def get_datetime_element():
    return gr.DateTime(
        value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        type="datetime",
        show_label=True,
        label="Select Preferred Datetime"
    )

def get_output_element():
    return gr.Textbox(label="Best Parking Location")


def calculate_distance(df, x, y):
    return np.sqrt((df['x_coord'] - x) ** 2 + (df['y_coord'] - y) ** 2)
