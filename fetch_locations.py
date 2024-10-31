import time
import pandas as pd
from geopy import Nominatim, Point


def get_singapore_locations():
    geolocator = Nominatim(user_agent="best_parking_location_finder_app")
    location = geolocator.geocode("Singapore")

    south, north, west, east = [float(coord) for coord in location.raw['boundingbox']]

    location_details = []
    seen_addresses = set()

    lat = south
    while lat <= north:
        lon = west
        while lon <= east:
            point = Point(lat, lon)
            try:
                reverse_location = geolocator.reverse(point, timeout=10)  # Add timeout for robustness
                if reverse_location and reverse_location.address not in seen_addresses:
                    location_details.append({
                        'latitude': lat,
                        'longitude': lon,
                        'address': reverse_location.address
                    })

                    seen_addresses.add(reverse_location.address)

            except Exception as e:
                print(f"Error at {point}: {e}")

            lon += 0.01  # Small increment for longitude
            time.sleep(1)  # Be respectful to Nominatim's servers (1-second delay per request)

        lat += 0.01  # Small increment for latitude

    if not location_details:
        print("No places found within the bounding box.")

    return list(location_details)

if __name__ == "__main__":
    # Don't call this unless needed to refresh the list of locations in SG - EXTREMELY EXPENSIVE CALL
    locations = get_singapore_locations()

    pd.DataFrame(locations).to_csv("data/sg_locations.csv")
