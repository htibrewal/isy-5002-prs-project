import requests


# Function to get user's approximate location using an IP geolocation service
def get_current_location():
    response = requests.get("https://ipinfo.io")
    data = response.json()
    latitude, longitude = map(float, data["loc"].split(","))
    return latitude, longitude


# Function to generate Google Maps embed URL and display in iframe
def generate_map(dest_lat, dest_lon):
    current_lat, current_lon = get_current_location()
    print(f"Current Latitude: {current_lat}, Current Longitude: {current_lon}")

    # Generate Google Maps Directions URL
    embed_url = f"https://www.google.com/maps/embed/v1/directions?key=AIzaSyAIVm0N0NA9hox8QIBMCEEeTso6ZmSluDQ&origin={current_lat},{current_lon}&destination={dest_lat},{dest_lon}&mode=driving"

    # Return HTML iframe
    html_code = f'<iframe src="{embed_url}" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe>'
    return html_code

