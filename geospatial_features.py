from pyproj import Proj, Transformer


def convert_xy_to_lat_long(x, y):
    svy21_proj = Proj("epsg:3414")
    wgs84_proj = Proj(proj="latlong", datum="WGS84")

    transformer = Transformer.from_proj(svy21_proj, wgs84_proj)
    longitude, latitude = transformer.transform(x, y)

    return latitude, longitude


def convert_lat_long_to_xy(latitude, longitude):
    wgs84_proj = Proj(proj="latlong", datum="WGS84")
    svy21_proj = Proj("epsg:3414")

    transformer = Transformer.from_proj(wgs84_proj, svy21_proj)
    x, y = transformer.transform(longitude, latitude)

    return x, y
