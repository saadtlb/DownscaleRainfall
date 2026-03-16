import numpy as np


def grid_coordinates(lat, lon):
    """Return the flattened longitude and latitude coordinates of the ERA5 grid."""
    lat_values = lat.squeeze().to_numpy(dtype=float)
    lon_values = lon.squeeze().to_numpy(dtype=float)
    lon_2d, lat_2d = np.meshgrid(lon_values, lat_values)
    return lon_values, lat_values, lon_2d.ravel(), lat_2d.ravel()


def haversine_km(lon1, lat1, lon2, lat2):
    """Compute the great-circle distance in kilometers."""
    radius = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c

