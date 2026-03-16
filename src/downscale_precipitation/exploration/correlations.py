import numpy as np
import pandas as pd

from .geographic import haversine_km


def compute_station_correlations(stations_data, mask):
    """Return occurrence and intensity correlation matrices."""
    rainfall = stations_data.loc[mask].astype(float)
    occurrence = (rainfall > 0).astype(int)
    intensity = rainfall.where(rainfall > 0, np.nan)
    return occurrence.corr(method="pearson"), intensity.corr(method="pearson")


def compute_station_distance_matrix(stations, station_ids):
    """Return the pairwise distance matrix between stations."""
    aligned = stations.loc[station_ids].copy()
    n_stations = len(aligned)
    distances = np.zeros((n_stations, n_stations), dtype=float)

    for i in range(n_stations):
        for j in range(n_stations):
            distances[i, j] = haversine_km(
                float(aligned.iloc[i]["LON"]),
                float(aligned.iloc[i]["LAT"]),
                float(aligned.iloc[j]["LON"]),
                float(aligned.iloc[j]["LAT"]),
            )

    return pd.DataFrame(distances, index=aligned.index, columns=aligned.index)

