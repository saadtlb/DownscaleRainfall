import numpy as np
import pandas as pd


def compute_rain_frequency(stations_data, mask):
    """Return the rain occurrence frequency by station."""
    rainfall = stations_data.loc[mask].astype(float)
    return (rainfall > 0).mean()


def find_extreme_days(stations_data):
    """Return one dry day and one very rainy day from the station observations."""
    rainfall = stations_data.astype(float)
    total_rain = rainfall.sum(axis=1)
    rainy_day = total_rain.idxmax()
    dry_candidates = total_rain[total_rain == 0]
    dry_day = dry_candidates.index[0]
    return rainy_day, dry_day, total_rain


def positive_rainfall_by_station(stations_data, mask):
    """Return positive rainfall values station by station."""
    rainfall = stations_data.loc[mask].astype(float)
    return {station_id: rainfall[station_id][rainfall[station_id] > 0] for station_id in rainfall.columns}


def mean_fields_for_dry_and_rainy_days(slp, d2, stations_data, mask):
    """Compare mean predictor fields for dry and rainy days."""
    rainfall = stations_data.loc[mask].astype(float)
    rainy_days = rainfall.sum(axis=1) > 0
    return {
        "slp_dry_mean": slp.loc[mask].loc[~rainy_days].mean(axis=0),
        "slp_rain_mean": slp.loc[mask].loc[rainy_days].mean(axis=0),
        "d2_dry_mean": d2.loc[mask].loc[~rainy_days].mean(axis=0),
        "d2_rain_mean": d2.loc[mask].loc[rainy_days].mean(axis=0),
    }

