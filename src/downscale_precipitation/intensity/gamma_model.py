from math import gamma as gamma_function

import numpy as np


def gamma_pdf(x, k, theta):
    """Compute the Gamma density."""
    x = np.asarray(x, dtype=float)
    return (x ** (k - 1) * np.exp(-x / theta)) / (gamma_function(k) * theta ** k)


def fit_gamma_station(r_positive):
    """Estimate Gamma parameters with the method of moments."""
    values = np.asarray(r_positive, dtype=float)
    mean_value = np.mean(values)
    variance = np.var(values)
    k = mean_value ** 2 / variance
    theta = variance / mean_value
    return k, theta


def fit_gamma_all_stations(stations_data, mask, min_positive_days=10):
    """Fit a Gamma distribution for each station."""
    gamma_params = {}
    rainfall = stations_data.loc[mask].astype(float)

    for station_id in rainfall.columns:
        values = rainfall[station_id]
        positive = values[values > 0]
        if len(positive) >= min_positive_days:
            k, theta = fit_gamma_station(positive)
            gamma_params[station_id] = {"k": float(k), "theta": float(theta)}

    return gamma_params

