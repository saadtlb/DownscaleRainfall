import numpy as np

from ..data.temporal_masks import winter_year
from ..occurrence.neighborhood import predict_neighborhood_probabilities
from .gamma_gpd import simulate_rainfall_gamma_gpd


def get_winter_year(date):
    """Compatibility helper mirroring the notebook function."""
    return int(winter_year([date])[0])


def run_stochastic_simulation(
    models,
    scaler,
    X_test,
    gamma_params,
    stations_data,
    mask_test,
    n_sim=200,
):
    """Simulate mean winter rainfall for a common predictor matrix."""
    dates_test = stations_data.loc[mask_test].index
    winter_years = winter_year(dates_test)
    unique_years = np.unique(winter_years)

    all_sim_means = {}
    X_scaled = scaler.transform(X_test)

    for station_id in stations_data.columns:
        if station_id not in gamma_params or station_id not in models:
            continue

        k = gamma_params[station_id]["k"]
        theta = gamma_params[station_id]["theta"]
        proba_rain = models[station_id].predict_proba(X_scaled)[:, 1]

        sim_winter_means = np.zeros((n_sim, len(unique_years)))
        for sim_index in range(n_sim):
            rng = np.random.default_rng(sim_index)
            is_rain = rng.random(len(dates_test)) <= proba_rain
            rainfall = np.zeros(len(dates_test), dtype=float)
            n_rain = int(is_rain.sum())
            if n_rain > 0:
                rainfall[is_rain] = rng.gamma(shape=k, scale=theta, size=n_rain)

            for year_index, year in enumerate(unique_years):
                sim_winter_means[sim_index, year_index] = rainfall[winter_years == year].mean()

        all_sim_means[station_id] = sim_winter_means

    return all_sim_means, unique_years


def simulate_winter_cumul(proba, k, theta_or_mu, winter_years, unique_years, n_sim=200, glm=False):
    """Simulate winter cumulative rainfall from occurrence probabilities."""
    sim_cumul = np.zeros((n_sim, len(unique_years)))
    n_days = len(winter_years)

    for sim_index in range(n_sim):
        rng = np.random.default_rng(sim_index)
        is_rain = rng.random(n_days) <= proba
        rainfall = np.zeros(n_days, dtype=float)

        if is_rain.sum() > 0:
            if glm:
                theta_t = theta_or_mu[is_rain] / k
                for local_index, theta_value in zip(np.where(is_rain)[0], theta_t):
                    rainfall[local_index] = rng.gamma(shape=k, scale=theta_value)
            else:
                rainfall[is_rain] = rng.gamma(shape=k, scale=theta_or_mu, size=is_rain.sum())

        for year_index, year in enumerate(unique_years):
            sim_cumul[sim_index, year_index] = rainfall[winter_years == year].sum()

    return sim_cumul


def simulate_mix_cumuls(proba, params_station, winter_years, unique_years, n_sim=200):
    """Simulate winter cumulative rainfall for the Gamma + GPD extension."""
    sim_cumul = np.zeros((n_sim, len(unique_years)))

    for sim_index in range(n_sim):
        rng = np.random.default_rng(sim_index)
        rainfall = simulate_rainfall_gamma_gpd(proba, params_station, rng=rng)
        for year_index, year in enumerate(unique_years):
            sim_cumul[sim_index, year_index] = rainfall[winter_years == year].sum()

    return sim_cumul


def predict_neighborhood_simulation_probabilities(
    station_id,
    slp,
    d2,
    mask,
    neighborhood_columns,
    scalers,
    models,
):
    """Return neighborhood probabilities for one station, ready for simulation."""
    return predict_neighborhood_probabilities(station_id, slp, d2, mask, neighborhood_columns, scalers, models)

