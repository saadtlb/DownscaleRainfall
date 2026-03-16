import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from ..exploration.geographic import grid_coordinates


def extract_station_neighborhood(distance_km, station_id, slp, d2, lat, lon, stations):
    """Extract the ERA5 pixels located within a radius around one station."""
    lon_values, lat_values, lon_flat, lat_flat = grid_coordinates(lat, lon)
    del lon_values, lat_values

    station = stations.loc[station_id]
    lat_station = float(station["LAT"])
    lon_station = float(station["LON"])

    radius = 6371.0
    phi_station = np.deg2rad(lat_station)
    phi_grid = np.deg2rad(lat_flat)
    dphi = phi_grid - phi_station
    dlambda = np.deg2rad(lon_flat - lon_station)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi_station) * np.cos(phi_grid) * np.sin(dlambda / 2) ** 2
    dist = 2 * radius * np.arcsin(np.sqrt(a))

    mask = dist <= distance_km
    pixel_indices = np.where(mask)[0]

    slp_local = slp.iloc[:, pixel_indices].copy() if slp is not None else None
    d2_local = d2.iloc[:, pixel_indices].copy() if d2 is not None else None
    coordinates = list(zip(lon_flat[mask], lat_flat[mask]))

    return slp_local, d2_local, coordinates


def build_neighborhood_columns(distance_km, slp, d2, lat, lon, stations):
    """Store the selected SLP and d2 columns for each station."""
    neighborhood_columns = {}
    for station_id in stations.index:
        slp_local, d2_local, _ = extract_station_neighborhood(distance_km, station_id, slp, d2, lat, lon, stations)
        neighborhood_columns[station_id] = {
            "slp_cols": list(slp_local.columns) if slp_local is not None else [],
            "d2_cols": list(d2_local.columns) if d2_local is not None else [],
        }
    return neighborhood_columns


def build_neighborhood_feature_matrix(station_id, slp, d2, mask, neighborhood_columns):
    """Build the local feature matrix for one station."""
    config = neighborhood_columns[station_id]
    features = []

    if slp is not None and config["slp_cols"]:
        features.append(slp.loc[mask, config["slp_cols"]].to_numpy())
    if d2 is not None and config["d2_cols"]:
        features.append(d2.loc[mask, config["d2_cols"]].to_numpy())

    if not features:
        raise ValueError(f"No neighborhood predictors available for station {station_id}.")

    return np.concatenate(features, axis=1)


def train_neighborhood_models(slp, d2, stations, mask_train, Y_train, neighborhood_columns):
    """Train one local logistic model per station."""
    models = {}
    scalers = {}

    for station_id in stations.index:
        X_train = build_neighborhood_feature_matrix(station_id, slp, d2, mask_train, neighborhood_columns)
        y_train = Y_train[station_id].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(max_iter=1200)
        model.fit(X_train_scaled, y_train)

        models[station_id] = model
        scalers[station_id] = scaler

    return scalers, models


def evaluate_neighborhood_models(slp, d2, stations, mask_test, Y_test, neighborhood_columns, scalers, models, threshold=0.5):
    """Evaluate the local neighborhood models on the test period."""
    results = {}

    for station_id in stations.index:
        X_test = build_neighborhood_feature_matrix(station_id, slp, d2, mask_test, neighborhood_columns)
        X_test_scaled = scalers[station_id].transform(X_test)
        y_true = Y_test[station_id].values
        y_proba = models[station_id].predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba > threshold).astype(int)

        results[station_id] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "y_true": y_true,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }

    return results


def predict_neighborhood_probabilities(station_id, slp, d2, mask, neighborhood_columns, scalers, models):
    """Predict probabilities for one station with the neighborhood features."""
    X = build_neighborhood_feature_matrix(station_id, slp, d2, mask, neighborhood_columns)
    X_scaled = scalers[station_id].transform(X)
    return models[station_id].predict_proba(X_scaled)[:, 1]
