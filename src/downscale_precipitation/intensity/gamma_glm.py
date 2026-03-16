import warnings

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def fit_gamma_k(stations_data, mask):
    """Estimate the Gamma shape parameter k for each station."""
    rainfall = stations_data.loc[mask].astype(float)
    k_params = {}

    for station_id in rainfall.columns:
        values = rainfall[station_id].values
        positive = values[values > 0]
        if len(positive) < 2:
            continue

        mean_value = positive.mean()
        variance = positive.var(ddof=1)
        k_params[station_id] = float(mean_value ** 2 / variance)

    return k_params


def fit_glm_gamma(X_train, stations_data, mask, verbose=False):
    """Fit one Gamma GLM per station on rainy days only."""
    rainfall = stations_data.loc[mask].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled = pd.DataFrame(X_scaled, index=X_train.index, columns=X_train.columns)

    models = {}
    for station_id in rainfall.columns:
        values = rainfall[station_id].values
        rainy_mask = values > 0
        if rainy_mask.sum() < 10:
            continue

        X_rain = X_scaled.loc[rainy_mask]
        y_rain = values[rainy_mask]
        X_rain = sm.add_constant(X_rain, has_constant="add")

        try:
            glm = sm.GLM(y_rain, X_rain, family=sm.families.Gamma(link=sm.families.links.Log()))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                models[station_id] = glm.fit(disp=0)
        except Exception as error:
            if verbose:
                print(f"Station {station_id}: GLM error - {error}")

    return scaler, models


def predict_glm_mu(model, scaler, X):
    """Predict the Gamma GLM conditional mean on new data."""
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    X_scaled = sm.add_constant(X_scaled, has_constant="add")
    predictions = model.predict(X_scaled)
    return predictions.values if hasattr(predictions, "values") else predictions
