import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def train_logistic_per_station(X_train, Y_zero_un, max_iter=1200):
    """Train one logistic regression per station with standardized inputs."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {}
    for station_id in tqdm(Y_zero_un.columns, desc="Logistic regression"):
        y_train = Y_zero_un[station_id].values
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train_scaled, y_train)
        models[station_id] = model

    return scaler, models


def evaluate_logistic_per_station(models, scaler, X_test, Y_test_zero_un, threshold=0.5):
    """Evaluate a dictionary of station-wise logistic models."""
    X_test_scaled = scaler.transform(X_test)
    results = {}

    for station_id, model in models.items():
        y_true = Y_test_zero_un[station_id].values
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba > threshold).astype(int)
        results[station_id] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "y_true": y_true,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }

    return results


def summarize_results(results):
    """Return mean accuracy and F1 from an evaluation dictionary."""
    accuracy_by_station = {station_id: values["accuracy"] for station_id, values in results.items()}
    f1_by_station = {station_id: values["f1"] for station_id, values in results.items()}

    return {
        "accuracy_mean": float(np.mean(list(accuracy_by_station.values()))),
        "f1_mean": float(np.mean(list(f1_by_station.values()))),
        "accuracy_by_station": accuracy_by_station,
        "f1_by_station": f1_by_station,
    }


def predict_probabilities(models, scaler, X):
    """Return station-wise probabilities on a common predictor matrix."""
    X_scaled = scaler.transform(X)
    return {station_id: model.predict_proba(X_scaled)[:, 1] for station_id, model in models.items()}


def results_to_frame(results, metric):
    """Convert a results dictionary to a one-column DataFrame."""
    return pd.DataFrame({metric: {station_id: values[metric] for station_id, values in results.items()}})
