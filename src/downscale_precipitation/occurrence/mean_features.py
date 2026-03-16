from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def train_mean_models(X_train_mean, Y_zero_un, max_iter=1200):
    """Train one logistic model per station on the mean SLP and mean d2 features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_mean)

    models = {}
    for station_id in Y_zero_un.columns:
        y_train = Y_zero_un[station_id].values
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_scaled, y_train)
        models[station_id] = model

    return scaler, models


def evaluate_mean_models(models, scaler, X_test_mean, Y_test_zero_un, threshold=0.5):
    """Evaluate the mean-feature configuration."""
    X_scaled = scaler.transform(X_test_mean)
    results = {}

    for station_id, model in models.items():
        y_true = Y_test_zero_un[station_id].values
        y_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_proba > threshold).astype(int)
        results[station_id] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "y_true": y_true,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }

    return results
