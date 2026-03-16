import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def train_logistic_pca_per_station(X_train, Y_zero_un, n_components):
    """Standardize predictors, apply PCA, then fit one logistic model per station."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    models = {}
    for station_id in tqdm(Y_zero_un.columns, desc="PCA logistic regression"):
        y_train = Y_zero_un[station_id].values
        model = LogisticRegression(max_iter=1200)
        model.fit(X_pca, y_train)
        models[station_id] = model

    return scaler, pca, models


def evaluate_logistic_pca_per_station(models, scaler, pca, X_test, Y_test_zero_un, threshold=0.5):
    """Evaluate the PCA-based logistic models."""
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    results = {}

    for station_id, model in models.items():
        y_true = Y_test_zero_un[station_id].values
        y_proba = model.predict_proba(X_test_pca)[:, 1]
        y_pred = (y_proba > threshold).astype(int)
        results[station_id] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "y_true": y_true,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }

    return results


def project_coefficients_back(model, pca):
    """Project a station coefficient vector back to the original feature space."""
    return (model.coef_ @ pca.components_).ravel()
