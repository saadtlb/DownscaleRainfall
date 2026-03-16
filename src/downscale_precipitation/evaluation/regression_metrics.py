import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_rmse(y_true, y_pred):
    """Return the root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true, y_pred):
    """Return the mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def build_error_frame(observed, predicted_by_station):
    """Build a station-wise RMSE and MAE table."""
    rows = []
    for station_id, y_pred in predicted_by_station.items():
        y_true = observed[station_id]
        rows.append(
            {
                "station_id": station_id,
                "RMSE": compute_rmse(y_true, y_pred),
                "MAE": compute_mae(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows).set_index("station_id")

