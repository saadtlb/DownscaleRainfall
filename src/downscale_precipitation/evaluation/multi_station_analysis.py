import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def compute_multi_station_error_tables(simulations_by_config, observed_by_station):
    """Compute RMSE and MAE for each configuration and station."""
    error_tables = {}

    for config_name, config_simulations in simulations_by_config.items():
        rows = []
        for station_id, sim_cumul in config_simulations.items():
            mean_sim = sim_cumul.mean(axis=0)
            observed = observed_by_station[station_id]
            rows.append(
                {
                    "station_id": station_id,
                    "RMSE": float(np.sqrt(mean_squared_error(observed, mean_sim))),
                    "MAE": float(mean_absolute_error(observed, mean_sim)),
                }
            )

        if rows:
            error_tables[config_name] = pd.DataFrame(rows).set_index("station_id")
        else:
            error_tables[config_name] = pd.DataFrame(columns=["RMSE", "MAE"])

    return error_tables


def best_configuration_from_errors(error_tables, metric="RMSE"):
    """Return the best configuration per station according to one error metric."""
    stations = set()
    for table in error_tables.values():
        stations.update(table.index.tolist())

    rows = []
    for station_id in sorted(stations):
        best_config = None
        best_value = np.inf
        for config_name, table in error_tables.items():
            if station_id in table.index:
                value = float(table.loc[station_id, metric])
                if value < best_value:
                    best_value = value
                    best_config = config_name
        rows.append({"station_id": station_id, "best_configuration": best_config, metric: best_value})

    return pd.DataFrame(rows).set_index("station_id")
