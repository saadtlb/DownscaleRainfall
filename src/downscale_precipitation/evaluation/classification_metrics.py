import numpy as np
import pandas as pd


def mean_metric(results, metric):
    """Return the average value of a metric over stations."""
    return float(np.mean([values[metric] for values in results.values()]))


def metric_by_station(results, metric):
    """Return the chosen metric by station."""
    return {station_id: values[metric] for station_id, values in results.items()}


def metrics_frame(results):
    """Convert accuracy and F1 results to a DataFrame."""
    rows = []
    for station_id, values in results.items():
        rows.append({"station_id": station_id, "accuracy": values["accuracy"], "f1": values["f1"]})
    return pd.DataFrame(rows).set_index("station_id")

