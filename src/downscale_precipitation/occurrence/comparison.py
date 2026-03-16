import numpy as np
import pandas as pd

from .logistic import summarize_results


def summarize_configuration(name, train_results, test_results):
    """Build the report-style summary for one configuration."""
    train_summary = summarize_results(train_results)
    test_summary = summarize_results(test_results)

    return {
        "name": name,
        "acc_train": train_summary["accuracy_mean"],
        "acc_test": test_summary["accuracy_mean"],
        "f1_train": train_summary["f1_mean"],
        "f1_test": test_summary["f1_mean"],
        "acc_train_per_station": train_summary["accuracy_by_station"],
        "acc_test_per_station": test_summary["accuracy_by_station"],
        "f1_train_per_station": train_summary["f1_by_station"],
        "f1_test_per_station": test_summary["f1_by_station"],
    }


def build_configuration_summary_table(configurations):
    """Return a tabular summary of all tested occurrence configurations."""
    rows = []
    for config in configurations:
        rows.append(
            {
                "Configuration": config["name"],
                "Acc Train": config["acc_train"],
                "Acc Test": config["acc_test"],
                "F1 Train": config["f1_train"],
                "F1 Test": config["f1_test"],
            }
        )
    return pd.DataFrame(rows)


def build_metric_radar_data(configurations, metric_key):
    """Return a dictionary ready to feed the radar plot helper."""
    return {config["name"]: config[metric_key] for config in configurations}


def best_configuration_per_station(configurations, metric_key="f1_test_per_station"):
    """Return the best configuration by station for a chosen metric."""
    stations = next(iter(configurations))[metric_key].keys()
    best = {}

    for station_id in stations:
        best_name = None
        best_score = -np.inf
        for config in configurations:
            score = config[metric_key][station_id]
            if score > best_score:
                best_score = score
                best_name = config["name"]
        best[station_id] = {"configuration": best_name, "score": float(best_score)}

    return best

