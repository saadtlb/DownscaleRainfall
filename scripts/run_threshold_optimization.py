from _bootstrap import build_parser, ensure_src_on_path, project_root, resolve_data_dir

ensure_src_on_path()

import matplotlib
import numpy as np

matplotlib.use("Agg")

from downscale_precipitation.data.dataset_builders import build_Y_zero_un, split_single_variable_features
from downscale_precipitation.data.loading import load_prepared_data
from downscale_precipitation.data.temporal_masks import build_train_test_masks
from downscale_precipitation.occurrence.logistic import evaluate_logistic_per_station, train_logistic_per_station
from downscale_precipitation.occurrence.threshold_optimization import compute_f1_by_threshold, compute_optimal_threshold_roc
from downscale_precipitation.visualization.threshold_plots import (
    plot_auc_by_station,
    plot_f1_default_vs_optimal,
    plot_f1_threshold_curve,
    plot_roc_curve_station,
)


def station_label(stations, station_id):
    if station_id in stations.index and "NOM_USUEL" in stations.columns:
        return str(stations.loc[station_id, "NOM_USUEL"])
    return str(station_id)


def main():
    parser = build_parser("Cherche un threshold pertinent sur la configuration SLP.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    data = load_prepared_data(data_dir)
    mask_train, mask_test = build_train_test_masks()

    Y_train = build_Y_zero_un(data["stations_data"], mask_train)
    Y_test = build_Y_zero_un(data["stations_data"], mask_test)
    X_train_slp, X_test_slp = split_single_variable_features(data["slp"], mask_train, mask_test)

    scaler_slp, models_slp = train_logistic_per_station(X_train_slp, Y_train)
    results_test = evaluate_logistic_per_station(models_slp, scaler_slp, X_test_slp, Y_test)

    station_id = Y_test.columns[0]
    y_true = results_test[station_id]["y_true"]
    y_proba = results_test[station_id]["y_proba"]

    thresholds, f1_curve, best_f1_threshold, best_f1_score = compute_f1_by_threshold(y_true, y_proba)
    fpr, tpr, roc_thresholds, best_roc_threshold, auc = compute_optimal_threshold_roc(y_true, y_proba)

    idx_best_roc = int(np.argmin(np.abs(roc_thresholds - best_roc_threshold)))

    station_ids = list(results_test.keys())
    station_names = [station_label(data["stations"], sid) for sid in station_ids]
    f1_default = [results_test[sid]["f1"] for sid in station_ids]
    f1_optimal = []
    auc_values = []

    for sid in station_ids:
        y_true_sid = results_test[sid]["y_true"]
        y_proba_sid = results_test[sid]["y_proba"]
        _, _, _, best_f1_sid = compute_f1_by_threshold(y_true_sid, y_proba_sid)
        _, _, _, _, auc_sid = compute_optimal_threshold_roc(y_true_sid, y_proba_sid)
        f1_optimal.append(best_f1_sid)
        auc_values.append(auc_sid)

    figures_dir = project_root() / "figures" / "occurrence"
    figures_dir.mkdir(parents=True, exist_ok=True)

    label = station_label(data["stations"], station_id)

    fig_f1_curve = plot_f1_threshold_curve(thresholds, f1_curve, best_f1_threshold, best_f1_score, station_label=label)
    fig_f1_curve.savefig(figures_dir / f"threshold_f1_curve_{station_id}.png", dpi=150, bbox_inches="tight")

    fig_f1_compare = plot_f1_default_vs_optimal(station_names, f1_default, f1_optimal)
    fig_f1_compare.savefig(figures_dir / "threshold_f1_default_vs_optimal.png", dpi=150, bbox_inches="tight")

    fig_roc = plot_roc_curve_station(
        fpr,
        tpr,
        auc,
        best_fpr=float(fpr[idx_best_roc]),
        best_tpr=float(tpr[idx_best_roc]),
        station_label=label,
    )
    fig_roc.savefig(figures_dir / f"threshold_roc_{station_id}.png", dpi=150, bbox_inches="tight")

    fig_auc = plot_auc_by_station(station_names, auc_values)
    fig_auc.savefig(figures_dir / "threshold_auc_by_station.png", dpi=150, bbox_inches="tight")

    print(f"Data directory     : {data_dir}")
    print(f"Station            : {station_id}")
    print(f"Best F1 threshold  : {best_f1_threshold:.3f}")
    print(f"Best F1 score      : {best_f1_score:.3f}")
    print(f"Best ROC threshold : {best_roc_threshold:.3f}")
    print(f"ROC AUC            : {auc:.3f}")
    print(f"Figures saved in   : {figures_dir}")


if __name__ == "__main__":
    main()
