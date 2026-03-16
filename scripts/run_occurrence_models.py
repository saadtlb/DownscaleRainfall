from _bootstrap import build_parser, ensure_src_on_path, project_root, resolve_data_dir

ensure_src_on_path()

import matplotlib

matplotlib.use("Agg")

from downscale_precipitation.data.dataset_builders import (
    build_Y_zero_un,
    build_mean_features,
    split_full_features,
    split_single_variable_features,
)
from downscale_precipitation.data.loading import load_prepared_data
from downscale_precipitation.data.temporal_masks import build_train_test_masks
from downscale_precipitation.occurrence.comparison import (
    build_metric_radar_data,
    build_configuration_summary_table,
    summarize_configuration,
)
from downscale_precipitation.occurrence.logistic import evaluate_logistic_per_station, train_logistic_per_station
from downscale_precipitation.occurrence.mean_features import evaluate_mean_models, train_mean_models
from downscale_precipitation.occurrence.neighborhood import (
    build_neighborhood_columns,
    evaluate_neighborhood_models,
    train_neighborhood_models,
)
from downscale_precipitation.occurrence.pca import (
    evaluate_logistic_pca_per_station,
    train_logistic_pca_per_station,
)
from downscale_precipitation.visualization.occurrence_plots import plot_all_configs_coefficients, plot_radar


def main():
    parser = build_parser("Entraine et compare les modeles d'occurrence.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    data = load_prepared_data(data_dir)
    mask_train, mask_test = build_train_test_masks()

    Y_train = build_Y_zero_un(data["stations_data"], mask_train)
    Y_test = build_Y_zero_un(data["stations_data"], mask_test)

    X_train_full, X_test_full = split_full_features(data["slp"], data["d2"], mask_train, mask_test)
    X_train_slp, X_test_slp = split_single_variable_features(data["slp"], mask_train, mask_test)
    X_train_d2, X_test_d2 = split_single_variable_features(data["d2"], mask_train, mask_test)
    X_train_mean = build_mean_features(data["slp"], data["d2"], mask_train)
    X_test_mean = build_mean_features(data["slp"], data["d2"], mask_test)

    scaler_full, models_full = train_logistic_per_station(X_train_full, Y_train)
    results_train_full = evaluate_logistic_per_station(models_full, scaler_full, X_train_full, Y_train)
    results_test_full = evaluate_logistic_per_station(models_full, scaler_full, X_test_full, Y_test)

    scaler_slp, models_slp = train_logistic_per_station(X_train_slp, Y_train)
    results_train_slp = evaluate_logistic_per_station(models_slp, scaler_slp, X_train_slp, Y_train)
    results_test_slp = evaluate_logistic_per_station(models_slp, scaler_slp, X_test_slp, Y_test)

    scaler_d2, models_d2 = train_logistic_per_station(X_train_d2, Y_train)
    results_train_d2 = evaluate_logistic_per_station(models_d2, scaler_d2, X_train_d2, Y_train)
    results_test_d2 = evaluate_logistic_per_station(models_d2, scaler_d2, X_test_d2, Y_test)

    neighborhood_columns = build_neighborhood_columns(100, data["slp"], data["d2"], data["lat"], data["lon"], data["stations"])
    scalers_neigh, models_neigh = train_neighborhood_models(
        data["slp"], data["d2"], data["stations"], mask_train, Y_train, neighborhood_columns
    )
    results_train_neigh = evaluate_neighborhood_models(
        data["slp"], data["d2"], data["stations"], mask_train, Y_train, neighborhood_columns, scalers_neigh, models_neigh
    )
    results_test_neigh = evaluate_neighborhood_models(
        data["slp"], data["d2"], data["stations"], mask_test, Y_test, neighborhood_columns, scalers_neigh, models_neigh
    )

    scaler_pca, pca, models_pca = train_logistic_pca_per_station(X_train_full, Y_train, n_components=20)
    results_train_pca = evaluate_logistic_pca_per_station(models_pca, scaler_pca, pca, X_train_full, Y_train)
    results_test_pca = evaluate_logistic_pca_per_station(models_pca, scaler_pca, pca, X_test_full, Y_test)

    scaler_mean, models_mean = train_mean_models(X_train_mean, Y_train)
    results_train_mean = evaluate_mean_models(models_mean, scaler_mean, X_train_mean, Y_train)
    results_test_mean = evaluate_mean_models(models_mean, scaler_mean, X_test_mean, Y_test)

    configurations = [
        summarize_configuration("SLP + d2", results_train_full, results_test_full),
        summarize_configuration("SLP only", results_train_slp, results_test_slp),
        summarize_configuration("d2 only", results_train_d2, results_test_d2),
        summarize_configuration("Neighborhood 100km", results_train_neigh, results_test_neigh),
        summarize_configuration("PCA", results_train_pca, results_test_pca),
        summarize_configuration("Mean", results_train_mean, results_test_mean),
    ]

    figures_dir = project_root() / "figures" / "occurrence"
    figures_dir.mkdir(parents=True, exist_ok=True)

    radar_accuracy = build_metric_radar_data(configurations, "acc_test_per_station")
    radar_f1 = build_metric_radar_data(configurations, "f1_test_per_station")

    fig_acc = plot_radar(radar_accuracy, data["stations"], title="Accuracy (test) by station")
    fig_acc.savefig(figures_dir / "radar_accuracy_test.png", dpi=150, bbox_inches="tight")

    fig_f1 = plot_radar(radar_f1, data["stations"], title="F1 (test) by station")
    fig_f1.savefig(figures_dir / "radar_f1_test.png", dpi=150, bbox_inches="tight")

    first_station = Y_train.columns[0]
    fig_coef = plot_all_configs_coefficients(
        first_station,
        models_full,
        models_slp,
        models_d2,
        models_neigh,
        models_pca,
        models_mean,
        pca,
        data["slp"],
        data["d2"],
        data["lat"],
        data["lon"],
        data["stations"],
        distance_km=100,
    )
    fig_coef.savefig(figures_dir / f"coefficients_{first_station}.png", dpi=150, bbox_inches="tight")

    summary = build_configuration_summary_table(configurations)
    print(f"Data directory: {data_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Occurrence figures saved in: {figures_dir}")


if __name__ == "__main__":
    main()
