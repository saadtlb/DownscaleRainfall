from _bootstrap import build_parser, ensure_src_on_path, project_root, resolve_data_dir

ensure_src_on_path()

import matplotlib

matplotlib.use("Agg")

from downscale_precipitation.data.loading import load_prepared_data
from downscale_precipitation.data.temporal_masks import build_train_test_masks
from downscale_precipitation.exploration.correlations import (
    compute_station_correlations,
    compute_station_distance_matrix,
)
from downscale_precipitation.exploration.diagnostics import compute_rain_frequency, find_extreme_days
from downscale_precipitation.exploration.diagnostics import mean_fields_for_dry_and_rainy_days
from downscale_precipitation.visualization.exploration_plots import (
    plot_altitude_vs_rain_frequency,
    plot_mean_fields_dry_vs_rainy,
    plot_positive_rainfall_histograms,
    plot_station_correlation_heatmaps,
    plot_station_locations,
)
from downscale_precipitation.visualization.maps import plot_compare_days, plot_slp_d2_graph


def main():
    parser = build_parser("Genere quelques figures d'exploration.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    data = load_prepared_data(data_dir)
    mask_train, _ = build_train_test_masks()

    rain_frequency = compute_rain_frequency(data["stations_data"], mask_train)
    rainy_day, dry_day, _ = find_extreme_days(data["stations_data"])
    corr_occ, corr_int = compute_station_correlations(data["stations_data"], mask_train)
    dist = compute_station_distance_matrix(data["stations"], corr_occ.columns)

    figures_dir = project_root() / "figures" / "exploration"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_station_correlation_heatmaps(corr_occ, corr_int, dist)
    fig1.savefig(figures_dir / "station_correlations.png", dpi=150, bbox_inches="tight")

    fig2 = plot_compare_days(
        data["slp"],
        data["d2"],
        data["lat"],
        data["lon"],
        data["stations"],
        rainy_day,
        dry_day,
        data["stations_data"],
    )
    fig2.savefig(figures_dir / "dry_vs_rainy_day.png", dpi=150, bbox_inches="tight")

    fig3 = plot_station_locations(data["stations"])
    fig3.savefig(figures_dir / "station_locations.png", dpi=150, bbox_inches="tight")

    fig4 = plot_slp_d2_graph(
        data["slp"],
        data["d2"],
        data["lat"],
        data["lon"],
        data["stations"],
        rainy_day,
    )
    fig4.savefig(figures_dir / "slp_d2_one_day_with_table.png", dpi=150, bbox_inches="tight")

    fig5 = plot_altitude_vs_rain_frequency(data["stations"], rain_frequency)
    fig5.savefig(figures_dir / "altitude_vs_rain_frequency.png", dpi=150, bbox_inches="tight")

    fig6 = plot_positive_rainfall_histograms(data["stations_data"], data["stations"], mask_train)
    fig6.savefig(figures_dir / "station_positive_histograms.png", dpi=150, bbox_inches="tight")

    mean_fields = mean_fields_for_dry_and_rainy_days(data["slp"], data["d2"], data["stations_data"], mask_train)
    fig7 = plot_mean_fields_dry_vs_rainy(mean_fields, data["lat"], data["lon"], data["stations"])
    fig7.savefig(figures_dir / "mean_fields_dry_vs_rainy.png", dpi=150, bbox_inches="tight")

    print("Exploration finished.")
    print(f"Data directory: {data_dir}")
    print(f"Example day for SLP/d2 table: {rainy_day}")
    print("Rain frequency by station:")
    print(rain_frequency.round(3))
    print(f"Figures saved in: {figures_dir}")


if __name__ == "__main__":
    main()
