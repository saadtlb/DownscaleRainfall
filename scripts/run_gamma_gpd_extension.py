from _bootstrap import build_parser, ensure_src_on_path, project_root, resolve_data_dir

ensure_src_on_path()

import matplotlib

matplotlib.use("Agg")

from downscale_precipitation.data.loading import load_prepared_data
from downscale_precipitation.data.temporal_masks import build_train_test_masks
from downscale_precipitation.intensity.gamma_gpd import fit_gamma_gpd_mixture_all_stations
from downscale_precipitation.visualization.validation_plots import plot_tail_comparison_station


def main():
    parser = build_parser("Ajuste l'extension Gamma + GPD et sauvegarde une figure de queue.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    data = load_prepared_data(data_dir)
    mask_train, _ = build_train_test_masks()

    params = fit_gamma_gpd_mixture_all_stations(data["stations_data"], mask_train)
    first_station = next(iter(params))

    figures_dir = project_root() / "figures" / "extensions"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_tail_comparison_station(first_station, data["stations_data"], data["stations"], mask_train)
    fig.savefig(figures_dir / f"tail_comparison_{first_station}.png", dpi=150, bbox_inches="tight")

    print(f"Data directory             : {data_dir}")
    print(f"Fitted Gamma + GPD stations: {len(params)}")
    print(f"Example station            : {first_station}")
    print(f"Parameters                 : {params[first_station]}")
    print(f"Figure saved in            : {figures_dir}")


if __name__ == "__main__":
    main()
