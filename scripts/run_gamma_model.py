from _bootstrap import build_parser, ensure_src_on_path, project_root, resolve_data_dir

ensure_src_on_path()

import matplotlib
import numpy as np

matplotlib.use("Agg")

from downscale_precipitation.data.dataset_builders import build_Y_zero_un, split_full_features
from downscale_precipitation.data.loading import load_prepared_data
from downscale_precipitation.data.temporal_masks import build_train_test_masks
from downscale_precipitation.data.temporal_masks import winter_year
from downscale_precipitation.evaluation.seasonal_stats import compute_winter_stats
from downscale_precipitation.intensity.gamma_model import fit_gamma_all_stations
from downscale_precipitation.intensity.simulation import run_stochastic_simulation, simulate_winter_cumul
from downscale_precipitation.occurrence.logistic import train_logistic_per_station
from downscale_precipitation.visualization.intensity_plots import plot_gamma_fit, plot_gamma_qq
from downscale_precipitation.visualization.seasonal_plots import plot_winter_cumulative_envelope


def station_label(stations, station_id):
    if station_id in stations.index and "NOM_USUEL" in stations.columns:
        return str(stations.loc[station_id, "NOM_USUEL"])
    return str(station_id)


def main():
    parser = build_parser("Ajuste la Gamma simple et lance une simulation stochastique.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    data = load_prepared_data(data_dir)
    mask_train, mask_test = build_train_test_masks()

    Y_train = build_Y_zero_un(data["stations_data"], mask_train)
    X_train_full, X_test_full = split_full_features(data["slp"], data["d2"], mask_train, mask_test)

    scaler_full, models_full = train_logistic_per_station(X_train_full, Y_train)
    gamma_params = fit_gamma_all_stations(data["stations_data"], mask_train)
    simulations, years = run_stochastic_simulation(
        models_full,
        scaler_full,
        X_test_full,
        gamma_params,
        data["stations_data"],
        mask_test,
        n_sim=50,
    )

    first_station = next(iter(gamma_params))
    station_name = station_label(data["stations"], first_station)

    rain_train = data["stations_data"].loc[mask_train, first_station].astype(float)
    rain_positive = rain_train[rain_train > 0].to_numpy()
    k = gamma_params[first_station]["k"]
    theta = gamma_params[first_station]["theta"]

    X_test_scaled = scaler_full.transform(X_test_full)
    proba_test = models_full[first_station].predict_proba(X_test_scaled)[:, 1]
    dates_test = data["stations_data"].loc[mask_test].index
    winter_years = winter_year(dates_test)
    unique_years = np.unique(winter_years)
    sim_cumul = simulate_winter_cumul(proba_test, k, theta, winter_years, unique_years, n_sim=200, glm=False)
    winter_stats = compute_winter_stats(sim_cumul, first_station, data["stations_data"], mask_test, aggregation="mean")

    figures_dir = project_root() / "figures" / "gamma"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_fit = plot_gamma_fit(rain_positive, k, theta, station_name=station_name)
    fig_fit.savefig(figures_dir / f"gamma_fit_{first_station}.png", dpi=150, bbox_inches="tight")

    fig_qq = plot_gamma_qq(rain_positive, k, theta, station_name=station_name)
    fig_qq.savefig(figures_dir / f"gamma_qq_{first_station}.png", dpi=150, bbox_inches="tight")

    fig_cumul = plot_winter_cumulative_envelope(
        winter_stats["years"],
        winter_stats["obs"],
        winter_stats["mean_sim"],
        winter_stats["p10"],
        winter_stats["p90"],
        title=f"Gamma simple - winter daily mean ({station_name})",
        ylabel="Daily mean rainfall (mm/day)",
    )
    fig_cumul.savefig(figures_dir / f"gamma_daily_mean_{first_station}.png", dpi=150, bbox_inches="tight")

    print(f"Data directory       : {data_dir}")
    print(f"Fitted Gamma stations: {len(gamma_params)}")
    print(f"Simulation winters   : {list(years)}")
    print(f"Example station      : {first_station}")
    print(f"Gamma parameters     : {gamma_params[first_station]}")
    print(f"Simulation shape     : {simulations[first_station].shape}")
    print(f"Figures saved in     : {figures_dir}")


if __name__ == "__main__":
    main()
