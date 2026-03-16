import pandas as pd

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
from downscale_precipitation.intensity.gamma_glm import fit_gamma_k, fit_glm_gamma, predict_glm_mu
from downscale_precipitation.intensity.gamma_model import fit_gamma_all_stations
from downscale_precipitation.intensity.simulation import simulate_winter_cumul
from downscale_precipitation.occurrence.logistic import train_logistic_per_station
from downscale_precipitation.visualization.seasonal_plots import (
    plot_gamma_vs_glm_cumulative,
    plot_winter_cumulative_envelope,
)


def station_label(stations, station_id):
    if station_id in stations.index and "NOM_USUEL" in stations.columns:
        return str(stations.loc[station_id, "NOM_USUEL"])
    return str(station_id)


def main():
    parser = build_parser("Ajuste les GLM Gamma par station.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    data = load_prepared_data(data_dir)
    mask_train, mask_test = build_train_test_masks()

    Y_train = build_Y_zero_un(data["stations_data"], mask_train)
    X_train_full, X_test_full = split_full_features(data["slp"], data["d2"], mask_train, mask_test)

    scaler_occurrence, occurrence_models = train_logistic_per_station(X_train_full, Y_train)

    gamma_params = fit_gamma_all_stations(data["stations_data"], mask_train)
    k_params = fit_gamma_k(data["stations_data"], mask_train)
    scaler_glm, glm_models = fit_glm_gamma(X_train_full, data["stations_data"], mask_train)

    first_station = next(iter(glm_models))
    station_name = station_label(data["stations"], first_station)
    mu = predict_glm_mu(glm_models[first_station], scaler_glm, X_test_full)

    X_test_scaled = scaler_occurrence.transform(X_test_full)
    proba_test = occurrence_models[first_station].predict_proba(X_test_scaled)[:, 1]

    dates_test = data["stations_data"].loc[mask_test].index
    winter_years = winter_year(dates_test)
    unique_years = np.unique(winter_years)

    k_glm = k_params[first_station]
    sim_cumul_glm = simulate_winter_cumul(proba_test, k_glm, mu, winter_years, unique_years, n_sim=200, glm=True)
    stats_glm = compute_winter_stats(sim_cumul_glm, first_station, data["stations_data"], mask_test, aggregation="mean")

    k_gamma = gamma_params[first_station]["k"]
    theta_gamma = gamma_params[first_station]["theta"]
    sim_cumul_gamma = simulate_winter_cumul(
        proba_test,
        k_gamma,
        theta_gamma,
        winter_years,
        unique_years,
        n_sim=200,
        glm=False,
    )
    stats_gamma = compute_winter_stats(sim_cumul_gamma, first_station, data["stations_data"], mask_test, aggregation="mean")

    figures_dir = project_root() / "figures" / "gamma_glm"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_glm = plot_winter_cumulative_envelope(
        stats_glm["years"],
        stats_glm["obs"],
        stats_glm["mean_sim"],
        stats_glm["p10"],
        stats_glm["p90"],
        title=f"Gamma GLM - winter daily mean ({station_name})",
        ylabel="Daily mean rainfall (mm/day)",
    )
    fig_glm.savefig(figures_dir / f"glm_daily_mean_{first_station}.png", dpi=150, bbox_inches="tight")

    fig_compare = plot_gamma_vs_glm_cumulative(
        stats_glm["years"],
        stats_glm["obs"],
        stats_gamma["mean_sim"],
        stats_gamma["p10"],
        stats_gamma["p90"],
        stats_glm["mean_sim"],
        stats_glm["p10"],
        stats_glm["p90"],
        title=f"Gamma vs Gamma-GLM daily mean ({station_name})",
        ylabel="Daily mean rainfall (mm/day)",
    )
    fig_compare.savefig(figures_dir / f"gamma_vs_glm_{first_station}.png", dpi=150, bbox_inches="tight")

    print(f"Data directory      : {data_dir}")
    print(f"Fitted GLM stations : {len(glm_models)}")
    print(f"Available k params  : {len(k_params)}")
    print(f"Example station     : {first_station}")
    print(f"First predicted mu  : {pd.Series(mu).head().round(3).tolist()}")
    print(f"Figures saved in    : {figures_dir}")


if __name__ == "__main__":
    main()
