from _bootstrap import build_parser, ensure_src_on_path, project_root, resolve_data_dir

ensure_src_on_path()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

from downscale_precipitation.data.dataset_builders import (
    build_Y_zero_un,
    build_mean_features,
    split_full_features,
    split_single_variable_features,
)
from downscale_precipitation.data.loading import load_prepared_data
from downscale_precipitation.data.temporal_masks import build_train_test_masks, winter_year
from downscale_precipitation.evaluation.multi_station_analysis import (
    best_configuration_from_errors,
    compute_multi_station_error_tables,
)
from downscale_precipitation.intensity.gamma_glm import fit_gamma_k, fit_glm_gamma, predict_glm_mu
from downscale_precipitation.intensity.gamma_model import fit_gamma_all_stations
from downscale_precipitation.intensity.simulation import (
    predict_neighborhood_simulation_probabilities,
    simulate_winter_cumul,
)
from downscale_precipitation.occurrence.comparison import (
    best_configuration_per_station,
    summarize_configuration,
)
from downscale_precipitation.occurrence.logistic import (
    evaluate_logistic_per_station,
    predict_probabilities,
    train_logistic_per_station,
)
from downscale_precipitation.occurrence.mean_features import evaluate_mean_models, train_mean_models
from downscale_precipitation.occurrence.neighborhood import (
    build_neighborhood_columns,
    build_neighborhood_feature_matrix,
    evaluate_neighborhood_models,
    train_neighborhood_models,
)
from downscale_precipitation.occurrence.pca import evaluate_logistic_pca_per_station, train_logistic_pca_per_station
from downscale_precipitation.visualization.comparison_plots import plot_error_heatmap

CONFIG_ORDER = ["SLP + d2", "SLP only", "d2 only", "Neighborhood 100km", "PCA", "Mean"]
CONFIG_COLORS = {
    "SLP + d2": "#1f77b4",
    "SLP only": "#ff7f0e",
    "d2 only": "#2ca02c",
    "Neighborhood 100km": "#d62728",
    "PCA": "#9467bd",
    "Mean": "#8c564b",
}
N_SIM = 120


def station_label(stations, station_id):
    if station_id in stations.index and "NOM_USUEL" in stations.columns:
        return str(stations.loc[station_id, "NOM_USUEL"])
    return str(station_id)


def to_daily_mean(values, day_counts):
    day_counts = np.asarray(day_counts, dtype=float)
    return np.asarray(values, dtype=float) / day_counts.reshape(1, -1)


def fit_neighborhood_glm_models(slp, d2, stations, stations_data, mask_train, neighborhood_columns):
    """Fit one Gamma-GLM per station using neighborhood-specific predictors."""
    rainfall = stations_data.loc[mask_train].astype(float)
    models = {}
    scalers = {}

    for station_id in stations.index:
        values = rainfall[station_id].values
        rainy_mask = values > 0
        if rainy_mask.sum() < 10:
            continue

        X_train = build_neighborhood_feature_matrix(station_id, slp, d2, mask_train, neighborhood_columns)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_rain = sm.add_constant(X_scaled[rainy_mask], has_constant="add")
        y_rain = values[rainy_mask]

        try:
            glm = sm.GLM(y_rain, X_rain, family=sm.families.Gamma(link=sm.families.links.Log()))
            models[station_id] = glm.fit(disp=0)
            scalers[station_id] = scaler
        except Exception:
            continue

    return scalers, models


def predict_neighborhood_glm_mu(station_id, slp, d2, mask_test, neighborhood_columns, scalers, models):
    if station_id not in models or station_id not in scalers:
        return None
    X_test = build_neighborhood_feature_matrix(station_id, slp, d2, mask_test, neighborhood_columns)
    X_scaled = scalers[station_id].transform(X_test)
    X_scaled = sm.add_constant(X_scaled, has_constant="add")
    mu = models[station_id].predict(X_scaled)
    return mu.values if hasattr(mu, "values") else np.asarray(mu, dtype=float)


def plot_configuration_envelope_grid(simulations_by_config, years, observed, title, ylabel):
    configs = [cfg for cfg in CONFIG_ORDER if cfg in simulations_by_config]
    n_cfg = len(configs)
    n_cols = min(3, n_cfg)
    n_rows = int(np.ceil(n_cfg / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, cfg in enumerate(configs):
        ax = axes[idx]
        sims = simulations_by_config[cfg]
        mean_sim = sims.mean(axis=0)
        p10 = np.percentile(sims, 10, axis=0)
        p90 = np.percentile(sims, 90, axis=0)
        rmse = np.sqrt(mean_squared_error(observed, mean_sim))
        mae = mean_absolute_error(observed, mean_sim)
        inside = np.mean((observed >= p10) & (observed <= p90)) * 100.0

        color = CONFIG_COLORS[cfg]
        ax.fill_between(years, p10, p90, alpha=0.25, color=color, label="P10-P90")
        ax.plot(years, mean_sim, "--", color=color, linewidth=2, label="Simulation mean")
        ax.plot(years, observed, "ko-", linewidth=2, markersize=4, label="Observed")
        ax.set_title(cfg, fontsize=11, fontweight="bold")
        ax.set_xlabel("Winter year")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.text(
            0.02,
            0.98,
            f"Inside: {inside:.0f}%\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        ax.legend(fontsize=7, loc="upper right")

    for idx in range(n_cfg, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_configuration_overlay(simulations_by_config, years, observed, title, ylabel):
    fig, ax = plt.subplots(figsize=(13, 7))

    for cfg in CONFIG_ORDER:
        if cfg not in simulations_by_config:
            continue
        sims = simulations_by_config[cfg]
        p10 = np.percentile(sims, 10, axis=0)
        p90 = np.percentile(sims, 90, axis=0)
        mean_sim = sims.mean(axis=0)
        color = CONFIG_COLORS[cfg]

        ax.fill_between(years, p10, p90, alpha=0.14, color=color, label=cfg)
        ax.plot(years, mean_sim, "--", color=color, linewidth=1.6)

    ax.plot(years, observed, "ko-", linewidth=2.5, markersize=5, label="Observed")
    ax.set_xlabel("Winter year")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    plt.tight_layout()
    return fig


def plot_gamma_vs_glm_grid(gamma_by_config, glm_by_config, years, observed, title, ylabel):
    configs = [cfg for cfg in CONFIG_ORDER if cfg in gamma_by_config and cfg in glm_by_config]
    fig, axes = plt.subplots(2, len(configs), figsize=(5 * len(configs), 8), squeeze=False)

    for idx, cfg in enumerate(configs):
        sims_gamma = gamma_by_config[cfg]
        sims_glm = glm_by_config[cfg]
        color = CONFIG_COLORS[cfg]

        mean_g = sims_gamma.mean(axis=0)
        p10_g = np.percentile(sims_gamma, 10, axis=0)
        p90_g = np.percentile(sims_gamma, 90, axis=0)
        rmse_g = np.sqrt(mean_squared_error(observed, mean_g))

        ax1 = axes[0, idx]
        ax1.fill_between(years, p10_g, p90_g, alpha=0.25, color=color, label="P10-P90")
        ax1.plot(years, mean_g, "--", color=color, linewidth=2, label="Simulation mean")
        ax1.plot(years, observed, "ko-", linewidth=2, markersize=4, label="Observed")
        ax1.set_title(f"{cfg}\nGamma", fontsize=10, fontweight="bold")
        ax1.set_ylabel(ylabel)
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=7)
        ax1.text(
            0.02,
            0.98,
            f"RMSE: {rmse_g:.2f}",
            transform=ax1.transAxes,
            fontsize=8,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        mean_glm = sims_glm.mean(axis=0)
        p10_glm = np.percentile(sims_glm, 10, axis=0)
        p90_glm = np.percentile(sims_glm, 90, axis=0)
        rmse_glm = np.sqrt(mean_squared_error(observed, mean_glm))

        ax2 = axes[1, idx]
        ax2.fill_between(years, p10_glm, p90_glm, alpha=0.25, color=color, label="P10-P90")
        ax2.plot(years, mean_glm, "--", color=color, linewidth=2, label="Simulation mean")
        ax2.plot(years, observed, "ko-", linewidth=2, markersize=4, label="Observed")
        ax2.set_title(f"{cfg}\nGamma-GLM", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Winter year")
        ax2.set_ylabel(ylabel)
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=7)
        ax2.text(
            0.02,
            0.98,
            f"RMSE: {rmse_glm:.2f}",
            transform=ax2.transAxes,
            fontsize=8,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_best_model_by_station(best_table, simulations_by_model, observed_by_station, years, stations):
    station_ids = list(best_table.index)
    n_cols = 4
    n_rows = int(np.ceil(len(station_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharey=True, squeeze=False)
    axes = axes.flatten()

    for idx, station_id in enumerate(station_ids):
        ax = axes[idx]
        model_name = best_table.loc[station_id, "best_configuration"]
        rmse = float(best_table.loc[station_id, "RMSE"])
        sims = simulations_by_model[model_name][station_id]
        observed = observed_by_station[station_id]

        mean_sim = sims.mean(axis=0)
        p10 = np.percentile(sims, 10, axis=0)
        p90 = np.percentile(sims, 90, axis=0)
        family = "Gamma-GLM" if "Gamma-GLM" in model_name else "Gamma"
        color = "#d62728" if family == "Gamma-GLM" else "#1f77b4"

        ax.fill_between(years, p10, p90, color=color, alpha=0.2)
        ax.plot(years, mean_sim, "s--", color=color, linewidth=1.7, markersize=3, label="Simulation")
        ax.plot(years, observed, "o-", color="black", linewidth=2, markersize=4, label="Observed")

        name = station_label(stations, station_id)
        ax.set_title(f"{name}\n{model_name}", fontsize=9, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.text(
            0.98,
            0.02,
            f"RMSE={rmse:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        if idx % n_cols == 0:
            ax.set_ylabel("Daily mean rainfall (mm/day)")
        ax.set_xlabel("Winter year")

    for idx in range(len(station_ids), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Best occurrence + intensity model by station (daily-mean RMSE)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    parser = build_parser("Compare occurrence x intensity configurations and select the best pair by station.")
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
    results_full = evaluate_logistic_per_station(models_full, scaler_full, X_test_full, Y_test)

    scaler_slp, models_slp = train_logistic_per_station(X_train_slp, Y_train)
    results_slp = evaluate_logistic_per_station(models_slp, scaler_slp, X_test_slp, Y_test)

    scaler_d2, models_d2 = train_logistic_per_station(X_train_d2, Y_train)
    results_d2 = evaluate_logistic_per_station(models_d2, scaler_d2, X_test_d2, Y_test)

    neighborhood_columns = build_neighborhood_columns(100, data["slp"], data["d2"], data["lat"], data["lon"], data["stations"])
    scalers_neigh, models_neigh = train_neighborhood_models(
        data["slp"], data["d2"], data["stations"], mask_train, Y_train, neighborhood_columns
    )
    results_neigh = evaluate_neighborhood_models(
        data["slp"], data["d2"], data["stations"], mask_test, Y_test, neighborhood_columns, scalers_neigh, models_neigh
    )

    scaler_pca, pca, models_pca = train_logistic_pca_per_station(X_train_full, Y_train, n_components=20)
    results_pca = evaluate_logistic_pca_per_station(models_pca, scaler_pca, pca, X_test_full, Y_test)

    scaler_mean, models_mean = train_mean_models(X_train_mean, Y_train)
    results_mean = evaluate_mean_models(models_mean, scaler_mean, X_test_mean, Y_test)

    occurrence_configs = [
        summarize_configuration("SLP + d2", results_full, results_full),
        summarize_configuration("SLP only", results_slp, results_slp),
        summarize_configuration("d2 only", results_d2, results_d2),
        summarize_configuration("Neighborhood 100km", results_neigh, results_neigh),
        summarize_configuration("PCA", results_pca, results_pca),
        summarize_configuration("Mean", results_mean, results_mean),
    ]
    best_occurrence = best_configuration_per_station(occurrence_configs)

    print(f"Data directory: {data_dir}")
    print("Best occurrence configuration by station (test F1):")
    for station_id, values in best_occurrence.items():
        print(f"{station_id}: {values['configuration']} ({values['score']:.3f})")

    occurrence_proba = {
        "SLP + d2": predict_probabilities(models_full, scaler_full, X_test_full),
        "SLP only": predict_probabilities(models_slp, scaler_slp, X_test_slp),
        "d2 only": predict_probabilities(models_d2, scaler_d2, X_test_d2),
        "Mean": predict_probabilities(models_mean, scaler_mean, X_test_mean),
    }

    X_test_full_scaled = scaler_pca.transform(X_test_full)
    X_test_pca = pca.transform(X_test_full_scaled)
    occurrence_proba["PCA"] = {sid: models_pca[sid].predict_proba(X_test_pca)[:, 1] for sid in models_pca}

    occurrence_proba["Neighborhood 100km"] = {}
    for station_id in data["stations"].index:
        occurrence_proba["Neighborhood 100km"][station_id] = predict_neighborhood_simulation_probabilities(
            station_id,
            data["slp"],
            data["d2"],
            mask_test,
            neighborhood_columns,
            scalers_neigh,
            models_neigh,
        )

    gamma_params = fit_gamma_all_stations(data["stations_data"], mask_train)
    k_params = fit_gamma_k(data["stations_data"], mask_train)

    X_train_full_scaled = scaler_pca.transform(X_train_full)
    X_train_pca = pd.DataFrame(
        pca.transform(X_train_full_scaled),
        index=X_train_full.index,
        columns=[f"PC{i+1}" for i in range(X_test_pca.shape[1])],
    )
    X_test_pca_df = pd.DataFrame(
        X_test_pca,
        index=X_test_full.index,
        columns=[f"PC{i+1}" for i in range(X_test_pca.shape[1])],
    )

    glm_scalers = {}
    glm_models = {}
    glm_scalers["SLP + d2"], glm_models["SLP + d2"] = fit_glm_gamma(X_train_full, data["stations_data"], mask_train)
    glm_scalers["SLP only"], glm_models["SLP only"] = fit_glm_gamma(X_train_slp, data["stations_data"], mask_train)
    glm_scalers["d2 only"], glm_models["d2 only"] = fit_glm_gamma(X_train_d2, data["stations_data"], mask_train)
    glm_scalers["Mean"], glm_models["Mean"] = fit_glm_gamma(X_train_mean, data["stations_data"], mask_train)
    glm_scalers["PCA"], glm_models["PCA"] = fit_glm_gamma(X_train_pca, data["stations_data"], mask_train)
    glm_scalers["Neighborhood 100km"], glm_models["Neighborhood 100km"] = fit_neighborhood_glm_models(
        data["slp"],
        data["d2"],
        data["stations"],
        data["stations_data"],
        mask_train,
        neighborhood_columns,
    )

    dates_test = data["stations_data"].loc[mask_test].index
    winter_years = winter_year(dates_test)
    unique_years = np.unique(winter_years)
    day_counts = np.array([np.sum(winter_years == year) for year in unique_years], dtype=float)

    model_names = [f"{cfg} | Gamma" for cfg in CONFIG_ORDER] + [f"{cfg} | Gamma-GLM" for cfg in CONFIG_ORDER]
    all_sims = {name: {} for name in model_names}
    observed_by_station = {}

    for station_id in data["stations_data"].columns:
        obs_daily = data["stations_data"].loc[mask_test, station_id].astype(float).values
        # Keep the daily-mean framing while being robust to missing observations.
        obs_cumul = np.array([np.nansum(obs_daily[winter_years == year]) for year in unique_years], dtype=float)
        observed_by_station[station_id] = obs_cumul / day_counts

        for cfg in CONFIG_ORDER:
            proba = occurrence_proba.get(cfg, {}).get(station_id)
            if proba is None:
                continue

            if station_id in gamma_params:
                sim_gamma = simulate_winter_cumul(
                    proba,
                    gamma_params[station_id]["k"],
                    gamma_params[station_id]["theta"],
                    winter_years,
                    unique_years,
                    n_sim=N_SIM,
                    glm=False,
                )
                all_sims[f"{cfg} | Gamma"][station_id] = to_daily_mean(sim_gamma, day_counts)

            if station_id not in k_params:
                continue

            if cfg == "Neighborhood 100km":
                mu = predict_neighborhood_glm_mu(
                    station_id,
                    data["slp"],
                    data["d2"],
                    mask_test,
                    neighborhood_columns,
                    glm_scalers[cfg],
                    glm_models[cfg],
                )
            elif cfg == "PCA":
                if station_id not in glm_models[cfg]:
                    mu = None
                else:
                    mu = predict_glm_mu(glm_models[cfg][station_id], glm_scalers[cfg], X_test_pca_df)
            elif cfg == "SLP + d2":
                if station_id not in glm_models[cfg]:
                    mu = None
                else:
                    mu = predict_glm_mu(glm_models[cfg][station_id], glm_scalers[cfg], X_test_full)
            elif cfg == "SLP only":
                if station_id not in glm_models[cfg]:
                    mu = None
                else:
                    mu = predict_glm_mu(glm_models[cfg][station_id], glm_scalers[cfg], X_test_slp)
            elif cfg == "d2 only":
                if station_id not in glm_models[cfg]:
                    mu = None
                else:
                    mu = predict_glm_mu(glm_models[cfg][station_id], glm_scalers[cfg], X_test_d2)
            else:
                if station_id not in glm_models[cfg]:
                    mu = None
                else:
                    mu = predict_glm_mu(glm_models[cfg][station_id], glm_scalers[cfg], X_test_mean)

            if mu is None:
                continue
            mu = np.asarray(mu, dtype=float)
            if mu.shape[0] != len(winter_years) or not np.isfinite(mu).all():
                continue

            sim_glm = simulate_winter_cumul(
                proba,
                k_params[station_id],
                mu,
                winter_years,
                unique_years,
                n_sim=N_SIM,
                glm=True,
            )
            all_sims[f"{cfg} | Gamma-GLM"][station_id] = to_daily_mean(sim_glm, day_counts)

    all_sims = {name: sims for name, sims in all_sims.items() if sims}
    error_tables = compute_multi_station_error_tables(all_sims, observed_by_station)
    best_models = best_configuration_from_errors(error_tables, metric="RMSE")

    print("\nBest occurrence + intensity pair by station (daily-mean RMSE):")
    for station_id, row in best_models.iterrows():
        print(f"{station_id}: {row['best_configuration']} ({row['RMSE']:.3f})")

    figures_dir = project_root() / "figures" / "comparison"
    figures_dir.mkdir(parents=True, exist_ok=True)

    example_station = max(
        observed_by_station.keys(),
        key=lambda sid: sum(1 for m in all_sims.values() if sid in m),
    )
    example_name = station_label(data["stations"], example_station)
    observed_example = observed_by_station[example_station]

    sims_gamma_example = {
        cfg: all_sims[f"{cfg} | Gamma"][example_station]
        for cfg in CONFIG_ORDER
        if f"{cfg} | Gamma" in all_sims and example_station in all_sims[f"{cfg} | Gamma"]
    }
    sims_glm_example = {
        cfg: all_sims[f"{cfg} | Gamma-GLM"][example_station]
        for cfg in CONFIG_ORDER
        if f"{cfg} | Gamma-GLM" in all_sims and example_station in all_sims[f"{cfg} | Gamma-GLM"]
    }

    fig_gamma_grid = plot_configuration_envelope_grid(
        sims_gamma_example,
        unique_years,
        observed_example,
        title=f"Gamma - all occurrence configurations ({example_name})",
        ylabel="Daily mean rainfall (mm/day)",
    )
    fig_gamma_grid.savefig(figures_dir / f"gamma_all_configs_daily_mean_{example_station}.png", dpi=150, bbox_inches="tight")

    fig_overlay = plot_configuration_overlay(
        sims_gamma_example,
        unique_years,
        observed_example,
        title=f"Gamma overlay - all occurrence configurations ({example_name})",
        ylabel="Daily mean rainfall (mm/day)",
    )
    fig_overlay.savefig(figures_dir / f"gamma_overlay_all_configs_daily_mean_{example_station}.png", dpi=150, bbox_inches="tight")

    fig_glm_grid = plot_configuration_envelope_grid(
        sims_glm_example,
        unique_years,
        observed_example,
        title=f"Gamma-GLM - all occurrence configurations ({example_name})",
        ylabel="Daily mean rainfall (mm/day)",
    )
    fig_glm_grid.savefig(figures_dir / f"glm_all_configs_daily_mean_{example_station}.png", dpi=150, bbox_inches="tight")

    common_gamma = {cfg: sims_gamma_example[cfg] for cfg in CONFIG_ORDER if cfg in sims_gamma_example and cfg in sims_glm_example}
    common_glm = {cfg: sims_glm_example[cfg] for cfg in common_gamma}
    fig_gamma_vs_glm = plot_gamma_vs_glm_grid(
        common_gamma,
        common_glm,
        unique_years,
        observed_example,
        title=f"Gamma vs Gamma-GLM by occurrence configuration ({example_name})",
        ylabel="Daily mean rainfall (mm/day)",
    )
    fig_gamma_vs_glm.savefig(figures_dir / f"gamma_vs_glm_by_config_daily_mean_{example_station}.png", dpi=150, bbox_inches="tight")

    fig_best_station = plot_best_model_by_station(best_models, all_sims, observed_by_station, unique_years, data["stations"])
    fig_best_station.savefig(figures_dir / "best_model_by_station_daily_mean.png", dpi=150, bbox_inches="tight")

    rmse_frame = pd.DataFrame({name: table["RMSE"] for name, table in error_tables.items() if not table.empty})
    if not rmse_frame.empty:
        rmse_frame = rmse_frame.sort_index()
        fig_rmse = plot_error_heatmap(rmse_frame, title="RMSE daily mean by station and model")
        if fig_rmse is not None:
            fig_rmse.savefig(figures_dir / "rmse_all_models_daily_mean.png", dpi=150, bbox_inches="tight")

    print(f"\nFigures saved in: {figures_dir}")


if __name__ == "__main__":
    main()
