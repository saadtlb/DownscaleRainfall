import matplotlib.pyplot as plt
import numpy as np

from ..occurrence.neighborhood import extract_station_neighborhood
from ..occurrence.pca import project_coefficients_back


def plot_radar(models_scores, stations, title="Model comparison"):
    """Plot a radar chart comparing station-wise scores across configurations."""
    model_names = list(models_scores.keys())
    station_ids = list(next(iter(models_scores.values())).keys())
    station_names = [stations.loc[sid, "NOM_USUEL"] if sid in stations.index else sid for sid in station_ids]

    angles = np.linspace(0, 2 * np.pi, len(station_names), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for index, name in enumerate(model_names):
        values = [models_scores[name][sid] for sid in station_ids]
        values = np.array(values + [values[0]], dtype=float)
        ax.plot(angles, values, marker="o", linewidth=2, label=name, color=colors[index % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[index % len(colors)])

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, station_names, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    plt.tight_layout()
    return fig


def plot_all_configs_coefficients(
    station_id,
    models_full,
    models_slp,
    models_d2,
    models_neighborhood,
    models_pca,
    models_mean,
    pca,
    slp,
    d2,
    lat,
    lon,
    stations,
    distance_km=100,
):
    """Plot coefficient maps for the six occurrence configurations."""
    lat_values = lat.squeeze().to_numpy(dtype=float)
    lon_values = lon.squeeze().to_numpy(dtype=float)
    n_lat = len(lat_values)
    n_lon = len(lon_values)
    grid_size = n_lat * n_lon
    lon_2d, lat_2d = np.meshgrid(lon_values, lat_values)

    station = stations.loc[station_id]
    lon_station = float(station["LON"])
    lat_station = float(station["LAT"])
    station_label = f"{station['NOM_USUEL']} ({station_id})"

    fig, axes = plt.subplots(6, 2, figsize=(15, 40), gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 0.45]})
    fig.suptitle(f"Logistic regression coefficients - {station_label}", fontsize=16, fontweight="bold", y=0.985)

    def plot_grid(axis, grid, title, vmin, vmax):
        pcm = axis.pcolormesh(lon_2d, lat_2d, grid, cmap="coolwarm", shading="auto", vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, ax=axis, label="Coefficient", fraction=0.046, pad=0.04)
        axis.set_title(title, fontsize=10)
        axis.scatter(lon_station, lat_station, s=140, edgecolor="black", facecolor="white", zorder=3)
        axis.text(lon_station, lat_station, "*", ha="center", va="center", fontsize=10, zorder=4)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")

    coef_full = models_full[station_id].coef_.ravel()
    slp_grid_full = coef_full[:grid_size].reshape(n_lat, n_lon)
    d2_grid_full = coef_full[grid_size:].reshape(n_lat, n_lon)

    slp_grid_only = models_slp[station_id].coef_.ravel().reshape(n_lat, n_lon)
    d2_grid_only = models_d2[station_id].coef_.ravel().reshape(n_lat, n_lon)

    slp_local, d2_local, coordinates = extract_station_neighborhood(distance_km, station_id, slp, d2, lat, lon, stations)
    del slp_local, d2_local
    n_local = len(coordinates)
    coef_local = models_neighborhood[station_id].coef_.ravel()
    slp_local_coef = coef_local[:n_local]
    d2_local_coef = coef_local[n_local:]
    slp_grid_local = np.full((n_lat, n_lon), np.nan)
    d2_grid_local = np.full((n_lat, n_lon), np.nan)
    for index, (lon_coord, lat_coord) in enumerate(coordinates):
        i_lat = np.argmin(np.abs(lat_values - lat_coord))
        i_lon = np.argmin(np.abs(lon_values - lon_coord))
        slp_grid_local[i_lat, i_lon] = slp_local_coef[index]
        d2_grid_local[i_lat, i_lon] = d2_local_coef[index]

    coef_pca = project_coefficients_back(models_pca[station_id], pca)
    slp_grid_pca = coef_pca[:grid_size].reshape(n_lat, n_lon)
    d2_grid_pca = coef_pca[grid_size:].reshape(n_lat, n_lon)

    coef_mean = models_mean[station_id].coef_.ravel()
    intercept_mean = float(models_mean[station_id].intercept_[0])

    vmax_full = np.abs(np.concatenate([slp_grid_full.ravel(), d2_grid_full.ravel()])).max()
    plot_grid(axes[0, 0], slp_grid_full, "Config 1 - SLP + d2 (SLP)", -vmax_full, vmax_full)
    plot_grid(axes[0, 1], d2_grid_full, "Config 1 - SLP + d2 (d2)", -vmax_full, vmax_full)

    vmax_slp = np.abs(slp_grid_only).max()
    plot_grid(axes[1, 0], slp_grid_only, "Config 2 - SLP only", -vmax_slp, vmax_slp)
    axes[1, 1].axis("off")

    axes[2, 0].axis("off")
    vmax_d2 = np.abs(d2_grid_only).max()
    plot_grid(axes[2, 1], d2_grid_only, "Config 3 - d2 only", -vmax_d2, vmax_d2)

    vmax_local = max(np.nanmax(np.abs(slp_grid_local)), np.nanmax(np.abs(d2_grid_local)))
    plot_grid(axes[3, 0], np.ma.masked_invalid(slp_grid_local), f"Config 4 - Neighborhood {distance_km} km (SLP)", -vmax_local, vmax_local)
    plot_grid(axes[3, 1], np.ma.masked_invalid(d2_grid_local), f"Config 4 - Neighborhood {distance_km} km (d2)", -vmax_local, vmax_local)

    vmax_pca = np.abs(np.concatenate([slp_grid_pca.ravel(), d2_grid_pca.ravel()])).max()
    plot_grid(axes[4, 0], slp_grid_pca, "Config 5 - PCA projected (SLP)", -vmax_pca, vmax_pca)
    plot_grid(axes[4, 1], d2_grid_pca, "Config 5 - PCA projected (d2)", -vmax_pca, vmax_pca)

    axes[5, 0].axis("off")
    axes[5, 1].axis("off")
    axes[5, 0].text(0.05, 0.75, "Config 6 - Mean", fontsize=12, fontweight="bold")
    axes[5, 0].text(0.05, 0.45, f"SLP mean coefficient : {coef_mean[0]:.4f}", fontsize=10)
    axes[5, 0].text(0.05, 0.25, f"d2 mean coefficient  : {coef_mean[1]:.4f}", fontsize=10)
    axes[5, 0].text(0.05, 0.05, f"Intercept            : {intercept_mean:.4f}", fontsize=10)

    plt.tight_layout()
    return fig

