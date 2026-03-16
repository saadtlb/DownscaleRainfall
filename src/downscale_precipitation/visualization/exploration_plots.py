import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_station_correlation_heatmaps(corr_occ, corr_int, distance_matrix):
    """Plot station correlation heatmaps with distance annotations."""
    annotations = distance_matrix.round(0).astype(int).astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(
        corr_occ,
        ax=axes[0],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=annotations,
        fmt="",
        annot_kws={"size": 7, "color": "black"},
        cbar_kws={"label": "Correlation"},
    )
    axes[0].set_title("Occurrence correlation (color) and distance in km (text)")
    axes[0].set_xlabel("Stations")
    axes[0].set_ylabel("Stations")

    sns.heatmap(
        corr_int,
        ax=axes[1],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=annotations,
        fmt="",
        annot_kws={"size": 7, "color": "black"},
        cbar_kws={"label": "Correlation"},
    )
    axes[1].set_title("Intensity correlation (color) and distance in km (text)")
    axes[1].set_xlabel("Stations")
    axes[1].set_ylabel("Stations")

    plt.tight_layout()
    return fig


def plot_station_locations(stations):
    """Plot station locations on a southern-France background map."""
    lon = stations["LON"].astype(float).to_numpy()
    lat = stations["LAT"].astype(float).to_numpy()
    names = stations["NOM_USUEL"].astype(str).to_numpy()
    lon_min = float(np.min(lon) - 1.5)
    lon_max = float(np.max(lon) + 1.5)
    lat_min = float(np.min(lat) - 1.2)
    lat_max = float(np.max(lat) + 1.2)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(9, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_facecolor("#eaf3ff")
        ax.add_feature(cfeature.LAND, facecolor="#f7f4ed")
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor="black")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.6, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False


        ax.scatter(lon, lat, s=90, color="tab:red", edgecolor="black", transform=ccrs.PlateCarree(), zorder=3)

        candidate_offsets = [
            (8, 8),
            (8, -8),
            (-8, 8),
            (-8, -8),
            (12, 0),
            (-12, 0),
            (0, 12),
            (0, -12),
            (16, 6),
            (-16, 6),
            (16, -6),
            (-16, -6),
        ]
        occupied_boxes = []

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        for index, (lon_value, lat_value, name) in enumerate(zip(lon, lat, names), start=1):
            label = f"{index}. {name}"
            placed = False

            for dx, dy in candidate_offsets:
                text = ax.annotate(
                    label,
                    xy=(lon_value, lat_value),
                    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    zorder=4,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
                )
                fig.canvas.draw()
                bbox = text.get_window_extent(renderer=renderer).expanded(1.05, 1.2)

                if any(bbox.overlaps(previous) for previous in occupied_boxes):
                    text.remove()
                    continue

                occupied_boxes.append(bbox)
                placed = True
                break

            if not placed:
                fallback = ax.annotate(
                    label,
                    xy=(lon_value, lat_value),
                    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=8,
                    zorder=4,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
                )
                fig.canvas.draw()
                occupied_boxes.append(fallback.get_window_extent(renderer=renderer).expanded(1.05, 1.2))
    except Exception:
        # Fallback used only when cartopy is not available.
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.scatter(lon, lat, s=120, color="tab:red", edgecolor="black")
        for index, (lon_value, lat_value, name) in enumerate(zip(lon, lat, names), start=1):
            ax.text(lon_value + 0.05, lat_value + 0.05, f"{index}. {name}", fontsize=8)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3)

    ax.set_title("Station locations")
    plt.tight_layout()
    return fig


def plot_altitude_vs_rain_frequency(stations, rain_frequency):
    """Plot station altitude against rain frequency."""
    station_order = list(stations.index)
    altitude = stations.loc[station_order, "ALTI"].astype(float).to_numpy()
    frequency = rain_frequency.reindex(station_order).astype(float).to_numpy()
    names = stations.loc[station_order, "NOM_USUEL"].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(altitude, frequency, s=120, color="tab:blue", edgecolor="black")
    for alt_value, freq_value, name in zip(altitude, frequency, names):
        ax.text(alt_value + 8, freq_value + 0.005, name, fontsize=8)

    ax.set_title("Altitude vs rain frequency")
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("Rain frequency")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_positive_rainfall_histograms(stations_data, stations, mask, ncols=4):
    """Plot one histogram of positive rainfall values per station."""
    rainfall = stations_data.loc[mask].astype(float)
    station_ids = list(rainfall.columns)
    n_stations = len(station_ids)
    nrows = int(np.ceil(n_stations / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for index, station_id in enumerate(station_ids):
        axis = axes[index]
        values = rainfall[station_id]
        positive = values[values > 0]
        axis.hist(positive, bins=30, color="steelblue", edgecolor="white", alpha=0.8)

        station_name = stations.loc[station_id, "NOM_USUEL"] if station_id in stations.index else station_id
        axis.set_title(str(station_name), fontsize=9)
        axis.set_xlabel("Rainfall (mm/day)")
        axis.set_ylabel("Count")

    for axis in axes[n_stations:]:
        axis.axis("off")

    fig.suptitle("Positive rainfall distribution by station", fontsize=12)
    plt.tight_layout()
    return fig


def plot_mean_fields_dry_vs_rainy(mean_fields, lat, lon, stations):
    """Plot mean SLP and d2 fields for dry and rainy days, plus differences."""
    lat_values = lat.squeeze().to_numpy(dtype=float)
    lon_values = lon.squeeze().to_numpy(dtype=float)
    n_lat = len(lat_values)
    n_lon = len(lon_values)
    lon_2d, lat_2d = np.meshgrid(lon_values, lat_values)

    slp_dry = mean_fields["slp_dry_mean"].to_numpy().reshape(n_lat, n_lon)
    slp_rain = mean_fields["slp_rain_mean"].to_numpy().reshape(n_lat, n_lon)
    d2_dry = mean_fields["d2_dry_mean"].to_numpy().reshape(n_lat, n_lon)
    d2_rain = mean_fields["d2_rain_mean"].to_numpy().reshape(n_lat, n_lon)
    slp_diff = slp_rain - slp_dry
    d2_diff = d2_rain - d2_dry

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    pcm1 = axes[0, 0].pcolormesh(lon_2d, lat_2d, slp_dry, cmap="viridis")
    pcm2 = axes[0, 1].pcolormesh(lon_2d, lat_2d, slp_rain, cmap="viridis")
    pcm3 = axes[0, 2].pcolormesh(lon_2d, lat_2d, slp_diff, cmap="RdBu_r")
    pcm4 = axes[1, 0].pcolormesh(lon_2d, lat_2d, d2_dry, cmap="coolwarm")
    pcm5 = axes[1, 1].pcolormesh(lon_2d, lat_2d, d2_rain, cmap="coolwarm")
    pcm6 = axes[1, 2].pcolormesh(lon_2d, lat_2d, d2_diff, cmap="RdBu_r")

    fig.colorbar(pcm1, ax=axes[0, 0], label="SLP (hPa)")
    fig.colorbar(pcm2, ax=axes[0, 1], label="SLP (hPa)")
    fig.colorbar(pcm3, ax=axes[0, 2], label="Delta SLP (hPa)")
    fig.colorbar(pcm4, ax=axes[1, 0], label="d2 (K)")
    fig.colorbar(pcm5, ax=axes[1, 1], label="d2 (K)")
    fig.colorbar(pcm6, ax=axes[1, 2], label="Delta d2 (K)")

    axes[0, 0].set_title("Mean SLP - dry days")
    axes[0, 1].set_title("Mean SLP - rainy days")
    axes[0, 2].set_title("SLP difference (rainy - dry)")
    axes[1, 0].set_title("Mean d2 - dry days")
    axes[1, 1].set_title("Mean d2 - rainy days")
    axes[1, 2].set_title("d2 difference (rainy - dry)")

    for _, row in stations.iterrows():
        lon_station = float(row["LON"])
        lat_station = float(row["LAT"])
        for axis in axes.flat:
            axis.scatter(lon_station, lat_station, s=35, edgecolor="black", facecolor="white", zorder=3)

    fig.suptitle("Mean fields: dry vs rainy days", fontsize=13)
    plt.tight_layout()
    return fig
