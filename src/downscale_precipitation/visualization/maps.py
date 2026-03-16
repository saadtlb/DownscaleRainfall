import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_slp_d2_graph(slp, d2, lat, lon, stations, date):
    """Plot SLP and d2 fields for one day together with station values."""
    date = pd.to_datetime(date).normalize()

    lat_values = lat.squeeze().to_numpy(dtype=float)
    lon_values = lon.squeeze().to_numpy(dtype=float)
    n_lat = len(lat_values)
    n_lon = len(lon_values)

    slp_2d = slp.loc[date].to_numpy().reshape(n_lat, n_lon)
    d2_2d = d2.loc[date].to_numpy().reshape(n_lat, n_lon)
    lon_2d, lat_2d = np.meshgrid(lon_values, lat_values)

    fig, (ax1, ax2, ax_table) = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        gridspec_kw={"width_ratios": [1, 1, 0.9]},
    )

    pcm1 = ax1.pcolormesh(lon_2d, lat_2d, slp_2d)
    fig.colorbar(pcm1, ax=ax1, label="SLP (hPa)")
    ax1.set_title(f"SLP (hPa) {date.date()}")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    pcm2 = ax2.pcolormesh(lon_2d, lat_2d, d2_2d)
    fig.colorbar(pcm2, ax=ax2, label="d2 (K)")
    ax2.set_title(f"d2 (K) {date.date()}")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    col_dates = pd.to_datetime(stations.columns.astype(str), format="%Y%m%d", errors="coerce").normalize()
    column_index = np.where(col_dates == date)[0][0]
    rain_column = stations.columns[column_index]
    rainfall = pd.to_numeric(stations[rain_column], errors="coerce")

    for number, (_, row) in enumerate(stations.iterrows(), start=1):
        lon_station = float(row["LON"])
        lat_station = float(row["LAT"])
        for axis in (ax1, ax2):
            axis.scatter(lon_station, lat_station, s=120, edgecolor="black", facecolor="white", zorder=3)
            axis.text(lon_station, lat_station, str(number), ha="center", va="center", fontsize=9, zorder=4)

    table_frame = pd.DataFrame(
        {
            "N": np.arange(1, len(stations) + 1),
            "Station": stations["NOM_USUEL"].astype(str).values,
            "Precipitations (mm)": np.round(rainfall.to_numpy(dtype=float), 2),
        }
    )

    ax_table.axis("off")
    ax_table.set_title(f"Stations {date.date()}")
    table = ax_table.table(
        cellText=table_frame.values,
        colLabels=table_frame.columns,
        loc="center",
        colWidths=[0.12, 0.58, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    plt.tight_layout()
    return fig


def plot_compare_days(slp, d2, lat, lon, stations, rainy_day, dry_day, stations_data):
    """Compare SLP and d2 fields between one dry and one rainy day."""
    lat_values = lat.squeeze().to_numpy(dtype=float)
    lon_values = lon.squeeze().to_numpy(dtype=float)
    n_lat = len(lat_values)
    n_lon = len(lon_values)
    lon_2d, lat_2d = np.meshgrid(lon_values, lat_values)

    slp_dry = slp.loc[dry_day].to_numpy().reshape(n_lat, n_lon)
    slp_rainy = slp.loc[rainy_day].to_numpy().reshape(n_lat, n_lon)
    d2_dry = d2.loc[dry_day].to_numpy().reshape(n_lat, n_lon)
    d2_rainy = d2.loc[rainy_day].to_numpy().reshape(n_lat, n_lon)

    vmin_slp = min(slp_dry.min(), slp_rainy.min())
    vmax_slp = max(slp_dry.max(), slp_rainy.max())
    vmin_d2 = min(d2_dry.min(), d2_rainy.min())
    vmax_d2 = max(d2_dry.max(), d2_rainy.max())

    total_rain = stations_data.loc[rainy_day].astype(float).sum()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pcm1 = axes[0, 0].pcolormesh(lon_2d, lat_2d, slp_dry, cmap="viridis", vmin=vmin_slp, vmax=vmax_slp)
    pcm2 = axes[0, 1].pcolormesh(lon_2d, lat_2d, slp_rainy, cmap="viridis", vmin=vmin_slp, vmax=vmax_slp)
    pcm3 = axes[1, 0].pcolormesh(lon_2d, lat_2d, d2_dry, cmap="coolwarm", vmin=vmin_d2, vmax=vmax_d2)
    pcm4 = axes[1, 1].pcolormesh(lon_2d, lat_2d, d2_rainy, cmap="coolwarm", vmin=vmin_d2, vmax=vmax_d2)

    fig.colorbar(pcm1, ax=axes[0, 0], label="SLP (hPa)")
    fig.colorbar(pcm2, ax=axes[0, 1], label="SLP (hPa)")
    fig.colorbar(pcm3, ax=axes[1, 0], label="d2 (K)")
    fig.colorbar(pcm4, ax=axes[1, 1], label="d2 (K)")

    axes[0, 0].set_title(f"SLP - Dry day ({pd.to_datetime(dry_day).date()})")
    axes[0, 1].set_title(f"SLP - Rainy day ({pd.to_datetime(rainy_day).date()}) - Total rain: {total_rain:.1f} mm")
    axes[1, 0].set_title("d2 - Dry day")
    axes[1, 1].set_title("d2 - Rainy day")

    for _, row in stations.iterrows():
        lon_station = float(row["LON"])
        lat_station = float(row["LAT"])
        for axis in axes.flat:
            axis.scatter(lon_station, lat_station, s=80, edgecolor="black", facecolor="white", zorder=3)

    plt.tight_layout()
    return fig
