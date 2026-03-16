import pandas as pd


def station_altitude_frequency_frame(stations, rain_frequency):
    """Build a small summary table for altitude and rain frequency."""
    return pd.DataFrame(
        {
            "station_name": stations["NOM_USUEL"].astype(str),
            "altitude": stations["ALTI"].astype(float),
            "rain_frequency": rain_frequency.reindex(stations.index).astype(float),
        },
        index=stations.index,
    )


def station_positive_amounts_frame(stations_data, mask):
    """Return a long-format table of positive rainfall amounts."""
    rainfall = stations_data.loc[mask].astype(float)
    rows = []
    for station_id in rainfall.columns:
        values = rainfall[station_id]
        positive = values[values > 0]
        for value in positive:
            rows.append({"station_id": station_id, "rainfall": float(value)})
    return pd.DataFrame(rows)

