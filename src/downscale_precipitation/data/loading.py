from pathlib import Path

import pandas as pd

from .preprocessing import add_time_index, prepare_stations


def load_data(data_dir):
    """Load the raw text files used in the original project."""
    data_path = Path(data_dir)

    stations = pd.read_csv(data_path / "station.data.81.10.txt", sep=r"\s+")
    stations = stations.transpose()

    slp = pd.read_csv(data_path / "ERA5.slp.81.10.txt", sep=r"\s+")
    d2 = pd.read_csv(data_path / "ERA5.d2.81.10.txt", sep=r"\s+")
    lon = pd.read_csv(data_path / "ERA5.lon.81.10.txt", sep=r"\s+", header=None)
    lat = pd.read_csv(data_path / "ERA5.lat.81.10.txt", sep=r"\s+", header=None)

    return stations, slp, d2, lon, lat


def load_prepared_data(data_dir):
    """Load raw files and return the prepared objects used by the models."""
    stations, slp, d2, lon, lat = load_data(data_dir)
    stations_data = prepare_stations(stations)
    slp, d2 = add_time_index(slp, d2)

    return {
        "stations": stations,
        "stations_data": stations_data,
        "slp": slp,
        "d2": d2,
        "lon": lon,
        "lat": lat,
    }

