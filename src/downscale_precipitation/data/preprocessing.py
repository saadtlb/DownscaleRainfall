import pandas as pd


def prepare_stations(stations):
    """Convert station date columns to datetime and return a time-indexed table."""
    new_columns = list(stations.columns)
    new_columns[4:] = pd.to_datetime(stations.columns[4:], format="%Y%m%d", errors="coerce")
    stations = stations.copy()
    stations.columns = new_columns

    date_columns = [col for col in stations.columns if isinstance(col, pd.Timestamp)]
    stations_data = stations[date_columns].transpose()
    stations_data.index = pd.to_datetime(stations_data.index)

    return stations_data


def add_time_index(slp, d2, start_date="1981-01-01"):
    """Add a shared daily datetime index to the ERA5 predictors."""
    n_days = slp.shape[0]
    time_index = pd.date_range(start_date, periods=n_days, freq="D")

    slp = slp.copy()
    d2 = d2.copy()
    slp.index = time_index
    d2.index = time_index

    return slp, d2

