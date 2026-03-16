import numpy as np
import pandas as pd

from ..data.temporal_masks import winter_year


def winter_season_year(index):
    """Return the winter year for a DatetimeIndex."""
    return winter_year(index)


def winter_totals_full_djf(series, min_days=90):
    """Aggregate daily rainfall into winter totals."""
    series = pd.Series(series).copy()
    years = winter_year(series.index)
    frame = pd.DataFrame({"value": series.values, "winter_year": years}, index=series.index)

    totals = frame.groupby("winter_year")["value"].agg(["sum", "count"])
    totals = totals[totals["count"] >= min_days]
    return totals["sum"]


def compute_winter_stats(sim_cumul, station_id, stations_data, mask_test, aggregation="sum"):
    """Return observed and simulated winter statistics for one station.

    Parameters
    ----------
    sim_cumul : array-like, shape (n_sim, n_years)
        Simulated winter totals (sum over days) for each simulation.
    station_id : hashable
        Station identifier.
    stations_data : DataFrame
        Daily observed rainfall by station.
    mask_test : array-like of bool
        Test-period mask over the daily timeline.
    aggregation : {"sum", "mean"}
        "sum" returns winter totals, "mean" returns winter daily means.
    """
    dates_test = stations_data.loc[mask_test].index
    winter_years = winter_year(dates_test)
    unique_years = np.unique(winter_years)
    day_counts = np.array([np.sum(winter_years == year) for year in unique_years], dtype=float)

    observed = stations_data.loc[mask_test, station_id].astype(float).values
    obs_cumul = np.array([observed[winter_years == year].sum() for year in unique_years])
    obs_mean = obs_cumul / day_counts

    sim_cumul = np.asarray(sim_cumul, dtype=float)
    if aggregation == "sum":
        sim_values = sim_cumul
        obs_values = obs_cumul
    elif aggregation == "mean":
        sim_values = sim_cumul / day_counts.reshape(1, -1)
        obs_values = obs_mean
    else:
        raise ValueError("aggregation must be 'sum' or 'mean'")

    return {
        "years": unique_years,
        "day_counts": day_counts,
        "obs": obs_values,
        "obs_cumul": obs_cumul,
        "obs_mean": obs_mean,
        "mean_sim": sim_values.mean(axis=0),
        "p10": np.percentile(sim_values, 10, axis=0),
        "p90": np.percentile(sim_values, 90, axis=0),
    }
