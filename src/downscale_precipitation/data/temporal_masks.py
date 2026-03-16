import numpy as np
import pandas as pd


def winter_year(dates):
    """Return the winter year for each timestamp."""
    dates = pd.to_datetime(dates)
    return np.where(dates.month == 12, dates.year + 1, dates.year)


def winter_mask(start_winter, end_winter, start_date="1981-01-01", n_days=10957):
    """Select full DJF winters between two winter years included."""
    time = pd.date_range(start_date, periods=n_days, freq="D")
    years = winter_year(time)
    winter_months = (time.month == 12) | (time.month == 1) | (time.month == 2)
    valid_years = (years >= start_winter) & (years <= end_winter)
    return valid_years & winter_months


def build_train_test_masks(train_start=1982, train_end=1995, test_start=1996, test_end=2010):
    """Return the default train and test masks used in the report."""
    mask_train = winter_mask(train_start, train_end)
    mask_test = winter_mask(test_start, test_end)
    return mask_train, mask_test

