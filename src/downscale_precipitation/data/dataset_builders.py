import numpy as np
import pandas as pd


def concat_reanalysis_features(slp, d2):
    """Concatenate SLP and d2 on columns."""
    return pd.concat([slp, d2], axis=1)


def build_X(data, mask):
    """Select rows matching the temporal mask."""
    return data.loc[mask]


def build_Y_zero_un(stations_data, mask):
    """Build the binary rain occurrence target for all stations."""
    rainfall = stations_data.loc[mask].astype(float)
    return (rainfall > 0).astype(int)


def build_Y_amount(stations_data, mask):
    """Build the rainfall amount table for all stations."""
    return stations_data.loc[mask].astype(float)


def split_full_features(slp, d2, mask_train, mask_test):
    """Return train and test matrices for the full SLP + d2 configuration."""
    data = concat_reanalysis_features(slp, d2)
    return build_X(data, mask_train), build_X(data, mask_test)


def split_single_variable_features(variable, mask_train, mask_test):
    """Return train and test matrices for a single predictor table."""
    return build_X(variable, mask_train), build_X(variable, mask_test)


def build_mean_features(slp, d2, mask):
    """Return the mean spatial SLP and d2 features used in the report."""
    slp_mean = slp.loc[mask].mean(axis=1).to_numpy().reshape(-1, 1)
    d2_mean = d2.loc[mask].mean(axis=1).to_numpy().reshape(-1, 1)
    data = np.hstack([slp_mean, d2_mean])
    index = slp.loc[mask].index
    return pd.DataFrame(data, index=index, columns=["slp_mean", "d2_mean"])

