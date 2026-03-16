from math import gamma as gamma_function

import numpy as np
from scipy.stats import genpareto


def gamma_pdf(r, k, theta):
    """Gamma density used by the Gamma + GPD extension."""
    r = np.asarray(r, dtype=float)
    pdf = (r ** (k - 1) * np.exp(-r / theta)) / (gamma_function(k) * theta ** k)
    pdf[r <= 0] = 0.0
    return pdf


def fit_gamma_gpd_mixture_station(r_positive, quantile_threshold=0.9, min_exceedances=20):
    """Fit a Gamma body and a GPD tail on one station."""
    values = np.asarray(r_positive, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if values.size < 30:
        raise ValueError("Not enough rainy days to fit Gamma + GPD.")

    threshold = float(np.quantile(values, quantile_threshold))
    body = values[values <= threshold]
    exceedances = values[values > threshold] - threshold

    if body.size < 10:
        raise ValueError("Not enough values in the Gamma body.")

    mean_value = float(np.mean(body))
    variance = float(np.var(body, ddof=1))
    if mean_value <= 0 or variance <= 0:
        raise ValueError("Invalid Gamma moments.")

    k = mean_value ** 2 / variance
    theta = variance / mean_value

    use_gpd = exceedances.size >= min_exceedances
    xi = np.nan
    sigma = np.nan

    if use_gpd:
        xi_fit, _, sigma_fit = genpareto.fit(exceedances, floc=0.0)
        xi = float(xi_fit)
        sigma = float(sigma_fit)
        if sigma <= 0:
            use_gpd = False
            xi = np.nan
            sigma = np.nan

    p_exceed = float(exceedances.size / values.size) if use_gpd else 0.0

    return {
        "k": float(k),
        "theta": float(theta),
        "u": threshold,
        "p_exceed": p_exceed,
        "xi": xi,
        "sigma": sigma,
        "use_gpd": bool(use_gpd),
        "n_pos": int(values.size),
        "n_body": int(body.size),
        "n_exceed": int(exceedances.size),
        "q_threshold": float(quantile_threshold),
    }


def fit_gamma_gpd_mixture_all_stations(stations_data, mask, quantile_threshold=0.9, min_exceedances=20):
    """Fit the Gamma + GPD extension for all stations."""
    rainfall = stations_data.loc[mask].astype(float)
    params = {}

    for station_id in rainfall.columns:
        values = rainfall[station_id].values
        positive = values[values > 0]
        try:
            params[station_id] = fit_gamma_gpd_mixture_station(
                positive,
                quantile_threshold=quantile_threshold,
                min_exceedances=min_exceedances,
            )
        except Exception:
            continue

    return params


def _sample_gamma_truncated_upper(k, theta, upper, size, rng, max_iter=12):
    """Sample from a Gamma distribution truncated above a threshold."""
    out = np.empty(size, dtype=float)
    filled = 0
    remaining = size
    iteration = 0

    while remaining > 0 and iteration < max_iter:
        draws = rng.gamma(shape=k, scale=theta, size=max(2 * remaining, 32))
        accepted = draws[(draws > 0) & (draws <= upper)]
        take = min(accepted.size, remaining)
        if take > 0:
            out[filled:filled + take] = accepted[:take]
            filled += take
            remaining -= take
        iteration += 1

    if remaining > 0:
        fallback = rng.gamma(shape=k, scale=theta, size=remaining)
        out[filled:] = np.minimum(np.maximum(fallback, 1e-8), upper)

    return out


def simulate_positive_gamma_gpd(n_samples, params, rng=None, theta_override=None):
    """Simulate positive rainfall amounts from the fitted Gamma + GPD mixture."""
    if rng is None:
        rng = np.random.default_rng(42)

    k = float(params["k"])
    theta_base = float(params["theta"])
    threshold = float(params["u"])
    p_exceed = float(params.get("p_exceed", 0.0))
    use_gpd = bool(params.get("use_gpd", False))
    xi = params.get("xi", np.nan)
    sigma = params.get("sigma", np.nan)

    is_exceed = rng.random(n_samples) < p_exceed if use_gpd else np.zeros(n_samples, dtype=bool)
    values = np.zeros(n_samples, dtype=float)

    n_body = int((~is_exceed).sum())
    n_tail = int(is_exceed.sum())

    if np.isscalar(theta_override) or theta_override is None:
        theta_value = theta_base if theta_override is None else float(theta_override)
        if n_body > 0:
            values[~is_exceed] = _sample_gamma_truncated_upper(k, theta_value, threshold, n_body, rng)
    else:
        theta_array = np.asarray(theta_override, dtype=float)
        if theta_array.size != n_samples:
            raise ValueError("theta_override must be scalar or of size n_samples.")
        body_indices = np.where(~is_exceed)[0]
        for index in body_indices:
            values[index] = _sample_gamma_truncated_upper(k, float(theta_array[index]), threshold, 1, rng)[0]

    if n_tail > 0:
        excess = genpareto.rvs(c=xi, loc=0.0, scale=sigma, size=n_tail, random_state=rng)
        values[is_exceed] = threshold + excess

    return values


def simulate_rainfall_gamma_gpd(proba_rain, params, rng=None, theta_override=None):
    """Simulate a full daily rainfall time series with occurrence and intensity."""
    if rng is None:
        rng = np.random.default_rng(42)

    proba_rain = np.asarray(proba_rain, dtype=float)
    n_days = proba_rain.size
    is_rain = rng.random(n_days) <= proba_rain

    rainfall = np.zeros(n_days, dtype=float)
    n_rain = int(is_rain.sum())
    if n_rain == 0:
        return rainfall

    theta_local = None
    if theta_override is not None:
        if np.isscalar(theta_override):
            theta_local = float(theta_override)
        else:
            theta_array = np.asarray(theta_override, dtype=float)
            if theta_array.size != n_days:
                raise ValueError("theta_override must be scalar or have the same size as proba_rain.")
            theta_local = theta_array[is_rain]

    rainfall[is_rain] = simulate_positive_gamma_gpd(n_rain, params, rng=rng, theta_override=theta_local)
    return rainfall

