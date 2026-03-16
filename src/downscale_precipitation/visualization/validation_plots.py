import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma as gamma_distribution, genpareto

from ..intensity.gamma_gpd import fit_gamma_gpd_mixture_station


def plot_tail_comparison_station(station_id, stations_data, stations, mask_train, q_threshold=0.90, min_exceedances=20):
    """Compare the empirical tail with Gamma and Gamma + GPD fits."""
    values = stations_data.loc[mask_train, station_id].astype(float).dropna().values
    values = values[values > 0]
    if len(values) < 30:
        raise ValueError(f"Station {station_id} does not have enough rainy days.")

    station_name = stations.loc[station_id, "NOM_USUEL"] if station_id in stations.index else station_id

    mean_value = values.mean()
    variance = values.var(ddof=1)
    k_gamma = mean_value ** 2 / variance
    theta_gamma = variance / mean_value

    mix = fit_gamma_gpd_mixture_station(values, quantile_threshold=q_threshold, min_exceedances=min_exceedances)
    threshold = mix["u"]
    p_exceed = mix["p_exceed"]
    k_body = mix["k"]
    theta_body = mix["theta"]
    xi = mix["xi"]
    sigma = mix["sigma"]

    x = np.linspace(max(1e-3, values.min()), values.max() * 1.1, 600)
    xs = np.sort(values)
    surv_emp = 1.0 - (np.arange(1, len(xs) + 1) / (len(xs) + 1.0))
    surv_gamma = 1.0 - gamma_distribution.cdf(x, a=k_gamma, scale=theta_gamma)

    f_u = gamma_distribution.cdf(threshold, a=k_body, scale=theta_body)
    f_u = max(f_u, 1e-8)

    surv_mix = np.empty_like(x)
    left = x <= threshold
    right = ~left
    surv_mix[left] = 1.0 - (1.0 - p_exceed) * (gamma_distribution.cdf(x[left], a=k_body, scale=theta_body) / f_u)
    y = np.maximum(x[right] - threshold, 0.0)
    surv_mix[right] = p_exceed * (1.0 - genpareto.cdf(y, c=xi, loc=0.0, scale=sigma))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, np.maximum(surv_emp, 1e-6), "k.", ms=3, label="Observations")
    ax.plot(x, np.maximum(surv_gamma, 1e-6), color="tab:blue", lw=2, label="Gamma")
    ax.plot(x, np.maximum(surv_mix, 1e-6), color="tab:red", lw=2, label="Gamma + GPD")
    ax.axvline(threshold, color="gray", ls="--", lw=1, label=f"u = q{int(q_threshold * 100)} = {threshold:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Rainfall (mm/day)")
    ax.set_ylabel("Exceedance probability P(R > x)")
    ax.set_title(f"{station_name} ({station_id}) - tail comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig
