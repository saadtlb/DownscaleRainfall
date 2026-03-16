import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma as gamma_distribution


def plot_gamma_fit(r_positive, k, theta, station_name="Station"):
    """Overlay a Gamma density on the histogram of positive rainfall."""
    values = np.asarray(r_positive, dtype=float)
    x = np.linspace(values.min(), values.max(), 300)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=25, density=True, alpha=0.4, color="tab:blue", label="Observed histogram")
    ax.plot(x, gamma_distribution.pdf(x, a=k, scale=theta), color="tab:red", linewidth=2, label="Gamma PDF")
    ax.set_title(f"Gamma fit - {station_name}")
    ax.set_xlabel("Rainfall (mm/day)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_gamma_qq(r_positive, k, theta, station_name="Station"):
    """Create a QQ-plot for the Gamma fit."""
    values = np.sort(np.asarray(r_positive, dtype=float))
    n_values = len(values)
    probs = (np.arange(1, n_values + 1) - 0.5) / n_values
    theoretical = gamma_distribution.ppf(probs, a=k, scale=theta)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theoretical, values, s=15, alpha=0.8)
    line_min = min(theoretical.min(), values.min())
    line_max = max(theoretical.max(), values.max())
    ax.plot([line_min, line_max], [line_min, line_max], color="tab:red", linestyle="--")
    ax.set_title(f"Gamma QQ-plot - {station_name}")
    ax.set_xlabel("Theoretical Gamma quantiles")
    ax.set_ylabel("Observed quantiles")
    plt.tight_layout()
    return fig

