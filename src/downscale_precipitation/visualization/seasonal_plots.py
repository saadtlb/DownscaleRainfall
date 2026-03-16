import matplotlib.pyplot as plt
import numpy as np


def plot_winter_cumulative_envelope(years, observed, mean_sim, p10, p90, title, ylabel="Daily mean rainfall (mm/day)"):
    """Plot observed winter statistics against simulation envelope."""
    years = np.asarray(years)
    observed = np.asarray(observed, dtype=float)
    mean_sim = np.asarray(mean_sim, dtype=float)
    p10 = np.asarray(p10, dtype=float)
    p90 = np.asarray(p90, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(years, p10, p90, color="tab:blue", alpha=0.2, label="P10-P90")
    ax.plot(years, mean_sim, "--", color="tab:blue", linewidth=2, label="Simulation mean")
    ax.plot(years, observed, "o-", color="black", linewidth=2, markersize=4, label="Observed")
    ax.set_xlabel("Winter year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_gamma_vs_glm_cumulative(
    years,
    observed,
    gamma_mean,
    gamma_p10,
    gamma_p90,
    glm_mean,
    glm_p10,
    glm_p90,
    title,
    ylabel="Daily mean rainfall (mm/day)",
):
    """Compare Gamma and Gamma-GLM simulation envelopes on winter statistics."""
    years = np.asarray(years)
    observed = np.asarray(observed, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(years, gamma_p10, gamma_p90, color="tab:blue", alpha=0.15, label="Gamma P10-P90")
    ax.fill_between(years, glm_p10, glm_p90, color="tab:red", alpha=0.15, label="Gamma-GLM P10-P90")
    ax.plot(years, gamma_mean, "--", color="tab:blue", linewidth=2, label="Gamma mean")
    ax.plot(years, glm_mean, "--", color="tab:red", linewidth=2, label="Gamma-GLM mean")
    ax.plot(years, observed, "o-", color="black", linewidth=2, markersize=4, label="Observed")
    ax.set_xlabel("Winter year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig
