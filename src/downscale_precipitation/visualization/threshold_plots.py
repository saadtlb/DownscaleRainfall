import matplotlib.pyplot as plt
import numpy as np


def plot_f1_threshold_curve(thresholds, f1_scores, best_threshold, best_f1, station_label="Station"):
    """Plot F1 score as a function of the decision threshold."""
    thresholds = np.asarray(thresholds, dtype=float)
    f1_scores = np.asarray(f1_scores, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_scores, color="tab:blue", linewidth=2, label="F1 score")
    ax.scatter([best_threshold], [best_f1], color="tab:green", s=90, zorder=3, label="Best threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.set_title(f"F1 vs threshold - {station_label}")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_f1_default_vs_optimal(station_labels, f1_default, f1_optimal):
    """Compare default and optimal F1 scores for all stations."""
    station_labels = list(station_labels)
    f1_default = np.asarray(f1_default, dtype=float)
    f1_optimal = np.asarray(f1_optimal, dtype=float)
    x = np.arange(len(station_labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(station_labels) * 0.8), 5))
    ax.bar(x - width / 2, f1_default, width, label="Threshold = 0.5", color="tab:orange")
    ax.bar(x + width / 2, f1_optimal, width, label="Best F1 threshold", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(station_labels, rotation=35, ha="right")
    ax.set_ylabel("F1")
    ax.set_title("F1 by station: default vs optimized threshold")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_roc_curve_station(fpr, tpr, auc, best_fpr=None, best_tpr=None, station_label="Station"):
    """Plot ROC curve for one station."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="tab:blue", linewidth=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    if best_fpr is not None and best_tpr is not None:
        ax.scatter([best_fpr], [best_tpr], color="tab:green", s=90, zorder=3, label="Best Youden point")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve - {station_label}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_auc_by_station(station_labels, auc_values):
    """Plot one AUC bar per station."""
    station_labels = list(station_labels)
    auc_values = np.asarray(auc_values, dtype=float)

    fig, ax = plt.subplots(figsize=(max(10, len(station_labels) * 0.8), 5))
    bars = ax.bar(station_labels, auc_values, color="tab:blue", edgecolor="black")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("ROC AUC by station")
    ax.tick_params(axis="x", rotation=35)

    for bar, value in zip(bars, auc_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return fig
