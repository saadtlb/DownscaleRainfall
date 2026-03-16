import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_configuration_metric_bars(metric_frame, metric_name):
    """Plot a simple bar chart for one metric across configurations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_frame.plot(kind="bar", x="Configuration", y=metric_name, ax=ax, legend=False, color="tab:blue")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by configuration")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig


def plot_error_heatmap(error_table, title="RMSE heatmap (daily mean)"):
    """Plot a heatmap for one multi-station error table."""
    if isinstance(error_table, pd.DataFrame) and not error_table.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(error_table, annot=True, fmt=".1f", cmap="viridis", ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig
    return None
