# this code should include functions to visualize the results 
# bar plots 
# distribution shift plots (?) 
# line plots for the simulations 

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
import umap.umap_ as umap
from sklearn.decomposition import PCA


def plot_metric_bars(df, metric="precision", title=None, save_path=None, padding_ratio=0.1):
    """
    Plot bar chart with confidence intervals for a given metric.

    Args:
        df: DataFrame with columns ['set', 'metric', 'mean', 'lower', 'upper'].
        metric: Which metric to plot (e.g. 'precision', 'recall', or 'MAE').
        title: Optional title.
        save_path: Optional path to save figure.
        padding_ratio: Extra space added above/below the data range (as fraction of range).
    """
    data = df[df["metric"] == metric].copy()
    if data.empty:
        raise ValueError(f"No data found for metric '{metric}'")

    # Compute error bars
    data["err_low"] = data["mean"] - data["lower"]
    data["err_high"] = data["upper"] - data["mean"]

    # Compute range dynamically
    y_min = float(data["lower"].min())
    y_max = float(data["upper"].max())

    # If all values >= 0, start from 0; otherwise, extend below min
    if y_min >= 0:
        y_min = 0
    y_range = y_max - y_min
    padding = y_range * padding_ratio if y_range > 0 else 0.05 * max(abs(y_max), 1)
    ylim = (y_min - padding if y_min < 0 else y_min, y_max + padding)

    # Plot
    plt.figure(figsize=(6, 5))
    bars = plt.bar(
        data["set"],
        data["mean"],
        yerr=[data["err_low"], data["err_high"]],
        capsize=6,
        color="skyblue",
        edgecolor="black",
    )

    plt.ylim(ylim)
    plt.title(title or metric.capitalize())
    plt.ylabel(metric.capitalize())
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add values above bars
    offset = y_range * 0.02 if y_range > 0 else 0.02
    for bar, mean in zip(bars, data["mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            mean + offset if mean >= 0 else mean - offset,
            f"{mean:.2f}" if y_max < 2 else f"{mean:.2f}",
            ha="center",
            va="bottom" if mean >= 0 else "top",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_metric_differences(
    results_plot,
    metrics=('precision', 'recall'),
    x_label="Target P(Y)",
    y_label="Absolute difference vs target",
    title_prefix="",
    figsize=(10, 5),
):
    """
    Plot absolute metric differences (with CI) vs target performance.
    Creates one separate plot for each metric.

    Args:
        results_plot: DataFrame or list of DataFrames with columns
            ['set', 'metric', 'mean_abs_diff', 'ci_2.5%', 'ci_97.5%', 'param_value'].
        metrics: Tuple/list of metric names to plot (default: ('precision', 'recall')).
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        title_prefix: Optional prefix added to each plot title (e.g. 'Metric: ').
        figsize: Tuple specifying figure size.
    """

    # Convert list of DataFrames to a single DataFrame if needed
    if isinstance(results_plot, list):
        df_plot = pd.concat(results_plot, ignore_index=True)
    else:
        df_plot = results_plot.copy()

    # Filter for selected metrics
    df_plot_filtered = df_plot[df_plot['metric'].isin(metrics)]
    df_plot_filtered = df_plot_filtered.sort_values(by='param_value')

    # Seaborn style
    sns.set_theme(style="whitegrid")

    figures = {}

    # Plot each metric separately
    for metric in metrics:
        df_metric = df_plot_filtered[df_plot_filtered['metric'] == metric]

        plt.figure(figsize=figsize)
        for method, group in df_metric.groupby("set"):
            group = group.sort_values("param_value")
            plt.plot(group["param_value"], group["mean_abs_diff"], marker='o', label=method)
            plt.fill_between(
                group["param_value"],
                group["ci_2.5%"],
                group["ci_97.5%"],
                alpha=0.2
            )

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{title_prefix}{metric.capitalize()}")
        plt.legend(title="Dataset")
        plt.tight_layout()
        plt.show()

        # Store figure handle for optional further use (saving, etc.)
        figures[metric] = plt.gcf()

    return figures


def plot_pca_tripanel_hexbin(df_source, df_target, df_sample, ignore_cols=None, gridsize=40):
    ignore_cols = ignore_cols or []

    def preprocess(df):
        return df.drop(columns=ignore_cols, errors="ignore") \
                 .select_dtypes(include=['number']) \
                 .dropna()

    Xs = preprocess(df_source)
    Xt = preprocess(df_target)
    Xp = preprocess(df_sample)

    # PCA on combined data
    X = pd.concat([Xs, Xt, Xp], axis=0)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    s_end = len(Xs)
    t_end = s_end + len(Xt)
    coords_s = coords[:s_end]
    coords_t = coords[s_end:t_end]
    coords_p = coords[t_end:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    datasets = [
        ("source", coords_s, "Blues"),
        ("target", coords_t, "Reds"),
        ("sample", coords_p, "Greens"),
    ]

    for ax, (label, pts, cmap) in zip(axes, datasets):
        hb = ax.hexbin(
            pts[:,0], pts[:,1],
            gridsize=gridsize,
            cmap=cmap,
            mincnt=1,
            linewidths=0.2
        )
        ax.set_title(label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(hb, ax=ax, label='count')

    plt.tight_layout()
    plt.show()


def plot_pca_joint_kde(df_source, df_target, df_sample, ignore_cols=None):
    ignore_cols = ignore_cols or []

    def preprocess(df):
        return (
            df.drop(columns=ignore_cols, errors="ignore")
              .select_dtypes(include=['number'])
              .dropna()
        )

    # Preprocess numerical features
    Xs = preprocess(df_source)
    Xt = preprocess(df_target)
    Xp = preprocess(df_sample)

    # PCA on combined data
    X_all = pd.concat([Xs, Xt, Xp], axis=0)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_all)
    
    # Split back
    s_end = len(Xs)
    t_end = s_end + len(Xt)

    df_s = pd.DataFrame(coords[:s_end], columns=["pca_0", "pca_1"])
    df_t = pd.DataFrame(coords[s_end:t_end], columns=["pca_0", "pca_1"])
    df_p = pd.DataFrame(coords[t_end:], columns=["pca_0", "pca_1"])

    # Add method labels
    df_s["method"] = "source"
    df_t["method"] = "target"
    df_p["method"] = "sample"

    # Combine into a single frame
    fulldata = pd.concat([df_s, df_t, df_p], axis=0)

    # Plot KDE jointplot
    g = sns.jointplot(
        data=fulldata,
        x="pca_0",
        y="pca_1",
        hue="method",
        kind="kde",
        joint_kws={"common_norm": False, "levels": 5, "linewidths": 1},
        marginal_kws={"common_norm": False},
    )

    return g

def plot_two_panel_barplot(results_dict):
    """
    Create a 2-panel grouped barplot (precision | recall)
    with perfectly aligned bars per experiment.

    Parameters
    ----------
    results_dict : dict
        Keys = experiment names (str)
        Values = pandas DataFrames with columns:
            ["set", "metric", "mean", "lower", "upper"]

    Returns
    -------
    A matplotlib figure object.
    """

    dfs = []
    for experiment_name, df in results_dict.items():
        df = df.copy()
        df["experiment"] = experiment_name
        dfs.append(df)

    df_plot = pd.concat(dfs, ignore_index=True)

    df_plot["err_low"]  = df_plot["mean"] - df_plot["lower"]
    df_plot["err_high"] = df_plot["upper"] - df_plot["mean"]

    sets = ["Source", "Target", "Sample", "Weighted"]
    df_plot["set"] = pd.Categorical(df_plot["set"], categories=sets, ordered=True)

    experiments = list(results_dict.keys())

    x = np.arange(len(experiments))  # base positions
    width = 0.18                  # bar width
    offsets = np.linspace(-0.27, 0.27, len(sets))  # 4 evenly spaced bars

    colors = sns.color_palette("Set1", len(sets))
    set_colors = dict(zip(sets, colors))

    def plot_panel(ax, metric):

        dfm = df_plot[df_plot["metric"] == metric]

        for i, s in enumerate(sets):
            sub = dfm[dfm["set"] == s]
            xpos = x + offsets[i]

            ax.bar(
                xpos,
                sub["mean"],
                width=width,
                color=set_colors[s],
                label=s if metric == "precision" else "",
                edgecolor="black"
            )

            # Add CI error bars
            ax.errorbar(
                xpos,
                sub["mean"],
                yerr=[sub["err_low"], sub["err_high"]],
                fmt='none',
                ecolor='black',
                elinewidth=1,
                capsize=3
            )

        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=25, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()}")
        sns.despine(ax=ax)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    plot_panel(axs[0], "precision")
    plot_panel(axs[1], "recall")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Set", bbox_to_anchor=(1.03, 0.95), loc="upper left")

    fig.suptitle("Precision and Recall by experiment", fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig
