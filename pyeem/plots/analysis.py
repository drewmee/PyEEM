import matplotlib.pyplot as plt
import numpy as np

from .base import _get_subplot_dims


def model_history(history):
    """[summary]

    Args:
        history ([type]): [description]

    Returns:
        [type]: [description]
    """
    fig, axes = plt.subplots(figsize=(8, 4), ncols=2, sharex=True)
    for i, metric in enumerate(["accuracy", "loss"]):
        ax = axes[i]
        ax.plot(history.history[metric])
        ax.plot(history.history["val_%s" % metric])
        ax.set_title("Model %s" % metric.title())
        ax.set_ylabel(metric.title())
        ax.set_xlabel("Epoch")
        ax.legend(["Train", "Val."], loc="upper left", fontsize=11)

    plt.tight_layout()
    return axes


def prediction_parity_plots(
    dataset, test_df, train_df=None, subplots=False, fig_kws={}, **kwargs
):
    """[summary]

    Args:
        dataset ([type]): [description]
        test_df ([type]): [description]
        train_df ([type], optional): [description]. Defaults to None.
        subplots (bool, optional): [description]. Defaults to False.
        fig_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        [type]: [description]
    """
    colors = plt.rcParams["axes.prop_cycle"]()
    sources = dataset.calibration_sources

    nsources = len(sources)
    nrows, ncols = _get_subplot_dims(nsources)
    nplots = nrows * ncols

    default_fig_kws = dict(figsize=(ncols ** 2, nrows * ncols), squeeze=False)
    fig_kws = dict(default_fig_kws, **fig_kws)

    fig, axes = plt.subplots(1, nsources, **fig_kws)

    def _get_regression_metric(source_df, metric):
        return source_df.index.get_level_values(level=metric).unique().item()

    pred_dfs = {"test": test_df, "train": train_df}

    ax_idx = 0
    lines = []
    labels = []
    for source in sources:
        for key, df in pred_dfs.items():
            if df is None:
                continue

            if key == "test":
                marker_color = next(colors)["color"]
                line_color = "black"
                alpha = 1
                zorder = 1
            else:
                marker_color = "lightblue"
                line_color = "grey"
                alpha = 0.25
                zorder = -1

            source_df = df.xs(source, level="source")
            source_units = _get_regression_metric(source_df, "units")
            slope = _get_regression_metric(source_df, "slope")
            y_intercept = _get_regression_metric(source_df, "intercept")
            r_squared = _get_regression_metric(source_df, "r_squared")
            cal_poly = np.poly1d([slope, y_intercept])

            x = source_df["true_concentration"]
            y = source_df["predicted_concentration"]
            axes.flat[ax_idx].scatter(
                x, y, label=key, color=marker_color, alpha=alpha, zorder=zorder
            )

            x = np.linspace(
                source_df["true_concentration"].min(),
                source_df["true_concentration"].max(),
            )
            axes.flat[ax_idx].plot(
                x,
                cal_poly(x),
                label="y = %s\n$R^2=%.2f$"
                % (str(cal_poly).replace("\n", ""), r_squared),
                color=line_color,
                linestyle="--",
                zorder=zorder,
            )

        formatted_source_str = source.replace("_", " ").title()
        xlabel_str = "True Conc., %s" % source_units
        ylabel_str = "Predicted Conc., %s" % source_units
        axes.flat[ax_idx].set_xlabel(xlabel_str, fontsize=14)
        axes.flat[ax_idx].set_ylabel(ylabel_str, fontsize=14)
        axes.flat[ax_idx].tick_params(axis="both", which="major", labelsize=12)
        axes.flat[ax_idx].set_title(
            "Parity Plot for\n%s Concentration" % formatted_source_str,
            pad=10,
            fontsize=16,
        )
        ax_line, ax_label = axes.flat[ax_idx].get_legend_handles_labels()
        lines.extend(ax_line)
        labels.extend(ax_label)
        axes.flat[ax_idx].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=11
        )
        ax_idx += 1

    hspace = kwargs.get("subplot_hspace", 0)
    wspace = kwargs.get("subplot_wspace", 0.3)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    return axes
