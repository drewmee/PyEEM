import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from celluloid import Camera

from .base import _get_subplot_dims, eem_plot


def plot_preprocessing(
    dataset,
    routine_results_df,
    sample_set,
    sample_name=None,
    include_complete=False,
    plot_type="imshow",
    fig=None,
    fig_kws={},
    plot_kws={},
    cbar_kws={},
    **kwargs
):
    """[summary]

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        routine_results_df (pandas.DataFrame): [description]
        sample_set (int): [description]
        sample_name (str, optional): [description]. Defaults to None.
        include_complete (bool, optional): [description]. Defaults to False.
        plot_type (str, optional): [description]. Defaults to "imshow".
        fig (matplotlib.pyplot.figure, optional): [description]. Defaults to None.
        fig_kws (dict, optional): [description]. Defaults to {}.
        plot_kws (dict, optional): [description]. Defaults to {}.
        cbar_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        matplotlib.axes.Axes: [description]
    """
    rr_df = routine_results_df.copy()
    if not include_complete:
        if "complete" in rr_df.index.get_level_values("step_name"):
            rr_df.drop("complete", level="step_name", inplace=True)

    rr_df = rr_df[rr_df["step_completed"].fillna(False)]
    blank_steps_dict = {}
    scan_set = rr_df.xs(sample_set, level="sample_set", drop_level=False)

    if "blank_eem" in scan_set.index.get_level_values("scan_type"):
        blank = scan_set.xs("blank_eem", level="scan_type")
        for step_name in blank.index.get_level_values("step_name"):
            units = blank.xs(step_name, level="step_name")["units"].unique().item()
            hdf_path = (
                blank.xs(step_name, level="step_name")["hdf_path"].unique().item()
            )
            title = "Blank: {0}".format(step_name.replace("_", " ").title())
            blank_steps_dict[title] = {
                "units": units,
                "eem_df": pd.read_hdf(dataset.hdf, key=hdf_path),
            }

    sample_steps_dict = {}
    if "sample_eem" in scan_set.index.get_level_values(
        "scan_type"
    ) and sample_name in scan_set.index.get_level_values("name"):
        sample = scan_set.xs(("sample_eem", sample_name), level=["scan_type", "name"])
        for step_name in sample.index.get_level_values("step_name"):
            units = sample.xs(step_name, level="step_name")["units"].unique().item()
            hdf_path = (
                sample.xs(step_name, level="step_name")["hdf_path"].unique().item()
            )
            title = "Sample: {0}".format(step_name.replace("_", " ").title())
            sample_steps_dict[title] = {
                "units": units,
                "eem_df": pd.read_hdf(dataset.hdf, key=hdf_path),
            }

    steps_spectra_dict = {**blank_steps_dict, **sample_steps_dict}

    nsteps = len(steps_spectra_dict.keys())
    nrows, ncols = _get_subplot_dims(nsteps)
    nplots = nrows * ncols

    # Set the fig_kws as a mapping of default and kwargs
    default_fig_kws = dict(
        tight_layout={"h_pad": 5, "w_pad": 0.05}, figsize=(ncols ** 2, nrows * ncols)
    )
    # Set the fig_kws
    fig_kws = dict(default_fig_kws, **fig_kws)
    fig = plt.figure(**fig_kws)

    projection = None
    if plot_type in ["surface", "surface_contour"]:
        projection = "3d"

    axes = []
    for i in range(1, ncols * nrows + 1):
        ax = fig.add_subplot(nrows, ncols, i, projection=projection)
        axes.append(ax)

    suptitle = "Results of Preprocessing Routine - Sample Set #{0}\nSample name: {1}".format(
        sample_set, sample_name
    )
    suptitle = kwargs.get("suptitle", suptitle)
    suptitle_fontsize = kwargs.get("suptitle_fontsize", 20)
    fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=1.05)

    for i in range(nsteps, nplots):
        axes[i].axis("off")
        axes[i].set_visible(False)
        # axes[i].remove()

    ax_idx = 0
    for title, step_dict in steps_spectra_dict.items():
        units = step_dict["units"]
        eem_df = step_dict["eem_df"]
        eem_plot(
            eem_df,
            ax=axes[ax_idx],
            plot_type=plot_type,
            intensity_units=units,
            plot_kws=plot_kws,
            cbar_kws=cbar_kws,
            title=title,
            **kwargs
        )
        ax_idx += 1

    pad = kwargs.get("tight_layout_pad", 1.08)
    h_pad = kwargs.get("tight_layout_hpad", None)
    w_pad = kwargs.get("tight_layout_wpad", None)
    rect = kwargs.get("tight_layout_rect", None)
    if plot_type in ["surface", "surface_contour"]:
        w_pad = kwargs.get("tight_layout_wpad", 25)
        h_pad = kwargs.get("tight_layout_hpad", 15)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)

    return axes


def plot_calibration_curves(dataset, cal_df, subplots=False, fig_kws={}, **kwargs):
    """[summary]

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        cal_df (pandas.DataFrame): [description]
        subplots (bool, optional): [description]. Defaults to False.
        fig_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        matplotlib.axes.Axes: [description]
    """
    colors = plt.rcParams["axes.prop_cycle"]()
    sources = cal_df.index.get_level_values(level="source").unique()

    nsources = sources.nunique()
    nrows, ncols = _get_subplot_dims(nsources)
    nplots = nrows * ncols

    default_fig_kws = dict(figsize=(ncols ** 2, nrows * ncols), squeeze=False)
    fig_kws = dict(default_fig_kws, **fig_kws)

    fig, axes = plt.subplots(1, nsources, **fig_kws)

    def _get_regression_metric(source_df, metric):
        return source_df.index.get_level_values(level=metric).unique().item()

    ax_idx = 0
    lines = []
    labels = []
    for source in sources:
        c = next(colors)["color"]
        source_df = cal_df.xs(source, level="source")
        source_units = _get_regression_metric(source_df, "source_units")
        measurement_units = _get_regression_metric(source_df, "measurement_units")
        slope = _get_regression_metric(source_df, "slope")
        y_intercept = _get_regression_metric(source_df, "intercept")
        r_squared = _get_regression_metric(source_df, "r_squared")

        cal_poly = np.poly1d([slope, y_intercept])

        proto_spectra_df = source_df[source_df["prototypical_sample"]]
        non_proto_spectra_df = source_df[~source_df["prototypical_sample"]]

        p_x = proto_spectra_df["concentration"].values
        p_y = proto_spectra_df["integrated_intensity"].values
        np_x = non_proto_spectra_df["concentration"].values
        np_y = non_proto_spectra_df["integrated_intensity"].values
        axes.flat[ax_idx].scatter(np_x, np_y, label="non-proto. sample", color=c)
        axes.flat[ax_idx].scatter(
            p_x, p_y, label="proto. sample", marker="*", color="black"
        )

        x = np.linspace(
            source_df["concentration"].min(), source_df["concentration"].max()
        )
        axes.flat[ax_idx].plot(
            x,
            cal_poly(x),
            label="y = %s\n$R^2=%.2f$" % (str(cal_poly).replace("\n", ""), r_squared),
            color="black",
            linestyle="--",
        )

        formatted_source_str = source.replace("_", " ").title()
        xlabel_str = "%s Concentration, %s" % (formatted_source_str, source_units)
        axes.flat[ax_idx].set_xlabel(xlabel_str, fontsize=14)
        axes.flat[ax_idx].set_ylabel(measurement_units, fontsize=14)
        axes.flat[ax_idx].tick_params(axis="both", which="major", labelsize=12)
        axes.flat[ax_idx].set_title(
            "%s Fluorescence Calibration Curve\nfor the %s %s"
            % (
                formatted_source_str,
                dataset.instruments_df.eem.item().manufacturer,
                dataset.instruments_df.eem.item().name.replace("_", " ").title(),
            ),
            pad=30,
            fontsize=14,
        )
        ax_line, ax_label = axes.flat[ax_idx].get_legend_handles_labels()
        lines.extend(ax_line)
        labels.extend(ax_label)
        axes.flat[ax_idx].legend(loc="upper left", fontsize=11)
        ax_idx += 1

    hspace = kwargs.get("subplot_hspace", 0)
    wspace = kwargs.get("subplot_wspace", 0.3)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    return axes
