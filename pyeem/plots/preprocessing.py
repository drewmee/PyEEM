import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from celluloid import Camera
from matplotlib import dates
from matplotlib.patches import Rectangle

from ..preprocessing.corrections import _calculate_raman_peak_area
from .base import _get_subplot_dims, eem_plot


def preprocessing_routine_plot(
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
        fig_kws (dict, optional): Optional keyword arguments to include for the figure. Defaults to {}.
        plot_kws (dict, optional): Optional keyword arguments to include. They are sent as an argument 
            to the matplotlib plot call. Defaults to {}.
        cbar_kws (dict, optional): Optional keyword arguments to include for the colorbar. Defaults to {}.

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


def water_raman_peak_plot(
    raman_df,
    excitation_wavelength,
    units="nm",
    ax=None,
    fig_kws={},
    plot_kws={},
    vline_kws={},
    fill_between_kws={},
    legend_kws={},
    **kwargs
):
    """[summary]

    Args:
        raman_df (pandas.DataFrame): [description]
        excitation_wavelength ([type]): [description]
        units (str, optional): [description]. Defaults to "nm".
        ax ([type], optional): [description]. Defaults to None.
        fig_kws (dict, optional): [description]. Defaults to {}.
        plot_kws (dict, optional): [description]. Defaults to {}.
        vline_kws (dict, optional): [description]. Defaults to {}.
        fill_between_kws (dict, optional): [description]. Defaults to {}.
        legend_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        [type]: [description]
    """
    if ax is None:
        default_fig_kws = dict()
        fig_kws = dict(default_fig_kws, **fig_kws)
        fig = plt.figure(**fig_kws)
        ax = plt.gca()

    raman_peak_area, (peak_position, a, b) = _calculate_raman_peak_area(
        raman_df, excitation_wavelength
    )

    ymin = kwargs.get("ymin", raman_df.min().item())
    ymax = kwargs.get("ymax", raman_df.max().item())
    datetime = kwargs.get("datetime", None)

    ax.plot(raman_df, color="black", linewidth=0.5, **plot_kws)

    peak_pos_vline = ax.vlines(
        peak_position,
        ymin=ymin,
        ymax=ymax,
        ls="--",
        color="black",
        alpha=0.75,
        linewidth=0.5,
        **vline_kws
    )
    a_vline = ax.vlines(
        a, ymin=ymin, ymax=ymax, ls="--", color="grey", linewidth=0.5, **vline_kws
    )
    b_vline = ax.vlines(
        b, ymin=ymin, ymax=ymax, ls="--", color="grey", linewidth=0.5, **vline_kws
    )

    tmp_df = raman_df.loc[a:b, raman_df.columns[0]]
    pos_fill = ax.fill_between(
        tmp_df.index,
        0,
        tmp_df,
        where=(tmp_df > 0),
        facecolor="g",
        alpha=0.5,
        **fill_between_kws
    )
    neg_fill = ax.fill_between(
        tmp_df.index,
        0,
        tmp_df,
        where=(tmp_df < 0),
        facecolor="r",
        alpha=0.5,
        **fill_between_kws
    )
    raman_area_placeholder = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0
    )
    wavelength_placeholder = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0
    )

    handle_label_dict = {}
    handle_label_dict[wavelength_placeholder] = "Excitation wavelength:\n%d%s" % (
        excitation_wavelength,
        units,
    )
    if datetime:
        datetime_placeholder = Rectangle(
            (0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0
        )
        handle_label_dict[datetime_placeholder] = "Datetime (UTC):\n%s" % datetime

    handle_label_dict[raman_area_placeholder] = (
        "Raman peak area:\n%.2f" % raman_peak_area
    )
    handle_label_dict[peak_pos_vline] = "peak position\n(%0.2f)" % peak_position
    handle_label_dict[a_vline] = "peak boundaries\n(%0.2f, %0.2f)" % (a, b)
    handle_label_dict[pos_fill] = "positive"
    handle_label_dict[neg_fill] = "negative"

    ax.legend(
        handle_label_dict.keys(),
        handle_label_dict.values(),
        fontsize=4,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        **legend_kws
    )

    ylabel_fontsize = kwargs.get("ylabel_fontsize", 8)
    xlabel_fontsize = kwargs.get("xlabel_fontsize", 8)
    title_fontsize = kwargs.get("title_fontsize", 8)
    tick_params_fontsize = kwargs.get("tick_params_fontsize", 8)
    offset_text_fontsize = kwargs.get("offset_text_fontsize", 8)

    ax.yaxis.offsetText.set_fontsize(offset_text_fontsize)

    ax.set_xlabel("Wavelength (%s)" % units, fontsize=xlabel_fontsize)
    ax.set_ylabel("Intensity (A.U.)", fontsize=ylabel_fontsize)
    title_str = "Water Raman scan"
    ax.set_title(title_str, fontsize=title_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_params_fontsize)
    return ax


def water_raman_peak_animation(dataset, excitation_wavelength, fig_kws={}, **kwargs):
    """[summary]

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        excitation_wavelength ([type]): [description]
        fig_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        [type]: [description]
    """
    # Set the default figure kws
    default_fig_kws = dict(figsize=(4, 4), dpi=500)
    fig_kws = dict(default_fig_kws, **fig_kws)

    fig = plt.figure(**fig_kws)
    rect = kwargs.get("rect", [0.15, 0.3, 0.5, 0.45])
    ax = fig.add_axes(rect)
    camera = Camera(fig)

    ymin = np.inf
    ymax = -(np.inf)
    raman_meta_df = dataset.meta_df.xs("water_raman", level="scan_type")
    raman_meta_df = raman_meta_df[
        raman_meta_df["water_raman_wavelength"] == excitation_wavelength
    ]
    for index, row in raman_meta_df.iterrows():
        raman_df = pd.read_hdf(dataset.hdf, key=row["hdf_path"])
        curr_min = raman_df.min().item()
        curr_max = raman_df.max().item()
        if curr_min < ymin:
            ymin = curr_min
        if curr_max > ymax:
            ymax = curr_max

    for index, row in raman_meta_df.iterrows():
        raman_df = pd.read_hdf(dataset.hdf, key=row["hdf_path"])
        raman_peak_area, (peak_position, a, b) = _calculate_raman_peak_area(
            raman_df, excitation_wavelength
        )
        kwargs = {"ymin": ymin, "ymax": ymax, "datetime": row["datetime_utc"]}
        water_raman_peak_plot(raman_df, excitation_wavelength, ax=ax, **kwargs)
        camera.snap()

    animation = camera.animate()
    return animation


def water_raman_timeseries(
    dataset,
    excitation_wavelength,
    units="nm",
    metric="area",
    fig_kws={},
    plot_kws={},
    **kwargs
):
    """[summary]

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        excitation_wavelength ([type]): [description]
        units (str, optional): [description]. Defaults to "nm".
        metric (str, optional): [description]. Defaults to "area".
        fig_kws (dict, optional): [description]. Defaults to {}.
        plot_kws (dict, optional): [description]. Defaults to {}.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    raman_meta_df = dataset.meta_df.xs("water_raman", level="scan_type")
    raman_meta_df = raman_meta_df[
        raman_meta_df["water_raman_wavelength"] == excitation_wavelength
    ]

    y = []
    for index, row in raman_meta_df.iterrows():
        raman_df = pd.read_hdf(dataset.hdf, key=row["hdf_path"])
        (raman_peak_area, (peak_position, a, b),) = _calculate_raman_peak_area(
            raman_df, excitation_wavelength
        )
        if metric == "area":
            metric_value = raman_peak_area
        elif metric == "peak_position":
            metric_value = peak_position
        elif metric == "peak_width":
            metric_value = b - a
        else:
            raise ValueError(
                "Invalid input for metric. Must be 'area', 'peak_position', or 'peak_width'"
            )
        y.append(metric_value)

    default_fig_kws = dict(figsize=(8, 6), dpi=100)
    fig_kws = dict(default_fig_kws, **fig_kws)
    fig = plt.figure(**fig_kws)
    ax = plt.gca()

    default_plot_kws = dict(fmt="o-")
    plot_kws = dict(default_plot_kws, **plot_kws)
    ax.plot_date(raman_meta_df["datetime_utc"], y, **plot_kws)

    byweekday = kwargs.get("byweekday", range(0, 7))
    interval = kwargs.get("interval", 1)
    minor_formatter_date_format = kwargs.get(
        "minor_formatter_date_format", "%d"
    )  #%d %a
    major_formatter_date_format = kwargs.get("major_formatter_date_format", "\n%b\n%Y")
    grid_alpha = kwargs.get("grid_alpha", 0.1)

    ax.xaxis.set_minor_locator(
        dates.WeekdayLocator(byweekday=byweekday, interval=interval)
    )
    ax.xaxis.set_minor_formatter(dates.DateFormatter(minor_formatter_date_format))
    ax.xaxis.grid(True, which="minor", alpha=grid_alpha)
    ax.yaxis.grid(alpha=grid_alpha)
    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter(major_formatter_date_format))

    ylabel_fontsize = kwargs.get("ylabel_fontsize", 14)
    xlabel_fontsize = kwargs.get("xlabel_fontsize", 14)
    title_fontsize = kwargs.get("title_fontsize", 14)
    x_minor_tick_labels_fontsize = kwargs.get("x_minor_tick_labels_fontsize", 14)

    metric_title_str = metric.replace("_", " ")
    plt.setp(
        ax.xaxis.get_minorticklabels(), fontsize=x_minor_tick_labels_fontsize,
    )
    ax.set_ylabel("Water Raman %s (A.U.)" % metric_title_str, fontsize=ylabel_fontsize)
    ax.set_xlabel("Datetime (UTC)", fontsize=xlabel_fontsize)
    title_str = "Timeseries of water Raman %s\nExcitation wavelength: %d%s" % (
        metric_title_str,
        excitation_wavelength,
        units,
    )
    ax.set_title(title_str, fontsize=title_fontsize)
    plt.tight_layout()
    return ax


def absorbance_plot(
    absorbance_df,
    wavelength_bounds=None,
    units="nm",
    ax=None,
    fig_kws={},
    plot_kws={},
    **kwargs
):
    """[summary]

    Args:
        absorbance_df (pandas.DataFrame): [description]
        wavelength_bounds ([type], optional): [description]. Defaults to None.
        units (str, optional): [description]. Defaults to "nm".
        ax ([type], optional): [description]. Defaults to None.
        fig_kws (dict, optional): [description]. Defaults to {}.
        plot_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        [type]: [description]
    """
    if ax is None:
        default_fig_kws = dict()
        fig_kws = dict(default_fig_kws, **fig_kws)
        fig = plt.figure(**fig_kws)
        ax = plt.gca()

    if wavelength_bounds:
        a = wavelength_bounds[0]
        b = wavelength_bounds[1]
        absorbance_df = absorbance_df.loc[a:b]

    ax.plot(absorbance_df, **plot_kws)

    ylabel_fontsize = kwargs.get("ylabel_fontsize", 12)
    xlabel_fontsize = kwargs.get("xlabel_fontsize", 12)
    title_fontsize = kwargs.get("title_fontsize", 14)
    xlabel = kwargs.get("xlabel", "Wavelength (%s)" % units)
    ylabel = kwargs.get("ylabel", "Absorbance (A.U.)")
    title = kwargs.get("title", "Absorbance spectra")
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    tick_params_fontsize = kwargs.get("tick_params_fontsize", 12)
    offset_text_fontsize = kwargs.get("offset_text_fontsize", 12)
    ax.yaxis.offsetText.set_fontsize(offset_text_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_params_fontsize)
    return ax


def calibration_curves_plot(dataset, cal_df, subplots=False, fig_kws={}, **kwargs):
    """[summary]

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        cal_df (pandas.DataFrame): [description]
        subplots (bool, optional): [description]. Defaults to False.
        fig_kws (dict, optional): Optional keyword arguments to include for the figure. Defaults to {}.

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
        # axes.flat[ax_idx].legend(loc="upper left", fontsize=11)
        axes.flat[ax_idx].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=11
        )
        ax_idx += 1

    hspace = kwargs.get("subplot_hspace", 0)
    wspace = kwargs.get("subplot_wspace", 0.3)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    return axes
