import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from celluloid import Camera

def _get_steps_set(routine_results_df):
    steps_set = (
        routine_results_df[routine_results_df["step_completed"] == True]
        .index.get_level_values("step_name")
        .unique()
        .values.tolist()
    )
    return steps_set


def _get_subplot_dims(n):
    ncols = 4
    if n % ncols:
        nplots = n + (ncols - n % ncols)
    else:
        nplots = n

    nrows = int(nplots / ncols)
    return nrows, ncols


def _get_nplots(grp, steps_set):
    # Add two steps for blank raw and blank crop
    nsteps = len(steps_set) + 2
    nrows, ncols = _get_subplot_dims(nsteps)
    nplots = nrows * ncols
    return nplots


def _plot_preprocessing_helper(
    steps_spectra_dict, steps_set, sample_set_name, name, fig
):
    # Add two steps for blank raw and blank crop
    nsteps = len(steps_set) + 2
    nrows, ncols = _get_subplot_dims(nsteps)
    nplots = nrows * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols ** 2, nrows * ncols))
    title_str = "Results of EEM Preprocessing Routine - Sample Set #{0}\n".format(
        sample_set_name
    )
    title_str += "Sample name: {0}".format(name)
    fig.suptitle(title_str, fontsize=20)

    ax_idx = 0

    for i in range(nsteps, nplots):
        axes.flat[i].axis("off")
        axes.flat[i].set_visible(False)

    for title, df in steps_spectra_dict.items():
        axes.flat[ax_idx].contourf(
            df.columns.to_numpy(), df.index.to_numpy(), df.to_numpy(),
        )
        axes.flat[ax_idx].set_title(title)
        ax_idx += 1
    plt.show()


def _handle_preprocessing_sample_group(
    sample_set_group, dataset, fig, include_complete
):
    sample_set_name = (
        sample_set_group.index.get_level_values("sample_set").unique().item()
    )

    steps_set = _get_steps_set(sample_set_group)
    # filter sample_set_group to keep only those rows with values step_name column
    # which are contained in steps_set.

    steps_spectra_dict = {}

    blank = sample_set_group.xs("blank_eem", level="scan_type")
    for step_name in blank.index.get_level_values("step_name"):
        hdf_path = blank.xs(step_name, level="step_name")["hdf_path"].unique().item()
        title = "Blank: {}".format(step_name)
        steps_spectra_dict[title] = pd.read_hdf(dataset.hdf, key=hdf_path)

    # handle those sample sets with on sample_eems?
    if "sample_eem" not in sample_set_group.index.get_level_values("scan_type"):
        _plot_preprocessing_helper(steps_spectra_dict, steps_set, sample_set_name, None)
        return

    samples = sample_set_group.xs("sample_eem", level="scan_type")
    for name in samples.index.get_level_values("name").unique().values:
        sample = samples.xs(name, level="name")
        for step_name in sample.index.get_level_values("step_name"):
            hdf_path = (
                sample.xs(step_name, level="step_name")["hdf_path"].unique().item()
            )
            title = "Sample: {0}".format(step_name)
            steps_spectra_dict[title] = pd.read_hdf(dataset.hdf, key=hdf_path)

        _plot_preprocessing_helper(
            steps_spectra_dict, steps_set, sample_set_name, name, fig
        )
        break
    return


def plot_preprocessing(
    dataset, routine_results_df, animate=False, include_complete=False
):
    if not include_complete:
        if "complete" in routine_results_df.index.get_level_values("step_name"):
            routine_results_df.drop("complete", level="step_name", inplace=True)

    fig = None
    if animate:
        # Determine what the number of plots and fig size should be.
        steps_set = _get_steps_set(routine_results_df)
        max_nplots = (
            routine_results_df.groupby(level="sample_set").apply(_test, steps_set).max()
        )
        fig = plt.figure()
        camera = Camera(fig)

    routine_results_df.groupby(level="sample_set").apply(
        _handle_preprocessing_sample_group,
        dataset=dataset,
        fig=fig,
        include_complete=include_complete,
    )

    if animate:
        animation = camera.animate()


def raman_area(dataset):
    return

def plot_calibration_curves(dataset, cal_df, subplots=False):
    colors = plt.rcParams["axes.prop_cycle"]()
    sources = cal_df.index.get_level_values(level="source").unique()

    # TODO - Just put all subplots on one row
    nsources = sources.nunique()
    nrows, ncols = _get_subplot_dims(nsources)
    nplots = nrows * ncols
    fig, axes = plt.subplots(1, nsources, figsize=(ncols ** 2, nrows * ncols))
    
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
        
        x = np.linspace(source_df["concentration"].min(),
                        source_df["concentration"].max())
        axes.flat[ax_idx].plot(
            x,
            cal_poly(x),
            label="y = %s\n$R^2=%.2f$" % (str(cal_poly).replace("\n", ""), r_squared),
            color="black",
            linestyle="--"
        )

        formatted_source_str = source.replace("_", " ").title()
        xlabel_str = "%s Concentration, %s" % (formatted_source_str, source_units)
        axes.flat[ax_idx].set_xlabel(xlabel_str, fontsize=12)
        axes.flat[ax_idx].set_ylabel(measurement_units, fontsize=12)
        axes.flat[ax_idx].tick_params(axis="both", which="major", labelsize=12)
        axes.flat[ax_idx].set_title(
            "%s Fluorescence Calibration Curve\nfor the %s %s"
            % (
                formatted_source_str,
                dataset.instruments_df.eem.item().manufacturer,
                dataset.instruments_df.eem.item().name.title(),
            ),
            pad=30,
            fontsize=14,
        )
        ax_line, ax_label = axes.flat[ax_idx].get_legend_handles_labels()
        lines.extend(ax_line)
        labels.extend(ax_label)
        axes.flat[ax_idx].legend(loc="upper left")
        ax_idx += 1
    
    plt.subplots_adjust(wspace=0.3)
    return axes


def plot_calibration_spectra(cal_df):
    return
