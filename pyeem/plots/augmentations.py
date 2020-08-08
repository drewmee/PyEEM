import matplotlib.pyplot as plt
import pandas as pd
from celluloid import Camera

from .base import (
    cbar_fontsize,
    cbar_label,
    combined_surface_contour_plot,
    contour_plot,
    eem_xlabel,
    eem_ylabel,
    units,
)

dpi = 100


def plot_prototypical_spectra(dataset, results_df, plot_type="contour"):
    for index, row in results_df.iterrows():
        proto_eem_df = pd.read_hdf(dataset.hdf, key=row["hdf_path"])
        if plot_type == "contour":
            contour_plot(proto_eem_df)
        elif plot_type == "surface":
            combined_surface_contour_plot(proto_eem_df)


def single_source_animation(dataset, ss_results_df, source, plot_type="contour"):
    """[summary]

    Args:
        source_name ([type]): [description]
        eem_df ([type]): [description]
        plot_type (str, optional): [description]. Defaults to "contour".
    """

    source_results_df = ss_results_df.xs(source, level="source")

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.xlabel(eem_xlabel)
    plt.ylabel(eem_ylabel)
    plt.tick_params(axis="both", which="major")

    camera = Camera(fig)

    hdf_path = source_results_df.index.get_level_values("hdf_path").unique().item()
    ss_df = pd.read_hdf(dataset.hdf, key=hdf_path)
    ss_np = ss_df.to_numpy()
    min_val = ss_np.min()
    max_val = ss_np.max()

    for concentration, eem_df in ss_df.groupby(source):
        drop_indices = list(eem_df.index.names)
        drop_indices.remove("emission_wavelength")
        eem_df.index = eem_df.index.droplevel(drop_indices)
        excitation_wavelengths = eem_df.columns.to_numpy()
        emission_wavelengths = eem_df.index.to_numpy()
        extent = [
            excitation_wavelengths[0],
            excitation_wavelengths[-1],
            emission_wavelengths[0],
            emission_wavelengths[-1],
        ]
        hmap = ax.imshow(
            eem_df, vmin=min_val, vmax=max_val, origin="lower", extent=extent,
        )
        concentration = round(concentration, 2)
        source = source.replace("_", " ")
        title = "Augmented Single Source Spectra:\n"
        title += "{0}: {1}ug/ml".format(source.title(), concentration)
        ax.text(0, 1.05, title, transform=ax.transAxes, fontsize=16)
        camera.snap()

    cbar = fig.colorbar(hmap, ax=ax)
    cbar.set_label(cbar_label, size=cbar_fontsize)
    cbar.ax.tick_params(labelsize=14)
    # For the saved animation the duration is going to be frames * (1 / fps) (in seconds)
    # For the display animation the duration is going to be frames * interval / 1000 (in seconds)
    animation = camera.animate()
    return animation


def mix_animation(dataset, mix_results_df):
    """[summary]

    Args:
        sources ([type]): [description]
        mix_df ([type]): [description]
    """

    hdf_path = mix_results_df.index.get_level_values("hdf_path").unique().item()
    aug_mix_df = pd.read_hdf(dataset.hdf, key=hdf_path)
    sources = mix_results_df.columns.to_list()

    """
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=5, ncols=2, width_ratios=[1, 1],
                           wspace=0.3, figure=fig)
    eem_ax = plt.subplot(gs[:, 0])
    bar_ax = plt.subplot(gs[2, 1])
    bar_ax.set_ylim([0, 5])
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.xlabel(eem_xlabel)
    plt.ylabel(eem_ylabel)
    plt.tick_params(axis="both", which="major")

    camera = Camera(fig)

    eem_np = aug_mix_df.to_numpy()
    min_val = eem_np.min()
    max_val = eem_np.max()

    excitation = aug_mix_df.columns.to_numpy()
    emission = aug_mix_df.index.droplevel(sources + ["hdf_path", "source"]).to_numpy()
    extent = [excitation[0], excitation[-1], emission[0], emission[-1]]

    for concentrations, g in aug_mix_df.groupby(level=list(sources)):
        rounded_conc = [str(round(conc, 2)) for conc in concentrations]
        title = "Augmented Mixture Spectra:\n"
        for key, value in dict(zip(sources, rounded_conc)).items():
            title += "%s: %s, " % (key.replace("_", " ").title(), value)
        title = title.rsplit(",", 1)[0] + " ug/ml"

        g.index = g.index.droplevel(sources + ["hdf_path", "source"])
        ax.text(0, 1.05, title, transform=ax.transAxes, fontsize=16)
        hmap = ax.imshow(
            g,
            cmap="viridis",
            vmin=min_val,
            vmax=max_val,
            origin="lower",
            extent=extent,
        )
        # bar_ax.bar(sources, concentrations, color='midnightblue')
        camera.snap()

    cbar = fig.colorbar(hmap, ax=ax)
    cbar.set_label(cbar_label, size=cbar_fontsize)
    cbar.ax.tick_params(labelsize=14)

    animation = camera.animate()
    return animation
