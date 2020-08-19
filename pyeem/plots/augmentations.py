import warnings

import matplotlib.pyplot as plt
import pandas as pd
from celluloid import Camera

from .base import _colorbar, _get_subplot_dims, eem_plot


def plot_prototypical_spectra(
    dataset,
    results_df,
    plot_type="imshow",
    fig=None,
    fig_kws={},
    plot_kws={},
    cbar_kws={},
    **kwargs
):
    """Plot the prototypical spectra from the calibration samples.

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        results_df (pandas.DataFrame): [description]
        plot_type (str, optional): [description]. Defaults to "imshow".
        fig (matplotlib.pyplot.figure, optional): [description]. Defaults to None.
        fig_kws (dict, optional): [description]. Defaults to {}.
        plot_kws (dict, optional): [description]. Defaults to {}.
        cbar_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        matplotlib.axes.Axes: [description]
    """

    nspectra = len(results_df.index.unique())
    nrows, ncols = _get_subplot_dims(nspectra)
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

    for i in range(nspectra, nplots):
        axes[i].axis("off")
        axes[i].set_visible(False)
        # axes[i].remove()

    ax_idx = 0
    for index, row in results_df.iterrows():
        proto_eem_df = pd.read_hdf(dataset.hdf, key=row["hdf_path"])
        source_name = proto_eem_df.index.get_level_values("source").unique().item()
        proto_conc = proto_eem_df.index.get_level_values("proto_conc").unique().item()
        source_units = (
            proto_eem_df.index.get_level_values("source_units").unique().item()
        )
        intensity_units = (
            proto_eem_df.index.get_level_values("intensity_units").unique().item()
        )

        title = "Prototypical Spectrum: {0}\n".format(source_name.title())
        title += "Concentration: {0} {1}".format(proto_conc, source_units)

        idx_names = proto_eem_df.index.names
        drop_idx_names = [
            idx_name for idx_name in idx_names if idx_name != "emission_wavelength"
        ]
        proto_eem_df = proto_eem_df.reset_index(level=drop_idx_names, drop=True)

        eem_plot(
            proto_eem_df,
            plot_type=plot_type,
            intensity_units=intensity_units,
            title=title,
            ax=axes[ax_idx],
            fig_kws=fig_kws,
            plot_kws=plot_kws,
            cbar_kws=cbar_kws,
            **kwargs
        )

        ax_idx += 1

    pad = kwargs.get("tight_layout_pad", 1.08)
    h_pad = kwargs.get("tight_layout_hpad", None)
    w_pad = kwargs.get("tight_layout_wpad", None)
    rect = kwargs.get("tight_layout_rect", None)
    if plot_type in ["surface", "surface_contour"]:
        w_pad = kwargs.get("tight_layout_wpad", 25)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
    return axes


def single_source_animation(
    dataset,
    ss_results_df,
    source,
    plot_type="imshow",
    fig_kws={},
    plot_kws={},
    cbar_kws={},
    animate_kws={},
    **kwargs
):
    """Create an animation of the augmented single source spectra.

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        ss_results_df (pandas.DataFrame): [description]
        source (str): [description]
        plot_type (str, optional): [description]. Defaults to "imshow".
        fig_kws (dict, optional): [description]. Defaults to {}.
        cbar_kws (dict, optional): [description]. Defaults to {}.
        animate_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        matplotlib.animation.ArtistAnimation: [description]
    """

    source_results_df = ss_results_df.xs(source, level="source")
    intensity_units = (
        ss_results_df.index.get_level_values(level="intensity_units").unique().item()
    )

    # Set the default figure kws
    default_fig_kws = dict()
    fig_kws = dict(default_fig_kws, **fig_kws)
    fig = plt.figure(**fig_kws)

    projection = None
    if plot_type in ["surface", "surface_contour"]:
        warnings.warn(
            "3D animation may take a considerable amount of time to complete. To speed things up (albeit with decreased resolution), consider increasing the values of the cstride and rstride keyword arguments."
        )
        projection = "3d"

    ax = plt.gca(projection=projection)

    camera = Camera(fig)

    hdf_path = source_results_df.index.get_level_values("hdf_path").unique().item()
    ss_df = pd.read_hdf(dataset.hdf, key=hdf_path)
    ss_np = ss_df.to_numpy()
    min_val = ss_np.min()
    max_val = ss_np.max()

    default_plot_kws = dict(vmin=min_val, vmax=max_val)
    plot_kws = dict(default_fig_kws, **plot_kws)

    default_kwargs = dict(zlim_min=min_val, zlim_max=max_val, title=None)
    kwargs = dict(default_kwargs, **kwargs)

    for concentration, eem_df in ss_df.groupby(source):
        drop_indices = list(eem_df.index.names)
        drop_indices.remove("emission_wavelength")
        eem_df.index = eem_df.index.droplevel(drop_indices)

        hmap = eem_plot(
            eem_df,
            ax=ax,
            plot_type=plot_type,
            include_cbar=False,
            plot_kws=plot_kws,
            **kwargs
        )
        hmap.set_clim(min_val, max_val)

        concentration = round(concentration, 2)
        source = source.replace("_", " ")
        title = "Augmented Single Source Spectra:\n"
        title += "{0}: {1} ug/ml".format(source.title(), concentration)
        if plot_type in ["surface", "surface_contour"]:
            ax.text2D(0.05, 0.95, title, transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0, 1.05, title, transform=ax.transAxes, fontsize=12)
        camera.snap()

    if plot_type in ["surface", "surface_contour"]:
        shrink = cbar_kws.get("shrink", 0.5)
        label_size = cbar_kws.get("size", 12)
        tick_params_labelsize = cbar_kws.get("labelsize", 11)
        cbar = plt.colorbar(hmap, ax=ax, shrink=shrink)
        cbar.set_label(intensity_units, size=label_size)
        cbar.ax.ticklabel_format(
            style="scientific", scilimits=(-2, 3), useMathText=True
        )
        cbar.ax.tick_params(labelsize=tick_params_labelsize)
    else:
        cbar = _colorbar(hmap, intensity_units, cbar_kws=cbar_kws)

    plt.tight_layout()

    # For the saved animation the duration is going to be frames * (1 / fps) (in seconds)
    # For the display animation the duration is going to be frames * interval / 1000 (in seconds)
    animation = camera.animate(**animate_kws)
    return animation


def mixture_animation(
    dataset,
    mix_results_df,
    plot_type="imshow",
    fig_kws={},
    plot_kws={},
    cbar_kws={},
    animate_kws={},
    **kwargs
):
    """Create an animation of the augmented mixture spectra.

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        mix_results_df (pandas.DataFrame): [description]
        plot_type (str, optional): [description]. Defaults to "imshow".
        fig_kws (dict, optional): [description]. Defaults to {}.
        cbar_kws (dict, optional): [description]. Defaults to {}.
        animate_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        matplotlib.animation.ArtistAnimation: [description]
    """

    # Set the default figure kws
    default_fig_kws = dict()
    fig_kws = dict(default_fig_kws, **fig_kws)
    fig = plt.figure(**fig_kws)

    projection = None
    if plot_type in ["surface", "surface_contour"]:
        warnings.warn(
            "3D animation may take a considerable amount of time to complete. To speed things up (albeit with decreased resolution), consider increasing the values of the cstride and rstride keyword arguments."
        )
        projection = "3d"

    ax = plt.gca(projection=projection)

    camera = Camera(fig)

    hdf_path = mix_results_df.index.get_level_values("hdf_path").unique().item()
    mix_df = pd.read_hdf(dataset.hdf, key=hdf_path)

    sources = list(dataset.calibration_sources.keys())
    source_units = (
        mix_results_df.index.get_level_values(level="source_units").unique().item()
    )
    intensity_units = (
        mix_results_df.index.get_level_values(level="intensity_units").unique().item()
    )

    mix_df = pd.read_hdf(dataset.hdf, key=hdf_path)
    mix_np = mix_df.to_numpy()
    min_val = mix_np.min()
    max_val = mix_np.max()

    default_plot_kws = dict(vmin=min_val, vmax=max_val)
    plot_kws = dict(default_fig_kws, **plot_kws)

    default_kwargs = dict(zlim_min=min_val, zlim_max=max_val, title=None)
    kwargs = dict(default_kwargs, **kwargs)

    for concentrations, eem_df in mix_df.groupby(level=sources):
        drop_indices = list(eem_df.index.names)
        drop_indices.remove("emission_wavelength")
        eem_df.index = eem_df.index.droplevel(drop_indices)

        hmap = eem_plot(
            eem_df,
            ax=ax,
            plot_type=plot_type,
            include_cbar=False,
            plot_kws=plot_kws,
            **kwargs
        )
        hmap.set_clim(min_val, max_val)

        rounded_conc = [str(round(conc, 2)) for conc in concentrations]
        title = "Augmented Mixture Spectra:\n"
        for key, value in dict(zip(sources, rounded_conc)).items():
            title += "%s: %s, " % (key.replace("_", " ").title(), value)
        title = title.rsplit(",", 1)[0]
        title += " %s" % source_units

        if plot_type in ["surface", "surface_contour"]:
            ax.text2D(0.05, 0.95, title, transform=ax.transAxes, fontsize=12)
        else:
            ax.text(-0.3, 1.05, title, transform=ax.transAxes, fontsize=12)
        camera.snap()

    if plot_type in ["surface", "surface_contour"]:
        shrink = cbar_kws.get("shrink", 0.5)
        label_size = cbar_kws.get("size", 12)
        tick_params_labelsize = cbar_kws.get("labelsize", 11)
        cbar = plt.colorbar(hmap, ax=ax, shrink=shrink)
        cbar.set_label(intensity_units, size=label_size)
        cbar.ax.ticklabel_format(
            style="scientific", scilimits=(-2, 3), useMathText=True
        )
        cbar.ax.tick_params(labelsize=tick_params_labelsize)
    else:
        cbar = _colorbar(hmap, intensity_units, cbar_kws=cbar_kws)

    plt.tight_layout()

    # For the saved animation the duration is going to be frames * (1 / fps) (in seconds)
    # For the display animation the duration is going to be frames * interval / 1000 (in seconds)
    animation = camera.animate(**animate_kws)
    return animation
