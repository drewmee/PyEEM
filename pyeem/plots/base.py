import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from celluloid import Camera
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D


def _get_subplot_dims(n):
    """[summary]

    Args:
        n (int): [description]

    Returns:
        tuple of int: [description]
    """
    ncols = 4
    if n % ncols:
        nplots = n + (ncols - n % ncols)
    else:
        nplots = n

    nrows = int(nplots / ncols)
    return nrows, ncols


def _colorbar(mappable, units, cbar_kws={}):
    """[summary]

    Args:
        mappable (AxesImage or QuadContourSet): [description]
        units (str): [description]
        cbar_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        matplotlib.colorbar.Colorbar: [description]
    """
    # https://joseph-long.com/writing/colorbars/
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)

    cbar_ax_size = cbar_kws.get("cbar_ax_size", "8%")
    cbar_ax_pad = cbar_kws.get("cbar_ax_pad", 0.05)
    cax = divider.append_axes("right", size=cbar_ax_size, pad=cbar_ax_pad)
    cbar = fig.colorbar(mappable, cax=cax)

    cbar_tick_params_labelsize = cbar_kws.get("cbar_tick_params_labelsize", 11)
    cbar.ax.tick_params(labelsize=cbar_tick_params_labelsize)
    cbar.formatter.set_powerlimits((-2, 3))
    plt.sca(last_axes)

    cbar_label_size = cbar_kws.get("cbar_label_size", 12)
    cbar_labelpad = cbar_kws.get("cbar_labelpad", 5)
    cbar.set_label(units, size=cbar_label_size, labelpad=cbar_labelpad)
    return cbar


def _eem_contour(
    eem, ax, intensity_units, include_cbar, plot_kws={}, cbar_kws={}, **kwargs
):
    """[summary]

    Args:
        eem (pandas.DataFrame): [description]
        ax (matplotlib.axes.Axes): [description]
        intensity_units (str): [description]
        include_cbar (bool): [description]
        plot_kws (dict, optional): [description]. Defaults to {}.
        cbar_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        QuadContourSet: [description]
    """
    # Set the default plot kws.
    # contourf doesn't take aspect as a kwarg...
    # so we have to remove it and pass it seperately
    # via set_aspect(). Clunky but oh well.
    default_aspect = "equal"
    aspect = plot_kws.get("aspect", default_aspect)
    contour_kws = plot_kws.copy()
    contour_kws.pop("aspect", None)
    default_contour_kws = dict()
    contour_kws = dict(default_contour_kws, **contour_kws)

    fl = eem.to_numpy()
    excitation = eem.columns.to_numpy()
    emission = eem.index.to_numpy()

    hmap = ax.contourf(excitation, emission, fl, **contour_kws)
    ax.set_aspect(aspect)

    if include_cbar:
        cbar = _colorbar(hmap, units=intensity_units, cbar_kws=cbar_kws)

    return hmap


def _eem_imshow(
    eem, ax, intensity_units, include_cbar, plot_kws={}, cbar_kws={}, **kwargs
):
    """[summary]

    Args:
        eem (pandas.DataFrame): [description]
        ax (matplotlib.axes.Axes): [description]
        intensity_units (str): [description]
        include_cbar (bool): [description]
        plot_kws (dict, optional): [description]. Defaults to {}.
        cbar_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        AxesImage: [description]
    """
    excitation = eem.columns.to_numpy()
    emission = eem.index.to_numpy()
    default_plot_kws = dict(
        origin="lower",
        extent=[excitation[0], excitation[-1], emission[0], emission[-1]],
        aspect="equal",
    )
    plot_kws = dict(default_plot_kws, **plot_kws)

    hmap = ax.imshow(eem, **plot_kws)
    if include_cbar:
        cbar = _colorbar(hmap, intensity_units, cbar_kws=cbar_kws)
    return hmap


def _eem_surface_contour(
    eem,
    ax,
    intensity_units,
    include_cbar,
    plot_type="surface",
    surface_plot_kws={},
    contour_plot_kws={},
    cbar_kws={},
    **kwargs
):
    """[summary]

    Args:
        eem (pandas.DataFrame): [description]
        ax (matplotlib.axes.Axes): [description]
        intensity_units (str): [description]
        include_cbar (bool): [description]
        plot_type (str, optional): [description]. Defaults to "surface".
        surface_plot_kws (dict, optional): [description]. Defaults to {}.
        contour_plot_kws (dict, optional): [description]. Defaults to {}.
        cbar_kws (dict, optional): [description]. Defaults to {}.

    Returns:
        mpl_toolkits.mplot3d.art3d.Poly3DCollection: [description]
    """
    excitation = eem.columns.to_numpy()
    emission = eem.index.to_numpy()
    fl = eem.to_numpy()
    excitation, emission = np.meshgrid(excitation, emission)

    default_surface_plot_kws = dict(
        rstride=1, cstride=1, alpha=0.75, cmap="viridis", shade=False,
    )
    surface_plot_kws = dict(default_surface_plot_kws, **surface_plot_kws)

    hmap = ax.plot_surface(excitation, emission, fl, **surface_plot_kws)

    zlim_min = kwargs.get("zlim_min", np.nanmin(fl))
    zlim_max = kwargs.get("zlim_max", np.nanmax(fl))
    z_offset = zlim_max * -2

    default_contour_plot_kws = dict(
        zdir="z", offset=z_offset, vmin=zlim_min, vmax=zlim_max,
    )
    contour_plot_kws = dict(default_contour_plot_kws, **contour_plot_kws)

    if plot_type == "surface_contour":
        ax.contourf(excitation, emission, fl, **contour_plot_kws)
        zlim_min += z_offset

    ax.set_zlim(zlim_min, zlim_max)
    ax.zaxis.set_ticks_position("none")
    ax.set_zticks([])

    elev = kwargs.get("elev", 20)
    azim = kwargs.get("azim", 135)
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.set_edgecolor("grey")
    ax.yaxis.pane.set_edgecolor("grey")
    ax.zaxis.pane.set_edgecolor("grey")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    title = kwargs.get("title", "Excitation Emission Matrix")
    title_fontsize = kwargs.get("title_fontsize", 14)
    title_fontweight = kwargs.get("title_fontweight", "bold")
    title_pad = kwargs.get("pad", 0)
    ax.set_title(
        title,
        wrap=True,
        fontsize=title_fontsize,
        fontweight=title_fontweight,
        pad=title_pad,
    )

    wavelength_units = kwargs.get("wavelength_units", "nm")
    xlabel = kwargs.get(
        "xlabel", "Excitation " + r"$\lambda$, %s" % str(wavelength_units)
    )
    ylabel = kwargs.get(
        "ylabel", "Emission " + r"$\lambda$, %s" % str(wavelength_units)
    )
    axis_label_fontsize = kwargs.get("axis_label_fontsize", 12)
    axis_labelpad = kwargs.get("axis_labelpad", 5)
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize, labelpad=axis_labelpad)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize, labelpad=axis_labelpad)

    tick_params_labelsize = kwargs.get("tick_params_labelsize", 10)
    ax.tick_params(axis="both", which="major", pad=0, labelsize=tick_params_labelsize)

    xaxis_major_maxnlocator = kwargs.get("xaxis_major_maxnlocator", 4)
    yaxis_major_maxnlocator = kwargs.get("yaxis_major_maxnlocator", 4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(xaxis_major_maxnlocator))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(yaxis_major_maxnlocator))

    if include_cbar:
        shrink = cbar_kws.get("shrink", 0.5)
        label_size = cbar_kws.get("size", 12)
        tick_params_labelsize = kwargs.get("labelsize", 11)
        cbar = plt.colorbar(hmap, ax=ax, shrink=shrink)
        cbar.set_label(intensity_units, size=label_size)
        cbar.ax.ticklabel_format(
            style="scientific", scilimits=(-2, 3), useMathText=True
        )
        cbar.ax.tick_params(labelsize=tick_params_labelsize)

    return hmap


def eem_plot(
    eem_df,
    ax=None,
    plot_type="imshow",
    wavelength_units="nm",
    intensity_units="unspecified",
    include_cbar=True,
    aspect="equal",
    fig_kws={},
    plot_kws={},
    cbar_kws={},
    **kwargs
):
    """[summary]

    Args:
        eem_df (pandas.DataFrame): [description]
        ax (matplotlib.axes.Axes, optional): [description]. Defaults to None.
        plot_type (str, optional): [description]. Defaults to "imshow".
        intensity_units (str, optional): [description]. Defaults to "unspecified".
        wavelength_units (str, optional): [description]. Defaults to "nm".
        aspect (str, optional): [description]. Defaults to "equal".
        include_cbar (bool, optional): [description]. Defaults to True.
        fig_kws (dict, optional): [description]. Defaults to {}.
        plot_kws (dict, optional): [description]. Defaults to {}.
        cbar_kws (dict, optional): [description]. Defaults to {}.

    Raises:
        ValueError: [description]

    Returns:
        QuadContourSet, AxesImage, or mpl_toolkits.mplot3d.art3d.Poly3DCollection: [description]
    """

    # Set the default figure kws
    default_fig_kws = dict()
    fig_kws = dict(default_fig_kws, **fig_kws)

    if ax is None:
        projection = None
        if plot_type == "surface_contour":
            projection = "3d"
        fig = plt.figure(**fig_kws)
        ax = plt.gca(projection=projection)

    if plot_type == "contour":
        hmap = _eem_contour(
            eem_df,
            ax,
            intensity_units,
            include_cbar,
            plot_kws=plot_kws,
            cbar_kws=cbar_kws,
            **kwargs
        )

    elif plot_type == "imshow":
        hmap = _eem_imshow(
            eem_df,
            ax,
            intensity_units,
            include_cbar,
            plot_kws=plot_kws,
            cbar_kws=cbar_kws,
            **kwargs
        )

    elif plot_type in ["surface", "surface_contour"]:
        hmap = _eem_surface_contour(
            eem_df, ax, intensity_units, include_cbar, plot_type=plot_type, **kwargs
        )
        return hmap

    else:
        raise ValueError("plot_type must be imshow, contour, or surface_contour")

    tick_params_labelsize = kwargs.get("tick_params_labelsize", 11)
    ax.tick_params(axis="both", which="major", labelsize=tick_params_labelsize)

    title = kwargs.get("title", "Excitation Emission Matrix")
    title_wrap = kwargs.get("title_wrap", True)
    title_fontsize = kwargs.get("title_fontsize", 14)
    title_pad = kwargs.get("title_pad", 20)
    fontweight = kwargs.get("title_fontweight", "bold")
    ax.set_title(
        title,
        wrap=title_wrap,
        fontsize=title_fontsize,
        fontweight=fontweight,
        pad=title_pad,
    )

    xlabel = kwargs.get(
        "xlabel", "Excitation " + r"$\lambda$, %s" % str(wavelength_units)
    )
    ylabel = kwargs.get(
        "ylabel", "Emission " + r"$\lambda$, %s" % str(wavelength_units)
    )
    axis_label_fontsize = kwargs.get("axis_label_fontsize", 12)
    axis_labelpad = kwargs.get("axis_labelpad", 5)
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize, labelpad=axis_labelpad)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize, labelpad=axis_labelpad)
    return hmap


def plot_absorbance(ax=None, plot_kws={}, fig_kws={}, **kwargs):
    return
