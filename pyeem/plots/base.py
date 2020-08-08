import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

units = "nm"
eem_xlabel = "Excitation " + r"$\lambda$" + " ({0})".format(units)
eem_ylabel = "Emission " + r"$\lambda$" + " ({0})".format(units)
cbar_label = "Normalized Intensity (Raman Units, R.U.)"
cbar_fontsize = 18


def contour_plot(eem, ax=None, plot_kws={}, fig_kws={}, **kwargs):
    """[summary]

    Args:
        eem ([type]): [description]
        title ([type]): [description]
    """
    fig, ax = plt.subplots()
    cf = ax.contourf(
        eem.columns.to_numpy(),
        eem.index.get_level_values("emission_wavelength").to_numpy(),
        eem.to_numpy(),
    )
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(cbar_label, size=cbar_fontsize)
    cbar.ax.tick_params()
    """
    source_name = eem.index.get_level_values("source").unique().item()
    proto_conc = eem.index.get_level_values("proto_conc").unique().item()
    title = "Prototypical Spectrum: {0}\n".format(source_name.title())
    title += "Concentration (ug/ml): {0}".format(proto_conc)
    plt.title(title)
    """
    plt.xlabel(eem_xlabel)
    plt.ylabel(eem_ylabel)
    plt.show()


def combined_surface_contour_plot(eem, ax=None, plot_kws={}, fig_kws={}, **kwargs):
    """[summary]

    Args:
        eem ([type]): [description]
        title ([type]): [description]
    """
    fig = plt.figure()
    ax = Axes3D(fig)

    X = eem.columns.to_numpy()
    X = np.array(X).astype(float)
    Y = eem.index.get_level_values("emission_wavelength").to_numpy()
    Y = np.array(Y).astype(float)
    Z = eem.to_numpy()
    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, alpha=0.75, cmap="viridis", shade=False
    )
    contourz = (Z.min() - Z.max()) * 0.4
    z_offset = -5  # this should be proportional to the max Z value, i think?
    ax.contourf(X, Y, Z, zdir="z", offset=z_offset)
    ax.set_zlim(Z.min() + z_offset, Z.max())
    # ax.set_zticks(np.round(np.linspace(Z.min(), Z.max(), num=5), 2))
    ax.zaxis.set_tick_params(pad=10)
    ax.ticklabel_format(
        style="scientific", scilimits=(-3, 4), useMathText=True, axis="z"
    )

    ax.view_init(elev=20, azim=135)
    ax.xaxis.pane.set_edgecolor("grey")
    ax.yaxis.pane.set_edgecolor("grey")
    ax.zaxis.pane.set_edgecolor("grey")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(50))

    cbar = fig.colorbar(surf, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, size=cbar_fontsize)
    cbar.ax.ticklabel_format(style="scientific", scilimits=(-3, 4), useMathText=True)
    cbar.ax.tick_params(labelsize=14)

    source_name = eem.index.get_level_values("source").unique().item()
    proto_conc = eem.index.get_level_values("proto_conc").unique().item()
    title = "Prototypical Spectrum: {0}\n".format(source_name.title())
    title += "Concentration (ug/ml): {0}".format(proto_conc)

    plt.title(title)
    plt.xlabel(eem_xlabel)
    plt.ylabel(eem_ylabel)
    plt.show()


def surface_plot():
    """[summary]
    """
    return


def contour_animation():
    """[summary]
    """
    return


def plot_absorbance(ax=None, plot_kws={}, fig_kws={}, **kwargs):
    return
