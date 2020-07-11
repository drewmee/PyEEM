import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from celluloid import Camera
import seaborn as sns
from tqdm import tqdm
import numpy as np
from matplotlib import rc

rc("font", family="serif")


class Visualizations:
    """[summary]
    """

    def __init__(self):
        self.font = "helvetica"
        self.viz_dir = ""
        self.figsize = (8, 8)
        self.dpi = 100
        self.cmap = "viridis"
        self.title_fontsize = 20
        self.axis_fontsize = 20
        self.tick_param_labelsize = 14
        self.units = "nm"
        self.eem_xlabel = "Excitation " + r"$\lambda$" + " ({0})".format(self.units)
        self.eem_ylabel = "Emission " + r"$\lambda$" + " ({0})".format(self.units)
        self.cbar_label = "Normalized Intensity (Raman Units, R.U.)"
        self.cbar_fontsize = 18

    def calibration_curve(self, cal_df):
        """[summary]

        Args:
            cal_df ([type]): [description]
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for index, row in cal_df.iterrows():
            """
            source_cal_coeffs = c.loc[:,
                        c.columns.str.startswith("cal_func_term")
                        ].iloc[0].values
            cal_func = np.poly1d(source_cal_coeffs)
            """
            display(row)
        return

    def contour_plot(self, eem, title):
        """[summary]

        Args:
            eem ([type]): [description]
            title ([type]): [description]
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        cf = ax.contourf(
            eem.columns.to_numpy(), eem.index.to_numpy(), eem.to_numpy(), cmap=self.cmap
        )
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(self.cbar_label, size=self.cbar_fontsize)
        cbar.ax.tick_params(labelsize=self.tick_param_labelsize)
        plt.title(title, fontsize=self.title_fontsize)
        plt.xlabel(self.eem_xlabel, fontsize=self.axis_fontsize)
        plt.ylabel(self.eem_ylabel, fontsize=self.axis_fontsize)
        plt.tick_params(axis="both", which="major", labelsize=self.tick_param_labelsize)
        plt.show()

    def combined_surface_contour_plot(self, eem, title):
        """[summary]

        Args:
            eem ([type]): [description]
            title ([type]): [description]
        """
        fig = plt.figure(figsize=self.figsize)
        ax = Axes3D(fig)

        X = eem.columns.to_numpy()
        X = np.array(X).astype(float)
        Y = eem.index.to_numpy()
        Y = np.array(Y).astype(float)
        Z = eem.to_numpy()
        X, Y = np.meshgrid(X, Y)

        surf = ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1, cmap=self.cmap, alpha=0.75, shade=False
        )
        contourz = (Z.min() - Z.max()) * 0.4
        z_offset = -5
        ax.contourf(X, Y, Z, zdir="z", offset=z_offset, cmap=self.cmap)
        ax.set_zlim(Z.min() + z_offset, Z.max())
        ax.set_zticks(np.round(np.linspace(Z.min(), Z.max(), num=5), 2))
        ax.zaxis.set_tick_params(pad=10)

        ax.view_init(elev=20, azim=135)
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor("black")
        ax.yaxis.pane.set_edgecolor("black")
        ax.zaxis.pane.set_edgecolor("black")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_locator(MultipleLocator(50))

        cbar = fig.colorbar(surf, ax=ax, shrink=0.8)
        cbar.set_label(self.cbar_label, size=self.cbar_fontsize)
        cbar.ax.tick_params(labelsize=self.tick_param_labelsize)

        plt.title(title, fontsize=self.title_fontsize)
        plt.xlabel(self.eem_xlabel, fontsize=self.axis_fontsize, labelpad=10)
        plt.ylabel(self.eem_ylabel, fontsize=self.axis_fontsize, labelpad=10)
        plt.tick_params(axis="both", which="major", labelsize=self.tick_param_labelsize)
        plt.show()

    def single_source_animation(self, source_name, eem_df, plot_type="contour"):
        """[summary]

        Args:
            source_name ([type]): [description]
            eem_df ([type]): [description]
            plot_type (str, optional): [description]. Defaults to "contour".
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.xlabel(self.eem_xlabel, fontsize=self.axis_fontsize)
        plt.ylabel(self.eem_ylabel, fontsize=self.axis_fontsize)
        plt.tick_params(axis="both", which="major", labelsize=self.tick_param_labelsize)

        camera = Camera(fig)

        eem_np = eem_df.to_numpy()
        min_val = eem_np.min()
        max_val = eem_np.max()

        excitation = eem_df.columns.to_numpy()
        emission = eem_df.index.droplevel(
            ["diesel", "cigarette", "wood_smoke", "source"]
        ).to_numpy()
        extent = [excitation[0], excitation[-1], emission[0], emission[-1]]
        for conc in eem_df.index.get_level_values(source_name).unique():
            ss = eem_df.xs(conc, level=source_name, drop_level=False)
            ss.index = ss.index.droplevel(
                ["diesel", "cigarette", "wood_smoke", "source"]
            )

            conc = round(conc, 2)
            title = "Augmented Single Source Spectra: {0}\n".format(source_name)
            title += "Concentration (ug/ml): {0}".format(conc)
            ax.text(
                -0.05, 1.05, title, transform=ax.transAxes, fontsize=self.title_fontsize
            )
            hmap = ax.imshow(
                ss,
                cmap=self.cmap,
                vmin=min_val,
                vmax=max_val,
                origin="lower",
                extent=extent,
            )
            camera.snap()

        cbar = fig.colorbar(hmap, ax=ax)
        cbar.set_label(self.cbar_label, size=self.cbar_fontsize)
        cbar.ax.tick_params(labelsize=self.tick_param_labelsize)

        animation = camera.animate()
        animation.save(source_name + ".mp4", dpi=self.dpi)

    def mix_animation(self, sources, mix_df):
        """[summary]

        Args:
            sources ([type]): [description]
            mix_df ([type]): [description]
        """
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
        eem_ax = plt.subplot(gs[0])
        bar_ax = plt.subplot(gs[1])

        camera = Camera(fig)

        eem_np = mix_df.to_numpy()
        min_val = eem_np.min()
        max_val = eem_np.max()

        excitation = mix_df.columns.to_numpy()
        emission = mix_df.index.droplevel(
            ["diesel", "cigarette", "wood_smoke"]
        ).to_numpy()
        extent = [excitation[0], excitation[-1], emission[0], emission[-1]]

        for conc_set, g in tqdm(mix_df.groupby(level=list(sources))):
            title = dict(zip(sources, conc_set))
            g.index = g.index.droplevel(["diesel", "cigarette", "wood_smoke"])

            eem_ax.text(0.2, 1, title, transform=eem_ax.transAxes)
            hmap = eem_ax.imshow(
                g,
                cmap=self.cmap,
                vmin=min_val,
                vmax=max_val,
                origin="lower",
                extent=extent,
            )

            data = [[30, 25, 50, 20], [40, 23, 51, 17], [35, 22, 45, 19]]
            X = np.arange(4)
            bar_ax.bar(X + 0.00, data[0], color="b", width=0.25)
            bar_ax.bar(X + 0.25, data[1], color="g", width=0.25)
            bar_ax.bar(X + 0.50, data[2], color="r", width=0.25)

            camera.snap()

        cbar = fig.colorbar(hmap, ax=eem_ax)
        cbar.set_label(self.cbar_label, size=self.cbar_fontsize)
        cbar.eem_ax.tick_params(labelsize=self.tick_param_labelsize)

        animation = camera.animate()
        animation.save("mix.mp4", dpi=self.dpi)

    def surface_plot(self):
        """[summary]
        """
        return

    def contour_animation(self):
        """[summary]
        """
        return
