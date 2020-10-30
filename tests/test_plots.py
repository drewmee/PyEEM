import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
import pandas as pd
import pyeem
import pytest


class TestPlots:
    @pytest.fixture(autouse=True)
    def setup(
        self,
        demo_datasets,
        demo_preprocessed_dataset,
        demo_calibration,
        demo_augmentation,
        demo_rutherfordnet,
    ):
        self.demo_datasets = demo_datasets
        self.preprocessed_dataset, self.routine_results_df = demo_preprocessed_dataset
        self.cal_df = demo_calibration
        (
            self.proto_results_df,
            self.ss_results_df,
            self.mix_results_df,
        ) = demo_augmentation
        self.rutherfordnet = demo_rutherfordnet

    def testEEMPlot(self):
        dataset = self.demo_datasets["rutherford"]
        hdf_path = dataset.meta_df.loc[1, "sample_eem"]["hdf_path"].iloc[0]
        eem_df = pd.read_hdf(dataset.hdf, key=hdf_path)

        ax = pyeem.plots.eem_plot(eem_df)
        assert isinstance(ax, matplotlib.image.AxesImage)

        fig_kws = {"dpi": 42}
        plot_kws = {"origin": "upper", "extent": [0, 1, 0, 1], "cmap": "plasma"}
        intensity_units = "test_units"
        ax = pyeem.plots.eem_plot(
            eem_df, intensity_units=intensity_units, plot_kws=plot_kws, fig_kws=fig_kws
        )
        assert plot_kws["cmap"] == ax.cmap.name
        assert intensity_units == ax.colorbar.ax.get_ylabel()
        assert ax.colorbar is not None
        assert plot_kws["origin"] == ax.origin
        assert plot_kws["extent"] == ax.get_extent()
        assert fig_kws["dpi"] == plt.gcf().properties()["dpi"]

        include_cbar = False
        ax = pyeem.plots.eem_plot(eem_df, plot_type="imshow", include_cbar=include_cbar)
        assert isinstance(ax, matplotlib.image.AxesImage)
        assert ax.colorbar is None

        ax = pyeem.plots.eem_plot(eem_df, plot_type="contour")
        assert isinstance(ax, matplotlib.contour.QuadContourSet)

        fig_kws = {"dpi": 42}
        plot_kws = {"origin": "upper", "cmap": "plasma", "levels": 5}
        intensity_units = "test_units"
        kwargs = {
            "title": "test_title",
            "title_fontsize": 16,
            "title_fontweight": "light",
            "wavelength_units": "test_wavelength_units",
        }
        ax = pyeem.plots.eem_plot(
            eem_df,
            plot_type="contour",
            intensity_units=intensity_units,
            plot_kws=plot_kws,
            fig_kws=fig_kws,
            **kwargs
        )
        assert fig_kws["dpi"] == plt.gcf().properties()["dpi"]
        assert kwargs["title"] == ax.axes.get_title()
        assert kwargs["title_fontsize"] == ax.axes.title.get_fontsize()
        assert kwargs["title_fontweight"] == ax.axes.title.get_fontweight()
        assert kwargs["wavelength_units"] in ax.axes.get_xlabel()
        assert kwargs["wavelength_units"] in ax.axes.get_ylabel()
        assert plot_kws["origin"] == ax.origin
        assert plot_kws["levels"] + 3 >= len(ax.levels) and plot_kws[
            "levels"
        ] - 3 <= len(ax.levels)
        assert plot_kws["cmap"] == ax.cmap.name
        assert intensity_units == ax.colorbar.ax.get_ylabel()
        assert ax.colorbar is not None

        include_cbar = False
        ax = pyeem.plots.eem_plot(
            eem_df, plot_type="contour", include_cbar=include_cbar,
        )
        assert isinstance(ax, matplotlib.contour.QuadContourSet)
        assert ax.colorbar is None

        ax = pyeem.plots.eem_plot(eem_df, plot_type="surface")
        assert isinstance(ax, mpl_toolkits.mplot3d.art3d.Poly3DCollection)

        fig_kws = {"dpi": 42}
        intensity_units = "test_units"
        kwargs = {
            "title": "test_title",
            "title_fontsize": 16,
            "title_fontweight": "light",
            "surface_plot_kws": {"cmap": "plasma", "alpha": 0.42},
        }
        ax = pyeem.plots.eem_plot(
            eem_df,
            plot_type="surface",
            intensity_units=intensity_units,
            plot_kws=plot_kws,
            fig_kws=fig_kws,
            **kwargs
        )
        assert isinstance(ax, mpl_toolkits.mplot3d.art3d.Poly3DCollection)
        assert kwargs["surface_plot_kws"]["alpha"] == ax.get_alpha()
        assert kwargs["surface_plot_kws"]["cmap"] == ax.get_cmap().name
        assert fig_kws["dpi"] == plt.gcf().properties()["dpi"]
        assert kwargs["title"] == ax.axes.get_title()
        assert kwargs["title_fontsize"] == ax.axes.title.get_fontsize()
        assert kwargs["title_fontweight"] == ax.axes.title.get_fontweight()
        assert intensity_units == ax.colorbar.ax.get_ylabel()
        assert ax.colorbar is not None

        include_cbar = False
        ax = pyeem.plots.eem_plot(
            eem_df, plot_type="surface", include_cbar=include_cbar,
        )
        assert isinstance(ax, mpl_toolkits.mplot3d.art3d.Poly3DCollection)
        assert ax.colorbar is None

        ax = pyeem.plots.eem_plot(eem_df, plot_type="surface_contour")
        assert isinstance(ax, mpl_toolkits.mplot3d.art3d.Poly3DCollection)

        fig_kws = {"dpi": 42}
        intensity_units = "test_units"
        kwargs = {
            "title": "test_title",
            "title_fontsize": 16,
            "title_fontweight": "light",
            "surface_plot_kws": {"cmap": "plasma", "alpha": 0.42},
        }
        ax = pyeem.plots.eem_plot(
            eem_df,
            plot_type="surface_contour",
            intensity_units=intensity_units,
            plot_kws=plot_kws,
            fig_kws=fig_kws,
            **kwargs
        )
        assert isinstance(ax, mpl_toolkits.mplot3d.art3d.Poly3DCollection)
        assert kwargs["surface_plot_kws"]["alpha"] == ax.get_alpha()
        assert kwargs["surface_plot_kws"]["cmap"] == ax.get_cmap().name
        assert fig_kws["dpi"] == plt.gcf().properties()["dpi"]
        assert kwargs["title"] == ax.axes.get_title()
        assert kwargs["title_fontsize"] == ax.axes.title.get_fontsize()
        assert kwargs["title_fontweight"] == ax.axes.title.get_fontweight()
        assert intensity_units == ax.colorbar.ax.get_ylabel()
        assert ax.colorbar is not None

        include_cbar = False
        ax = pyeem.plots.eem_plot(
            eem_df, plot_type="surface_contour", include_cbar=include_cbar,
        )
        assert ax.colorbar is None
        assert isinstance(ax, mpl_toolkits.mplot3d.art3d.Poly3DCollection)

    def testAbsorbancePlot(self):
        return

    def testWaterRamanPeakPlot(self):
        return

    def testWaterRamanPeakAnimation(self):
        return

    def testWaterRamanTimeseries(self):
        return

    def testPreprocessingPlot(self):
        sample_set = 2
        sample_name = "sample_eem1"
        axes = pyeem.plots.preprocessing_routine_plot(
            self.preprocessed_dataset,
            self.routine_results_df,
            sample_set=sample_set,
            sample_name=sample_name,
            plot_type="imshow",
        )
        assert isinstance(axes, list)
        for ax in axes:
            # assert isinstance(ax, matplotlib.axes._subplots.AxesSubplot)
            pass

    def testCalibrationCurvesPlot(self):
        axes = pyeem.plots.calibration_curves_plot(
            self.preprocessed_dataset, self.cal_df
        )
        cal_sources_list = (
            self.cal_df.index.get_level_values("source").unique().tolist()
        )
        assert isinstance(axes, np.ndarray)
        assert len(axes.flatten()) == len(cal_sources_list)
        for ax in axes.flatten():
            assert isinstance(ax, matplotlib.axes.Axes)
            # assert ax.get_title()
            # assert ax.get_xlabel()
            # assert ax.get_ylabel()
            # assert ax.get_legend().texts

    def testPrototypicalSpectraPlot(self):
        proto_results_df = pyeem.augmentation.create_prototypical_spectra(
            self.preprocessed_dataset, self.cal_df
        )

        axes = pyeem.plots.prototypical_spectra_plot(
            self.preprocessed_dataset, self.proto_results_df, plot_type="contour"
        )

        assert isinstance(axes, list)
        for ax in axes:
            # assert isinstance(ax, matplotlib.axes._subplots.AxesSubplot)
            pass

    def testSingleSourceAnimation(self):
        source = "wood_smoke"
        anim = pyeem.plots.single_source_animation(
            self.preprocessed_dataset,
            self.ss_results_df,
            source=source,
            plot_type="imshow",
            fig_kws={"dpi": 120},
            animate_kws={"interval": 100, "blit": True},
            surface_plot_kws={"rstride": 10, "cstride": 10},
        )
        assert isinstance(anim, matplotlib.animation.ArtistAnimation)

    def testMixtureAnimation(self):
        anim = pyeem.plots.mixture_animation(
            self.preprocessed_dataset,
            self.mix_results_df,
            plot_type="contour",
            fig_kws={"dpi": 120},
            animate_kws={"interval": 100, "blit": True},
            surface_plot_kws={"rstride": 10, "cstride": 10},
        )
        assert isinstance(anim, matplotlib.animation.ArtistAnimation)

    def testModelHistoryPlot(self):
        history = self.rutherfordnet.model.history
        axes = pyeem.plots.model_history_plot(history)
        assert isinstance(axes, np.ndarray)

    def testPredictionParityPlots(self):
        (x_train, y_train), (x_test, y_test) = self.rutherfordnet.prepare_data(
            self.preprocessed_dataset,
            self.ss_results_df,
            self.mix_results_df,
            self.routine_results_df,
        )

        train_predictions = self.rutherfordnet.model.predict(x_train)
        test_predictions = self.rutherfordnet.model.predict(x_test)

        train_pred_results_df = self.rutherfordnet.get_prediction_results(
            self.preprocessed_dataset, train_predictions, y_train
        )
        test_pred_results_df = self.rutherfordnet.get_prediction_results(
            self.preprocessed_dataset, test_predictions, y_test
        )

        axes = pyeem.plots.prediction_parity_plot(
            self.preprocessed_dataset,
            test_pred_results_df,
            train_df=train_pred_results_df,
        )

        assert isinstance(axes, np.ndarray)
