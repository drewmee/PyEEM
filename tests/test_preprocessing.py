import itertools

import pandas as pd
import pyeem
import pytest


class TestPreprocessing:
    @pytest.fixture(autouse=True)
    def setup(self, demo_datasets, demo_preprocessed_dataset):
        self.demo_datasets = demo_datasets
        self.preprocessed_dataset, self.routine_results_df = demo_preprocessed_dataset

    def testCreateRoutine(self):
        steps = [
            "crop",
            "discrete_wavelengths",
            "gaussian_smoothing",
            "blank_subtraction",
            "inner_filter_effect",
            "raman_normalization",
            "scatter_removal",
            "dilution",
        ]
        perm_steps = ["raw", "complete"]
        step_bool = [True, False]
        # Generate routine for all permutations of True/False for steps
        for bools in list(itertools.product(step_bool, repeat=len(steps))):
            args = dict(zip(steps, bools))
            routine_df = pyeem.preprocessing.create_routine(**args)
            assert isinstance(routine_df, pd.DataFrame)

            for key, value in args.items():
                if not value:
                    continue

                assert (
                    routine_df[routine_df["step_name"].str.contains(key)].shape[0] == 1
                )

                for perm_step in perm_steps:
                    assert (
                        routine_df[
                            routine_df["step_name"].str.contains(perm_step)
                        ].shape[0]
                        == 1
                    )

    def testPerformRoutine(self):
        dataset = self.demo_datasets["rutherford"]

        routine_df = pyeem.preprocessing.create_routine(
            crop=True,
            discrete_wavelengths=False,
            gaussian_smoothing=False,
            blank_subtraction=True,
            inner_filter_effect=True,
            raman_normalization=True,
            scatter_removal=True,
            dilution=False,
        )

        crop_dimensions = {
            "emission_bounds": (246, 573),
            "excitation_bounds": (224, float("inf")),
        }
        routine_results_df = pyeem.preprocessing.perform_routine(
            dataset,
            routine_df,
            crop_dims=crop_dimensions,
            raman_source_type="metadata",
            fill="interp",
            progress_bar=True,
        )
        assert isinstance(routine_results_df, pd.DataFrame)
        assert routine_results_df[routine_results_df["step_completed"] == False].empty
        assert routine_results_df[routine_results_df["step_exception"].notna()].empty
        assert routine_results_df["hdf_path"].apply(lambda x: x in dataset.hdf).all()
        metadata_sample_sets = (
            dataset.meta_df.index.get_level_values("sample_set").unique().tolist()
        )
        assert (
            metadata_sample_sets
            == routine_results_df.index.get_level_values("sample_set").unique().tolist()
        )

    def testCalibration(self):
        cal_df = pyeem.preprocessing.calibration(
            self.preprocessed_dataset, self.routine_results_df
        )
        required_columns = [
            "concentration",
            "integrated_intensity",
            "prototypical_sample",
            "hdf_path",
        ]
        required_indices = [
            "source",
            "source_units",
            "intensity_units",
            "measurement_units",
            "slope",
            "intercept",
            "r_squared",
        ]
        assert isinstance(cal_df, pd.DataFrame)
        assert sorted(required_columns) == sorted(cal_df.columns.tolist())
        assert sorted(required_indices) == sorted(cal_df.index.names)
        assert (
            list(self.preprocessed_dataset.calibration_sources.keys())
            == cal_df.index.get_level_values("source").unique().tolist()
        )

        num_cal_samples = self.preprocessed_dataset.meta_df[
            self.preprocessed_dataset.meta_df["calibration_sample"]
        ].shape[0]
        num_proto_samples = self.preprocessed_dataset.meta_df[
            self.preprocessed_dataset.meta_df["prototypical_sample"]
        ].shape[0]
        assert cal_df.shape[0] == num_cal_samples
        assert cal_df[cal_df["prototypical_sample"]].shape[0] == num_proto_samples

        cal_source_units = list(
            set(self.preprocessed_dataset.calibration_sources.values())
        )[0]
        assert (
            cal_source_units
            == cal_df.index.get_level_values("source_units").unique().item()
        )

        cal_summary_df = pyeem.preprocessing.calibration_summary_info(cal_df)
        required_columns = [
            "source",
            "source_units",
            "intensity_units",
            "measurement_units",
            "slope",
            "intercept",
            "r_squared",
            "Number of Samples",
            "Min. Concentration",
            "Max. Concentration",
        ]
        assert isinstance(cal_summary_df, pd.DataFrame)
        assert sorted(required_columns) == sorted(cal_summary_df.columns.tolist())
        assert cal_summary_df["source"].values.tolist() == list(
            self.preprocessed_dataset.calibration_sources.keys()
        )
