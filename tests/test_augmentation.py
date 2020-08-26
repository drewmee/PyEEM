import pandas as pd
import pytest

import pyeem


class TestAugmentation:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir_fixture, demo_preprocessed_dataset, demo_calibration):
        self.dataset, self.routine_results_df = demo_preprocessed_dataset
        self.cal_df = demo_calibration

    def testCreatePrototypicalSpectra(self):
        proto_results_df = pyeem.augmentation.create_prototypical_spectra(
            self.dataset, self.cal_df
        )
        assert isinstance(proto_results_df, pd.DataFrame)

    def testCreateSingleSourceSpectra(self):
        ss_results_df = pyeem.augmentation.create_single_source_spectra(
            self.dataset, self.cal_df, conc_range=(0, 5), num_spectra=10
        )
        assert isinstance(ss_results_df, pd.DataFrame)

    def testCreateMixtures(self):
        mix_results_df = pyeem.augmentation.create_mixtures(
            self.dataset, self.cal_df, conc_range=(0.01, 6.3), num_steps=5
        )
        assert isinstance(mix_results_df, pd.DataFrame)
