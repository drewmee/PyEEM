import os

import pandas as pd
import pytest

import pyeem


class TestDatasets:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir_fixture, demo_datasets):
        self.data_dir = tmp_dir_fixture
        self.demo_datasets = demo_datasets

    def testDownloadDemo(self):
        bucket_name = "pyeem-demo-datasets"
        bucket_dirs = ["rutherford", "dreem"]
        for demo in bucket_dirs:
            # Get list of files in the bucket.
            file_list = pyeem.datasets._get_bucket_file_list(bucket_name, demo)
            # Download the files from the bucket.
            pyeem.datasets.download_demo(self.data_dir, demo_name=demo)
            for f in file_list:
                # Make sure each file that was in the bucket was
                # Downloaded.
                assert os.path.isfile(os.path.join(self.data_dir, f))

    def testCreateMetadataTemplate(self):
        meta_df_columns = [
            "datetime_utc",
            "sample_set",
            "scan_type",
            "filename",
            "description",
            "comments",
            "collected_by",
            "dilution",
        ]
        cal_columns = ["calibration_sample", "prototypical_sample", "test_sample"]

        template_filename = "template.csv"
        template_filepath = os.path.join(self.data_dir, template_filename)
        meta_df = pyeem.datasets.create_metadata_template(template_filepath)
        assert os.path.isfile(template_filepath)
        assert isinstance(meta_df, pd.DataFrame)
        assert sorted(meta_df.columns.to_list()) == sorted(meta_df_columns)

        template_filename = "template_calibration.csv"
        template_filepath = os.path.join(self.data_dir, template_filename)
        cal_sources = ["source_a", "source_b", "source_c"]
        required_columns = meta_df_columns + cal_columns + cal_sources
        meta_df = pyeem.datasets.create_metadata_template(
            template_filepath, calibration_sources=cal_sources
        )
        assert os.path.isfile(template_filepath)
        assert isinstance(meta_df, pd.DataFrame)
        assert sorted(meta_df.columns.to_list()) == sorted(required_columns)

    def checkWarnings(self, warning_record, num_warnings, required_warnings):
        assert len(warning_record) == num_warnings
        warning_record_df = pd.DataFrame(warning_record, columns=["warning"])
        warning_record_df["message"] = (
            warning_record_df["warning"].apply(lambda x: x.message).astype(str)
        )

        for warn, freq in required_warnings:
            assert (
                warning_record_df[
                    warning_record_df["message"].str.contains(warn)
                ].shape[0]
                == freq
            )

    def checkMetadataSummary(self, summary_df, required_columns):
        assert isinstance(summary_df, pd.DataFrame)
        assert summary_df.shape[0] == 1
        assert sorted(summary_df.columns.to_list()) == sorted(required_columns.keys())
        for key, value in required_columns.items():
            if "UTC" in key:
                continue
            assert summary_df[key].item() == value

    def checkDatasetAttributes(self, dataset, data_dir, calibration_sources):
        assert dataset.data_dir == os.path.abspath(data_dir)

        scan_sets_subdir = "raw_sample_sets"
        assert dataset.scan_sets_subdir == os.path.join(
            os.path.abspath(data_dir), scan_sets_subdir
        )

        assert dataset.calibration_sources == calibration_sources
        assert dataset.progress_bar == False
        assert isinstance(dataset.instruments_df, pd.DataFrame)
        # TODO - check some more things about the instruments df

        metadata_filename = "metadata.csv"
        assert dataset.metadata_path == os.path.join(
            os.path.abspath(data_dir), metadata_filename
        )
        assert isinstance(dataset.meta_df, pd.DataFrame)
        # TODO - check some more things about the meta df

        hdf_filename = "root.hdf5"
        assert dataset.hdf_path == os.path.join(os.path.abspath(data_dir), hdf_filename)
        assert isinstance(dataset.hdf, pd.HDFStore)
        # Makes sure Dataset.load_sample_sets was successful
        assert dataset.meta_df["hdf_path"].apply(lambda x: x in dataset.hdf).all()

    def testRutherfordDemoDataset(self):
        demo_data_dir = pyeem.datasets.download_demo(
            self.data_dir, demo_name="rutherford"
        )

        calibration_sources = {
            "cigarette": "ug/ml",
            "diesel": "ug/ml",
            "wood_smoke": "ug/ml",
        }

        raman_instrument = None
        absorbance_instrument = "aqualog"
        eem_instrument = "aqualog"

        with pytest.warns(None) as warning_record:
            dataset = pyeem.datasets.Dataset(
                data_dir=demo_data_dir,
                raman_instrument=raman_instrument,
                absorbance_instrument=absorbance_instrument,
                eem_instrument=eem_instrument,
                calibration_sources=calibration_sources,
            )

        assert isinstance(dataset, pyeem.datasets.Dataset)
        self.checkDatasetAttributes(
            dataset, demo_data_dir, calibration_sources=calibration_sources
        )

        num_warning = 18
        required_warnings = [
            ("No Water Raman scan found in sample set", 14),
            ("No Sample EEM scans were found in sample set", 1),
            ("More than one Blank EEM found in sample set", 3),
        ]
        self.checkWarnings(warning_record, num_warning, required_warnings)

        # Check metadata_summary_info
        summary_df = dataset.metadata_summary_info()
        required_columns = {
            "Start datetime (UTC)": None,
            "End datetime (UTC)": None,
            "Number of sample sets": 14,
            "Number of blank EEMs": 20,
            "Number of sample EEMs": 107,
            "Number of water raman scans": 0,
            "Number of absorbance scans": 107,
        }
        self.checkMetadataSummary(summary_df, required_columns)

    def testDreemDemoDataset(self):
        demo_data_dir = pyeem.datasets.download_demo(self.data_dir, demo_name="drEEM")

        raman_instrument = "fluorolog"
        absorbance_instrument = "cary_4e"
        eem_instrument = "fluorolog"

        with pytest.warns(None) as warning_record:
            dataset = pyeem.datasets.Dataset(
                data_dir=demo_data_dir,
                raman_instrument=raman_instrument,
                absorbance_instrument=absorbance_instrument,
                eem_instrument=eem_instrument,
            )

        assert isinstance(dataset, pyeem.datasets.Dataset)
        self.checkDatasetAttributes(dataset, demo_data_dir, calibration_sources=None)

        num_warnings = 2
        required_warnings = [
            ("No corresponding absorbance scan for sample", 1),
            ("No Sample EEM scans were found in sample set", 1),
        ]
        self.checkWarnings(warning_record, num_warnings, required_warnings)

        # Check metadata_summary_info
        summary_df = dataset.metadata_summary_info()
        required_columns = {
            "Start datetime (UTC)": None,
            "End datetime (UTC)": None,
            "Number of sample sets": 16,
            "Number of blank EEMs": 16,
            "Number of sample EEMs": 73,
            "Number of water raman scans": 16,
            "Number of absorbance scans": 77,
        }
        self.checkMetadataSummary(summary_df, required_columns)
