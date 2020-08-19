import operator
import os
import traceback
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from tables.exceptions import NaturalNameWarning
from tqdm import tqdm

from .._utils import std_out_err_redirect_tqdm
from ..instruments import _get_dataset_instruments_df, _supported

# TODO - debug the following warnings...
warnings.simplefilter(action="ignore", category=NaturalNameWarning)
warnings.simplefilter(action="ignore", category=PerformanceWarning)


def custom_format_warning(msg, *args, **kwargs):
    # ignore everything except the message
    return "WARNING: " + str(msg) + "\n"


warnings.formatwarning = custom_format_warning


def _metadata_template(calibration_sources=None):
    # There are the columns that are required for the metadata file.
    meta_df_cols = [
        "datetime_utc",
        "sample_set",
        "scan_type",
        "filename",
        "description",
        "comments",
        "collected_by",
        "dilution",
    ]
    cal_cols = ["calibration_sample", "prototypical_sample", "test_sample"]
    if calibration_sources:
        meta_df_cols = meta_df_cols + calibration_sources + cal_cols

    return pd.DataFrame(columns=meta_df_cols)


def create_metadata_template(filepath, calibration_sources=None):
    """[summary]

    Args:
        filepath (str): [description]
        calibration_sources (dict of {str : str}, optional): [description]. Defaults to None.

    Returns:
        DataFrame: [description]
    """
    abs_filepath = os.path.abspath(filepath)
    meta_df = _metadata_template(calibration_sources=calibration_sources)
    meta_df.to_csv(abs_filepath, index=False)
    return meta_df


class Dataset:
    """[summary]
    """

    def __init__(
        self,
        data_dir,
        raman_instrument,
        absorbance_instrument,
        eem_instrument,
        scan_sets_subdir="raw_sample_sets",
        metadata_filename="metadata.csv",
        hdf_filename="root.hdf5",
        calibration_sources=None,
        progress_bar=False,
        **kwargs,
    ):
        """[summary]

        Args:
            data_dir (str): [description]
            raman_instrument (str, optional): [description]. Defaults to None.
            absorbance_instrument (str, optional): [description]. Defaults to None.
            eem_instrument (str, optional): [description]. Defaults to None.
            scan_sets_subdir (str, optional): [description]. Defaults to "raw_sample_sets".
            metadata_filename (str, optional): [description]. Defaults to "metadata.csv".
            hdf_filename (str, optional): [description]. Defaults to "root.hdf5".
            calibration_sources (dict of {str : str}, optional): [description]. Defaults to None.
            progress_bar (bool, optional): [description]. Defaults to False.
        """
        self.data_dir = os.path.abspath(data_dir)
        self.scan_sets_subdir = os.path.join(self.data_dir, scan_sets_subdir)
        self.metadata_path = os.path.join(self.data_dir, metadata_filename)
        self.hdf_path = os.path.join(self.data_dir, hdf_filename)
        self.hdf = pd.HDFStore(
            self.hdf_path,
            mode=kwargs.get("mode", "a"),
            complevel=kwargs.get("complevel", 0),
            complib=kwargs.get("complib", None),
            fletcher32=kwargs.get("fletcher32", False),
        )
        self.calibration_sources = calibration_sources
        self.progress_bar = progress_bar
        self.meta_df = self.load_metadata()
        self.instruments_df = _get_dataset_instruments_df(
            raman_instrument, absorbance_instrument, eem_instrument
        )
        self.load_sample_sets()

    # In the future, consider using something like the library  formencode
    # to validate the inputs to this class. There has to be a cleaner way
    # to do this.
    data_dir = property(operator.attrgetter("_data_dir"))
    scan_sets_subdir = property(operator.attrgetter("_scan_sets_subdir"))
    metadata_path = property(operator.attrgetter("_metadata_path"))

    @data_dir.setter
    def data_dir(self, d):
        # Ensure the data directory exists.
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            raise FileNotFoundError(d)

        self._data_dir = d

    @scan_sets_subdir.setter
    def scan_sets_subdir(self, s):
        # Ensure the scan sets subdirectory exists.
        if not os.path.isdir(s):
            raise FileNotFoundError(s)

        self._scan_sets_subdir = s

    @metadata_path.setter
    def metadata_path(self, m):
        # Ensure the metadata file exists.
        if not os.path.exists(m):
            raise FileNotFoundError(m)

        self._metadata_path = m

    def _calibration_metadata(self, meta_df):
        cal_source_names = list(self.calibration_sources)
        # Ensure the metadata file contains all of the source columns
        if not set(cal_source_names).issubset(meta_df.columns):
            raise Exception("Not all calibration source columns exist in metadata.")

        # Ensure the metadata file contains prototypical and validation columns
        if not set(
            ("calibration_sample", "prototypical_sample", "test_sample")
        ).issubset(meta_df.columns):
            raise Exception(
                "calibration_sample/prototypical_sample/test_sample/ "
                "columns do not exist in metadata."
            )

        # Set NaN values to 0 for the each of the sources columns
        meta_df[cal_source_names] = meta_df[cal_source_names].fillna(0)

        cal_sample_types = ["calibration_sample", "prototypical_sample", "test_sample"]
        # Convert columns to lower case
        meta_df[cal_sample_types].applymap(lambda s: s.lower() if type(s) == str else s)

        yes_list = ["y", "yes", "ye"]
        # Convert calibration sample type columns to boolean
        meta_df[cal_sample_types] = meta_df[cal_sample_types].isin(yes_list)

        """
        # TODO - Is this correct?
        # Create prototypical_source column
        cols = meta_df[cal_source_names].columns
        ps = meta_df[cols].astype(bool)
        meta_df["prototypical_source"] = ps.apply(
            lambda x: "" if x.sum() != 1 else cols[x.values].item(), axis=1,
        )

        # TODO - Is this really needed?
        # Also, it seems to be broken at the moment...
        # Create test_sources column
        ts = meta_df[cols].astype(bool)
        meta_df["test_sources"] = ts.apply(
            lambda x: [] if x.sum() <= 1 else cols[x.values].to_list(), axis=1,
        )
        """
        return meta_df

    def _qc_metadata(self, meta_df, meta_df_cols):
        # Ensure the metadata file contains all of the required columns
        if not set(meta_df_cols).issubset(meta_df.columns):
            raise Exception("Not all required columns exist in metadata.")

        # Ensure datetime column is in the correct format
        try:
            pd.to_datetime(
                meta_df["datetime_utc"],
                format="%YYYY-%mm-%dd %HH:%MM:%SS",
                errors="raise",
            )
        except ValueError:
            warnings.warn(
                (
                    "Incorrect datetime format in the datetime_utc column of metadata.csv, "
                    "requires %YYYY-%mm-%dd %HH:%MM:%SS"
                )
            )

        # Ensure datetime values are all unique
        if not meta_df["datetime_utc"].is_unique:
            warnings.warn(
                "Non-unique datetime values present in datetime_utc column of metadata.csv."
            )

        """
        # Ensure no values are NULL in all columns except for the
        # description and comments columns.
        if meta_df[meta_df.columns.difference(["description", "comments"])
                    ].isnull().values.any():
            # raise warning
            raise Exception("NULL values found in columns besides "
                            "description and comments.")
        """

    def load_metadata(self):
        """[summary]

        Returns:
            DataFrame: [description]
        """
        template = _metadata_template()
        meta_df_cols = list(template.columns)

        # Load the metadata csv into a dataframe
        meta_df = pd.read_csv(self.metadata_path, parse_dates=["datetime_utc"])

        meta_df["filepath"] = meta_df.apply(
            lambda row: os.path.join(
                *[self.scan_sets_subdir, str(row["sample_set"]), row["filename"],]
            ),
            axis=1,
        )
        meta_df["name"] = meta_df["filename"].str.rsplit(".", n=1, expand=True)[0]

        self._qc_metadata(meta_df, meta_df_cols)

        # Set NaN values to empty strings for the columns:
        # "description", "comments", "collected_by"
        nan_str_cols = ["description", "comments", "collected_by"]
        meta_df[nan_str_cols] = meta_df[nan_str_cols].fillna("")

        # set NaN values to 1.0 for the column dilution
        meta_df["dilution"] = meta_df["dilution"].fillna(1)

        if self.calibration_sources:
            meta_df = self._calibration_metadata(meta_df)

        # Add multi-index with sample_set and scan_type
        meta_df.set_index(["sample_set", "scan_type"], inplace=True)
        meta_df.to_hdf(self.hdf, key=os.path.join("metadata"))
        return meta_df

    def metadata_summary_info(self):
        """[summary]

        Returns:
            DataFrame: [description]
        """
        # self.summary_info["metadata"] = meta_summary_df
        # self.metadata_summary_info = {}

        date_range = (
            self.meta_df["datetime_utc"].min(),
            self.meta_df["datetime_utc"].max(),
        )
        num_sample_sets = self.meta_df.groupby(level="sample_set").ngroups

        summary_dict = {
            "Date Range": str(date_range),
            "Number of Sample Sets": str(num_sample_sets),
        }

        scan_types = {
            "blank_eem": {"Number of blank EEMs": 0},
            "sample_eem": {"Number of sample EEMs": 0},
            "water_raman": {"Number of water raman scans": 0},
            "absorb": {"Number of absorbance scans": 0},
        }

        scan_type_counts = self.meta_df.groupby(level="scan_type").size()
        for st, st_dict in scan_types.items():
            key = list(scan_types[st].keys())[0]
            if st in scan_type_counts:
                scan_types[st][key] = scan_type_counts[st]
            summary_dict[key] = scan_types[st][key]

        summary_df = pd.DataFrame(summary_dict, index=[0])
        return summary_df

    def _process_scan_type(self, scan_type_row):
        try:
            sample_set = str(scan_type_row.name[0])
            scan_type = scan_type_row.name[1]
            name = scan_type_row["name"]
            filepath = scan_type_row["filepath"]

            if not os.path.isfile(filepath):
                raise Exception("The file %s does not exist." % (filepath))

            if scan_type == "absorb":
                instrument = self.instruments_df["absorbance"].item()
                df = instrument.load_absorbance(filepath)

            elif scan_type == "water_raman":
                instrument = self.instruments_df["water_raman"].item()
                df = instrument.load_water_raman(filepath)

            elif "eem" in scan_type:
                instrument = self.instruments_df["eem"].item()
                df = instrument.load_eem(filepath)

            else:
                raise Exception(
                    "Invalid scan_type for %s in sample_set %s" % (name, sample_set)
                )

            hdf_path = os.path.join(*["raw_sample_sets", sample_set, name])
            df.to_hdf(self.hdf, key=hdf_path)

        except Exception as e:
            hdf_path = None
            warnings.warn(str(e))

        return hdf_path

    def _qc_scan_type_group(self, scan_type):
        # Check that filenames are monotonically increasing
        pass

    def _process_scan_type_group(self, scan_type_group):
        scan_type_group["hdf_path"] = scan_type_group.apply(
            self._process_scan_type, axis=1
        )
        return scan_type_group

    def _check_unique_scan_types(self):
        pass

    def _check_blank_raman_scans(self):
        pass

    def _check_sample_absorbance_scans(self):
        pass

    def _qc_sample_set(self, sample_set):
        # This function is UGLY! REFACTOR!
        sample_set_name = str(
            sample_set.index.get_level_values(level="sample_set").unique().item()
        )

        # There should only be one blank scan and one water raman scan for
        # each scan set. If there are more than one, just use the first one.
        mono_types = {"blank_eem": "Blank EEM", "water_raman": "Water Raman"}
        for key, value in mono_types.items():
            # check to see if any exists first
            if key in sample_set.index.get_level_values(level="scan_type"):
                nunique = sample_set.xs(key, level="scan_type")["filename"].nunique()
                if nunique > 1:
                    first_scan = sample_set.xs(key, level="scan_type")[
                        "filename"
                    ].unique()[0]
                    msg = (
                        "More than one %s found in sample set %s, only %s will be used going forward."
                        % (value, sample_set_name, first_scan)
                    )
                    warnings.warn(msg)
            else:
                msg = "No %s scan found in sample set %s." % (value, sample_set_name)
                warnings.warn(msg)

        # Ensure there are N absorbance scan_types for N sample EEM scan_types
        if "sample_eem" in sample_set.index.get_level_values(level="scan_type"):
            sample_eem_rows = sample_set.xs("sample_eem", level="scan_type")

            if "absorb" in sample_set.index.get_level_values(level="scan_type"):
                absorb_rows = sample_set.xs("absorb", level="scan_type")
            else:
                absorb_rows = pd.DataFrame()

            for index, row in sample_eem_rows.iterrows():
                absorbance_filename = "absorb" + row["filename"].split("sample_eem")[-1]
                if absorb_rows.empty:
                    pass
                elif (
                    not absorb_rows["filename"].str.contains(absorbance_filename).any()
                ):
                    pass
                else:
                    continue
                msg = (
                    "No corresponding absorbance scan for sample EEM %s in sample set %s. There should be an absorbance measurement named %s in this sample set."
                    % (row["filename"], sample_set_name, absorbance_filename)
                )
                warnings.warn(msg)
        else:
            msg = "No Sample EEM scans were found in sample set %s." % (sample_set_name)
            warnings.warn(msg)

        return sample_set

    def _process_sample_set(self, sample_set_group):
        sample_set = (
            sample_set_group.index.get_level_values(level="sample_set").unique().item()
        )

        sample_set_group = self._qc_sample_set(sample_set_group)

        # Group by scan types
        return sample_set_group.groupby(level="scan_type", as_index=False).apply(
            self._process_scan_type_group
        )

    def load_sample_sets(self):
        """[summary]
        """
        # Group by sample sets
        if self.progress_bar:
            with std_out_err_redirect_tqdm() as orig_stdout:
                tqdm.pandas(
                    desc="Loading scan sets", file=orig_stdout, dynamic_ncols=True
                )
                self.meta_df = self.meta_df.groupby(
                    level="sample_set", as_index=False
                ).progress_apply(self._process_sample_set)
        else:
            self.meta_df = self.meta_df.groupby(
                level="sample_set", as_index=False
            ).apply(self._process_sample_set)
