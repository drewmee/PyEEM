import os
import operator
import numpy as np
import pandas as pd
import h5py
from ..instruments import _supported, Aqualog, Fluorolog, Cary
import warnings
from tables.exceptions import NaturalNameWarning
from pandas.errors import PerformanceWarning
warnings.simplefilter(action='ignore', category=NaturalNameWarning)
warnings.simplefilter(action='ignore', category=PerformanceWarning)


class Load:
    def __init__(self, data_dir):
        self.raw_subdir = "raw_sample_sets"
        self.data_dir = data_dir
        self.hdf = pd.HDFStore(os.path.join(self.data_dir, 'root.hdf5'))

    data_dir = property(operator.attrgetter('_data_dir'))

    @data_dir.setter
    def data_dir(self, d):
        # Ensure the data directory exists and is not empty.
        if not d:
            raise Exception("data_dir cannot be empty.")
        if not os.path.isdir(d):
            raise FileNotFoundError(d)

        # Ensure the data directory has a subdirectory named
        # raw_sample_sets.
        if not os.path.isdir(os.path.join(d, self.raw_subdir)):
            raise Exception("Data directory has no subdirectory "
                            "named raw_sample_sets")

        self._data_dir = d

    def spectral_correction_files(self):
        return

    def metadata(self, sources=None):
        """[summary]

        Args:
            sources ([type], optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
            Exception: [description]
            Exception: [description]
            warning: [description]
            Exception: [description]
            warning: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]

        Returns:
            [type]: [description]
        """
        fp = os.path.join(*[self.data_dir, "metadata.csv"])
        # There are the columns that are required for the metadata file.
        meta_df_cols = [
            'datetime_utc', 'sample_set', 'scan_type',
            'filename', 'description', 'comments', 'collected_by'
        ]

        # Ensure the metadata file exists where it should.
        if not os.path.isfile(fp):
            raise Exception("Metadata does not exist at {0}".format(fp))

        try:
            # Load the metadata csv into a dataframe
            meta_df = pd.read_csv(fp, parse_dates=["datetime_utc"])
        except:
            raise Exception("Unable to process metadata.csv, check format.")

        # Ensure the metadata file contains all of the required columns
        if not set(meta_df_cols).issubset(meta_df.columns):
            raise Exception("Not all required columns exist in "
                            "metadata.")

        # Ensure datetime column is in the correct format
        # TODO - raise warning instead
        try:
            pd.to_datetime(meta_df['datetime_utc'],
                           format='%YYYY-%mm-%dd %HH:%MM:%SS',
                           errors='raise')
        except ValueError:
            raise Exception("Incorrect datetime format in metadata.csv. "
                            "Requires %YYYY-%mm-%dd %HH:%MM:%SS")

        # Ensure datetime values are all unique
        # TODO - raise warning instead
        if not meta_df["datetime_utc"].is_unique:
            raise Exception("Non-unique datetime values.")

        '''
        # Ensure no values are NULL in all columns except for the
        # description and comments columns.
        if meta_df[meta_df.columns.difference(["description", "comments"])
                   ].isnull().values.any():
            raise Exception("NULL values found in columns besides "
                            "description and comments.")
        '''

        # TODO - move to qc_sample_set()
        # Ensure scan_type only contains the following values:
        scan_types = ["water_raman",
                      "blank_eem", "absorb", "sample_eem"]
        if not set(meta_df["scan_type"].unique().tolist()).issubset(
                scan_types):
            raise Exception(
                "Unexpected scan_type types included in scan_type column")

        # TODO - move to qc_sample_set()
        # Group by scan sets (Scan sets are defined by one blank scan_type,
        # at least one sample scan_type and no more than N absorbance
        # scan_types for N sample scan_types.)
        for name, grp in meta_df.groupby("sample_set"):
            # Ensure there is only one blank scan in the scan set
            nunique_blanks = grp[
                grp["scan_type"] == "blank_eem"
            ]["scan_type"].nunique()
            if nunique_blanks != 1:
                raise Exception("Only one blank allowed per scan set.")

            # Ensure there is at least one sample scan_type
            if grp[grp["scan_type"] == "sample_eem"]["scan_type"].nunique() < 1:
                pass
                # raise Exception("Sample set requires at least 1 sample.")

            '''
            # Ensure there are no more than N absorbance scan_types for N
            # sample scan_types
            subgrp = grp["scan_type"].value_counts()
            print(subgrp)
            if subgrp["abs"] > subgrp["sample"] + subgrp["blank"]:
                raise Exception("More absorbance scans than expected.")
            '''

        # TODO - move to qc_sample_set()
        # Ensure filenames are properly named and actually exist in their
        # proper scan set subdirectory.
        meta_df["filepath"] = meta_df.apply(lambda row: os.path.join(
            *[self.data_dir, self.raw_subdir, str(row["sample_set"]), row["filename"]]),
            axis=1)

        for index, item in meta_df["filepath"].iteritems():
            if not os.path.isfile(item):
                print(item)
                raise Exception("File does not exist")

        # If there are no calibration sources,
        # return meta_df as it stands
        if sources == None:
            # TODO - this section of code is repeated twice -- refactor
            # Add multi-index with sample_set and scan_type
            meta_df.set_index(['sample_set', 'scan_type'], inplace=True)
            meta_df.to_hdf(self.hdf, key=os.path.join('metadata'))
            self.meta_df = meta_df
            return meta_df

        # TODO - break this out into another function for dealing w/
        # calibration stuff. Concat resulting dataframe w/ meta_df
        # and have one return in this function.

        # Ensure the metadata file contains all of the source columns
        if not set(sources).issubset(meta_df.columns):
            raise Exception("Not all source columns exist in metadata.")

        # Ensure the metadata file contains prototypical and
        # validation columns
        if not set(
            ("calibration_sample", "prototypical_sample", "test_sample")
        ).issubset(meta_df.columns):
            raise Exception("calibration_sample/prototypical_sample/"
                            "test_sample/"
                            "columns do not exist in metadata.")

        # Create prototypical_source column
        cols = meta_df[sources].columns
        ps = meta_df[
            meta_df["prototypical_sample"] == "y"
        ][cols].apply(lambda x: x > 0)
        meta_df['prototypical_source'] = ps.apply(
            lambda x: np.nan if len(cols[x.values].values) != 1
            else cols[x.values].item(), axis=1)

        # Create validation_sources column
        vs = meta_df[
            meta_df["test_sample"] == "y"
        ][cols].apply(lambda x: x > 0)
        meta_df['test_sources'] = vs.apply(
            lambda x: np.nan if len(cols[x.values].values) <= 1
            else ','.join(cols[x.values]), axis=1)

        # Add multi-index with sample_set and scan_type
        meta_df.set_index(['sample_set', 'scan_type'], inplace=True)
        meta_df.to_hdf(self.hdf, key=os.path.join('metadata'))
        self.meta_df = meta_df

        return meta_df

    def calibration(self, sources):
        cal_df = None
        return cal_df
        fp = os.path.join(*[self.data_dir,
                            self.raw_subdir,
                            "calibration.csv"])
        # There are the columns that are required for the calibration file.
        cal_df_cols = [
            'source', 'proto_conc', 'cal_func_term1'
        ]

        # Ensure the calibration file exists where it should.
        if not os.path.isfile(fp):
            raise Exception(
                "Calibration file does not exist at {0}".format(fp))

        try:
            # Load the calibration csv into a dataframe.
            cal_df = pd.read_csv(fp)
        except:
            raise Exception("Unable to process calibration.csv, check format.")

        # Ensure the calibration file contains all of the required columns.
        if not set(cal_df_cols).issubset(cal_df.columns):
            raise Exception("Not all required columns exist in "
                            "calibration.csv.")

        return cal_df

    # Fluorescence EEM data loaders
    def _load_aqualog_eem(self, data_filename):
        eem = pd.read_csv(data_filename, sep='\t', index_col=0)
        eem.columns = eem.columns.astype(float)
        eem.index.name = "emission_wavelength"
        eem = eem.sort_index(axis=0)
        eem = eem.sort_index(axis=1)
        return eem

    def _load_fluorolog_eem(self, data_filename):
        if data_filename != "data/drEEM_modified/raw_sample_sets/1/sample_eem1.csv":
            return
        try:
            print("\n")
            print(data_filename)
            names = pd.read_csv(data_filename, header=None, nrows=1)
            names = names.iloc[0].values
            names = names[~np.isnan(names)].astype(int).tolist()
            print(names)

            # Here's what we'll do:
            # Drop nan from numpy array -- 0th and sometimes last index
            # Specify index_col=0, skiprows=1, names=names
            eem = pd.read_csv(data_filename, sep=',',
                              skiprows=1, usecols=names)
            #eem.set_index(names[0], inplace=True)
            #eem.columns = eem.columns.astype(int)
            #
            #eem = eem.sort_index(axis=0)
            #eem = eem.sort_index(axis=1)
            if data_filename == "data/drEEM_modified/raw_sample_sets/1/sample_eem1.csv":
                print(data_filename)
                print(eem.columns)
                display(eem)
            '''
            eem = pd.read_csv(data_filename, sep=',', index_col=0)
            eem.columns = eem.columns.astype(int)
            eem.index.name = "emission_wavelength"
            eem = eem.sort_index(axis=0)
            eem = eem.sort_index(axis=1)
            '''
        except Exception as e:
            print("UMMMMM", data_filename, e)
            #print(eem.index.dtype, eem.columns.dtype)
        return eem

    def _load_cary_eem(self, data_filename):
        raise NotImplementedError()

    # Absorbance data loaders
    def _load_aqualog_abs(self, data_filename):
        absorb = pd.read_csv(data_filename, sep='\t',
                             index_col=0, header=0, skiprows=[1, 2])
        absorb.index.name = "wavelength"
        absorb = absorb.sort_index()
        absorb = absorb[['Abs']]
        absorb.rename(columns={'Abs': 'absorbance'}, inplace=True)
        return absorb

    def _load_cary_abs(self, data_filename):
        absorb = pd.read_csv(data_filename, sep=',',
                             header=None, index_col=0,
                             names=["wavelength", "absorbance"])
        absorb = absorb.sort_index()
        return absorb

    # Raman data loaders
    def _load_fluorolog_raman(self, data_filename):
        water_raman = pd.read_csv(data_filename, sep=',',
                                  index_col=0, skiprows=1,
                                  names=["emission_wavelength",
                                         "intensity"])
        water_raman = water_raman.sort_index()
        return water_raman

    def _load_metadata_raman(self):
        raise NotImplementedError()

    def _load_blank_raman(self, data_filename):
        raise NotImplementedError()

    def _process_scan_type(self, scan_type_row):
        sample_set = scan_type_row.name[0]
        scan_type = scan_type_row.name[1]
        filename = scan_type_row["filename"]
        filepath = scan_type_row["filepath"]

        try:
            if scan_type == "absorb":
                df = self.absorbance_instrument.load_absorbance(filepath)

            elif scan_type == "water_raman":
                df = self.raman_instrument.load_water_raman(filepath)

            elif "eem" in scan_type:
                df = self.eem_instrument.load_eem(filepath)

            attrs = {}
            # Add attributes from row in scan set dataframe
            for key, value in scan_type_row.to_dict().items():
                attrs[key] = value
            df.attrs = attrs

            df.to_hdf(
                self.hdf,
                key=os.path.join(*[
                    "raw_sample_sets",
                    str(sample_set),
                    filename
                ]))

        except Exception as e:
            pass
            '''
            print(os.path.join(*[
                "raw_sample_sets",
                str(sample_set),
                filename
            ]))
            print(e)
            warnings.warn("Unable to load {0} with exception: {1}".format(
                filename, e))
            '''

    def _process_scan_type_group(self, scan_type_group):
        scan_type_group.apply(self._process_scan_type, axis=1)
        return

    def _qc_sample_set(self, sample_set):
        scan_types = [
            "water_raman", "blank_eem", "sample_eem", "absorb"
        ]
        return 1

    def _process_sample_set(self, sample_set_group):
        sample_set = sample_set_group.index.get_level_values('sample_set')
        sample_set = sample_set.unique().item()

        '''
        qc_error = self.qc_sample_set(sample_set)
        if qc_error:
            print(qc_error)
            warnings.warn("""\
                Issues with structure of sample set {0}.\
                Omitting sample set from dataset.\
                Exception: {1}""".format(set_name, qc_error))
            continue
        '''
        sample_set_group.groupby(level="scan_type").apply(
            self._process_scan_type_group)

        return

    def sample_sets(self, raman_instrument, absorbance_instrument,
                    eem_instrument):
        """[summary]

        Arguments:
            file_type {[type]} -- [description]
            metadata {[type]} -- [description]

        Raises:
            Exception: [description]

        Returns:
            [type] -- [description]
        """

        if eem_instrument not in _supported['name'].values:
            raise Exception("EEM collected by unsupported instrument.")
        if absorbance_instrument not in _supported['name'].values:
            raise Exception("Absorbance collected by unsupported instrument.")
        if (raman_instrument not in _supported['name'].values) and \
                (raman_instrument is not None):
            raise Exception("Raman collected by unsupported instrument.")

        if raman_instrument is not None:
            self.raman_instrument = _supported[
                _supported['name'] == raman_instrument
            ]['object'].unique().item()
        
        self.absorbance_instrument = _supported[
            _supported['name'] == absorbance_instrument
        ]['object'].unique().item()
        
        self.eem_instrument = _supported[
            _supported['name'] == eem_instrument
        ]['object'].unique().item()

        self.meta_df.groupby(level="sample_set").apply(
            self._process_sample_set)

        return self.hdf


def load_dreem():
    module_path = os.path.dirname(__file__)
    base_dir = os.path.join(module_path, 'data')
    demo_dir = os.path.join(base_dir, 'drEEM')

    load = Load(data_dir=demo_dir)
    meta_df = load.metadata()
    hdf = load.sample_sets(raman_instrument="fluorolog",
                        eem_instrument="fluorolog",
                        absorbance_instrument="cary")

    return load


def load_rutherford():
    module_path = os.path.dirname(__file__)
    base_dir = os.path.join(module_path, 'data')
    demo_dir = os.path.join(base_dir, 'rutherford')

    load = Load(data_dir=demo_dir)
    calibration_sources = ['cigarette', 'diesel', 'wood_smoke']
    meta_df = load.metadata(sources=calibration_sources)
    cal_df = load.calibration(sources=calibration_sources)
    hdf = load.sample_sets(raman_instrument=None,
                        absorbance_instrument="aqualog",
                        eem_instrument="aqualog")
    
    return load
