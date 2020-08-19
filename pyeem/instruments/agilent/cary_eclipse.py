import pandas as pd


class CaryEclipse:
    """[summary]

    Raises:
        NotImplementedError: [description]
        NotImplementedError: [description]
        NotImplementedError: [description]
    """

    manufacturer = "Agilent"
    """Docstring for class attribute CaryEclipse.manufacturer."""

    name = "cary_eclipse"
    """Docstring for class attribute CaryEclipse.name."""

    supported_models = ["Cary Eclipse"]
    """Docstring for class attribute CaryEclipse.supported_models."""

    def __init__(self, model, sn=None):
        """[summary]

        Args:
            model (str): [description]
            sn (str or int, optional): [description]. Defaults to None.
        """
        self.model = model
        self.sn = sn

    @staticmethod
    def load_eem(filepath):
        """[summary]

        Args:
            filepath (str): [description]

        Returns:
            DataFrame: [description]
        """
        # Read csv, skip the 2nd row which contains repeating
        # columns pairs of "Wavelength (nm)", "Intensity (a.u.)"
        # df = pd.read_csv("sample_3D.csv", skiprows=[1])
        df = pd.read_csv(filepath, skiprows=[1])

        # Drop columns and rows with all NaN values, these sometimes
        # appear at the end for some reason
        df.dropna(how="all", axis=1, inplace=True)
        df.dropna(how="all", axis=0, inplace=True)

        # Get the columns and extract the first one
        # We'll use the first one to remove all rows below
        # the actual EEM. These rows are filled with metadata about
        # each excitation scan. We'll throw them away for the point being.
        columns = df.columns.tolist()
        first_ex_wl = columns[0]
        df[first_ex_wl] = df[first_ex_wl].apply(pd.to_numeric, errors="coerce")
        df.drop(df.loc[pd.isna(df[first_ex_wl]), :].index, inplace=True)

        # Due to the very odd format of the raw EEM, there is an emission wavelength
        # column and an intensity column for each excitation wavelength.
        # Maybe the emission wavelengths can be changed between excitation scans and that's
        # why it is stored in that format. Regardless, what we want is an index containing emission
        # wavelengths and column names which are excitation wavelengths.

        # We zip together excitation column names and the corresponding intensity columns
        zipped = dict(zip(columns[1::2], columns[0::2],))

        # Force all columns to be numeric
        df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")

        # We will only keep the emission wavelengths that correspond with the first excitation scan.
        # These emission wavelengths should correspond with all of the excitation scans, assuming this
        # setting is not changed between excitation scans. Which should be true.
        em_wls = df[first_ex_wl]
        em_wls.name = "emission_wavelength"
        em_wls.index.name = "emission_wavelength"

        # Get the excitation column names and drop them from the dataframe
        # Since these columns only contain emission wavelengths and we
        # already extracted the emission wavelengths for the first excitation scan
        # which we will use for all the scans.
        ex_cols = [col for col in columns if "_EX_" in col]
        df.drop(columns=ex_cols, inplace=True)

        # Rename the emission intensity columns to their corresponding
        # excitation wavelength in order to get Emission (rows) by Excitation (cols)
        df.rename(columns=zipped, inplace=True)
        # Remove the additional substring preprended to each excitation wavelength.
        new_cols = df.columns.str.rsplit("_", 1)
        new_cols = [i[-1] for i in new_cols.to_list()]
        df.columns = new_cols
        df.columns = df.columns.astype(float)

        # Set the index of the dataframe to be the emission wavelengths
        df.set_index([pd.Index(em_wls)], inplace=True)
        return df

    @staticmethod
    def load_absorbance(filepath):
        """[summary]

        Args:
            filepath (str): [description]

        Returns:
            DataFrame: [description]
        """
        absorb_df = pd.read_csv(
            filepath,
            sep=",",
            header=None,
            index_col=0,
            names=["wavelength", "absorbance"],
        )
        absorb_df["absorbance"] = absorb_df["absorbance"].apply(
            pd.to_numeric, errors="coerce"
        )
        absorb_df.fillna(0, inplace=True)
        absorb_df.index = absorb_df.index.astype("float64")
        absorb_df = absorb_df.sort_index()
        return absorb_df

    @staticmethod
    def load_water_raman(filepath):
        """[summary]

        Args:
            filepath (str): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()

    @staticmethod
    def load_spectral_corrections():
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()
