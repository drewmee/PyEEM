import pandas as pd


class Fluorolog:
    """The Horiba Fluorolog Steady State Spectrofluorometer."""

    manufacturer = "Horiba"
    """Name of Manufacturer."""

    name = "fluorolog"
    """Name of Instrument."""

    supported_models = ["SPEX Fluorolog-3"]
    """List of supported models."""

    def __init__(self, model, sn=None):
        """
        Args:
            model (str): The model name of the instrument.
            sn (str or int, optional): The serial number of the instrument.
                Defaults to None.
        """
        self.model = model
        self.sn = sn

    @staticmethod
    def load_eem(filepath):
        """Loads an Excitation Emission Matrix which is generated by the instrument.

        Args:
            filepath (str): The filepath of the data file.

        Returns:
            pandas.DataFrame: An Excitation Emission Matrix.
        """
        df = pd.read_csv(filepath, index_col=0, nrows=0)
        df.drop(
            df.columns[df.columns.str.contains("unnamed", case=False)],
            axis=1,
            inplace=True,
        )
        col_names = df.columns.to_list()
        col_names.insert(0, "Unnamed: 0")

        eem_df = pd.read_csv(filepath, sep=",", index_col=[0], usecols=col_names)
        eem_df.dropna(how="all", axis=1, inplace=True)
        eem_df.dropna(how="all", axis=0, inplace=True)
        eem_df.columns = eem_df.columns.astype(float)
        eem_df = eem_df.sort_index(axis=0)
        eem_df = eem_df.sort_index(axis=1)
        eem_df = eem_df[eem_df.index.notnull()]
        eem_df.index.name = "emission_wavelength"
        return eem_df

    @staticmethod
    def load_absorbance():
        """Loads an absorbance spectrum which is generated by the instrument.

        Raises:
            NotImplementedError: On the TODO list...
        """
        raise NotImplementedError()

    @staticmethod
    def load_water_raman(filepath):
        """Loads a water Raman spectrum which is generated by the instrument.

        Args:
            filepath (str): The filepath of the data file.

        Returns:
            pandas.DataFrame:[description]
        """
        water_raman_df = pd.read_csv(
            filepath,
            sep=",",
            index_col=0,
            skiprows=1,
            names=["emission_wavelength", "intensity"],
        )
        water_raman_df = water_raman_df.sort_index()
        return water_raman_df

    @staticmethod
    def load_spectral_corrections():
        """TODO - Should load instrument specific spectral corrections which
        will be used in data preprocessing.

        Raises:
            NotImplementedError: On the TODO list...
        """
        raise NotImplementedError()
