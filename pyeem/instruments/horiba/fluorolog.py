import pandas as pd


class Fluorolog:
    """[summary]

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """

    manufacturer = "Horiba"
    """Docstring for class attribute Fluorolog.manufacturer."""

    name = "fluorolog"
    """Docstring for class attribute Fluorolog.name."""

    supported_models = ["SPEX Fluorolog-3"]
    """Docstring for class attribute Fluorolog.supported_models."""

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
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()

    @staticmethod
    def load_water_raman(filepath):
        """[summary]

        Args:
            filepath (str): [description]

        Returns:
            DataFrame: [description]
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
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()
