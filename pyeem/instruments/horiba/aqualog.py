import pandas as pd


class Aqualog:
    """[summary]

    Raises:
        NotImplementedError: [description]
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """

    manufacturer = "Horiba"
    """Docstring for class attribute Aqualog.manufacturer."""

    name = "aqualog"
    """Docstring for class attribute Aqualog.name."""

    supported_models = ["Aqualog-880-C"]
    """Docstring for class attribute Aqualog.supported_models."""

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
        eem_df = pd.read_csv(filepath, sep="\t", index_col=0)
        eem_df.dropna(how="all", axis=1, inplace=True)
        eem_df.dropna(how="all", axis=0, inplace=True)
        eem_df.columns = eem_df.columns.astype(float)
        eem_df = eem_df.sort_index(axis=0)
        eem_df = eem_df.sort_index(axis=1)
        eem_df.index.name = "emission_wavelength"
        return eem_df

    @staticmethod
    def load_absorbance(filepath):
        """[summary]

        Args:
            filepath (str): [description]

        Returns:
            DataFrame: [description]
        """
        absorb = pd.read_csv(filepath, sep="\t", index_col=0, header=0, skiprows=[1, 2])
        absorb = absorb[["Abs"]]
        absorb.rename(columns={"Abs": "absorbance"}, inplace=True)
        absorb["absorbance"] = absorb["absorbance"].apply(
            pd.to_numeric, errors="coerce"
        )
        absorb.fillna(0, inplace=True)
        absorb.index = absorb.index.astype("float64")
        absorb = absorb.sort_index()
        absorb.index.name = "wavelength"
        return absorb

    @staticmethod
    def load_water_raman():
        """[summary]

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
