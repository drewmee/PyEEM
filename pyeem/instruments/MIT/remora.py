import pandas as pd


class Remora:
    """The MIT REMORA, a field compact deployable spectrofluorometer."""

    manufacturer = "MIT"
    """Name of Manufacturer."""

    name = "REMORA"
    """Name of Instrument."""

    supported_models = ["REMORA-V1"]
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
        eem_df = pd.read_csv(filepath, index_col=0)
        eem_df.columns = eem_df.columns.astype(float)
        eem_df = eem_df.sort_index(axis=0)
        eem_df = eem_df.sort_index(axis=1)
        eem_df.index.name = "emission_wavelength"
        return eem_df

    def load_absorbance(filepath):
        """Loads an absorbance spectrum which is generated by the instrument.

        Args:
            filepath (str): The filepath of the data file.

        Returns:
            pandas.DataFrame: An absorbance spectrum.
        """
        absorb_df = pd.read_csv(filepath, index_col=0)
        absorb_df.index.name = "excitation_wavelength"
        absorb_df.sort_index(axis=0)
        absorb_df.index = absorb_df.index.astype("float64")
        return absorb_df

    def load_water_raman(filepath):
        """Loads a water Raman spectrum which is generated by the instrument.

        Args:
            filepath (str): The filepath of the data file.

        Returns:
            pandas.DataFrame: An absorbance spectrum.
        """
        raman_df = pd.read_csv(filepath, index_col=0)
        raman_df.columns = raman_df.columns.astype(float)
        raman_df = raman_df.sort_index(axis=0)

        raman_df = raman_df.rename(columns={raman_df.columns[0]: "intensity"})
        raman_df.index.name = "emission_wavelength"
        return raman_df

    @staticmethod
    def load_spectral_corrections():
        """TODO - Should load instrument specific spectral corrections which will
        be used in data preprocessing.

        Raises:
            NotImplementedError: On the TODO list...
        """
        raise NotImplementedError()