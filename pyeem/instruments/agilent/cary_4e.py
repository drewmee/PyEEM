import pandas as pd


class Cary4E:
    """[summary]

    Raises:
        NotImplementedError: [description]
        NotImplementedError: [description]
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """

    manufacturer = "Agilent"
    """Docstring for class attribute Cary4E.manufacturer."""

    name = "cary_4e"
    """Docstring for class attribute Cary4E.name."""

    supported_models = ["Cary 4E"]
    """Docstring for class attribute Cary4E.supported_models."""

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

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()

    @staticmethod
    def load_absorbance(filepath):
        """[summary]

        Args:
            filepath ([type]): [description]

        Returns:
            [type]: [description]
        """
        absorb_df = pd.read_csv(
            filepath,
            sep=",",
            header=None,
            index_col=0,
            names=["wavelength", "absorbance"],
        )
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
