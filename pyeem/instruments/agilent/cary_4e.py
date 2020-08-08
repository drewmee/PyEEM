import pandas as pd


class Cary4E:
    """[summary]

    Raises:
        NotImplementedError: [description]
        NotImplementedError: [description]
        NotImplementedError: [description]
    """

    manufacturer = "Agilent"
    name = "cary_4e"
    supported_models = ["Cary 4E"]

    def __init__(self, model, sn=None):
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

        Raises:
            NotImplementedError: [description]
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
            filepath ([type]): [description]

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
