import numpy as np
import pandas as pd


class Aqualog:
    manufacturer = "Horiba"
    name = "aqualog"
    supported_models = ["Aqualog-880-C"]

    def __init__(self, model=None, sn=None):
        self.model = model
        self.sn = sn

    @staticmethod
    def load_eem(filename):
        eem = pd.read_csv(filename, sep='\t', index_col=0)
        eem.columns = eem.columns.astype(float)
        eem.index.name = "emission_wavelength"
        eem = eem.sort_index(axis=0)
        eem = eem.sort_index(axis=1)
        return eem

    @staticmethod
    def load_absorbance(filename):
        absorb = pd.read_csv(filename, sep='\t',
                             index_col=0, header=0, skiprows=[1, 2])
        absorb.index.name = "wavelength"
        absorb = absorb.sort_index()
        absorb = absorb[['Abs']]
        absorb.rename(columns={'Abs': 'absorbance'}, inplace=True)
        return absorb

    @staticmethod
    def load_water_raman():
        raise NotImplementedError()

    @staticmethod
    def load_spectral_corrections():
        raise NotImplementedError()


class Fluorolog:
    manufacturer = "Horiba"
    name = "fluorolog"
    supported_models = ["SPEX Fluorolog-3"]

    def __init__(self, model=None, sn=None):
        self.model = model
        self.sn = sn

    @staticmethod
    def load_eem(filename):
        return

    @staticmethod
    def load_absorbance():
        raise NotImplementedError()

    @staticmethod
    def load_water_raman(filename):
        water_raman = pd.read_csv(filename, sep=',',
                                  index_col=0, skiprows=1,
                                  names=["emission_wavelength",
                                         "intensity"])
        water_raman = water_raman.sort_index()
        return water_raman

    @staticmethod
    def load_spectral_corrections():
        return


class Cary:
    manufacturer = "Agilent"
    name = "cary"
    supported_models = ["Cary 4E"]

    def __init__(self, model=None, sn=None):
        self.model = model
        self.sn = sn

    @staticmethod
    def load_eem(filename):
        raise NotImplementedError()

    @staticmethod
    def load_absorbance():
        return

    @staticmethod
    def load_water_raman(self, filename):
        raise NotImplementedError()

    @staticmethod
    def load_spectral_corrections():
        raise NotImplementedError()


def _get_supported_instruments():
    instruments = [Aqualog, Fluorolog, Cary]
    df = pd.DataFrame()
    for i in instruments:
        for j in i.supported_models:
            d = {
                "manufacturer": i.manufacturer,
                "name": i.name,
                "supported_models": j,
                "object": i
            }
            df = df.append(d, ignore_index=True)
    df.set_index(['manufacturer', 'supported_models'], inplace=True)
    df_display = df.drop(columns=['object'])
    return df_display, df


supported, _supported = _get_supported_instruments()
