import pandas as pd
import pyeem
import pytest


class TestInstruments:
    manufacturers = ["Agilent", "Horiba", "Tecan"]
    """
    manuf_instruments = {
        pyeem.instruments.agilent.name: pyeem.instruments.agilent.instruments,
        pyeem.instruments.horiba.name: pyeem.instruments.horiba.instruments,
    }
    """

    def testGetSupportedInstruments(self):
        supported, _supported = pyeem.instruments.get_supported_instruments()
        for supp in [supported, _supported]:
            assert isinstance(supp, pd.DataFrame)
            assert supp.index.names == ["manufacturer", "supported_models"]
            assert (
                supp.index.get_level_values("manufacturer").unique().to_list()
                == self.manufacturers
            )

        assert "object" in _supported.columns
        assert "object" not in supported.columns
