import pandas as pd

from . import agilent, horiba, tecan


def get_supported_instruments():
    """Gets list of instruments which are supported by PyEEM. These instruments 
    can be passed as arguments to pyeem.datasets.Dataset to load raw data files.

    Returns:
        tuple of pandas.DataFrame: Two tables of supported instruments. The first including 
        only human-readable information. The second containing instances of each of the 
        instrument classes which is eventually used in pyeem.datasets.Dataset to load 
        raw data files.
    """
    manuf_instruments = {
        agilent.name: agilent.instruments,
        horiba.name: horiba.instruments,
        tecan.name: tecan.instruments,
    }
    # instruments = [Aqualog, Fluorolog, Cary]
    df = pd.DataFrame()
    for manuf, instruments in manuf_instruments.items():
        for i in instruments:
            for j in i.supported_models:
                d = {
                    "manufacturer": manuf,
                    "name": i.name,
                    "supported_models": j,
                    "object": i,
                }
                df = df.append(d, ignore_index=True)

    df.set_index(["manufacturer", "supported_models"], inplace=True)
    df_display = df.drop(columns=["object"])
    return df_display, df


def _get_dataset_instruments_df(
    raman_instrument, absorbance_instrument, eem_instrument
):
    _, _supported = get_supported_instruments()

    passed_instruments = {
        "Water Raman": raman_instrument,
        "Absorbance": absorbance_instrument,
        "EEM": eem_instrument,
    }

    dataset_instruments = {}
    for key, value in passed_instruments.items():
        if (value not in _supported["name"].values) and (value is not None):
            raise Exception("%s scans collected by unsupported instrument." % key)

        if value is not None:
            instrument_obj = (
                _supported[_supported["name"] == value]["object"].unique().item()
            )
        else:
            instrument_obj = None

        dataset_instruments[key.lower().replace(" ", "_")] = instrument_obj

    dataset_instruments_df = pd.DataFrame(dataset_instruments, index=[0])
    return dataset_instruments_df
