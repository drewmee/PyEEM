import pandas as pd

from . import agilent, horiba


def get_supported_instruments():
    """[summary]

    Returns:
        [type]: [description]
    """
    manuf_instruments = {
        agilent.name: agilent.instruments,
        horiba.name: horiba.instruments,
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


supported, _supported = get_supported_instruments()


def _get_dataset_instruments_df(
    raman_instrument, absorbance_instrument, eem_instrument
):
    global _supported
    if eem_instrument not in _supported["name"].values:
        raise Exception("EEM scans collected by unsupported instrument.")
    if absorbance_instrument not in _supported["name"].values:
        raise Exception("Absorbance scans collected by unsupported instrument.")
    if (raman_instrument not in _supported["name"].values) and (
        raman_instrument is not None
    ):
        raise Exception("Raman scans collected by unsupported instrument.")

    if raman_instrument is not None:
        raman_instrument_obj = (
            _supported[_supported["name"] == raman_instrument]["object"].unique().item()
        )
    else:
        raman_instrument_obj = None

    absorbance_instrument_obj = (
        _supported[_supported["name"] == absorbance_instrument]["object"]
        .unique()
        .item()
    )

    eem_instrument_obj = (
        _supported[_supported["name"] == eem_instrument]["object"].unique().item()
    )

    return pd.DataFrame(
        {
            "water_raman": raman_instrument_obj,
            "absorbance": absorbance_instrument_obj,
            "eem": eem_instrument_obj,
        },
        index=[0],
    )
