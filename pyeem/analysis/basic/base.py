import numpy as np


def fluorescence_regional_integration(eem_df, region_bounds=None):
    """[summary]

    Args:
        eem_df (DataFrame): [description]
        region_bounds ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    # Fluorescence Excitation−Emission Matrix Regional
    # Integration to Quantify Spectra for Dissolved Organic Matter
    fl = eem_df.to_numpy()
    em = eem_df.index.values
    ex = eem_df.columns.to_numpy()

    # return np.trapz(em, np.trapz(ex, fl))
    return fl.sum()
