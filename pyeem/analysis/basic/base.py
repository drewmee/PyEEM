import numpy as np


def fluorescence_regional_integration(
    eem_df,
    region_bounds={
        "excitation_bounds": (-float("inf"), float("inf")),
        "emission_bounds": (-float("inf"), float("inf")),
    },
):
    """Fluorescence Excitation−Emission Matrix Regional Integration to
    Quantify Spectra for Dissolved Organic Matter. Chen et al. 2003

    Args:
        eem_df (pandas.DataFrame): An Excitation Emission matrix.
        region_bounds (dict of {str: tuple of (int or float)}, optional):
            A dictionary containing the upper and lower wavelength integration
            bounds for both excitation and emission wavelengths. Defaults to
            { "excitation_bounds": (-float("inf"), float("inf")), "emission_bounds": (-float("inf"), float("inf")) }.

    Returns:
        float: Integrated fluorescence intensity.
    """

    fl = eem_df.to_numpy()
    # em = eem_df.index.values
    # ex = eem_df.columns.to_numpy()
    # return np.trapz(em, np.trapz(ex, fl))
    return fl.sum()
