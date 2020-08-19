import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def _get_steps():
    hdf_subdir = "preprocessing/filters/"
    steps = {"step_name": ["crop", "discrete_wavelengths", "gaussian_smoothing"]}
    steps_df = pd.DataFrame(steps)
    steps_df["hdf_path"] = hdf_subdir + steps_df["step_name"]
    return steps_df


def _QC_crop_dims(crop_dims):
    # Make sure the crop_dim argument is a dictionary
    if not isinstance(crop_dims, dict):
        raise ValueError("Argument crop_dims must be of type dict.")

    # Make sure the crop_dims dictionary contains the required keys
    req_keys = ["emission_bounds", "excitation_bounds"]
    if not all(dim in crop_dims for dim in req_keys):
        raise ValueError(
            "Argument crop_dims must contain the following keys: "
            "emission_bounds, excitation_bounds"
        )

    # Make sure the values are tuples containing two numbers in
    # ascending order.
    # some_tuple == tuple(sorted(some_tuple)):
    if not all(isinstance(bounds, tuple) for bounds in list(crop_dims.values())):
        raise ValueError(
            "Argument crop_dims must be of type dict with values of type tuple."
        )

    # if type(z) == int or type(z) == float:
    for bounds in list(crop_dims.values()):
        if not isinstance(bounds, tuple):
            raise ValueError(
                "Argument crop_dims must be of type dict with values of type tuple."
            )
        for bound in bounds:
            if not isinstance(bound, (np.number, float, int)):
                raise ValueError("Crop dimensions must be numeric.")

        if not bounds == tuple(sorted(bounds)):
            raise ValueError("")


def crop(eem_df, crop_dims):
    """[summary]

    Args:
        eem_df (~pandas.DataFrame): [description]
        crop_dims (dict of {str : tuple of (int, float)}): [description]

    Returns:
        DataFrame: [description]
    """

    _QC_crop_dims(crop_dims)

    #  Rows (axis=0) are Emission wavelengths
    eem_df = eem_df.truncate(
        before=crop_dims["emission_bounds"][0],
        after=crop_dims["emission_bounds"][1],
        axis=0,
    )

    # Columns (axis=1) are Excitation wavelengths
    eem_df = eem_df.truncate(
        before=crop_dims["excitation_bounds"][0],
        after=crop_dims["excitation_bounds"][1],
        axis=1,
    )

    return eem_df


def discrete_excitations(eem_df, selected_wavelengths):
    """[summary]

    Args:
        eem_df (~pandas.DataFrame): [description]
        ex_wl (list of int or float): [description]

    Returns:
        DataFrame: [description]
    """
    eem_tdf = eem_df.transpose()
    ilocs = []
    for wl in selected_wavelengths:
        ilocs.append(eem_tdf.index.get_loc(wl, method="nearest"))
    eem_df = eem_tdf.iloc[ilocs].transpose()
    return eem_df


def gaussian_smoothing(eem_df, sigma, truncate):
    """This function does a gaussian_blurr on the 2D spectra image from
    the input sigma and truncation sigma.

    Args:
        eem_df (~pandas.DataFrame): [description]
        sig (int): Sigma of the gaussian distribution weight for
        the data smoothing.
        trun (int): Truncate in 'sigmas' the gaussian distribution.

    Returns:
        DataFrame: [description]
    """

    # smooth the data with the gaussian filter
    eem_blurred = gaussian_filter(eem_df.to_numpy(), sigma=sigma, truncate=truncate)
    return eem_blurred
