import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def _get_steps():
    hdf_subdir = "preprocessing/filters/"
    steps = {
        "step_name": [
            "crop",
            "fill_missing_values",
            "discrete_wavelengths",
            "gaussian_smoothing",
        ]
    }
    steps_df = pd.DataFrame(steps)
    steps_df["hdf_path"] = hdf_subdir + steps_df["step_name"]
    return steps_df


def fill_missing_values(eem_df, fill):
    """Fills NA/NAN values within an excitation-emission matrix with a user-selectable value.

    Args:
        eem_df (pandas.DataFrame): An excitation-emission matrix.
        fill (str): The value to replace NA/NAN values with.

    Raises:
        ValueError: Raised if fill passed value other than "zeros" or "interp".

    Returns:
        pandas.DataFrame: An excitation-emission matrix with missing values filled in.
    """
    valid_fill = {"zeros", "interp"}
    if fill not in valid_fill:
        raise ValueError("fill_missing_values: fill must be one of %r." % valid_fill)
    if fill == "zeros":
        eem_df.fillna(0, inplace=True)
    elif fill == "interp":
        eem_df = eem_df.interpolate(method="linear", axis=0).ffill().bfill()
    return eem_df


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
    """Crops an excitation-emission matrix to user-selectable dimensions.

    Args:
        eem_df (pandas.DataFrame): An excitation-emission matrix.
        crop_dims (dict of {str : tuple of (int, float)}): A dictionary containing the
            upper and lower bounds for both the excitation and emission wavelengths for
            the EEM region that you would like to keep. These bounds are inclusive.

    Returns:
        pandas.DataFrame: The cropped excitation-emission matrix.
    """

    # TODO - rethink if this is necessary...
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
        eem_df (pandas.DataFrame): An excitation-emission matrix.
        ex_wl (list of int or float): [description]

    Returns:
        pandas.DataFrame:[description]
    """
    eem_tdf = eem_df.transpose()
    ilocs = []
    for wl in selected_wavelengths:
        ilocs.append(eem_tdf.index.get_loc(wl, method="nearest"))
    eem_df = eem_tdf.iloc[ilocs].transpose()
    return eem_df


def gaussian_smoothing(eem_df, sigma, truncate):
    """Performs gaussian smooothing on the excitation-emission matrix from the input
    sigma and truncation sigma.

    Args:
        eem_df (pandas.DataFrame): An excitation-emission matrix.
        sig (int): Sigma of the gaussian distribution weight for the data smoothing.
        trun (int): Truncate in 'sigmas' the gaussian distribution.

    Returns:
        pandas.DataFrame: A guassian smoothed excitation-emission matrix.
    """

    # smooth the data with the gaussian filter
    eem_blurred = gaussian_filter(eem_df, sigma=sigma, truncate=truncate)
    eem_df[:] = eem_blurred
    return eem_df
