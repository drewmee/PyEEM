import warnings

import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.integrate import trapz

from ..filters import crop


def _get_steps():
    # Step names are human readable/interpretable and are to be used in figures, reporting, etc.
    # Step aliases must correspond to functions and will be used to link arguments in
    # preprocessing.routine() to their corresponding preprocessing function.
    hdf_subdir = "preprocessing/corrections/"
    steps = {
        "step_name": [
            "blank_subtraction",
            "inner_filter_effect",
            "raman_normalization",
            "scatter_removal",
            "dilution",
        ]
    }
    steps_df = pd.DataFrame(steps)
    steps_df["hdf_path"] = hdf_subdir + steps_df["step_name"]
    return steps_df


def blank_subtraction(sample_df, blank_df):
    """[summary]

    Args:
        sample_df (~pandas.DataFrame): [description]
        blank_df (~pandas.DataFrame): [description]

    Returns:
        DataFrame: [description]
    """
    if sample_df.shape != blank_df.shape:
        # warnings.warn("Sample EEM and blank EEM have different dimensions, attempting to crop the sample to the size of the blank.")

        # check if bounds overlap at all
        emission_overlap = blank_df.index.intersection(sample_df.index)
        excitation_overlap = blank_df.T.index.intersection(sample_df.T.index)

        if any(a.size == 0 for a in [emission_overlap, excitation_overlap]):
            raise ValueError(
                "Unable to perform blank subtraction. The sample EEM and the blank EEM have no overlapping Excitation-Emission pairs."
            )

        crop_dimensions = {
            "emission_bounds": (blank_df.index.min(), blank_df.index.max()),
            "excitation_bounds": (blank_df.columns.min(), blank_df.columns.max()),
        }

        sample_df = crop(sample_df, crop_dimensions)

    sample_df = sample_df.subtract(blank_df, axis=1)
    sample_df.clip(lower=0, inplace=True)
    # Just in case
    sample_df.dropna(how="all", axis=1, inplace=True)
    sample_df.dropna(how="all", axis=0, inplace=True)
    return sample_df


def inner_filter_effect(
    eem_df, absorb_df, pathlength=1, unit="absorbance", threshold=0.03
):
    """Based on Kothawala, D. N., Murphy, K. R., Stedmon, C. A., Weyhenmeyer,
    G. A., & Tranvik, L. J. (2013). Inner filter correction of dissolved
    organic matter fluorescence. Limnology and Oceanography: Methods, 11(12),
    616-630. http://doi.org/10.4319/lom.2013.11.616
    
    TODO - add the IFE correction formula in RST format
    
    .. math::
        \sum_{i=1}^{\\infty} x_{i}

    Args:
        eem_df (~pandas.DataFrame): [description]
        abs_df (~pandas.DataFrame): [description]
        pathlength (int or float): [description]
        unit (str, optional): [description]. Defaults to "absorbance".

    Returns:
        DataFrame: [description]
    """
    # "From the ABA algorithm, the onset of significant IFE (>5%)
    # occurs when absorbance exceeds 0.042"

    # "For rare EEMs with ATotal> 1.5 (3.0% of the lakes in the Swedish
    # survey), a 2-fold dilution is recommended followed by ABA or CDA
    # correction"

    # Fcorr = Fobs * 10**((Aex + Aem)/pathlength*2)
    # ife_correction_factor = 10^(total_absorbance * pathlength/2)

    def _process(row):
        row_df = pd.DataFrame(row)
        excitation_wavelength = row.name
        excitation_absorbance = absorb_df.iloc[
            absorb_df.index.get_loc(excitation_wavelength, method="nearest")
        ]["absorbance"]
        merged_df = pd.merge_asof(
            row_df, absorb_df, left_index=True, right_index=True, direction="nearest"
        )
        merged_df["a_total"] = merged_df["absorbance"] + [excitation_absorbance]
        merged_df["f_corr"] = merged_df[excitation_wavelength] * 10 ** (
            merged_df["a_total"] / (pathlength * 2)
        )
        return merged_df["f_corr"]

    return eem_df.apply(_process, axis=0)


def raman_normalization(eem_df, raman_source_type, raman_source, method="gradient"):
    """Element-wise division of the EEM spectra by area under the
    ramam peak. See reference Murphy et al. Measurement of Dissolved
    Organic Matter Fluorescence in Aquatic Environments:
    An Interlaboratory Comparison" 2010 Environmental Science and
    Technology.

    Args:
        eem_df (pandas.DataFrame): [description]
        blank_df (pandas.DataFrame): [description]
        method (str, optional): [description]. Defaults to "gradient".

    Returns:
        DataFrame: Raman normalized EEM spectrum in Raman Units (R.U.)
    """
    # TODO - The Raman area is calculated using the  baseline-corrected
    # peak boundary definition (Murphy and others, 2011)
    # raman_sources = ['water_raman', 'blank', 'metadata']

    if raman_source_type in ["blank", "water_raman"]:
        a = 371  # lower limit
        b = 428  # upper limit
        raman_peak_area = trapz(raman_source[350].loc[a:b])

    elif raman_source_type == "metadata":
        # Raise warning
        raman_peak_area = raman_source

    else:
        # raise Exception
        raise ValueError(
            "Invalid input for raman_source_type. Must be 'metadata', 'water_raman', or 'blank'"
        )

    return eem_df / raman_peak_area


def scatter_bands():
    # pd.DataFrame
    data = [
        {"band": "Raleigh", "order": "first", "poly1d": np.poly1d([0, 1.0000, 0])},
        {
            "band": "Raman",
            "order": "first",
            "poly1d": np.poly1d([0.0006, 0.8711, 18.7770]),
        },
        {"band": "Raleigh", "order": "second", "poly1d": np.poly1d([0, 2.0000, 0])},
        {
            "band": "Raman",
            "order": "second",
            "poly1d": np.poly1d([-0.0001, 2.4085, -47.2965]),
        },
    ]
    return pd.DataFrame.from_records(data)


def scatter_removal(
    eem_df, band="both", order="both", excision_width=20, fill="interp", truncate=None
):
    """Function for removing Raleigh and Raman scatter by excising values
    in the areas where scatter is expected and replacing the missing
    values using 2d interpolation. This function is based on the
    following publication: Zepp et al. Dissolved organic fluorophores
    in southeastern US coastal waters: correction method for eliminating
    Rayleigh and Raman scattering peaks in excitation–emission matrices.
    Marine Chemistry. 2004

    Args:
        eem_df (pandas.DataFrame): Excitation Emission Matrix
        band (str, optional): The scatter band (Raleigh/Raman) to be removed. Defaults to "both".
        order (str, optional): The scatter band order (first/second) to be removed. Defaults to "both".
        excision_width (int, optional): The width of excision that each band will be removed with. Defaults to 20.
        fill (str, optional): The values which will fill the excised scatter region. Defaults to "interp".
        truncate (str, optional): The option to remove all values above and/or below the excised bands. Defaults to None.

    Returns:
        DataFrame: EEM with Raleigh/Raman scatter bands removed.
    """
    fl = eem_df.to_numpy()
    em = eem_df.index.values
    ex = eem_df.columns.to_numpy()
    grid_ex, grid_em = np.meshgrid(ex, em)
    values_to_excise = np.zeros(eem_df.shape, dtype=bool)

    bands_df = scatter_bands()
    r = excision_width / 2
    bands_df["above"], bands_df["below"] = [r, r]

    band = band.lower()
    if band in ["raleigh", "raman"]:
        bands_df = bands_df[bands_df["band"].str.lower() == band]

    order = order.lower()
    if order in ["first", "second"]:
        bands_df = bands_df[bands_df["order"].str.lower() == order]

    def _truncation(row):
        if truncate == "below":
            if row["order"] == "first":
                row["below"] = np.inf

        elif truncate == "above":
            if row["order"] == "second":
                row["above"] = np.inf

        elif truncate == "both":
            if row["order"] == "first":
                row["below"] = np.inf

            if row["order"] == "second":
                row["above"] = np.inf

        return row[["above", "below"]]

    bands_df[["above", "below"]] = bands_df.apply(_truncation, axis=1)

    for index, row in bands_df.iterrows():
        band_name, band_order = row[["band", "order"]]
        peaks = np.polyval(row["poly1d"], ex)
        peaks_grid = np.tile(peaks.reshape(1, -1), (em.size, 1))

        # Create logical arrays with 'True' where flourescent values
        # should be kept.
        keep_above = (grid_em - np.subtract(peaks_grid, row["below"])) <= 0
        keep_below = (grid_em - np.add(peaks_grid, row["above"])) >= 0

        # Update locations of fluorescent values to excise.
        values_to_excise = values_to_excise + np.invert(keep_above + keep_below)

    if fill == None:
        # Create an array with 'nan' in the place of values where scatter
        # is located. This may be used for vizualizing the locations of
        # scatter removal.
        fl_NaN = np.array(fl)
        fl_NaN[values_to_excise] = np.nan
        eem_df = pd.DataFrame(data=fl_NaN, index=em, columns=ex)

    elif fill == "zeros":
        fl_zeros = np.array(fl)
        fl_zeros[values_to_excise] = 0
        eem_df = pd.DataFrame(data=fl_zeros, index=em, columns=ex)

    else:
        # Any other input for fill treat as default fill value of "interp"
        # Create a boolean array of values to keep to use when interpolating.
        values_to_keep = np.invert(values_to_excise)

        # Interpolate to fill the missing values.
        # 'points' is a 'Number of Points' x 2 array containing coordinates
        # of datapoints to be used when interpolating to fill in datapoints.
        points = np.array(
            [
                np.reshape(grid_ex[values_to_keep], (-1)),
                np.reshape(grid_em[values_to_keep], (-1)),
            ]
        )
        points = np.transpose(points)
        values = fl[values_to_keep]

        fl_interp = scipy.interpolate.griddata(
            points, values, (grid_ex, grid_em), fill_value=0
        )
        # Replace excised values with interpolated values.
        fl_clean = np.array(fl)
        fl_clean[values_to_excise] = fl_interp[values_to_excise]
        eem_df = pd.DataFrame(data=fl_clean, index=em, columns=ex)

    return eem_df


def dilution(eem_df, dilution_factor):
    """[summary]

    Args:
        eem_df (~pandas.DataFrame): [description]
        dilution_factor (int or float): [description]

    Returns:
        DataFrame: [description]
    """
    return eem_df * dilution_factor


def pseudo_pivot(meta_df):
    """Think about using melt()

    DataFrame.pivot_table
    Generalization of pivot that can handle duplicate values for one index/column pair.
    """
    m = []
    for name, group in meta_df.groupby(level="sample_set"):
        blank_name = group.xs("blank", level="scan_type")["filename"].item()
        blank_abs = blank_name.rsplit(".dat", 1)[0] + "_abs.dat"

        if blank_abs in group["filename"].values:
            blank_abs = group[group["filename"] == blank_abs]["filename"].item()
        else:
            blank_abs = np.nan

        for index, row in group[
            group.index.get_level_values("scan_type") == "sample"
        ].iterrows():
            sample_name = row["filename"]
            sample_abs = row["filename"].rsplit(".dat", 1)[0] + "_abs.dat"
            if sample_abs in group["filename"].values:
                sample_abs = group[group["filename"] == sample_abs]["filename"].item()
            else:
                sample_abs = np.nan

            m.append(
                {
                    "sample_set": name,
                    "blank": blank_name,
                    "blank_abso": blank_abs,
                    "sample": sample_name,
                    "sample_abs": sample_abs,
                }
            )

    return pd.DataFrame(m)
