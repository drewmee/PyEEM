import warnings

import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.integrate import trapz

from ..filters import crop


def _get_steps():
    """
    Step names are human readable/interpretable and are to be used in figures, reporting, etc.
    Step aliases must correspond to functions and will be used to link arguments in
    preprocessing.routine() to their corresponding preprocessing function.
    """
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
    """Subtract the blank Excitation Emission Matrix (EEM) signal from a sample EEM. This step helps to
    reduce the effect of Raman and Rayleigh scattering.

    Args:
        sample_df (pandas.DataFrame): Excitation Emission Matrix of a sample.
        blank_df (pandas.DataFrame): Excitation Emission Matrix of a blank.

    Raises:
        ValueError: Raised if the sample EEM and the blank EEM have no overlapping Excitation-Emission pairs.

    Returns:
        pandas.DataFrame: Blank subtracted Excitation Emission Matrix of sample.
    """
    if sample_df.shape != blank_df.shape:
        # warnings.warn("Sample EEM and blank EEM have different dimensions, attempting to crop the sample to the size of the blank.")
        # Check if bounds overlap at all
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
    # Just in case, drop rows and columns with only nan values
    sample_df.dropna(how="all", axis=1, inplace=True)
    sample_df.dropna(how="all", axis=0, inplace=True)
    return sample_df


def inner_filter_effect(eem_df, absorb_df, pathlength=1, threshold=0.03, limit=1.5):
    """Uses an absorbance measurement to correct the Excitation Emission matrix for the inner-filter effect.
    Based on Kothawala, D. N., Murphy, K. R., Stedmon, C. A., Weyhenmeyer,
    G. A., & Tranvik, L. J. (2013). Inner filter correction of dissolved
    organic matter fluorescence. Limnology and Oceanography: Methods, 11(12),
    616-630. http://doi.org/10.4319/lom.2013.11.616

    TODO - add the IFE correction formula in RST format

    .. math::
        \sum_{i=1}^{\\infty} x_{i}

    Args:
        eem_df (pandas.DataFrame): Excitation Emission Matrix of a sample.
        abs_df (pandas.DataFrame): Absorbance spectrum of a sample
        pathlength (int or float): Pathlength of the cuvette with which the sample was measured.
        threshold (float, optional): The threshold of total absorbance after which IFE correction
            will be applied. Defaults to 0.03.
        limit (float, optional): The total absorbance level at which IFE correction can no longer be effective.
            If this level of total absorbance is measured, it is reccomended to perform a 2-fold dilution of
            the sample and perform measurements again.

    Returns:
        pandas.DataFrame: Inner-filter Effect corrected Excitation Emission Matrix.
    """

    # "From the ABA algorithm, the onset of significant IFE (>5%)
    # occurs when absorbance exceeds 0.042"

    # "For rare EEMs with ATotal> 1.5 (3.0% of the lakes in the Swedish
    # survey), a 2-fold dilution is recommended followed by ABA or CDA
    # correction"

    # Fcorr = Fobs * 10**((Aex + Aem)/pathlength*2)
    # ife_correction_factor = 10^(total_absorbance * pathlength/2)

    tmp_df_list = []
    for index, row in eem_df.iterrows():
        row_df = pd.DataFrame(row)
        excitation_wavelength = row.name
        excitation_absorbance = absorb_df.iloc[
            absorb_df.index.get_loc(excitation_wavelength, method="nearest")
        ]["absorbance"]
        tmp_df = pd.merge_asof(
            row_df, absorb_df, left_index=True, right_index=True, direction="nearest"
        )
        tmp_df["a_total"] = tmp_df["absorbance"] + [excitation_absorbance]
        tmp_df = tmp_df.drop(columns=[excitation_wavelength, "absorbance"])
        tmp_df = tmp_df.rename(columns={"a_total": excitation_wavelength})
        tmp_df_list.append(tmp_df)

    a_total_df = pd.concat(tmp_df_list, axis=1).T
    if a_total_df.to_numpy().max() >= limit:
        raise ValueError(
            "Found absorbance total greater than 1.5, a 2-fold dilution is reccomended. Inner-filter effect correction will not be applied to this sample."
        )
    elif (a_total_df.values > threshold).any():
        corr_df = a_total_df.applymap(lambda x: 10 ** (x / (pathlength * 2)))
        eem_df = eem_df * corr_df
    else:
        # All absorbance total values below threshold, no need to perform IFE correction
        pass

    return eem_df


def _get_peak_position(excitation_wavelength):
    peak_position = 1e7 / ((1e7 / excitation_wavelength) - 3400)
    return peak_position


def _calculate_raman_peak_area(raman_source, excitation_wavelength):
    peak_position = _get_peak_position(excitation_wavelength)
    peak_width = 55.6
    a = peak_position - peak_width / 2
    b = peak_position + peak_width / 2
    raman_peak_area = trapz(raman_source.loc[a:b, raman_source.columns[0]])
    return raman_peak_area, (peak_position, a, b)


def raman_normalization(eem_df, raman_source_type, raman_source, excitation_wavelength):
    """Element-wise division of the EEM spectra by area under the Raman peak. See
    reference Murphy et al. Measurement of Dissolved Organic Matter Fluorescence
    in Aquatic Environments: An Interlaboratory Comparison" 2010 Environmental
    Science and Technology.

    Args:
        eem_df (pandas.DataFrame): Excitation Emission matrix of a sample.
        raman_source_type ([type]): [description]
        raman_source ([type]): [description]
        excitation_wavelength ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        pandas.DataFrame: Raman normalized Excitation Emission Matrix in Raman Units (R.U.).
    """
    if raman_source_type in ["blank", "water_raman"]:
        raman_peak_area, (peak_position, a, b) = _calculate_raman_peak_area(
            raman_source, excitation_wavelength
        )
    elif raman_source_type == "metadata":
        # Raise warning
        raman_peak_area = raman_source
    else:
        raise ValueError(
            "Invalid input for raman_source_type. Must be 'metadata', 'water_raman', or 'blank'"
        )
    return eem_df / raman_peak_area


def _scatter_bands():
    data = [
        {"band": "Rayleigh", "order": "first", "poly1d": np.poly1d([0, 1.0000, 0])},
        {
            "band": "Raman",
            "order": "first",
            "poly1d": np.poly1d([0.0006, 0.8711, 18.7770]),
        },
        {"band": "Rayleigh", "order": "second", "poly1d": np.poly1d([0, 2.0000, 0])},
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
    """Removal of Rayleigh and Raman scatter by excising values in the areas where
    scatter is expected and replacing the excised values with a user-selectable
    value. This function is based on the following publication: Zepp et al.,
    Dissolved organic fluorophores in southeastern US coastal waters: correction
    method for eliminating Rayleigh and Raman scattering peaks in excitation–emission
    matrices. Marine Chemistry. 2004

    Args:
        eem_df (pandas.DataFrame): Excitation Emission Matrix of a sample.
        band (str, optional): The scatter band (Rayleigh/Raman) to be removed. Defaults to "both".
        order (str, optional): The scatter band order (first/second) to be removed. Defaults to "both".
        excision_width (int, optional): The width of excision that each band will be removed with. Defaults to 20.
        fill (str, optional): The values which will fill the excised scatter region. Defaults to "interp".
        truncate (str, optional): The option to remove all values above and/or below the excised bands. Defaults to None.

    Returns:
        pandas.DataFrame: Excitation Emission Matrix with Rayleigh/Raman scatter bands removed.
    """
    fl = eem_df.to_numpy()
    em = eem_df.index.values
    ex = eem_df.columns.to_numpy()
    grid_ex, grid_em = np.meshgrid(ex, em)
    values_to_excise = np.zeros(eem_df.shape, dtype=bool)

    bands_df = _scatter_bands()
    r = excision_width / 2
    bands_df["above"], bands_df["below"] = [r, r]

    band = band.lower()
    if band in ["rayleigh", "raman"]:
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
        fl_nan = np.array(fl)
        fl_nan = fl_nan.astype(object)
        fl_nan[values_to_excise] = fl_nan[values_to_excise].astype("float64")
        fl_nan[values_to_excise] = np.nan
        eem_df = pd.DataFrame(data=fl_nan, index=em, columns=ex)

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
    """Corrects for sample dilution. Samples can be diluted for a variety of reasons,
    for example for total absorbance values greater than X, N et al. states that a
    two-factor dilution is necessary.

    Args:
        eem_df (pandas.DataFrame): Excitation Emission Matrix of a sample.
        dilution_factor (int or float): Dilution of factor of original sample, (0, 1].

    Returns:
        pandas.DataFrame: Dilution corrected Excitation Emission Matrix.
    """
    return eem_df * dilution_factor
