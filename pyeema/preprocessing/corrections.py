import scipy.interpolate
from scipy.ndimage import gaussian_filter
from scipy.integrate import trapz
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

'''
def spectral_correction(em_corr, ex_cor):
    return
'''

# TODO = Probably can go without this


def absorbance_baseline_correction(abs_df, wl_range=(680, 700)):
    """Correct the instrumental baseline drift in absorbance
    data by subtracting the mean of the absorbance at higher
    wavelengths (Li and Hur 2017).

    Arguments:
        wl_range {[type]} -- [description]
    """
    mean_abs = abs_df.loc[wl_range[0]:wl_range[1]].mean()
    abs_df = abs_df - mean_abs
    return abs_df


def subtract_blank(sample_df, blank_df):
    """[summary]

    Arguments:
        sample_df {[type]} -- [description]
        blank_df {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # Need to make sure they're the same shape
    # and have the same ex, em wavelengths
    sub_eem_df = sample_df.subtract(blank_df)
    # Set all negative values to zero
    sub_eem_df.clip(lower=0, inplace=True)
    return sub_eem_df


def remove_scatter_bands(eem_df):
    """Function for removing Raleigh and Raman scatter by excising values
    in the areas where scatter is expected and replacing the missung 
    values using 2d interpolation.  This function is based on the 
    following publication: Zepp et al. Dissolved organic fluorophores 
    in southeastern US coastal waters: correction method for eliminating
    Rayleigh and Raman scattering peaks in excitation–emission matrices.
    Marine Chemistry. 2004

    Arguments:
        eem_df {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    coeff = np.array(([0, 1.0000, 0],
                      [0.0006, 0.8711, 18.7770],
                      [0, 2.0000, 0],
                      [-0.0001, 2.4085, -47.2965]))

    tol = np.array([[10,  10],
                    [10,   10],
                    [10,   10],
                    [10,   10]])

    fl = eem_df.to_numpy()
    em = eem_df.index.values
    ex = eem_df.columns.to_numpy()
    grid_ex, grid_em = np.meshgrid(ex, em)
    values_to_excise = np.zeros(eem_df.shape, dtype=bool)

    for n in range(len(tol)):
        # Only remove scatter if the tolerance is greater than 0.
        if tol[n, 0] > 0 or tol[n, 1] > 0:

            peaks = np.polyval(coeff[n, :], ex)
            # Peaks is a 1 x length(ex) vector containing the emission
            # wavelength of expected scatter peaks at each
            # exctation wavelenth.
            peaks_grid = np.tile(peaks.reshape(1, -1), (em.size, 1))

            # Create logical arrays with 'True' where flourescent values
            # should be kept.
            keep_above = (
                grid_em - np.subtract(peaks_grid, tol[n, 0])) <= 0
            keep_below = (grid_em - np.add(peaks_grid, tol[n, 1])) >= 0

            # Update locations of flourecent values to excise.
            values_to_excise = values_to_excise + \
                np.invert(keep_above + keep_below)

    # Create a boolean array of values to keep to use when interpolating.
    values_to_keep = np.invert(values_to_excise)

    # Create an array with 'nan' in the place of values where scatter
    # is located. This may be used for vizualizing the locations of
    # scatter removal.
    fl_NaN = np.array(fl)
    fl_NaN[values_to_excise] = np.nan

    # Interpolate to fill the missing values.
    # 'points' is a 'Number of Points' x 2 array containing coordinates
    # of datapoints to be used when interpolating to fill in datapoints.
    points = np.array([np.reshape(grid_ex[values_to_keep], (-1)),
                       np.reshape(grid_em[values_to_keep], (-1))])
    points = np.transpose(points)
    values = fl[values_to_keep]

    fl_interp = scipy.interpolate.griddata(
        points, values, (grid_ex, grid_em), fill_value=0)

    # Replace excised values with interpolated values.
    fl_clean = np.array(fl)
    fl_clean[values_to_excise] = fl_interp[values_to_excise]

    eem_df = pd.DataFrame(data=fl_clean, index=em, columns=ex)
    return eem_df


def ife_correction(eem_df, abs_df, cuvl, unit="absorbance"):
    """https://github.com/PMassicotte/eemR/blob/7f6843b1490237922dabea8da49d21a724018048/R/eem_inner_filter_effect.R

    Arguments:
        eem_df {[type]} -- [description]
        abs_df {[type]} -- [description]
        cuvl {[type]} -- [description]

    Keyword Arguments:
        unit {str} -- [description] (default: {"absorbance"})
    """
    return


def raman_normalization(eem_df, blank_df):
    # TODO - The Raman area is calculated using the  baseline-corrected
    # peak boundary definition (Murphy and others, 2011)
    """Raman normaization - element-wise division of the eem spectra by 
    area under the ramam peak. See reference Murphy et al. "scan_type 
    of Dissolved Organic Matter Fluorescence in Aquatic Environments: 
    An Interlaboratory Comparison" 2010 Environmental Science and 
    Technology.

    Arguments:
        eem_df {[type]} -- [description]
        blank_df {[type]} -- [description]

    Returns:
        [type] -- Raman normalized EEM spectrum in Raman Units (R.U.)
    """
    a = 371  # lower limit
    b = 428  # upper limit
    raman_peak_area = trapz(blank_df[350].loc[a:b])
    norm_eem_df = eem_df / raman_peak_area
    return norm_eem_df


def dilution(eem, dilution_factor):
    """I don't think this is right

    Arguments:
        eem {[type]} -- [description]
        dilution_factor {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # check some things about the inputs
    return eem * dilution_factor


def pseudo_pivot(meta_df):
    """
    Think about using melt()

    DataFrame.pivot_table
    Generalization of pivot that can handle duplicate values for one index/column pair.
    """
    m = []
    for name, group in meta_df.groupby(level='sample_set'):
        blank_name = group.xs('blank', level='scan_type')['filename'].item()
        blank_abs = blank_name.rsplit(".dat", 1)[0] + "_abs.dat"

        if blank_abs in group['filename'].values:
            blank_abs = group[group['filename']
                              == blank_abs]['filename'].item()
        else:
            blank_abs = np.nan

        for index, row in group[
            group.index.get_level_values('scan_type') == 'sample'
        ].iterrows():
            sample_name = row['filename']
            sample_abs = row['filename'].rsplit(".dat", 1)[0] + "_abs.dat"
            if sample_abs in group['filename'].values:
                sample_abs = group[group['filename']
                                   == sample_abs]['filename'].item()
            else:
                sample_abs = np.nan

            m.append({
                "sample_set": name, "blank": blank_name,
                "blank_abso": blank_abs, "sample": sample_name,
                "sample_abs": sample_abs})

    return pd.DataFrame(m)


def __save_intermediate():
    return


def routine(meta_df, hdf, intermediate_store=True,
            spectral_corrections=False,
            crop_dims=None, blank_subtract=True, ife_correction=True,
            scatter_removal=True, scatter_fill='interp',
            raman_norm=True, raman_source='water_raman',
            smoothing=False):

    # These should be defined in their respective functions
    #raman_sources = ['water_raman', 'blank', 'metadata']
    #scatter_fill = ['interp', 'zeros', None]

    ex_keep_range = (224, np.inf)
    em_keep_range = (246, 573)
    #discrete_ex_wl = [225, 240, 275, 290, 300, 425]

    for set_name, sample_set in meta_df.groupby(level='sample_set'):
        blank_name = sample_set.xs(
            'blank_eem',
            level='scan_type')['filename'].item()

        blank_eem = pd.read_hdf(hdf, key=os.path.join(
            *["raw_sample_sets", str(set_name), blank_name]))
        blank_eem = crop(blank_eem, ex_keep_range, em_keep_range)
        blank_eem.to_hdf(hdf, key=os.path.join(*[
            "corrections", "sample_sets_crop",
            str(set_name), blank_name
        ]))

        blank_abs_name = blank_name.rsplit(".dat", 1)[0] + "_abs.dat"
        if blank_abs_name not in sample_set['filename'].values:
            blank_abs_name = np.nan

        # Absorbance baseline correction
        # pyeema.absorbance_baseline()

        if 'sample_eem' not in sample_set.index.get_level_values('scan_type'):
            continue

        for index, row in sample_set.xs('sample_eem',
                                        level='scan_type').iterrows():
            sample_name = row['filename']
            eem = pd.read_hdf(hdf, key=os.path.join(*[
                "raw_sample_sets", str(set_name), sample_name
            ]))

            # Crop sample EEM
            if crop_dims:
                # call crop() here

                # include this QC in the crop function
                # Make sure the crop_dim argument is a dictionary
                if not isinstance(crop_dims, dict):
                    #raise exception
                    pass
                # Make sure the crop_dims dictionary contains the
                # required keys
                req_keys = ["emission_bounds", "excitation_bounds"]
                if not all(dim in crop_dims for dim in req_keys):
                    #raise exception
                    pass

                em_bounds = crop_dims["em"]
                ex_bounds = crop_dims["ex"]

                # Make sure the values are tuples containing two numbers
                # in ascending order.
                # if type(z) == int or type(z) == float:
                # some_tuple == tuple(sorted(some_tuple)):
                if not all(isinstance(bounds, tuple)
                           for bounds in [em_bounds, ex_bounds]):
                    #raise exception
                    pass

                eem = crop(eem, ex_bounds, ex_bounds)
                eem.to_hdf(hdf, key=os.path.join(*[
                    "corrections", "sample_sets_crop",
                    str(set_name), sample_name
                ]))

            eem = crop(eem, ex_keep_range, em_keep_range)
            eem.to_hdf(hdf, key=os.path.join(*[
                "corrections", "sample_sets_crop",
                str(set_name), sample_name
            ]))

            # Absorbance baseline correction
            # pyeema.absorbance_baseline()

            # Subtract blank from sample
            eem = subtract_blank(eem, blank_eem)
            eem.to_hdf(hdf, key=os.path.join(*[
                "corrections", "sample_sets_subtract_blank",
                str(set_name), sample_name
            ]))

            # Remove Raman and Rayleigh scattering
            eem = remove_scatter_bands(eem)
            eem.to_hdf(hdf, key=os.path.join(*[
                "corrections", "sample_sets_remove_scatter",
                str(set_name), sample_name
            ]))

            # Perform Raman normalization
            eem = raman_normalization(eem, blank_eem)
            eem.to_hdf(hdf, key=os.path.join(*[
                "corrections", "sample_sets_raman_normalization",
                str(set_name), sample_name
            ]))

            # inner-filter effect correction
            # pyeema.ife_correction()

            # Knockout regions of EEM to simulate only having a few
            # discrete excitation wavelengths.
            '''
            eem = simulate_discrete_excitations(eem, discrete_ex_wl)
            eem.to_hdf(hdf, key=os.path.join(*[
                "corrections", "sample_sets_discrete_excitations",
                str(set_name), sample_name
            ]))
            '''
            # gaussian smoothing
            #eem = pyeema.gaussian_smoothing(eem, sig=2, trun=4)
            # eem = pyeema.plot_data(eem, sig=2, trun=4)

            sample_abs_name = sample_name.rsplit(".dat", 1)[0] + "_abs.dat"
            if sample_abs_name != row['filename']:
                sample_abs_name = np.nan

            # save all intermediate steps here
