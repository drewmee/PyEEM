from .corrections import *
from .filters import *


def __save_intermediate():
    return

def test():
    """Trying to get documentation to work
    """
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
