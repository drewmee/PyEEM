def crop(eem_df, crop_dims):
    """[summary]

    Arguments:
        eem_df {[type]} -- [description]
        ex_range {[type]} -- [description]
        em_range {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    #  Rows (axis=0) are Emission wavelengths
    eem_df = eem_df.truncate(before=em_range[0], after=em_range[1], axis=0)

    # Columns (axis=1) are Excitation wavelengths
    eem_df = eem_df.truncate(before=ex_range[0], after=ex_range[1], axis=1)

    return eem_df


def simulate_discrete_excitations(eem_df, ex_wl):
    """[summary]

    Arguments:
        eem_df {[type]} -- [description]
        ex_wl {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    eem_tdf = eem_df.transpose()
    ilocs = []
    for wl in ex_wl:
        ilocs.append(eem_tdf.index.get_loc(wl, method="nearest"))
    eem_df = eem_tdf.iloc[ilocs].transpose()
    return eem_df


def gaussian_smoothing(eem_df, sig, trun):
    """This function does a gaussian_blurr on the 2D spectra image from
    the input sigma and truncation sigma.

    Arguments:
        eem_df {[type]} -- [description]
        sig {integer} -- Sigma of the gaussian distribution weight for 
        the data smoothing
        trun {integer} -- truncate in 'sigmas' the gaussian distribution

    Returns:
        [type] -- [description]
    """

    # smooth the data with the gaussian filter
    eem_blurred = gaussian_filter(eem_df.to_numpy(), sigma=sig, truncate=trun)
    return eem_blurred
