import itertools
import os
import random

import numpy as np
import pandas as pd


def prototypical_spectrum(dataset, source_df, aug_steps_df):
    """Weighted average of calibration spectra with randomly
    assigned weights between 0 and 1.

    Args:
        dataset ([type]): [description]
        source_df (DataFrame): [description]
        aug_steps_df (DataFrame): [description]

    Returns:
        DataFrame: [description]
    """
    source_name = source_df.index.get_level_values("source").unique().item()
    source_units = source_df.index.get_level_values("source_units").unique().item()
    intensity_units = (
        source_df.index.get_level_values("intensity_units").unique().item()
    )

    proto_eems = []
    for index, row in source_df.iterrows():
        eem_path = row["hdf_path"]
        eem = pd.read_hdf(dataset.hdf, key=eem_path)
        proto_eems.append(eem)

    proto_concentration = source_df[source_df["prototypical_sample"]][
        "concentration"
    ].mean()

    weights = []
    for i in range(len(proto_eems)):
        weights.append(random.uniform(0, 1))

    proto_eem = np.average([eem.values for eem in proto_eems], axis=0, weights=weights)

    proto_eem = pd.DataFrame(
        data=proto_eem, index=proto_eems[0].index, columns=proto_eems[0].columns
    )
    proto_eem.index.name = "emission_wavelength"

    hdf_path = aug_steps_df[aug_steps_df["step_name"] == "prototypical"][
        "hdf_path"
    ].item()
    hdf_path = os.path.join(hdf_path, source_name)

    new_indices = np.array(
        ["source", "proto_conc", "source_units", "intensity_units", "hdf_path"]
    )
    proto_eem = proto_eem.assign(
        **{
            "source": source_name,
            "proto_conc": proto_concentration,
            "source_units": source_units,
            "intensity_units": intensity_units,
            "hdf_path": hdf_path,
        }
    )
    proto_eem.set_index(new_indices.tolist(), append=True, inplace=True)
    new_indices = np.append(new_indices, ("emission_wavelength"))
    proto_eem = proto_eem.reorder_levels(new_indices)
    proto_eem.to_hdf(dataset.hdf, key=hdf_path)
    return proto_eem


def single_source(dataset, source_df, cal_df, aug_steps_df, conc_range, num_spectra):
    """[summary]

    Args:
        dataset ([type]): [description]
        source_df ([type]): [description]
        cal_df ([type]): [description]
        aug_steps_df ([type]): [description]
        conc_range (tuple of int or float): [description]
        num_spectra (int): [description]

    Returns:
        DataFrame: [description]
    """
    # Get the source's name
    source_name = source_df.index.get_level_values("source").unique().item()

    # Get the HDF5 path to the source's prototypical EEM
    proto_hdf_path = aug_steps_df[aug_steps_df["step_name"] == "prototypical"][
        "hdf_path"
    ].item()
    proto_hdf_path = os.path.join(proto_hdf_path, source_name)

    # Read in the prototypical EEM
    proto_eem = pd.read_hdf(dataset.hdf, key=proto_hdf_path)

    # Get the source's prototypical concentration
    proto_concentration = proto_eem.index.get_level_values("proto_conc").unique().item()

    # Remove the concentration index from the dataframe
    proto_eem.reset_index(level=["proto_conc"], drop=True, inplace=True)

    # Get the slope and intercept of the source's calibration function
    slope = (
        cal_df.xs(source_name, level="source")
        .index.get_level_values("slope")
        .unique()
        .item()
    )
    y_intercept = (
        cal_df.xs(source_name, level="source")
        .index.get_level_values("intercept")
        .unique()
        .item()
    )

    # Generate the 1D polynomial
    cal_func = np.poly1d([slope, y_intercept])

    # Generate the concentration range based on the argument's
    concentration_range = np.linspace(conc_range[0], conc_range[1], num=num_spectra)

    # Create a new HDF5 path for the single source spectra
    hdf_path = aug_steps_df[aug_steps_df["step_name"] == "single_sources"][
        "hdf_path"
    ].item()
    hdf_path = os.path.join(hdf_path, source_name)

    # aug_ss_dfs: A list which we will iteratively append single source spectra to. For each
    # concentration in the concentration range. Then we will turn the list of DFs
    # into a single DF by using concat()
    aug_ss_dfs = []
    sources = list(dataset.calibration_sources)
    for new_concentration in concentration_range:
        scalar = cal_func(new_concentration) / cal_func(proto_concentration)
        ss_eem = proto_eem * scalar
        # Make sure there are no negative values
        ss_eem.clip(lower=0, inplace=True)
        label = np.zeros(len(sources))
        source_index = sources.index(source_name)
        label[source_index] = new_concentration

        ss_eem.index.name = "emission_wavelength"
        ss_eem = ss_eem.assign(**dict(zip(sources, label)))

        new_indices = sources
        ss_eem.set_index(new_indices, append=True, inplace=True)
        new_indices = [
            "source",
            "source_units",
            "intensity_units",
            "hdf_path",
        ] + new_indices
        new_indices.append("emission_wavelength")
        ss_eem = ss_eem.reorder_levels(new_indices)
        ss_eem.rename(index={proto_hdf_path: hdf_path}, inplace=True)
        aug_ss_dfs.append(ss_eem)

    aug_ss_df = pd.concat(aug_ss_dfs)
    aug_ss_df.to_hdf(dataset.hdf, key=hdf_path)
    return aug_ss_df


'''
def mixtures(sources, cal_df, hdf, conc_range, num_steps, scale="linear"):
    """[summary]

    Args:
        sources ([type]): [description]
        cal_df (~pandas.DataFrame): [description]
        hdf (pandas.io.pytables.HDFStore): [description]
        conc_range (tuple of int or float): [description]
        num_steps (int): [description]
        scale (str, optional): [description]. Defaults to "linear".

    Raises:
        ValueError: [description]

    Returns:
        DataFrame: [description]
    """
    # conc_range=(0.01, 6.3), num_steps=15, scale = logarithmic

    proto_spectra = []
    for source in sources:
        proto_eem = pd.read_hdf(
            hdf, key=os.path.join(*["augmented", "prototypical_spectra", source])
        )
        proto_spectra.append(proto_eem)
    proto_eem_df = pd.concat(proto_spectra)

    if scale == "logarithmic":
        number_range = np.geomspace(conc_range[0], conc_range[1], num=num_steps)
    elif scale == "linear":
        number_range = np.linspace(conc_range[0], conc_range[1], num=num_steps)
    else:
        raise ValueError("scale must be 'logarithmic' or 'linear'")

    cartesian_product = [p for p in itertools.product(number_range.tolist(), repeat=3)]

    aug = []
    for conc_set in cartesian_product:
        mix = []
        for index, label in enumerate(zip(sources, conc_set)):
            source_name = label[0]
            new_concentration = label[1]

            c = cal_df[cal_df["source"] == source_name]
            source_cal_coeffs = (
                c.loc[:, c.columns.str.startswith("cal_func_term")].iloc[0].values
            )
            cal_func = np.poly1d(source_cal_coeffs)

            proto_eem = proto_eem_df.xs(source_name, level="source", drop_level=False)

            proto_concentration = (
                proto_eem.index.get_level_values("proto_conc").unique().item()
            )
            proto_eem.reset_index(level=["proto_conc"], drop=True, inplace=True)

            scalar = cal_func(new_concentration) / cal_func(proto_concentration)
            new_eem = proto_eem * scalar
            mix.append(new_eem)

        mix_eem = pd.concat(mix).sum(level="emission_wavelength")
        mix_eem = mix_eem.assign(**dict(zip(sources, conc_set)))

        new_indices = sources
        mix_eem.set_index(new_indices.tolist(), append=True, inplace=True)
        new_indices = np.append(new_indices, ("emission_wavelength"))
        mix_eem = mix_eem.reorder_levels(new_indices)
        aug.append(mix_eem)

    aug_mix_df = pd.concat(aug)
    aug_mix_df.to_hdf(hdf, key=os.path.join("augmented", "mixtures"))
    return aug_mix_df
'''
