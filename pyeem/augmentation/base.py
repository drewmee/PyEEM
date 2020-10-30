import itertools
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm


def _get_steps():
    hdf_subdir = "augmentation/"
    steps = {"step_name": ["prototypical", "single_sources", "mixtures"]}
    steps_df = pd.DataFrame(steps)
    steps_df["hdf_path"] = hdf_subdir + steps_df["step_name"]
    # Impose order on the augmentation steps:
    steps_df["step_name"] = pd.Categorical(
        steps_df["step_name"], ["prototypical", "single_sources", "mixtures"],
    )
    steps_df.sort_values("step_name", inplace=True, ignore_index=True)
    return steps_df


def prototypical_spectrum(dataset, source_df):
    """Weighted average of calibration spectra with randomly assigned weights
    between 0 and 1.

    Args:
        dataset (pyeem.datasets.Dataset): Your PyEEM dataset.
        source_df (pandas.DataFrame): Calibration information for a single source.

    Returns:
        pandas.DataFrame: A prototypical Excitation Emission Matrix for a single source.
    """
    aug_steps_df = _get_steps()

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

    # TODO - IMPORTANT: This can't just be the mean of the prototypical samples...
    # Need to use the same weighted average as the intensity values!
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


def create_prototypical_spectra(dataset, cal_df):
    """Creates a protoypical spectrum for each calibration source in the PyEEM 
    dataset. 

    Args:
        dataset (pyeem.datasets.Dataset): Your PyEEM dataset.
        cal_df (pandas.DataFrame): Calibration information for your dataset 
            returned from :meth:`pyeem.preprocessing.calibration()`

    Returns:
        pandas.DataFrame: A table describing the prototypical spectra and their
        paths within the HDF5 store.
    """

    results_rows = []
    for source_name, group in cal_df.groupby(level="source", as_index=False):
        proto_eem_df = prototypical_spectrum(dataset, group)
        new_indices = proto_eem_df.index.droplevel("emission_wavelength").unique()
        result = dict(zip(list(new_indices.names), list(new_indices.item())))
        results_rows.append(result)

    results_df = pd.DataFrame(results_rows)
    results_index = "source"
    results_df.set_index(results_index, inplace=True)
    return results_df


def single_source(dataset, source_df, conc_range, num_spectra):
    """Creates augmented single source spectra for a single calibration source.

    Args:
        dataset (pyeem.datasets.Dataset): Your PyEEM dataset.
        source_df (pandas.DataFrame): Calibration information for a single source.
        conc_range (tuple of (int, float)): The concentration range which the 
            augmented single source spectra will occupy.
        num_spectra (int): The number of augmented single source spectra to create.

    Returns:
        pandas.DataFrame: A table describing the source's augmented spectra and their
        paths within the HDF5 store.
    """
    aug_steps_df = _get_steps()
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
    slope = source_df.index.get_level_values("slope").unique().item()
    y_intercept = source_df.index.get_level_values("intercept").unique().item()
    """
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
    """
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


def create_single_source_spectra(dataset, cal_df, conc_range, num_spectra):
    """Creates augmented single source spectra for each calibration source in the
    PyEEM dataset.

    Args:
        dataset (pyeem.datasets.Dataset): Your PyEEM dataset.
        cal_df (pandas.DataFrame): Calibration information for your dataset 
            returned from :meth:`pyeem.preprocessing.calibration()`
        conc_range (tuple of (int, float)): The concentration range which the 
            augmented single source spectra will occupy.
        num_spectra (int): The number of augmented single source spectra for each
            calibration source.

    Returns:
        pandas.DataFrame: A table describing the augmented single source spectra
        and their paths within the HDF5 store.
    """

    aug_ss_dfs = []
    for source_name, group in tqdm(cal_df.groupby(level="source", as_index=False)):
        ss_df = single_source(
            dataset, group, conc_range=conc_range, num_spectra=num_spectra,
        )
        ss_df = (
            ss_df.index.droplevel(["emission_wavelength"])
            .unique()
            .to_frame()
            .reset_index(drop=True)
        )
        ss_df.set_index(
            ["source", "source_units", "intensity_units", "hdf_path"], inplace=True
        )
        aug_ss_dfs.append(ss_df)

    aug_ss_df = pd.concat(aug_ss_dfs)
    return aug_ss_df


"""
def mixture():
    return
"""


def create_mixture_spectra(dataset, cal_df, conc_range, num_steps, scale="logarithmic"):
    """Creates augmented mixture spectra by summing together augmented single source spectra.
    The number of augmented mixtures created is equal to the Cartesian product composed of...

    Args:
        dataset (pyeem.datasets.Dataset): Your PyEEM dataset.
        cal_df (pandas.DataFrame): Calibration information for your dataset 
            returned from :meth:`pyeem.preprocessing.calibration()`
        conc_range (tuple of (int, float)): The concentration range which the 
            augmented spectra mixtures will occupy.
        num_steps (int): The number of intervals within the concentration range.
        scale (str, optional): Determines how the concentrations will be spaced along
            the given concentration range. Options are "linear" and "logarithmic". Defaults to "logarithmic".

    Raises:
        Exception: Raised if calibration sources are reported in different units.
        ValueError: Raised if the scale argument is a value other than linear" or "logarithmic".

    Returns:
        pandas.DataFrame: A table describing the augmented mixture spectra
        and their paths within the HDF5 store.
    """
    if cal_df.index.get_level_values("source_units").nunique() != 1:
        raise Exception(
            "Sources must be reported in the same units in order create augmented mixtures."
        )

    sources = cal_df.index.get_level_values(level="source").unique().to_list()
    source_units = cal_df.index.get_level_values("source_units").unique().item()
    intensity_units = (
        cal_df.index.get_level_values(level="intensity_units").unique().item()
    )

    aug_steps_df = _get_steps()

    hdf_path = aug_steps_df[aug_steps_df["step_name"] == "mixtures"]["hdf_path"].item()

    proto_spectra = []
    for source_name, group in cal_df.groupby(level="source", as_index=False):
        # Get the HDF5 path to the source's prototypical EEM
        proto_hdf_path = aug_steps_df[aug_steps_df["step_name"] == "prototypical"][
            "hdf_path"
        ].item()
        proto_hdf_path = os.path.join(proto_hdf_path, source_name)
        # Read in the prototypical EEM
        proto_eem = pd.read_hdf(dataset.hdf, key=proto_hdf_path)
        proto_spectra.append(proto_eem)

    proto_eem_df = pd.concat(proto_spectra)

    if scale == "logarithmic":
        number_range = np.geomspace(conc_range[0], conc_range[1], num=num_steps)
    elif scale == "linear":
        number_range = np.linspace(conc_range[0], conc_range[1], num=num_steps)
    else:
        raise ValueError("scale must be 'logarithmic' or 'linear'")

    cartesian_product = [
        p for p in itertools.product(number_range.tolist(), repeat=len(sources))
    ]
    aug = []
    for conc_set in tqdm(cartesian_product, desc="Creating Augmented Mixtures"):
        mix = []
        # TODO - it'd be a good idea to break this out into another function.
        # Call it mixture() -- returns a single mixture EEM
        for index, label in enumerate(zip(sources, conc_set)):
            source_name = label[0]
            new_concentration = label[1]

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
            cal_func = np.poly1d([slope, y_intercept])

            proto_eem = proto_eem_df.xs(source_name, level="source", drop_level=False)
            proto_concentration = (
                proto_eem.index.get_level_values("proto_conc").unique().item()
            )
            proto_eem.reset_index(level=["proto_conc"], drop=True, inplace=True)

            scalar = cal_func(new_concentration) / cal_func(proto_concentration)
            new_eem = proto_eem * scalar
            # Make sure there are no negative values
            new_eem.clip(lower=0, inplace=True)
            mix.append(new_eem)

        mix_eem = pd.concat(mix).sum(level="emission_wavelength")
        mix_eem = mix_eem.assign(**dict(zip(sources, conc_set)))
        mix_eem["hdf_path"] = hdf_path
        mix_eem["source"] = "mixture"
        mix_eem["source_units"] = source_units
        mix_eem["intensity_units"] = intensity_units
        new_indices = [
            "source",
            "source_units",
            "intensity_units",
            "hdf_path",
        ] + sources
        mix_eem.set_index(new_indices, append=True, inplace=True)
        new_indices = np.append(new_indices, ("emission_wavelength"))
        mix_eem = mix_eem.reorder_levels(new_indices)
        aug.append(mix_eem)

    aug_mix_df = pd.concat(aug)
    aug_mix_df.to_hdf(dataset.hdf, key=hdf_path)
    aug_mix_df = (
        aug_mix_df.index.droplevel(["emission_wavelength"])
        .unique()
        .to_frame()
        .reset_index(drop=True)
    )
    aug_mix_df.set_index(
        ["source", "source_units", "intensity_units", "hdf_path"], inplace=True
    )
    return aug_mix_df
