import itertools
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import prototypical_spectrum, single_source


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


def create_prototypical_spectra(dataset, cal_df):
    """[summary]

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        cal_df (pandas.DataFrame): [description]

    Returns:
        DataFrame: [description]
    """
    aug_steps_df = _get_steps()

    results_rows = []
    for source_name, group in cal_df.groupby(level="source", as_index=False):
        proto_eem_df = prototypical_spectrum(dataset, group, aug_steps_df)
        new_indices = proto_eem_df.index.droplevel("emission_wavelength").unique()
        result = dict(zip(list(new_indices.names), list(new_indices.item())))
        results_rows.append(result)

    results_df = pd.DataFrame(results_rows)
    results_index = "source"
    results_df.set_index(results_index, inplace=True)
    return results_df


def create_single_source_spectra(dataset, cal_df, conc_range, num_spectra):
    """[summary]

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        cal_df (pandas.DataFrame): [description]
        conc_range (tuple of (int, float)): [description]
        num_spectra (int): [description]

    Returns:
        DataFrame: [description]
    """

    aug_steps_df = _get_steps()
    aug_ss_dfs = []
    for source_name, group in tqdm(cal_df.groupby(level="source", as_index=False)):
        ss_df = single_source(
            dataset,
            group,
            cal_df,
            aug_steps_df,
            conc_range=conc_range,
            num_spectra=num_spectra,
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


def create_mixtures(dataset, cal_df, conc_range, num_steps, scale="logarithmic"):
    """[summary]

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        cal_df (pandas.DataFrame): [description]
        conc_range (tuple of (int, float)): [description]
        num_steps (int): [description]
        scale (str, optional): [description]. Defaults to "logarithmic".

    Raises:
        Exception: [description]
        ValueError: [description]

    Returns:
        DataFrame: [description]
    """
    if cal_df.index.get_level_values("source_units").nunique() != 1:
        raise Exception(
            "Sources are must reported in the same units in order create augmented mixtures."
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

    cartesian_product = [p for p in itertools.product(number_range.tolist(), repeat=3)]
    aug = []
    for conc_set in tqdm(cartesian_product, desc="Creating Augmented Mixtures"):
        mix = []
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
