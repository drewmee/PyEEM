import os

import numpy as np
import pandas as pd
from scipy import stats

from pyeem.analysis.basic import fluorescence_regional_integration


def calibration(dataset, routine_results_df, step="complete"):
    """[summary]

    Args:
        dataset (pyeem.datasets.Dataset): [description]
        routine_results_df (pandas.DataFrame): [description]
        step (str, optional): [description]. Defaults to "complete".

    Returns:
        DataFrame: [description]
    """
    cal_df = pd.DataFrame()
    for source, units in dataset.calibration_sources.items():
        # Filter meta_df for calibration samples and the source's concentration
        # is greater than zero.
        meta_source_df = dataset.meta_df[
            (dataset.meta_df["calibration_sample"]) & (dataset.meta_df[source] > 0)
        ]

        # Drop the hdf_path column from the metadata, routine_results_df contains a
        # more relevant hdf_path column in this context.
        meta_source_df = meta_source_df.drop(columns=["hdf_path"])

        # Filter the routine_results_df step_name is equal to the passed
        # step type (defaults to the complete step).
        step_results = routine_results_df.xs(step, level="step_name")

        # Join the filtered meta_df with the filtered routine_results_df.
        meta_step_df = pd.merge(
            meta_source_df, step_results, on=["sample_set", "scan_type", "name"]
        )

        def _get_integrated_intensity(hdf_path):
            eem_df = pd.read_hdf(dataset.hdf, key=hdf_path)
            return fluorescence_regional_integration(eem_df, region_bounds=None)

        # Calculate the surface integral of the EEM.
        meta_step_df["integrated_intensity"] = (
            meta_step_df.xs("sample_eem", level="scan_type")["hdf_path"]
            .apply(_get_integrated_intensity)
            .values
        )
        meta_step_df.rename(columns={"units": "intensity_units"}, inplace=True)
        meta_step_df["measurement_units"] = "Integrated " + meta_step_df[
            "intensity_units"
        ].astype(str)

        # Reset index in preparation for re-indexing for the cal_df format.
        meta_step_df.reset_index(drop=True, inplace=True)

        # Store source into the source column, and rename the source's named column
        # to concentration.
        meta_step_df["source"] = source
        meta_step_df["source_units"] = units
        meta_step_df.rename(columns={source: "concentration"}, inplace=True)

        # Perform linear regression w/ concentration & surface integral
        (
            meta_step_df["slope"],
            meta_step_df["intercept"],
            meta_step_df["r_value"],
            _,
            _,
        ) = stats.linregress(
            meta_step_df["concentration"], meta_step_df["integrated_intensity"]
        )

        # Convert R value to R^2.
        meta_step_df["r_squared"] = meta_step_df["r_value"] ** 2

        # Set index to cal_df format.
        meta_step_df.set_index(
            [
                "source",
                "source_units",
                "intensity_units",
                "measurement_units",
                "slope",
                "intercept",
                "r_squared",
            ],
            inplace=True,
        )

        # Build up cal_df w/ each source.
        cal_df = pd.concat(
            [
                cal_df,
                meta_step_df[
                    [
                        "concentration",
                        "integrated_intensity",
                        "prototypical_sample",
                        "hdf_path",
                    ]
                ],
            ]
        )

    return cal_df


def calibration_summary_info(cal_df):
    """[summary]

    Args:
        cal_df (pandas.DataFrame): [description]

    Returns:
        DataFrame: [description]
    """
    summary_df = pd.DataFrame(
        cal_df.index.unique().values.tolist(), columns=list(cal_df.index.unique().names)
    )

    def _get_summary_info(row):
        source_df = cal_df.xs(row["source"], level="source")
        row["Number of Samples"] = source_df.shape[0]
        row["Min. Concentration"] = source_df["concentration"].min()
        row["Max. Concentration"] = source_df["concentration"].max()
        return row

    summary_df = summary_df.apply(_get_summary_info, axis=1)
    return summary_df
