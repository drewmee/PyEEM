import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import corrections, filters


def _get_steps():
    steps_df = pd.concat([filters._get_steps(), corrections._get_steps()])
    steps = [
        {"step_name": "raw", "hdf_path": "raw_sample_sets/"},
        {"step_name": "complete", "hdf_path": "preprocessing/complete/"},
    ]
    steps_df = steps_df.append(pd.DataFrame.from_records(steps), ignore_index=True)

    # Impose order on the preprocessing steps:
    steps_df["step_name"] = pd.Categorical(
        steps_df["step_name"],
        [
            "raw",
            "crop",
            "fill_missing_values",
            "blank_subtraction",
            "inner_filter_effect",
            "raman_normalization",
            "scatter_removal",
            "dilution",
            "gaussian_smoothing",
            "discrete_wavelengths",
            "complete",
        ],
    )
    steps_df.sort_values("step_name", inplace=True, ignore_index=True)
    return steps_df


def create_routine(
    crop=False,
    fill_missing_values=False,
    discrete_wavelengths=False,
    gaussian_smoothing=False,
    blank_subtraction=False,
    inner_filter_effect=False,
    raman_normalization=False,
    scatter_removal=False,
    dilution=False,
):
    """Create the preprocessing routine for correcting and filtering the data.

    Args:
        crop (bool, optional): Determines whether or not to include
            :meth:`~pyeem.preprocessing.filters.crop` in the routine. Defaults to False.
        fill_missing_values (bool, optional): Determines whether or not to include
            :meth:`~pyeem.preprocessing.filters.fill_missing_values` in the routine. Defaults to False.
        discrete_wavelengths (bool, optional): Determines whether or not to include
            :meth:`~pyeem.preprocessing.filters.filter_wavelengths` in the routine. Defaults to False.
        gaussian_smoothing (bool, optional): Determines whether or not to include
            :meth:`~pyeem.preprocessing.filters.gaussian_smoothing` in the routine. Defaults to False.
        blank_subtraction (bool, optional): Determines whether or not to include
            :meth:`~pyeem.preprocessing.corrections.blank_subtraction` in the routine. Defaults to False.
        inner_filter_effect (bool, optional): Determines whether or not to include
            :meth:`~pyeem.preprocessing.corrections.inner_filter_effect` in the routine. Defaults to False.
        raman_normalization (bool, optional): Determines whether or not to include
            :meth:`~pyeem.preprocessing.corrections.raman_normalization` in the routine. Defaults to False.
        scatter_removal (bool, optional): Determines whether or not to include
            :meth:`~pyeem.preprocessing.corrections.scatter_removal` in the routine. Defaults to False.
        dilution (bool, optional): Determines whether or not to include
            :meth:`~pyeem.preprocessing.corrections.dilution` in the routine. Defaults to False.

    Returns:
        pandas.DataFrame: A table containing the steps of the preprocessing routine in the order
        which they will be executed by pyeem.preprocessing.perform_routine().
    """
    raw = True
    complete = True
    steps = locals()
    steps = {k: v for k, v in steps.items() if v}

    steps_df = _get_steps()
    routine_df = steps_df[steps_df["step_name"].isin(steps)].reset_index(drop=True)
    routine_df.index.name = "step_order"
    return routine_df


def _generate_result_row(
    scan_type, name, step_name, step_completed, step_exception, hdf_path
):
    result_row = {
        "scan_type": scan_type,
        "name": name,
        "step_name": step_name,
        "step_completed": step_completed,
        "step_exception": step_exception,
        "hdf_path": hdf_path,
    }
    return result_row


def _create_results_df():
    return


def _process_blank():
    return


def _process_sample():
    return


def _assign_units(steps_group):
    def _initial_label(row):
        step_name = row.name[2]
        if not row["step_completed"]:
            units = None
        elif step_name == "raman_normalization":
            units = "Intensity, RU"
        else:
            units = "Intensity, AU"
        return units

    steps_group["units"] = steps_group.apply(_initial_label, axis=1)

    raman_str = "raman_normalization"
    if raman_str in steps_group.index.get_level_values("step_name").values:
        step_row = steps_group.xs(raman_str, level="step_name", drop_level=False)
        if step_row["step_completed"].item():
            units = step_row["units"].item()
            steps_group = steps_group.reset_index()
            raman_idx = steps_group.index[steps_group["step_name"] == raman_str].item()
            steps_group.loc[
                (steps_group.index > raman_idx) & (steps_group["step_completed"]),
                ("units"),
            ] = units
            steps_group = steps_group.set_index(["scan_type", "name", "step_name"])

    return steps_group


def _process_sample_set(sample_set_group, dataset, routine_df, **kwargs):
    sample_set = sample_set_group.index.get_level_values("sample_set").unique().item()

    # TODO - Breakout into another function - _create_results_df()
    results_columns = [
        "scan_type",
        "name",
        "step_name",
        "step_completed",
        "step_exception",
        "hdf_path",
    ]
    results_df = pd.DataFrame(columns=results_columns)
    results_indices = ["scan_type", "name", "step_name"]
    results_df.set_index(results_indices, inplace=True)

    # Handle the blank scan corresponding with the sample_set
    # TODO - Breakout blank handling into a helper function
    blank_name = None
    if "blank_eem" in sample_set_group.index.get_level_values("scan_type"):
        blank_name = sample_set_group.xs("blank_eem", level="scan_type")[
            "name"
        ].unique()[0]

    if "water_raman" in sample_set_group.index.get_level_values("scan_type"):
        water_raman_wavelength = kwargs.get("water_raman_wavelength", None)
        water_raman_scans = sample_set_group.xs("water_raman", level="scan_type")
        water_raman_scans = water_raman_scans[
            water_raman_scans["water_raman_wavelength"] == water_raman_wavelength
        ]
        if not water_raman_scans.empty:
            water_raman_name = water_raman_scans["name"].unique()[0]

    # Perform loading of blank EEM
    step_name = "raw"
    try:
        hdf_path = routine_df[routine_df["step_name"] == step_name]["hdf_path"].item()
        hdf_path = os.path.join(*[hdf_path, str(sample_set), blank_name])
        blank_df = pd.read_hdf(dataset.hdf, key=hdf_path)
        step_completed = True
        step_exception = None
    except Exception as e:
        blank_df = None
        step_completed = False
        step_exception = e
        hdf_path = None

    results_row = _generate_result_row(
        "blank_eem", blank_name, step_name, step_completed, step_exception, hdf_path
    )
    row_df = pd.DataFrame([results_row])
    row_df.set_index(results_indices, inplace=True)
    results_df = pd.concat([results_df, row_df])

    # Perform cropping of blank EEM
    step_name = "crop"
    if step_name in routine_df["step_name"].values:
        try:
            hdf_path = routine_df[routine_df["step_name"] == "crop"]["hdf_path"].item()
            hdf_path = os.path.join(*[hdf_path, str(sample_set), blank_name])
            crop_dims = kwargs.get("crop_dims", None)
            blank_df = filters.crop(blank_df, crop_dims)
            blank_df.to_hdf(dataset.hdf, key=hdf_path)
            step_completed = True
            step_exception = None

        except Exception as e:
            blank_df = None
            step_completed = False
            step_exception = e
            hdf_path = None

        results_row = _generate_result_row(
            "blank_eem", blank_name, step_name, step_completed, step_exception, hdf_path
        )

        row_df = pd.DataFrame([results_row])
        row_df.set_index(results_indices, inplace=True)
        results_df = pd.concat([results_df, row_df])

    # Perform filling of missing values
    step_name = "fill_missing_values"
    if step_name in routine_df["step_name"].values:
        try:
            hdf_path = routine_df[routine_df["step_name"] == "fill_missing_values"][
                "hdf_path"
            ].item()
            hdf_path = os.path.join(*[hdf_path, str(sample_set), blank_name])
            fill = kwargs.get("fill", None)
            blank_df = filters.fill_missing_values(blank_df, fill)
            blank_df.to_hdf(dataset.hdf, key=hdf_path)
            step_completed = True
            step_exception = None

        except Exception as e:
            blank_df = None
            step_completed = False
            step_exception = e
            hdf_path = None

        results_row = _generate_result_row(
            "blank_eem", blank_name, step_name, step_completed, step_exception, hdf_path
        )

        row_df = pd.DataFrame([results_row])
        row_df.set_index(results_indices, inplace=True)
        results_df = pd.concat([results_df, row_df])

    step_name = "raman_normalization"
    if step_name in routine_df["step_name"].values:
        raman_source_type = kwargs.get("raman_source_type", None)
        raman_source = None
        if raman_source_type == "water_raman":
            try:
                hdf_path = os.path.join(
                    *["raw_sample_sets", str(sample_set), water_raman_name]
                )
                raman_source = pd.read_hdf(dataset.hdf, key=hdf_path)
            except Exception as e:
                pass

        elif raman_source_type == "blank":
            raman_source = blank_df

        elif raman_source_type == "metadata":
            pass

    # If there are no samples within this sample set, return the results_df as is.
    if "sample_eem" not in sample_set_group.index.get_level_values("scan_type"):
        results_df = results_df.groupby(level=["name"]).apply(_assign_units)
        return results_df

    # TODO - Refactor to use groupby/apply or just break it out into another function
    results_rows = []
    for sample_index, sample_row in sample_set_group.xs(
        "sample_eem", level="scan_type"
    ).iterrows():
        sample_name = sample_row["name"]
        step_name = "raw"
        hdf_path = routine_df[routine_df["step_name"] == step_name]["hdf_path"].item()
        hdf_path = os.path.join(*[hdf_path, str(sample_set), sample_name])
        try:
            eem_df = pd.read_hdf(dataset.hdf, key=hdf_path)
        except Exception as e:
            eem_df = None
            step_completed = False
            step_exception = e
            hdf_path = None

        # TODO - consider using apply or breaking out into another function.
        for routine_index, routine_row in routine_df.iterrows():
            try:
                step_name = routine_row["step_name"]
                hdf_path = routine_row["hdf_path"]
                hdf_path = os.path.join(*[hdf_path, str(sample_set), sample_name])

                if step_name == "raw":
                    pass

                elif step_name == "complete":
                    pass

                elif step_name == "crop":
                    crop_dims = kwargs.get("crop_dims", None)
                    eem_df = filters.crop(eem_df, crop_dims)

                elif step_name == "fill_missing_values":
                    fill = kwargs.get("fill", None)
                    eem_df = filters.fill_missing_values(eem_df, fill)

                elif step_name == "blank_subtraction":
                    eem_df = corrections.blank_subtraction(eem_df, blank_df)

                elif step_name == "inner_filter_effect":
                    sample_abs_name = sample_name.replace("sample_eem", "absorb")
                    abs_hdf_path = routine_df[routine_df["step_name"] == "raw"][
                        "hdf_path"
                    ].item()
                    abs_hdf_path = os.path.join(
                        *[abs_hdf_path, str(sample_set), sample_abs_name]
                    )
                    abs_df = pd.read_hdf(dataset.hdf, key=abs_hdf_path)
                    eem_df = corrections.inner_filter_effect(
                        eem_df,
                        abs_df,
                        pathlength=kwargs.get("pathlength", 1),
                        threshold=kwargs.get("threshold", 0.03),
                        limit=kwargs.get("limit", 1.5),
                    )

                elif step_name == "raman_normalization":
                    if raman_source_type == "metadata":
                        raman_source = sample_row["water_raman_area"]
                        water_raman_wavelength = None

                    eem_df = corrections.raman_normalization(
                        eem_df, raman_source_type, raman_source, water_raman_wavelength
                    )

                elif step_name == "scatter_removal":
                    eem_df = corrections.scatter_removal(
                        eem_df,
                        band=kwargs.get("band", "both"),
                        order=kwargs.get("order", "both"),
                        excision_width=kwargs.get("excision_width", 20),
                        fill=kwargs.get("fill", "interp"),
                        truncate=kwargs.get("truncate", None),
                    )

                elif step_name == "dilution":
                    dilution_factor = sample_row["dilution_factor"]
                    eem_df = corrections.dilution(eem_df, dilution_factor)

                elif step_name == "gaussian_smoothing":
                    # TODO get default values of sigma and truncate
                    eem_df = filters.gaussian_smoothing(
                        eem_df,
                        sigma=kwargs.get("sigma", 2),
                        truncate=kwargs.get("truncate", 4),
                    )

                elif step_name == "discrete_wavelengths":
                    # TODO create datastructure to pass selected wavelengths
                    # similar to crop_dims in crop()
                    selected_wavelengths = None
                    eem_df = filters.discrete_excitations(eem_df, selected_wavelengths)

                eem_df.to_hdf(dataset.hdf, key=hdf_path)
                step_completed = True
                step_exception = None

            except Exception as e:
                step_completed = False
                step_exception = e
                hdf_path = None

            results_row = _generate_result_row(
                "sample_eem",
                sample_name,
                step_name,
                step_completed,
                step_exception,
                hdf_path,
            )
            results_rows.append(results_row)

    row_df = pd.DataFrame(results_rows)
    row_df.set_index(results_indices, inplace=True)
    results_df = pd.concat([results_df, row_df])
    results_df = results_df.groupby(level=["name"]).apply(_assign_units)
    return results_df


def perform_routine(dataset, routine_df, progress_bar=False, **kwargs):
    """Perform the data preprocessing routine which was created by
    pyeem.preprocessing.create_routine()

    Args:
        dataset ([type]): [description]
        routine_df ([type]): [description]
        progress_bar (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    if progress_bar:
        tqdm.pandas(desc="Preprocessing scan sets")
        results_df = dataset.meta_df.groupby(level="sample_set").progress_apply(
            _process_sample_set, dataset=dataset, routine_df=routine_df, **kwargs
        )
    else:
        results_df = dataset.meta_df.groupby(level="sample_set").apply(
            _process_sample_set, dataset, routine_df, **kwargs
        )
    return results_df
