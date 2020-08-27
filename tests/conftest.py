import shutil

import pytest

import pyeem


@pytest.fixture(scope="session", autouse=True)
def tmp_dir_fixture(tmpdir_factory):
    # setup section
    #tmp_data_dir = tmpdir_factory.mktemp("demo_data")
    tmp_data_dir = "local_test_data"
    yield tmp_data_dir
    # teardown section
    if tmp_data_dir != "local_test_data":
        shutil.rmtree(tmp_data_dir.strpath)


def _get_rutherford_dataset(data_dir):
    demo_data_dir = pyeem.datasets.download_demo(data_dir, demo_name="rutherford")
    calibration_sources = {
        "cigarette": "ug/ml",
        "diesel": "ug/ml",
        "wood_smoke": "ug/ml",
    }
    dataset = pyeem.datasets.Dataset(
        data_dir=demo_data_dir,
        raman_instrument=None,
        absorbance_instrument="aqualog",
        eem_instrument="aqualog",
        calibration_sources=calibration_sources,
    )
    return dataset


def _get_dreem_dataset(data_dir):
    demo_data_dir = pyeem.datasets.download_demo(data_dir, demo_name="drEEM")
    dataset = pyeem.datasets.Dataset(
        data_dir=demo_data_dir,
        raman_instrument="fluorolog",
        absorbance_instrument="cary_4e",
        eem_instrument="fluorolog",
    )
    return dataset


@pytest.fixture(scope="session", autouse=True)
def demo_datasets(tmp_dir_fixture):
    demo_dataset_dict = {
        "rutherford": _get_rutherford_dataset(tmp_dir_fixture),
        "drEEM": _get_dreem_dataset(tmp_dir_fixture),
    }
    return demo_dataset_dict


@pytest.fixture(scope="session", autouse=True)
def demo_preprocessed_dataset(tmp_dir_fixture):
    dataset = _get_rutherford_dataset(tmp_dir_fixture)

    routine_df = pyeem.preprocessing.create_routine(
        crop=True,
        discrete_wavelengths=False,
        gaussian_smoothing=False,
        blank_subtraction=True,
        inner_filter_effect=True,
        raman_normalization=True,
        scatter_removal=True,
        dilution=False,
    )

    crop_dimensions = {
        "emission_bounds": (246, 573),
        "excitation_bounds": (224, float("inf")),
    }
    routine_results_df = pyeem.preprocessing.perform_routine(
        dataset,
        routine_df,
        crop_dims=crop_dimensions,
        raman_source_type="metadata",
        fill="interp",
        progress_bar=True,
    )
    return dataset, routine_results_df


@pytest.fixture(scope="session", autouse=True)
def demo_calibration(tmp_dir_fixture, demo_preprocessed_dataset):
    dataset, routine_results_df = demo_preprocessed_dataset
    cal_df = pyeem.preprocessing.calibration(dataset, routine_results_df)
    return cal_df


@pytest.fixture(scope="session", autouse=True)
def demo_augmentation(tmp_dir_fixture, demo_preprocessed_dataset, demo_calibration):
    dataset, routine_results_df = demo_preprocessed_dataset
    cal_df = demo_calibration
    proto_results_df = pyeem.augmentation.create_prototypical_spectra(dataset, cal_df)
    ss_results_df = pyeem.augmentation.create_single_source_spectra(
        dataset, cal_df, conc_range=(0, 5), num_spectra=10
    )
    mix_results_df = pyeem.augmentation.create_mixtures(
        dataset, cal_df, conc_range=(0.01, 6.3), num_steps=5
    )
    return proto_results_df, ss_results_df, mix_results_df


@pytest.fixture(scope="session", autouse=True)
def demo_rutherfordnet(
    tmp_dir_fixture, demo_preprocessed_dataset, demo_calibration, demo_augmentation
):
    dataset, routine_results_df = demo_preprocessed_dataset
    cal_df = demo_calibration
    (_, ss_results_df, mix_results_df,) = demo_augmentation

    rutherfordnet = pyeem.analysis.models.RutherfordNet()
    (x_train, y_train), (x_test, y_test) = rutherfordnet.prepare_data(
        dataset, ss_results_df, mix_results_df, routine_results_df
    )
    rutherfordnet.train(x_train, y_train)
    return rutherfordnet
