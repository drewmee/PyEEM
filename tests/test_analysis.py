import os  # isort:skip

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pytest
import tensorflow as tf

import pyeem


class TestModels:
    @pytest.fixture(autouse=True)
    def setup(
        self,
        demo_datasets,
        demo_preprocessed_dataset,
        demo_calibration,
        demo_augmentation,
    ):
        self.demo_datasets = demo_datasets
        self.preprocessed_dataset, self.routine_results_df = demo_preprocessed_dataset
        self.cal_df = demo_calibration
        (
            self.proto_results_df,
            self.ss_results_df,
            self.mix_results_df,
        ) = demo_augmentation

    def testRutherfordnet(self):
        rutherfordnet = pyeem.analysis.models.RutherfordNet()
        assert isinstance(rutherfordnet, pyeem.analysis.models.RutherfordNet)
        assert isinstance(
            rutherfordnet.model, tf.python.keras.engine.sequential.Sequential
        )
        # rutherfordnet.model.summary()
        # Make sure the model is shaped how it should be

        (x_train, y_train), (x_test, y_test) = rutherfordnet.prepare_data(
            self.preprocessed_dataset,
            self.ss_results_df,
            self.mix_results_df,
            self.routine_results_df,
        )
        assert all(
            isinstance(i, np.ndarray) for i in [x_train, y_train, x_test, y_test]
        )
        # assert x_train.shape[?] == y_train.shape[?]
        # assert x_test.shape == y_test.shape
        history = rutherfordnet.train(x_train, y_train)
        assert isinstance(history, tf.python.keras.callbacks.History)

        predictions = rutherfordnet.model.predict(x_train)
        assert isinstance(predictions, np.ndarray)


class TestBasic:
    @pytest.fixture(autouse=True)
    def setup(self, demo_datasets):
        self.demo_datasets = demo_datasets

    def testFluorescenceRegionalIntegration(self):
        return
