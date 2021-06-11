import os  # isort:skip

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

# from tensorflow.keras.optimizers import Adam


class RutherfordNet:
    """The convolutional neural network (CNN) described in Rutherford et al. 2020."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", None)
        self.input_shape = kwargs.get("input_shape", [142, 139, 1])
        self.output_dense_units = kwargs.get("output_dense_units", 3)
        self.compile_kws = kwargs.get("compile_kws", {})
        self.model = self.create_model(
            name=self.name,
            input_shape=self.input_shape,
            output_dense_units=self.output_dense_units,
            compile_kws=self.compile_kws,
        )

    def create_model(
        self,
        name="rutherfordnet",
        input_shape=[142, 139, 1],
        output_dense_units=3,
        compile_kws={},
    ):
        """Builds and compiles the CNN.

        Args:
            name (str, optional): The name of the model. Defaults to "rutherfordnet".
            compile_kws (dict, optional): Additional keyword arguments which
                will be passed to tensorflow.keras.Model.compile(). Defaults to {}.

        Returns:
            tensorflow.keras.Model: The compiled CNN model.
        """
        model = Sequential(name=name)

        # Convolution layers
        # first layer
        model.add(
            Conv2D(
                20, (5, 5), padding="same", input_shape=input_shape, activation="elu"
            )
        )
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.2))

        # second layer
        model.add(Conv2D(10, (10, 10), padding="same", activation="elu"))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.2))

        # third layer
        model.add(Conv2D(10, (15, 15), padding="same", activation="elu"))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Flatten())  # converts 3D feature maps to 1D feature maps
        model.add(Dropout(0.2))

        # Dense Layers
        model.add(Dense(512, activation="elu"))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation="elu"))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation="elu"))

        # Output layer
        model.add(Dropout(0.2))
        model.add(Dense(output_dense_units, activation="linear"))

        default_compile_kws = dict(
            loss="mean_squared_error", optimizer="adam", metrics=["accuracy"]
        )
        """
        opt = Adam(learning_rate=0.0001)
        default_compile_kws = dict(
            loss="mean_squared_error", optimizer=opt, metrics=["accuracy"]
        )
        """
        compile_kws = dict(default_compile_kws, **compile_kws)
        model.compile(**compile_kws)
        return model

    def get_training_data(self, dataset, ss_results_df, mix_results_df):
        """Assembles a training data in a format that is able to be ingested by the
        Keras CNN model.

        Args:
            dataset (pyeem.datasets.Dataset): The PyEEM dataset being used to
                generate training data.
            ss_results_df (pandas.DataFrame): The augmented single source spectra results.
            mix_results_df (pandas.DataFrame): The augmented mixture spectra results.

        Returns:
            tuple of numpy.ndarray: The formatted training data to be used in
            pyeem.analysis.models.RutherfordNet.train()
        """
        sources = list(dataset.calibration_sources.keys())
        aug_results_df = pd.concat([ss_results_df, mix_results_df])
        aug_df = []
        for p in aug_results_df.index.get_level_values("hdf_path").unique().to_list():
            aug_df.append(pd.read_hdf(dataset.hdf, key=p))

        aug_df = pd.concat(aug_df)

        drop_indices = list(aug_df.index.names)
        keep_indices = sources + ["source", "emission_wavelength"]
        for keep in keep_indices:
            drop_indices.remove(keep)

        X, y = [], []

        aug_df.index = aug_df.index.droplevel(drop_indices)
        # shuffle
        aug_df = aug_df.sample(frac=1)
        for concentrations, eem_df in aug_df.groupby(
            sources + ["source"], as_index=False
        ):
            drop_indices = list(eem_df.index.names)
            drop_indices.remove("emission_wavelength")
            eem_df.index = eem_df.index.droplevel(drop_indices)

            eem_np = eem_df.values
            eem_np = eem_np.reshape(eem_df.shape[0], eem_df.shape[1], 1)

            X.append(eem_np)
            y.append(concentrations[:-1])

        X = np.asarray(X)
        y = np.asarray(y)

        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        y = y[randomize]

        return X, y

    def _isolate_test_samples(self, dataset, routine_results_df):
        # Isolate test samples from the metadata
        samples = dataset.meta_df[dataset.meta_df["test_sample"]].xs(
            "sample_eem", level="scan_type", drop_level=False
        )
        samples.rename(columns={"hdf_path": "raw_hdf_path"}, inplace=True)

        # Isolate sample EEMs from preprocessing routine results
        samples_rr_df = routine_results_df.xs(
            "sample_eem", level="scan_type", drop_level=False
        )

        # Filter out samples which failed any step in the preprocessing routine
        samples_rr_df = samples_rr_df.groupby(
            level=["sample_set", "scan_type", "name"]
        ).filter(lambda x: x["step_completed"].all())

        # Isolate the complete step
        samples_rr_df = samples_rr_df.xs(
            "complete", level="step_name", drop_level=False
        )[["step_completed", "hdf_path", "units"]]
        samples_rr_df = samples_rr_df.reset_index(level=["name", "step_name"])

        # Join the filtered metadata and filtered preprocessing routine results
        test_samples_df = pd.merge(
            samples, samples_rr_df, on=["sample_set", "scan_type", "name"]
        )
        test_samples_df.rename(columns={"units": "intensity_units"}, inplace=True)

        # Get the calibration sources
        sources = list(dataset.calibration_sources.keys())
        # Get the calibration source units. There should only be one unique value here.
        source_units = list(set(dataset.calibration_sources.values()))
        if len(source_units) != 1:
            raise Exception(
                "All calibration/test sources are must reported in the same units."
            )
        source_units = source_units[0]
        test_samples_df["source_units"] = source_units

        # Filter out columns not of interest
        keep_cols = ["intensity_units", "hdf_path", "source_units"]
        keep_cols += sources
        test_samples_df = test_samples_df[keep_cols].reset_index(drop=True)

        def _get_source(row):
            row_df = row.to_frame().T[sources]
            test_sources = row_df.columns[row_df[sources].any()].values

            source = np.nan
            if len(test_sources) == 1:
                source = test_sources[0]
            elif len(test_sources) > 1:
                source = "mixture"

            return source

        # Get the source name for each test sample.
        test_samples_df["source"] = test_samples_df.apply(_get_source, axis=1)
        # Sort the columns for asthetic reasons
        sort_cols = ["source"] + sources
        test_samples_df.sort_values(sort_cols, inplace=True, ignore_index=True)
        test_samples_df = test_samples_df.set_index(
            ["source", "source_units", "intensity_units", "hdf_path"]
        )

        return test_samples_df

    def get_test_data(self, dataset, routine_results_df):
        """Assembles the test data in a format that is able to be ingested by the
        Keras CNN model. This data will be fed into the trained CNN for it to
        make predictions with.

        Args:
            dataset (pyeem.datasets.Dataset): The PyEEM dataset being used to
                generate test data.
            routine_results_df (pandas.DataFrame): The results of the preprocessing routine.

        Returns:
            tuple of numpy.ndarray: The formatted test data to be used in
            pyeem.analysis.models.RutherfordNet.model.predict()
        """
        test_samples_df = self._isolate_test_samples(dataset, routine_results_df)

        sources = (
            test_samples_df.index.get_level_values("source").unique().dropna().values
        )
        sources = np.delete(sources, np.where(sources == "mixture"))

        X = []
        y = []

        for hdf_path, group in test_samples_df.groupby(level="hdf_path"):
            eem_df = pd.read_hdf(dataset.hdf, key=hdf_path)
            eem_np = eem_df.values
            eem_np = eem_np.reshape(eem_df.shape[0], eem_df.shape[1], 1)
            concentrations = group[sources].values[0]

            X.append(eem_np)
            y.append(concentrations)

        return np.asarray(X), np.asarray(y)

    def prepare_data(self, dataset, ss_results_df, mix_results_df, routine_results_df):
        """Assembles both training and test data in a format that is able to be ingested by the
        Keras CNN model.

        Args:
            dataset (pyeem.datasets.Dataset): A PyEEM dataset
            ss_results_df (pandas.DataFrame): The augmented single source spectra results.
            mix_results_df (pandas.DataFrame): The augmented mixture spectra results.
            routine_results_df (pandas.DataFrame): The results of the preprocessing routine.

        Returns:
            tuple of (tuple of numpy.ndarray): Training and test data.
        """
        x_train, y_train = self.get_training_data(
            dataset, ss_results_df, mix_results_df
        )
        x_test, y_test = self.get_test_data(dataset, routine_results_df)
        return (x_train, y_train), (x_test, y_test)

    def train(self, X, y, fit_kws={}):
        """Train the CNN model with a call to Keras' fit().

        Args:
            X (numpy.ndarray): Training Spectra.
            y (numpy.ndarray): Concentration labels.
            fit_kws (dict, optional): Additional key word arguments which will be used in the
                call to Kera's fit(). Defaults to {}.

        Returns:
            tensorflow.python.keras.callbacks.History: The model's training history which contains information
            about model accuracy and loss across training epochs.
        """
        default_fit_kws = dict(
            batch_size=32, epochs=5, validation_split=0.3, shuffle=True
        )
        fit_kws = dict(default_fit_kws, **fit_kws)
        history = self.model.fit(X, y, **fit_kws)
        return history

    def get_prediction_results(self, dataset, predictions, y):
        cal_sources = list(dataset.calibration_sources.keys())
        true_df = pd.DataFrame(y, columns=cal_sources)
        pred_df = pd.DataFrame(predictions, columns=cal_sources)

        results_df = pd.DataFrame()
        for source, units in dataset.calibration_sources.items():
            tmp_df = pd.concat(
                [
                    true_df[source].to_frame(name="true_concentration"),
                    pred_df[source].to_frame(name="predicted_concentration"),
                ],
                axis=1,
            )
            tmp_df[["source", "units"]] = source, units
            (
                tmp_df["slope"],
                tmp_df["intercept"],
                tmp_df["r_value"],
                _,
                _,
            ) = stats.linregress(
                tmp_df["true_concentration"], tmp_df["predicted_concentration"]
            )
            tmp_df["r_squared"] = tmp_df["r_value"] ** 2
            tmp_df = tmp_df.set_index(
                ["source", "units", "slope", "intercept", "r_squared"]
            )
            tmp_df = tmp_df.drop(columns="r_value")
            results_df = pd.concat([results_df, tmp_df])

        return results_df
