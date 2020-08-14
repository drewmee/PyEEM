# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import pyeem

#%%

# 1 DOCSTRINGS!
# 2 Refactor preprocessing routine
# 3 Finalize preprocessing steps (Raman, QS units, spectral corrections)
# 4 Refactor dataset.load() and create tests
# 5 Rutherford model - training data, train, analyze
# 6 Paper
# 7 Contact rutherford

# Download a demo dataset from S3
demo_data_dir = pyeem.datasets.download_demo(
    "examples/demo_data", demo_name="rutherford"
)


# demo_data_dir = "examples/demo_data/rutherford"
cal_sources = {"cigarette": "ug/ml", "diesel": "ug/ml", "wood_smoke": "ug/ml"}
dataset = pyeem.datasets.Load(
    data_dir=demo_data_dir,
    raman_instrument=None,
    absorbance_instrument="aqualog",
    eem_instrument="aqualog",
    calibration_sources=cal_sources,
    mode="w",
)
"""
demo_data_dir = "examples/demo_data/drEEM"
dataset = pyeem.datasets.Load(
    data_dir=demo_data_dir,
    raman_instrument="fluorolog",
    eem_instrument="fluorolog",
    absorbance_instrument="cary_4e",
    progress_bar=True,
)
"""
# Run the preprocessing routine which includes several
# filtering and correction steps.
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

# discrete_ex_wl = [225, 240, 275, 290, 300, 425]
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


i = 0
for name, group in routine_results_df.groupby(level=["sample_set", "name"]):
    sample_set = name[0]
    sample_name = name[1]
    kwargs = {}
    axes = pyeem.plots.plot_preprocessing(
        dataset,
        routine_results_df,
        sample_set=sample_set,
        sample_name=sample_name,
        plot_type="surface_contour",
        fig_kws={"dpi": 200},
    )
    plt.show()
    i += 1
    if i == 2:
        break

cal_df = pyeem.preprocessing.calibration(dataset, routine_results_df)
cal_summary_df = pyeem.preprocessing.calibration_summary_info(cal_df)
axes = pyeem.plots.plot_calibration_curves(dataset, cal_df)
plt.show()

proto_results_df = pyeem.augmentation.create_prototypical_spectra(dataset, cal_df)
axes = pyeem.plots.plot_prototypical_spectra(
    dataset, proto_results_df, plot_type="contour"
)
plt.show()

ss_results_df = pyeem.augmentation.create_single_source_spectra(
    dataset, cal_df, conc_range=(0, 5), num_spectra=100
)

source = "wood_smoke"
anim = pyeem.plots.single_source_animation(
    dataset,
    ss_results_df,
    source=source,
    plot_type="contour",
    fig_kws={"dpi": 175},
    animate_kws={"interval": 100, "blit": True},
)

mix_results_df = pyeem.augmentation.create_mixtures(
    dataset, cal_df, conc_range=(0.01, 6.3), num_steps=5
)
anim = pyeem.plots.mixture_animation(
    dataset,
    mix_results_df,
    plot_type="imshow",
    fig_kws={"dpi": 175},
    animate_kws={"interval": 100, "blit": True},
)

# pyeem.plots.plot_preprocessing(dataset, routine_results_df, animate=False)

# Visualize Raman peak characterization
# pyeem.plots.raman_peak_characterization()

# Time-series analysis of various fluorescence indices
# pyeem.plots.timeseries(dataset, timeframe=() metric="")

# Download as zip or R or matlab data files
# dataset.download()

# Perfrom calibration if there are any sources.
cal_df = pyeem.preprocessing.calibration(dataset, routine_results_df)
# cal_df.to_pickle("cal_df.pkl")
# cal_df = pd.read_pickle("cal_df.pkl")

# Visualize the calibration functions
pyeem.plots.plot_calibration_curves(dataset, cal_df)
#%%
# Visualize the calibration spectra
# pyeem.plots.plot_calibration_spectra(cal_df)

# Get the calibration summary information
pyeem.preprocessing.calibration_summary_info(cal_df)


results_df = pyeem.augmentation.create_prototypical_spectra(dataset, cal_df)
pyeem.plots.augmentations.plot_prototypical_spectra(
    dataset, results_df, plot_type="surface"
)


results_df = pyeem.augmentation.create_single_source_spectra(
    dataset, cal_df, conc_range=(0, 5), num_spectra=100
)
"""
results_df = pyeem.plots.augmentations.plot_single_source_spectra(
    dataset, results_df, conc_range=(0.01, 6.3), num_steps=15, scale="logarithmic"
)
"""
results_df = pyeem.augmentation.create_mixtures(
    dataset, cal_df, conc_range=(0.01, 6.3), num_steps=15
)
# pyeem.plots.augmentations.plot_mixture_spectra(dataset, results_df)

# rutherford_net = pyeem.analysis.models.RutherfordNet()
# rutherford_net.create_training_data(dataset)
# rutherford_net.train()

# %%
sources = list(np.sort(meta_df["prototypical_source"].dropna().unique()))


def prepare_training_data(df, sources, X, y):
    for name, group in df.groupby(level=sources):
        eem = group.values
        eem = eem.reshape(eem.shape[0], eem.shape[1], 1)
        X.append(eem)

        label = group.droplevel("emission_wavelength")
        label = label.index.unique().item()
        y.append(label)
    return X, y


X_train = []
y_train = []
for source in sources:
    path = os.path.join(*["augmented", "single_sources", source])
    s = pd.read_hdf(hdf, key=path)
    s = s.droplevel("source")
    X_train, y_train = prepare_training_data(s, sources, X_train, y_train)

m = pd.read_hdf(hdf, key=os.path.join(*["augmented", "mixtures"]))
X_train, y_train = prepare_training_data(m, list(sources), X_train, y_train)

X_train = np.array(X_train)
y_train = np.array(y_train)

model = pyeem.analyze.rutherfordnet(X_train, y_train)

# %%
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

model = tf.keras.models.load_model("rutherford-net.h5")

X_test = []
y_test = []
for sample_set, row in meta_df[meta_df["validation_sources"].notna()].iterrows():

    path = os.path.join(
        *[
            "corrections",
            "sample_sets_raman_normalization",
            str(sample_set[0]),
            row["filename"],
        ]
    )
    t = pd.read_hdf(hdf, key=path)
    t = t.values
    t = t.reshape(t.shape[0], t.shape[1], 1)
    X_test.append(t)

    label = row[sources].to_numpy()
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)
test_p = model.predict(X_test)

train_p = model.predict(X_train)
train_pred_df = pd.DataFrame(data=train_p, columns=sources)
train_true_df = pd.DataFrame(data=y_train, columns=sources)

test_pred_df = pd.DataFrame(data=test_p, columns=sources)
test_true_df = pd.DataFrame(data=y_test, columns=sources)
for source in sources:
    print(source)
    x1 = test_true_df[source]
    y1 = test_pred_df[source]

    x2 = train_true_df[source]
    y2 = train_pred_df[source]

    df = pd.concat(
        [
            pd.DataFrame({"true": x1, "pred": y1, "category": "test"}),
            pd.DataFrame({"true": x2, "pred": y2, "category": "train"}),
        ],
        ignore_index=True,
    )
    df["true"] = df["true"].astype("float64")
    df["pred"] = df["pred"].astype("float64")

    g = sns.lmplot(
        x="true",
        y="pred",
        hue="category",
        data=df,
        ci=None,
        palette=dict(test="darkorange", train="lightblue"),
    )
    g = g.set_axis_labels("True Conc. (ug/ml)", "Predicted Conc. (ug/ml)").set(
        xlim=(-1, 7), ylim=(-1, 7)
    )
    plt.show()

# display(pd.concat([true_df, pred_df], axis=1))
# %%


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2


sns.set(font_scale=1.5)
sns.set_style("whitegrid")
for source in sources:

    x2 = df[df["category"] == "test"][source + "_true"].values.tolist()
    y2 = df[df["category"] == "test"][source + "_pred"].values.tolist()

    g = sns.lmplot(
        x=source + "_true",
        y=source + "_pred",
        data=df,
        hue="category",
        palette=dict(test="darkorange", train="lightblue"),
        ci=None,
    )
    plt.title(source)
    g = g.set_axis_labels("True Conc. (ug/ml)", "Predicted Conc. (ug/ml)").set(
        xlim=(-1, 7), ylim=(-1, 7)
    )
    print(round(r2(x2, y2), 3))

    plt.show()
# %%
