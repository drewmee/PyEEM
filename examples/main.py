# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats

# import seaborn as sns
import pandas as pd
import numpy as np
import time
import pyeem

#%%
# Check out the supported instruments
display(pyeem.instruments.supported)

# Load the Rutherford et al. dataset and check out the metadata
demo_dataset = pyeem.datasets.load_rutherford("demo_data/")
display(demo_dataset.meta_df)
display(demo_dataset.hdf)

# Load the drEEM dataset and check out the metadata
# demo_dataset = pyeem.datasets.load_dreem("demo_data/")
# display(demo_dataset.meta_df)
#%%
# Run the preprocessing routine which includes several
# filtering and correction steps.
crop_dimensions = {"emission_bounds": (246, 573), "excitation_bounds": (224, np.inf)}
pyeem.preprocessing.routine(
    demo_dataset.meta_df,
    demo_dataset.hdf,
    crop_dims=crop_dimensions,
    raman_source="metadata",
)
#%%
# Visualize each step of the corrections.
# pyeem.visualization.corrections()

# Visualize the water raman peak characterization
# pyeem.visualization.water_raman()

# Visualize the calibration functions
# pyeem.visualization.calibration()

# Run the augmentation routine which includes creating prototypical
# spectra from the calibration. These spectra are then scaled across
# a concentration range and mixed together to create a large number of
# single source and mixture spectra.
# pyeem.augmentation.routine()

# Visualize the augmented spectra
# pyeem.visualization.augmentations()

# pyeem.analysis.basic.
# pyeem.analysis.timeseries.
# pyeem.analysis.models.rutherfordnet()
# pyeem.analysis.models.pca()
# pyeem.analysis.models.lda()
# pyeem.analysis.models.parafac()
# %%
sources = np.sort(meta_df["prototypical_source"].dropna().unique())

proto_groups = (
    meta_df[meta_df["prototypical_source"].notna()]
    .sort_values("prototypical_source")
    .groupby("prototypical_source")
)
# %%
for source_name, group in proto_groups:
    # This can be put into augment.py for the functions
    # proto_spectra/single_source/mixture to  use
    c = cal_df[cal_df["source"] == source_name]
    proto_eem = pyeem.prototypical_spectra(source_name, group, c, hdf)

    display(proto_eem)
    break
    # These calls should not be here, put into a function within
    # visualizations.py... something like parse(df_type="proto/ss/mix")
    conc = proto_eem.index.get_level_values("proto_conc").unique().item()
    source = proto_eem.index.get_level_values("source").unique().item()
    proto_eem.index = proto_eem.index.droplevel(["source", "proto_conc"])

    plt_title = "Prototypical Spectrum: {0}\n".format(source_name)
    plt_title += "Concentration (ug/ml): {0}".format(conc)
    pyeem.Visualizations().contour_plot(proto_eem, plt_title)
    pyeem.Visualizations().combined_surface_contour_plot(proto_eem, plt_title)

    s = pyeem.single_sources(
        source_name, sources, c, hdf, conc_range=(0, 5), num_spectra=1000
    )

    # pyeem.Visualizations().single_source_animation(source_name, s)

# %%
# m = pyeem.mixtures(sources, cal_df, hdf, conc_range=(0, 5), num_steps=15)
m = pyeem.mixtures(
    sources, cal_df, hdf, conc_range=(0.01, 6.3), num_steps=15, scale="logarithmic"
)
# pyeem.Visualizations().mix_animation(sources, m)

t1 = time.time()
total = t1 - t0
print(total)

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
