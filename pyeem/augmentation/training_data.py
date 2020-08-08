import numpy as np
import pandas as pd


# This should be rolled into pyeem.analysis.models
# Since it's model specific
def create_train_data(aug_ss, aug_mix):
    train_arr = np.concatenate([aug_ss, aug_mix])
    np.random.shuffle(train_arr)

    X = np.stack(train_arr[:, 0])
    y = np.stack(train_arr[:, 1])

    # width=142; height=139
    # TODO - These values can't be hardcoded...
    X = np.array(X).reshape(-1, 250, 151, 1)
    return (X, y)


def create_test_data():
    test_arr = []
    for name, group in meta_df.groupby("Type"):
        if "Test" not in name:
            continue

        for row_index, row in group.iterrows():
            fname = poststardom_dir + row.File_Name + ".csv"
            data = pd.read_csv(fname, index_col=0)
            data.columns = data.columns.map(int)
            data = data.truncate(before=246, after=573, axis=0)  # rows:emission
            data = data.truncate(before=224, axis=1)  # column:excitation
            data = data.to_numpy()
            # width=142; height=139
            # TODO - These values can't be hardcoded...
            data = np.array(data).reshape(-1, 142, 139, 1)
            display(data.shape)
            display(data)
            labels = row[[i.split("_")[1] for i in SOURCES]].to_numpy()
            test_arr.append([data, labels])

    test_arr = np.asarray(test_arr)
