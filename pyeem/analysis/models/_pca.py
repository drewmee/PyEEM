from sklearn.decomposition import FastICA, PCA
from sklearn import datasets as sklearn_datasets

# def ICA():
# https://stats.stackexchange.com/questions/267497/using-only-one-mic-in-cocktail-party-algorithm


def pca_summary(pca, standardised_data, out=True):
    names = ["PC" + str(i) for i in range(1, len(pca.explained_variance_ratio_) + 1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [
        np.sum(pca.explained_variance_ratio_[:i])
        for i in range(1, len(pca.explained_variance_ratio_) + 1)
    ]
    columns = pd.MultiIndex.from_tuples(
        [
            ("sdev", "Standard deviation"),
            ("varprop", "Proportion of Variance"),
            ("cumprop", "Cumulative Proportion"),
        ]
    )
    summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
    if out:
        print("Importance of components:")
        print(summary)
    return summary


def screeplot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0) ** 2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp." + str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()


def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color="r", alpha=0.5)
        if labels is None:
            plt.text(
                coeff[i, 0] * 1.15,
                coeff[i, 1] * 1.15,
                "Var" + str(i + 1),
                color="green",
                ha="center",
                va="center",
            )
        else:
            plt.text(
                coeff[i, 0] * 1.15,
                coeff[i, 1] * 1.15,
                labels[i],
                color="g",
                ha="center",
                va="center",
            )

    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.show()


def perform_pca(X, y, target_names):
    iris = sklearn_datasets.load_iris()

    sc = StandardScaler()
    # X = sc.fit_transform(iris.data)
    X = sc.fit_transform(X)
    # y = iris.target
    # target_names = iris.target_names

    pca = PCA().fit(X)
    X_r = pca.transform(X)
    summary = pca_summary(pca, X)
    print(pca.components_[0])
    screeplot(pca, X)

    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_r[y == i, 0],
            X_r[y == i, 1],
            color=color,
            alpha=0.8,
            lw=lw,
            label=target_name,
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of IRIS dataset")
    plt.show()

    # biplot(pca[:,0:2],np.transpose(pcamodel.components_[0:2, :]),list(x.columns))

    return
