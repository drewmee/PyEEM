from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def perform_lda(X1, y1, target_names):
    iris = sklearn_datasets.load_iris()

    # X = iris.data
    # y = iris.target
    # target_names = iris.target_names

    # print(X.shape)
    # print(y.shape)
    # print(target_names)

    X = X1
    y = y1
    print(X1.shape)
    print(y1.shape)

    lda = LDA(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # z_labels = lda.predict(Z) #gives you the predicted label for each sample
    # z_prob = lda.predict_proba(Z) #the probability of each sample to belong to each class
    print(lda.scalings_)

    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA of IRIS dataset")
    plt.show()

    def pretty_scalings(lda, X, out=False):
        ret = pd.DataFrame(
            lda.scalings_,
            index=X.columns,
            columns=["LD" + str(i + 1) for i in range(lda.scalings_.shape[1])],
        )
        if out:
            print("Coefficients of linear discriminants:")
            print(ret)
        return ret

    pretty_scalings_ = pretty_scalings(lda, X, out=True)

    """
    #https://stackoverflow.com/questions/13973096/how-do-i-get-the-components-for-lda-in-scikit-learn
    def myplot(score,coeff,labels=None):
        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]

        plt.scatter(xs ,ys, c = y) #without scaling
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
            if labels is None:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

    plt.xlabel("LD{}".format(1))
    plt.ylabel("LD{}".format(2))
    plt.grid()

    #Call the function. 
    myplot(X_r2[:,0:2], lda.scalings_) 
    plt.show()
    """
    return
