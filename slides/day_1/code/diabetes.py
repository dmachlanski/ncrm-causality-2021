import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, tree, ensemble, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
# Code source: Jaques Grobler
# License: BSD 3 clause

from sklearn.model_selection import cross_val_score

if __name__ == "__main__":

    #df = pd.read_csv("./diab_class.csv")
    #x_columns = list(df.columns[:])
    #x_columns.remove("Outcome")

    #df_X = df[[df.colums]]
    #print(df[x_columns].head().to_latex(float_format="%.2f"))
    #print(df[["Outcome"]].head().to_latex(float_format="%.2f"))
    #

    #clf = tree.DecisionTreeClassifier(max_depth=2)
    #clf.fit(df[x_columns], df["Outcome"])

    #print(clf.coef_, clf.intercept_)
    #tree.plot_tree(clf, feature_names = x_columns, class_names = ["0","1"])
    # plt.figure(figsize=(16, 9))
    # plt.tight_layout()
    #plt.show()
    #plt.savefig("./tmp/diab_class_tree.pdf")




    #Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    diabetes_X = diabetes_X * diabetes_X.std(axis = 0)
    #print(diabetes_X.head().to_latex(float_format="%.2f"))
    clf = linear_model.LinearRegression()
    import sklearn
    print(sorted(sklearn.metrics.SCORERS.keys()))

    X_train, X_test, y_train, y_test = train_test_split( diabetes_X, diabetes_y, test_size = 0.2, random_state = 0)
    clf.fit(X_train, y_train)

    scores = -cross_val_score(clf, diabetes_X, diabetes_y, cv=5, scoring="neg_mean_squared_error")
    print(scores, scores.mean(), scores.std())

    clf = tree.DecisionTreeRegressor()

    X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.2, random_state=0)
    clf.fit(X_train, y_train)

    scores = -cross_val_score(clf, diabetes_X, diabetes_y, cv=5, scoring="neg_mean_squared_error")
    print(scores, scores.mean(), scores.std())

    alphas = np.logspace(-4, -0.5, 30)


    print("| alpha  | scores  | mean | std |")
    print("|---|---|---|---|")
    for alpha in alphas:

        clf = linear_model.Lasso(alpha=alpha)
        scores =  -cross_val_score(clf, diabetes_X, diabetes_y, cv=5, scoring="neg_mean_squared_error")
        print("%.4f"%(alpha), "|",["%.0f"%(score) for score in scores],"|", "%.4f"%(scores.mean()), "|", "%.4f"%(scores.std()))

    # params = {'n_estimators': 500,
    #           'max_depth': 1,
    #           'min_samples_split': 5,
    #           'learning_rate': 0.01,
    #           'loss': 'ls'}
    # clf = ensemble.GradientBoostingRegressor(**params)
    #
    # X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.2, random_state=0)
    # clf.fit(X_train, y_train)
    #
    # scores = -cross_val_score(clf, diabetes_X, diabetes_y, cv=5, scoring="neg_mean_squared_error")
    # print(scores, scores.mean(), scores.std())

    #print(clf.coef_, clf.intercept_)



