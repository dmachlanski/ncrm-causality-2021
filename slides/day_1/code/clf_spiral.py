import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from sklearn import tree



def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))



def filter_one(X,y,a1_max, a1_min, a2_max, a2_min):
    X_new = []
    y_new = []
    for (a1,a2), c in zip(X,y):
        #print(a1,a2,c)
        if(a1 < a1_max and a1 > a1_min):
            if(a2 < a2_max and a2 > a2_min):
                X_new.append([a1,a2])
                y_new.append(c)
    X = np.array(X_new)
    y = np.array(y_new)
    return X,y


def clear():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

if __name__ == "__main__":
    for clf in [LinearRegression(), DecisionTreeRegressor(max_depth=3), RandomForestRegressor(max_depth=3), CatBoostRegressor(verbose=False, max_depth=3)]:
        for n_points in [200,2000]:
            np.random.seed(100)

            X, y = twospirals(n_points)
            #print(X.max(), X.min())
            name = (clf.__class__.__name__)


            a1_max, a1_min, a2_max, a2_min = 0,-6,8,2.3

            X,y = filter_one(X,y, a1_max, a1_min, a2_max, a2_min)
            #

            clf.fit(X[y == 0,0:1],X[y == 0,1])
            #print(X[y == 0,0:1].shape)
            fake_x = np.array([np.linspace(-13, 13, num=10000)]).T
            #print(fake_x.shape)
            clf.predict(fake_x)

            if(name == "DecisionTreeRegressor"):
                clear()
                #plt.figure(figsize=(16, 9))
                tree.plot_tree(clf)
                #plt.figure(figsize=(16, 9))
                #plt.tight_layout()
                plt.savefig("./tmp/reg_vis_tree_intra_spiral_%d_%s.pdf"%(n_points,clf.__class__.__name__))
                clear()

            plt.title('training set')
            axes = plt.gca()
            axes.set_xlim([-13, 13])
            axes.set_ylim([-13, 13])

            plt.scatter(X[y == 0, 0], X[y == 0, 1], 1, label='data',c = "red")
            plt.scatter(fake_x, clf.predict(fake_x), 1, label="Regressor", c="gray", alpha=0.1)

            plt.legend()
            name = "./tmp/reg_intra_spiral_%d_%s.pdf"%(n_points,clf.__class__.__name__)
            print(name)
            plt.savefig(name)
            clear()



    for clf in [LogisticRegression(), DecisionTreeClassifier(max_depth=3), RandomForestClassifier(max_depth=3), CatBoostClassifier(verbose=False, max_depth=3)]:
        for n_points in [200,2000]:
            np.random.seed(100)

            X, y = twospirals(n_points)
            #print(X.max(), X.min())
            name = (clf.__class__.__name__)




            clf.fit(X, y)
            #print(X[y == 0,0:1].shape)
            fake_x = np.array([np.linspace(-13, 13, num=1000)]).T
            print(fake_x.shape)
            twodfake_x = np.array(list(zip(fake_x,fake_x)))
            clf.predict(fake_x)

            if(name == "DecisionTreeClassifier"):
                clear()
                #plt.figure(figsize=(16, 9))
                tree.plot_tree(clf)
                #plt.figure(figsize=(16, 9))
                #plt.tight_layout()
                plt.savefig("./tmp/class_vis_tree_intra_spiral_%d_%s.pdf"%(n_points,clf.__class__.__name__))
                clear()

            plt.title('training set')
            axes = plt.gca()
            axes.set_xlim([-13, 13])
            axes.set_ylim([-13, 13])

            plt.scatter(X[y == 0, 0], X[y == 0, 1], 1, label='data',c = "red")
            plt.scatter(fake_x, clf.predict(fake_x), 1, label="Regressor", c="gray", alpha=0.1)

            plt.legend()
            name = "./tmp/class_intra_spiral_%d_%s.pdf"%(n_points,clf.__class__.__name__)
            print(name)
            plt.savefig(name)
            clear()
